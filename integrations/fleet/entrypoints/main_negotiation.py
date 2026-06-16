"""
Negotiation training entrypoint for SkyRL.

Identical to ``main_fleet`` (S3 checkpoint management via ``FleetPPOExp``) but uses
a trainer subclass that ALSO runs the free, local exploitation probe inside the
in-loop eval cycle, logging ``eval/probe/*`` to wandb.

The probe plays the live policy (served over vLLM's OpenAI-compatible HTTP
endpoint at ``http://<host>:<port>/v1``) against a pure-Python scripted conceder
-> $0, no external API calls. It runs every eval cycle, blocking, during the eval
window when the colocated inference engine is awake with freshly-synced weights.

Paid evals (cross-play, NegotiationArena) intentionally stay offline/manual.

Usage:
    python -m integrations.fleet.entrypoints.main_negotiation \
        environment.env_class=negotiation \
        generator.inference_engine.enable_http_endpoint=true \
        generator.inference_engine.served_model_name=policy \
        ...

Env vars (read by the trainer subclass):
    PROBE_EVAL: "true"/"false" (default true) - toggle the in-loop probe
    PROBE_N: scenarios per eval (default 16)
    PROBE_DATASET: scenario dataset (default dnd)
    NEGOTIATION_PROTOCOL: single|dual (default single) - must match training
    ENABLE_THINKING: if "true", the policy emits <think> (probe no_think follows)
Plus all S3 vars honored by FleetPPOExp.
"""

import logging
import os
import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from integrations.fleet.entrypoints.main_fleet import FleetPPOExp, _strip_hydra_prefixes

logger = logging.getLogger(__name__)


def _truthy(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")


class NegotiationRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer that additionally runs the (free, local) exploitation probe
    in the in-loop eval cycle against the live policy, logging ``eval/probe/*``.
    """

    async def eval(self):
        # Standard env-loop eval first (e.g. eval/negotiation_synthetic/*). Weight
        # sync has already happened, so the inference engine is awake with current
        # weights -> the HTTP endpoint is reachable for the probe below.
        eval_metrics = await super().eval()

        if not _truthy(os.environ.get("PROBE_EVAL", "true")):
            return eval_metrics

        # Never let an eval-side failure kill training.
        try:
            probe_metrics = await self._run_exploitation_probe()
            eval_metrics.update(probe_metrics)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"In-loop exploitation probe failed (skipping): {e}", exc_info=True)

        return eval_metrics

    async def _run_exploitation_probe(self):
        """Play the live policy vs the scripted conceder and emit eval/probe/* metrics."""
        # The eval harness is a script-style module (relative imports + sys.path
        # tricks), so add its directory to sys.path and import it lazily.
        import skyrl_gym.envs.negotiation as _neg

        eval_dir = os.path.join(os.path.dirname(_neg.__file__), "eval")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        import run_probe  # noqa: E402  (script-style module)

        ie_cfg = self.cfg.generator.inference_engine
        served = getattr(ie_cfg, "served_model_name", None) or "policy"
        host = getattr(ie_cfg, "http_endpoint_host", None) or "127.0.0.1"
        port = getattr(ie_cfg, "http_endpoint_port", None) or 8000
        base_url = f"http://{host}:{port}/v1"

        # Match the policy's training-time thinking mode.
        enable_thinking = _truthy(os.environ.get("ENABLE_THINKING", "false"))
        participant = {
            "slug": served,
            "label": "Policy",
            "no_think": not enable_thinking,
            "base_url": base_url,
            "role": "policy",
        }

        n = int(os.environ.get("PROBE_N", "16"))
        dataset = os.environ.get("PROBE_DATASET", "dnd")
        protocol = os.environ.get("NEGOTIATION_PROTOCOL", "single")
        max_turns = int(getattr(self.cfg.generator, "max_turns", 6) or 6)

        payload = await run_probe.run_probe(
            [participant],
            dataset=dataset,
            split="val",
            n=n,
            max_turns=max_turns,
            temperature=0.7,
            protocol=protocol,
            seed=1,
            write=False,  # skip disk + matplotlib; we only want the metrics
        )

        agg = payload["per_model"]["Policy"]["aggregate"]
        metrics = {}
        for key in (
            "opp_norm",            # HEADLINE: pushover's score; lower = squeezed harder
            "policy_norm",
            "pool_take_fraction",
            "value_capture",
            "gratuitous_take",
            "agreement_rate",
            "no_deal_rate",
            "avg_turns",
        ):
            val = agg.get(key)
            if val is not None:
                metrics[f"eval/probe/{key}"] = float(val)

        # Exploitation gap vs the step-0 (pre-train) policy: how much LESS the
        # pushover walks away with after training. No paid base model needed -- we
        # cache the first eval's opp_norm (eval_before_train -> global_step 0).
        opp_norm = agg.get("opp_norm")
        if opp_norm is not None:
            base0 = getattr(self, "_probe_step0_opp_norm", None)
            if base0 is None:
                base0 = float(opp_norm)
                self._probe_step0_opp_norm = base0
            metrics["eval/probe/exploitation_gap_vs_step0"] = float(base0 - opp_norm)

        logger.info(
            f"[probe] step={self.global_step} opp_norm={agg.get('opp_norm')} "
            f"pool_take={agg.get('pool_take_fraction')} gratuitous={agg.get('gratuitous_take')} "
            f"agree={agg.get('agreement_rate')}"
        )
        return metrics


class NegotiationPPOExp(FleetPPOExp):
    """FleetPPOExp (S3 checkpoint management) wired to the negotiation trainer."""

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return NegotiationRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    """Ray remote function that runs negotiation training."""
    # negotiation env is auto-registered by skyrl_gym.envs.__init__
    exp = NegotiationPPOExp(cfg)
    exp.run()


def main() -> None:
    """Main entry point for negotiation training."""
    args = _strip_hydra_prefixes(sys.argv[1:])
    cfg = SkyRLTrainConfig.from_cli_overrides(args)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
