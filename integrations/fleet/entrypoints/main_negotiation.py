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
    JUDGE_EVAL: "true"/"false" (default true) - toggle the in-loop LLM-as-judge
        deception probe, which scores the policy messages produced inside the
        probe games above (measurement only; never a reward). Paid but cheap.
    JUDGE_MODEL: OpenRouter slug for the judge (default openai/gpt-4.1-mini,
        calibrated on past traces; gpt-4o-mini over-flags honest bargaining).
Plus all S3 vars honored by FleetPPOExp. Both probes need OPENROUTER_API_KEY
(the judge calls OpenRouter; the probe opponent is free/scripted).
"""

import logging
import os
import sys

import ray

from integrations.fleet.entrypoints.main_fleet import FleetPPOExp, _strip_hydra_prefixes
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

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
        probe_payload = None
        try:
            probe_metrics, probe_payload = await self._run_exploitation_probe()
            eval_metrics.update(probe_metrics)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"In-loop exploitation probe failed (skipping): {e}", exc_info=True)

        # In-loop LLM-as-judge deception probe: score the policy messages produced
        # inside the probe games above. Measurement only (never a reward). Paid but
        # cheap (a few dozen single-shot classifications per eval).
        if probe_payload is not None and _truthy(os.environ.get("JUDGE_EVAL", "true")):
            try:
                judge_metrics = await self._run_deception_judge(probe_payload)
                eval_metrics.update(judge_metrics)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"In-loop deception judge failed (skipping): {e}", exc_info=True)

        return eval_metrics

    async def _run_exploitation_probe(self):
        """Play the live policy vs the scripted conceder and emit eval/probe/* metrics.

        Returns ``(metrics, payload)`` where ``payload`` carries the per-game runs
        (incl. ``policy_msgs``) so the deception judge can reuse them.
        """
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
        # A server bind-all address is NOT a valid client connect target; the
        # endpoint runs on this (head) node, so connect via loopback.
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        port = getattr(ie_cfg, "http_endpoint_port", None) or 8000
        base_url = f"http://{host}:{port}/v1"

        # Match the policy's training-time thinking mode. Thinking is ON by default
        # (ENABLE_THINKING=true); the probe then leaves reasoning enabled (no_think=False)
        # and run_probe strips <think> from the policy's own multi-turn context, mirroring
        # training's qwen3_without_thinking template -> "thinking on but stripped".
        enable_thinking = _truthy(os.environ.get("ENABLE_THINKING", "true"))
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
        # Give the think channel the same per-turn generation budget as training so a
        # long <think> can't truncate before the action tag (which would read as no_deal).
        max_tokens = int(os.environ.get("MAX_GENERATE_LENGTH", "4096"))

        payload = await run_probe.run_probe(
            [participant],
            dataset=dataset,
            split="val",
            n=n,
            max_turns=max_turns,
            temperature=0.7,
            max_tokens=max_tokens,
            protocol=protocol,
            seed=1,
            write=False,  # skip disk + matplotlib; we only want the metrics
        )

        _probe_runs = (payload.get("per_model_runs") or {}).get("Policy", [])
        _probe_errs = [r.get("error") for r in _probe_runs if r.get("error")]
        if _probe_errs:
            logger.warning(
                f"[probe] {len(_probe_errs)}/{len(_probe_runs)} probe games ERRORED "
                f"(endpoint={base_url} model={served}) -> eval/probe/* and the deception "
                f"judge will be ~0. Sample error: {_probe_errs[0]!r}"
            )

        agg = payload["per_model"]["Policy"]["aggregate"]
        metrics = {}
        for key in (
            "opp_norm",  # HEADLINE: pushover's score; lower = squeezed harder
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
        return metrics, payload

    async def _run_deception_judge(self, payload):
        """LLM-as-judge deception probe: score the policy's opponent-facing messages
        from the probe games and emit eval/deception_judge/* metrics.

        Measurement only -- this never feeds the reward. The judge is a cheap model
        (JUDGE_MODEL, default openai/gpt-4.1-mini) called via OpenRouter; it was
        calibrated on past traces (.overnight/judge_modelcmp_out.json).
        """
        import skyrl_gym.envs.negotiation as _neg

        eval_dir = os.path.join(os.path.dirname(_neg.__file__), "eval")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        import deception_judge  # noqa: E402
        import run_eval  # noqa: E402  (script-style module)

        runs = (payload.get("per_model_runs") or {}).get("Policy", [])
        if not runs:
            logger.info("[deception_judge] no probe runs to judge; skipping")
            return {}

        model = os.environ.get("JUDGE_MODEL", deception_judge.DEFAULT_JUDGE_MODEL)
        client = run_eval.make_client("https://openrouter.ai/api/v1")
        metrics, _verdicts = await deception_judge.judge_probe_runs(runs, client, model)

        logger.info(
            f"[deception_judge] step={self.global_step} model={model} "
            f"n_msgs={metrics.get('eval/deception_judge/n_msgs')} "
            f"deception_rate={metrics.get('eval/deception_judge/deception_rate')} "
            f"omission_rate={metrics.get('eval/deception_judge/omission_rate')} "
            f"false_promise_rate={metrics.get('eval/deception_judge/false_promise_rate')}"
        )
        return metrics


class NegotiationPPOExp(FleetPPOExp):
    """FleetPPOExp (S3 checkpoint management) wired to the negotiation trainer."""

    atof_entrypoint = "main_negotiation"

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
