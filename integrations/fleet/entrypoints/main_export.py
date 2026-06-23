"""
FSDP -> HuggingFace checkpoint export entrypoint for SkyRL.

Training only persists *sharded* FSDP checkpoints (per-rank
`model_world_size_{N}_rank_{r}.pt` files) under `trainer.ckpt_path`. vLLM
cannot serve those directly. This entrypoint loads a sharded policy checkpoint
back into the distributed model and re-emits it as HuggingFace safetensors
under `trainer.export_path/global_step_<N>/policy`, which `vllm serve` (and the
negotiation eval harnesses) can consume.

It is the offline equivalent of what `trainer.save_models()` does mid-run when
`trainer.hf_save_interval > 0`. Use it for runs that trained without HF export
enabled, or to re-export an older checkpoint.

IMPORTANT: FSDP shard files encode the training `world_size` in their names and
the full-state-dict gather is a collective across all ranks. You must run this
on the *same node/GPU topology* (same world size) the checkpoint was trained
with, otherwise the load will not find its shards.

Usage (mirrors scripts/fleet-export-run.sh):
    # Export the latest checkpoint pulled from S3 for a given run:
    RESUME_RUN_NAME=<wandb_run> \
    python -m integrations.fleet.entrypoints.main_export \
        trainer.policy.model.path=Qwen/Qwen3.5-9B \
        trainer.ckpt_path=$HOME/ckpts/<run> \
        trainer.export_path=$HOME/exports \
        trainer.resume_mode=latest

    # Export a specific local checkpoint directory:
    python -m integrations.fleet.entrypoints.main_export \
        trainer.resume_mode=from_path \
        trainer.resume_path=$HOME/ckpts/<run>/global_step_70 ...

    # Export EVERY local global_step_* dir found under trainer.ckpt_path:
    EXPORT_ALL_LOCAL_STEPS=1 python -m integrations.fleet.entrypoints.main_export ...

Environment Variables:
    RESUME_RUN_NAME: W&B run name to pull the latest FSDP checkpoint from S3.
        If unset, only local checkpoints under trainer.ckpt_path are used.
    EXPORT_ALL_LOCAL_STEPS: If "1"/"true", export every local global_step_* dir
        under trainer.ckpt_path (ignores resume_mode). Otherwise export a single
        checkpoint selected by resume_mode (latest / from_path).
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION / S3_CHECKPOINT_BUCKET:
        S3 credentials/config (only used when RESUME_RUN_NAME is set).
"""

import logging
import os
import sys
from pathlib import Path

import ray

from integrations.fleet.entrypoints.main_eval import (
    FleetEvalExp,
    _strip_hydra_prefixes,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

logger = logging.getLogger(__name__)


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


class FleetExportExp(FleetEvalExp):
    """Load sharded policy checkpoint(s) and re-emit them as HF safetensors.

    Reuses `FleetEvalExp`'s policy-only FSDP load (`_load_policy_only`) and S3
    download/broadcast, but skips weight sync + the eval loop. Instead it calls
    `trainer.save_models()`, which writes HF safetensors to
    `export_path/global_step_<trainer.global_step>/policy`.
    """

    def get_eval_dataset(self):
        """No eval dataset needed for export-only runs."""
        return None

    def get_inference_client(self):
        """Export does a pure FSDP-load -> save_hf; it never generates rollouts, so skip
        building the vLLM inference engines entirely. They are heavy and, for the 35B-A3B
        GDN MoE, flaky to init (16 engines JIT-compiling the GDN kernel). `_setup_trainer`
        builds the client BEFORE `build_models`, so skipping it here avoids that failure;
        `build_models` only reads the inference-engine *config*, not this client object."""
        logger.info("Export run: skipping inference engine creation (not needed for FSDP->HF export).")
        return None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """No generator needed for export (we only call save_models)."""
        return None

    def run(self):
        trainer = self._setup_trainer()

        # Optional S3 resume: download latest FSDP shards on this VM and
        # broadcast to the rest of the cluster. Mirrors FleetEvalExp.run().
        resume_run_name = os.environ.get("RESUME_RUN_NAME", "")
        if resume_run_name:
            try:
                from integrations.fleet.s3_checkpoints import (
                    broadcast_checkpoint_to_workers,
                    download_checkpoint_from_s3,
                )

                ckpt_path = trainer.cfg.trainer.ckpt_path
                model_path = getattr(trainer.cfg.trainer.policy.model, "path", "unknown-model")
                model_name = Path(model_path).name
                project_name = getattr(trainer.cfg.trainer, "project_name", "skyrl")
                download_checkpoint_from_s3(
                    ckpt_path=ckpt_path,
                    run_name=resume_run_name,
                    project_name=project_name,
                    model_name=model_name,
                )
                broadcast_checkpoint_to_workers(ckpt_path)
            except Exception as e:
                logger.warning(f"Failed to download checkpoint from S3: {e}")

        if _truthy(os.environ.get("EXPORT_ALL_LOCAL_STEPS", "")):
            self._export_all_local_steps(trainer)
        else:
            # Reuse FleetEvalExp's path resolution (latest via
            # latest_ckpt_global_step.txt, or from_path via resume_path). It
            # loads policy-only shards and sets trainer.global_step.
            self._load_policy_only(trainer)
            self._export_loaded(trainer)

        try:
            trainer.tracker.finish()
        except Exception:
            pass

    def _export_loaded(self, trainer):
        """Export whatever policy weights are currently loaded into the model."""
        export_root = trainer.cfg.trainer.export_path
        out_dir = os.path.join(export_root, f"global_step_{trainer.global_step}", "policy")
        logger.info(f"Exporting HF safetensors for step {trainer.global_step} -> {out_dir}")
        trainer.save_models()
        logger.info(f"Export complete: {out_dir}")

    def _export_all_local_steps(self, trainer):
        """Load and export every local `global_step_*` checkpoint dir in turn."""
        from skyrl.backends.skyrl_train.utils.io import io
        from skyrl.train.utils.trainer_utils import (
            GLOBAL_STEP_PREFIX,
            extract_step_from_path,
        )

        ckpt_root = Path(trainer.cfg.trainer.ckpt_path)
        if not ckpt_root.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_root}")

        step_dirs = sorted(
            (d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith(GLOBAL_STEP_PREFIX)),
            key=lambda d: extract_step_from_path(d),
        )
        if not step_dirs:
            raise FileNotFoundError(
                f"No '{GLOBAL_STEP_PREFIX}*' checkpoints found under {ckpt_root}"
            )

        logger.info(f"Exporting {len(step_dirs)} local checkpoints: {[d.name for d in step_dirs]}")
        for ckpt_dir in step_dirs:
            step = extract_step_from_path(ckpt_dir)
            policy_ckpt_dir = os.path.join(str(ckpt_dir), "policy")
            if not io.exists(policy_ckpt_dir):
                logger.warning(f"Skipping {ckpt_dir.name}: no policy/ subdir")
                continue
            logger.info(f"Loading policy checkpoint {policy_ckpt_dir} (step {step})")
            trainer.dispatch.load_checkpoint(
                "policy",
                policy_ckpt_dir,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )
            trainer.global_step = step
            self._export_loaded(trainer)


@ray.remote(num_cpus=1)
def skyrl_export_entrypoint(cfg: SkyRLTrainConfig):
    """Ray remote function that runs the FSDP -> HF export."""
    # negotiation / fleet_task envs are auto-registered by skyrl_gym.envs.__init__
    exp = FleetExportExp(cfg)
    exp.run()


def main() -> None:
    """Main entry point for FSDP -> HF checkpoint export."""
    args = _strip_hydra_prefixes(sys.argv[1:])
    cfg = SkyRLTrainConfig.from_cli_overrides(args)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_export_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
