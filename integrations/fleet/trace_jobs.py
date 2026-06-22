"""Fleet dashboard trace-job rotation for SkyRL rollouts.

The Fleet task env uploads a trace when its class-level trace config is set.
This module keeps that wiring out of the generic generator while letting the
Fleet SkyRL entrypoints mirror the Tinker harness: separate dashboard jobs for
train and eval phases.
"""

from __future__ import annotations

import inspect
import os
import re
from asyncio import Lock
from typing import Any, Optional

from loguru import logger

from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
)


def _clean_trace_part(value: object) -> str:
    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:80] or "unknown"


def build_trace_job_stem(
    *,
    run_name: str,
    prefix: str = "skyrl",
    dataset_key: Optional[str] = None,
) -> str:
    """Build a Fleet trace-job stem.

    Mirrors the Tinker trace naming convention Deniz added:
    ``<prefix>_<DATASET_KEY?>_<run_name>``. ``run_name`` is the stable run
    identifier, and Fleet launchers may already include timestamps or random
    suffixes there, so this helper deliberately does not add another timestamp.
    """
    if dataset_key is None:
        dataset_key = os.environ.get("DATASET_KEY", "") or os.environ.get("DATA_VERSION", "")

    parts = [_clean_trace_part(prefix)]
    dataset_key = dataset_key.strip()
    if dataset_key:
        parts.append(_clean_trace_part(dataset_key))
    parts.append(_clean_trace_part(run_name))
    return "_".join(parts)


def trace_label_for_input(
    input_batch: GeneratorInput,
    *,
    force_eval_only: bool = False,
    total_training_steps: Optional[int] = None,
) -> Optional[str]:
    """Return the Fleet trace-job label for a SkyRL generator batch."""
    batch_metadata = input_batch.get("batch_metadata")
    if batch_metadata is None:
        return None

    global_step = batch_metadata.global_step
    if batch_metadata.training_phase == "train":
        return f"train_step_{global_step}"

    if force_eval_only:
        return "eval_only"
    if global_step is None:
        return "eval_only"
    if global_step == 0:
        return "eval_pre"
    if total_training_steps is not None and total_training_steps > 0 and global_step == total_training_steps:
        return "eval_final"
    return f"eval_step_{global_step}"


class FleetTraceJobRotator:
    """Create one Fleet trace job per rollout phase and set FleetTaskEnv config."""

    def __init__(
        self,
        *,
        run_name: str,
        model: str,
        api_key: Optional[str] = None,
        prefix: str = "skyrl",
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("FLEET_API_KEY")
        self.model = model
        self._current_label: Optional[str] = None
        self._current_job_id: Optional[str] = None
        self._job_stem = build_trace_job_stem(run_name=run_name, prefix=prefix)

    async def rotate(self, label: Optional[str]) -> Optional[str]:
        """Rotate to ``label`` unless already active.

        Trace upload is best-effort: missing credentials or Fleet-side hiccups
        never block rollout collection.
        """
        if not label or not self.api_key:
            self.clear()
            return None
        if label == self._current_label:
            return self._current_job_id

        self.clear()
        try:
            job_name = f"{self._job_stem}_{_clean_trace_part(label)}"
            job_id = await self._create_trace_job(job_name)
            self._set_trace_config(job_id)
            self._current_label = label
            self._current_job_id = job_id
            logger.info(f"Fleet trace job ({label}): {job_id} ({job_name})")
            return job_id
        except Exception as e:
            self.clear()
            logger.warning(f"Fleet trace job creation failed for {label}: {e}")
            return None

    def clear(self) -> None:
        self._current_label = None
        self._current_job_id = None
        self._clear_trace_config()

    async def _create_trace_job(self, name: str) -> str:
        from envs.fleet_env.trace import create_trace_job

        return await create_trace_job(self.api_key, name)

    def _set_trace_config(self, job_id: str) -> None:
        from skyrl_gym.envs.fleet_task.env import FleetTaskEnv

        FleetTaskEnv.set_trace_config(job_id=job_id, model=self.model)

    def _clear_trace_config(self) -> None:
        from skyrl_gym.envs.fleet_task.env import FleetTaskEnv

        FleetTaskEnv.clear_trace_config()


class FleetTraceWrappedGenerator(GeneratorInterface):
    """Generator wrapper that rotates Fleet trace jobs before rollout batches."""

    def __init__(
        self,
        generator: GeneratorInterface,
        rotator: FleetTraceJobRotator,
        *,
        force_eval_only: bool = False,
        total_training_steps: Optional[int] = None,
    ):
        self._generator = generator
        self._rotator = rotator
        self._force_eval_only = force_eval_only
        self._total_training_steps = total_training_steps
        self._lock = Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        if not self._contains_fleet_task(input_batch):
            return await self._generate_inner(input_batch, disable_tqdm=disable_tqdm)

        # FleetTaskEnv reads trace config from class state at upload time.
        # Keep rotate+generate atomic so overlapping eval/train generate calls
        # cannot move in-flight rollouts into another phase's dashboard job.
        async with self._lock:
            await self._rotator.rotate(
                trace_label_for_input(
                    input_batch,
                    force_eval_only=self._force_eval_only,
                    total_training_steps=self._total_training_steps,
                )
            )
            return await self._generate_inner(input_batch, disable_tqdm=disable_tqdm)

    def set_total_training_steps(self, total_training_steps: Optional[int]) -> None:
        self._total_training_steps = total_training_steps

    async def _generate_inner(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        params = inspect.signature(self._generator.generate).parameters
        if "disable_tqdm" in params:
            return await self._generator.generate(input_batch, disable_tqdm=disable_tqdm)
        return await self._generator.generate(input_batch)

    @staticmethod
    def _contains_fleet_task(input_batch: GeneratorInput) -> bool:
        return any(env_class == "fleet_task" for env_class in input_batch.get("env_classes") or [])


def wrap_generator_for_fleet_traces(
    generator: GeneratorInterface,
    *,
    run_name: str,
    model: str,
    force_eval_only: bool = False,
    total_training_steps: Optional[int] = None,
) -> GeneratorInterface:
    """Wrap a SkyRL generator with Fleet trace rotation."""
    return FleetTraceWrappedGenerator(
        generator,
        FleetTraceJobRotator(run_name=run_name, model=model),
        force_eval_only=force_eval_only,
        total_training_steps=total_training_steps,
    )


def set_fleet_trace_total_training_steps(
    generator: GeneratorInterface,
    total_training_steps: Optional[int],
) -> None:
    """Teach a Fleet trace wrapper where the final training eval lands."""
    setter = getattr(generator, "set_total_training_steps", None)
    if setter is not None:
        setter(total_training_steps)
