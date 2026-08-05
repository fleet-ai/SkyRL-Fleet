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
from asyncio import Lock, create_task, gather
from collections import defaultdict
from typing import Any, Optional

from loguru import logger

from integrations.fleet.session_bridge import upload_group_session
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
)


def clean_trace_part(value: object) -> str:
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

    parts = [clean_trace_part(prefix)]
    dataset_key = dataset_key.strip()
    if dataset_key:
        parts.append(clean_trace_part(dataset_key))
    parts.append(clean_trace_part(run_name))
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
        self.current_label: Optional[str] = None
        self.current_job_id: Optional[str] = None
        self.job_stem = build_trace_job_stem(run_name=run_name, prefix=prefix)

    async def rotate(self, label: Optional[str]) -> Optional[str]:
        """Rotate to ``label`` unless already active.

        Trace upload is best-effort: missing credentials or Fleet-side hiccups
        never block rollout collection.
        """
        if not label or not self.api_key:
            self.clear()
            return None
        if label == self.current_label:
            return self.current_job_id

        self.clear()
        try:
            job_name = f"{self.job_stem}_{clean_trace_part(label)}"
            job_id = await self.create_trace_job(job_name)
            self.set_trace_config(job_id)
            self.current_label = label
            self.current_job_id = job_id
            logger.info(f"Fleet trace job ({label}): {job_id} ({job_name})")
            return job_id
        except Exception as e:
            self.clear()
            logger.warning(f"Fleet trace job creation failed for {label}: {e}")
            return None

    def clear(self) -> None:
        self.current_label = None
        self.current_job_id = None
        self.clear_trace_config()

    async def create_trace_job(self, name: str) -> str:
        from envs.fleet_env.trace import create_trace_job

        return await create_trace_job(self.api_key, name)

    def set_trace_config(self, job_id: str) -> None:
        from skyrl_gym.envs.fleet_task.env import FleetTaskEnv

        FleetTaskEnv.set_trace_config(job_id=job_id, model=self.model)

    def clear_trace_config(self) -> None:
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
        self.generator = generator
        self.rotator = rotator
        self.force_eval_only = force_eval_only
        self.total_training_steps = total_training_steps
        self.lock = Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.generator, name)

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        if not self.contains_fleet_task(input_batch):
            return await self.generate_inner(input_batch, disable_tqdm=disable_tqdm)

        # FleetTaskEnv reads trace config from class state at upload time.
        # Keep rotate+generate atomic so overlapping eval/train generate calls
        # cannot move in-flight rollouts into another phase's dashboard job.
        async with self.lock:
            job_id = await self.rotator.rotate(
                trace_label_for_input(
                    input_batch,
                    force_eval_only=self.force_eval_only,
                    total_training_steps=self.total_training_steps,
                )
            )
            groups = self.prepare_group_sessions(input_batch, job_id)
            pending_group_upload = create_task(self.upload_group_sessions(groups))
            try:
                output = await self.generate_inner(input_batch, disable_tqdm=disable_tqdm)
            finally:
                await pending_group_upload
            await self.upload_group_sessions(groups, output)
            return output

    def prepare_group_sessions(self, input_batch: GeneratorInput, job_id: Optional[str]) -> dict[str, dict[str, Any]]:
        emitter = getattr(self.generator, "atof_emitter", None)
        if not job_id or emitter is None:
            return {}

        batch_metadata = input_batch.get("batch_metadata")
        global_step = batch_metadata.global_step if batch_metadata is not None else None
        phase = (
            f"{batch_metadata.training_phase}_step_{batch_metadata.global_step}" if batch_metadata is not None else None
        )
        groups: dict[str, dict[str, Any]] = {}
        for env_class, env_extras in zip(input_batch.get("env_classes") or [], input_batch.get("env_extras") or []):
            if env_class != "fleet_task":
                continue
            task_key = env_extras.get("task_key")
            if not task_key:
                continue
            session_id = emitter.producer_session_id(
                task_key=task_key,
                global_step=global_step,
                phase=phase,
                job_id=job_id,
            )
            env_extras["skyrl_group_session_id"] = session_id
            env_extras["skyrl_trace_job_id"] = job_id
            group = groups.setdefault(
                task_key,
                {
                    "session_id": session_id,
                    "env_key": env_extras.get("env_key") or env_extras.get("data_source"),
                    "global_step": global_step,
                    "phase": phase,
                    "expected_rollouts": 0,
                },
            )
            group["expected_rollouts"] += 1
        return groups

    async def upload_group_sessions(
        self,
        groups: dict[str, dict[str, Any]],
        output: Optional[GeneratorOutput] = None,
    ) -> None:
        emitter = getattr(self.generator, "atof_emitter", None)
        api_key = getattr(self.rotator, "api_key", None)
        job_id = getattr(self.rotator, "current_job_id", None)
        model = getattr(self.rotator, "model", None)
        if not groups or emitter is None or not api_key or not job_id or not model:
            return

        scores_by_task: dict[str, list[float]] = defaultdict(list)
        completed_by_task: dict[str, int] = defaultdict(int)
        if output is not None:
            env_metrics = output.get("env_metrics") or []
            is_last_step = output.get("is_last_step")
            for index, metrics in enumerate(env_metrics):
                if not isinstance(metrics, dict):
                    continue
                if is_last_step is not None and not is_last_step[index]:
                    continue
                task_key = metrics.get("task_key")
                if task_key not in groups:
                    continue
                completed_by_task[task_key] += 1
                score = metrics.get("final_reward")
                if isinstance(score, (int, float)):
                    scores_by_task[task_key].append(float(score))

        uploads = []
        for task_key, group in groups.items():
            scores = scores_by_task[task_key]
            metadata = {
                "skyrl_session_kind": "group",
                "skyrl_expected_rollouts": group["expected_rollouts"],
                "env_key": group["env_key"],
                "phase": group["phase"],
                "global_step": group["global_step"],
            }
            if output is not None:
                metadata["skyrl_completed_rollouts"] = completed_by_task[task_key]
            uploads.append(
                upload_group_session(
                    api_key=api_key,
                    session_id=group["session_id"],
                    job_id=job_id,
                    task_key=task_key,
                    model=model,
                    score=max(scores) if scores else None,
                    metadata=metadata,
                    status="completed" if output is not None else None,
                )
            )
        if not uploads:
            return
        try:
            await gather(*uploads)
        except Exception as exc:
            logger.warning(f"SkyRL group session uploads failed; training will continue: {exc}")

    def set_total_training_steps(self, total_training_steps: Optional[int]) -> None:
        self.total_training_steps = total_training_steps

    async def generate_inner(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        params = inspect.signature(self.generator.generate).parameters
        if "disable_tqdm" in params:
            return await self.generator.generate(input_batch, disable_tqdm=disable_tqdm)
        return await self.generator.generate(input_batch)

    @staticmethod
    def contains_fleet_task(input_batch: GeneratorInput) -> bool:
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
