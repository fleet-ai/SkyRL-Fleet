from __future__ import annotations

import pytest

from integrations.fleet.trace_jobs import (
    FleetTraceJobRotator,
    FleetTraceWrappedGenerator,
    build_trace_job_stem,
    trace_label_for_input,
)
from skyrl.train.generators.base import BatchMetadata


def _batch(phase: str, step: int | None, env_class: str = "fleet_task"):
    return {
        "prompts": [[{"role": "user", "content": "hi"}]],
        "env_classes": [env_class],
        "env_extras": [{}],
        "sampling_params": None,
        "trajectory_ids": None,
        "batch_metadata": BatchMetadata(global_step=step, training_phase=phase),
    }


def test_trace_label_for_input_matches_phase_taxonomy():
    assert trace_label_for_input(_batch("eval", 0)) == "eval_pre"
    assert trace_label_for_input(_batch("train", 7)) == "train_step_7"
    assert trace_label_for_input(_batch("eval", 7)) == "eval_step_7"
    assert trace_label_for_input(_batch("eval", 7), total_training_steps=7) == "eval_final"
    assert trace_label_for_input(_batch("eval", 7), force_eval_only=True) == "eval_only"
    assert trace_label_for_input(_batch("eval", None)) == "eval_only"


def test_trace_job_stem_uses_dataset_key_without_extra_timestamp(monkeypatch):
    monkeypatch.setenv("DATASET_KEY", "JUNE16-PSI-HEALTH")
    monkeypatch.setenv("DATA_VERSION", "fallback-dataset")

    stem = build_trace_job_stem(run_name="fleet_qwen35_tool_use_abcd")

    assert stem == "skyrl_JUNE16-PSI-HEALTH_fleet_qwen35_tool_use_abcd"


def test_trace_job_stem_falls_back_to_data_version(monkeypatch):
    monkeypatch.delenv("DATASET_KEY", raising=False)
    monkeypatch.setenv("DATA_VERSION", "JUNE16-PSI-HEALTH")

    stem = build_trace_job_stem(run_name="fleet_qwen35_tool_use_abcd")

    assert stem == "skyrl_JUNE16-PSI-HEALTH_fleet_qwen35_tool_use_abcd"


def test_trace_job_stem_omits_empty_dataset_key(monkeypatch):
    monkeypatch.delenv("DATASET_KEY", raising=False)
    monkeypatch.delenv("DATA_VERSION", raising=False)

    stem = build_trace_job_stem(run_name="fleet_qwen35_tool_use_abcd")

    assert stem == "skyrl_fleet_qwen35_tool_use_abcd"


@pytest.mark.asyncio
async def test_rotator_is_noop_without_api_key():
    rotator = FleetTraceJobRotator(run_name="run", model="model", api_key="")
    cleared = []
    rotator._clear_trace_config = lambda: cleared.append(None)

    assert await rotator.rotate("train_step_1") is None
    assert cleared == [None]


@pytest.mark.asyncio
async def test_rotator_reuses_current_label_without_new_job(monkeypatch):
    created = []
    configured = []
    cleared = []
    rotator = FleetTraceJobRotator(run_name="run", model="model", api_key="key")

    async def fake_create(name: str) -> str:
        created.append(name)
        return f"job-{len(created)}"

    monkeypatch.setattr(rotator, "_create_trace_job", fake_create)
    monkeypatch.setattr(rotator, "_set_trace_config", configured.append)
    monkeypatch.setattr(rotator, "_clear_trace_config", lambda: cleared.append(None))

    assert await rotator.rotate("train_step_3") == "job-1"
    assert await rotator.rotate("train_step_3") == "job-1"
    assert await rotator.rotate("eval_step_3") == "job-2"
    assert len(created) == 2
    assert configured == ["job-1", "job-2"]
    assert len(cleared) == 2


@pytest.mark.asyncio
async def test_rotator_clears_stale_config_when_new_job_fails(monkeypatch):
    configured = []
    cleared = []
    rotator = FleetTraceJobRotator(run_name="run", model="model", api_key="key")
    monkeypatch.setattr(rotator, "_set_trace_config", configured.append)
    monkeypatch.setattr(rotator, "_clear_trace_config", lambda: cleared.append(None))

    async def create_success(name: str) -> str:
        return "job-1"

    async def create_failure(name: str) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(rotator, "_create_trace_job", create_success)
    assert await rotator.rotate("train_step_1") == "job-1"

    monkeypatch.setattr(rotator, "_create_trace_job", create_failure)
    assert await rotator.rotate("eval_step_1") is None
    assert rotator._current_label is None
    assert rotator._current_job_id is None
    assert configured == ["job-1"]
    assert len(cleared) == 3


class _DummyGenerator:
    def __init__(self):
        self.calls = []

    async def generate(self, input_batch, disable_tqdm: bool = False):
        self.calls.append((input_batch, disable_tqdm))
        return {
            "prompt_token_ids": [[1]],
            "response_ids": [[2]],
            "rewards": [1.0],
            "loss_masks": [[1]],
            "stop_reasons": ["stop"],
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": None,
            "env_metrics": None,
            "is_last_step": None,
            "is_hinted": None,
            "rollout_expert_indices": None,
        }


class _DummyRotator:
    def __init__(self):
        self.labels = []

    async def rotate(self, label):
        self.labels.append(label)


@pytest.mark.asyncio
async def test_wrapped_generator_rotates_only_for_fleet_batches():
    generator = _DummyGenerator()
    rotator = _DummyRotator()
    wrapped = FleetTraceWrappedGenerator(generator, rotator)

    await wrapped.generate(_batch("train", 4), disable_tqdm=True)
    await wrapped.generate(_batch("train", 5, env_class="gsm8k"))

    assert rotator.labels == ["train_step_4"]
    assert generator.calls[0][1] is True
    assert generator.calls[1][1] is False


@pytest.mark.asyncio
async def test_wrapped_generator_uses_final_eval_label():
    generator = _DummyGenerator()
    rotator = _DummyRotator()
    wrapped = FleetTraceWrappedGenerator(generator, rotator)

    wrapped.set_total_training_steps(5)
    await wrapped.generate(_batch("eval", 5))

    assert rotator.labels == ["eval_final"]


@pytest.mark.asyncio
async def test_wrapped_generator_can_force_eval_only_label():
    generator = _DummyGenerator()
    rotator = _DummyRotator()
    wrapped = FleetTraceWrappedGenerator(generator, rotator, force_eval_only=True)

    wrapped.set_total_training_steps(5)
    await wrapped.generate(_batch("eval", 5))

    assert rotator.labels == ["eval_only"]


class _BlockingGenerator:
    def __init__(self):
        self.started = []

    async def generate(self, input_batch, disable_tqdm: bool = False):
        import asyncio

        event = asyncio.Event()
        self.started.append((input_batch, event))
        await event.wait()
        return {
            "prompt_token_ids": [[1]],
            "response_ids": [[2]],
            "rewards": [1.0],
            "loss_masks": [[1]],
            "stop_reasons": ["stop"],
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": None,
            "env_metrics": None,
            "is_last_step": None,
            "is_hinted": None,
            "rollout_expert_indices": None,
        }


@pytest.mark.asyncio
async def test_wrapped_generator_serializes_fleet_batches():
    import asyncio

    generator = _BlockingGenerator()
    rotator = _DummyRotator()
    wrapped = FleetTraceWrappedGenerator(generator, rotator)

    first = asyncio.create_task(wrapped.generate(_batch("train", 4)))
    while not generator.started:
        await asyncio.sleep(0)

    second = asyncio.create_task(wrapped.generate(_batch("eval", 4)))
    await asyncio.sleep(0)

    assert rotator.labels == ["train_step_4"]
    assert len(generator.started) == 1

    generator.started[0][1].set()
    await first
    while len(generator.started) < 2:
        await asyncio.sleep(0)

    assert rotator.labels == ["train_step_4", "eval_step_4"]
    generator.started[1][1].set()
    await second
