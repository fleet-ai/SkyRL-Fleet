"""ATOF hooks in the Tinker rollout path (item 3 of the ATOF orchestration).

Behavioral tests drive the real collect_fleet_rollout / collect_batch_rollouts
with a fake env + sampling client and a recording emitter. Provider calls run
through the shared NeMo helper while rollout and environment marks stay on the
SkyRL emitter.

AST contract tests (test_main_fleet_tinker_rollout_dump.py style) pin the
main() wiring: init_atof after wandb setup, drain_atof on both exits, and
metadata threaded into every rollout-collecting call site.

Run:
    uv run --extra dev --extra tinker pytest \
        integrations/fleet/tests/test_atof_tinker.py
"""

from __future__ import annotations

import ast
import asyncio
import sys
import types
from pathlib import Path

# Make sure skyrl-gym is importable for the parent module.
ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "skyrl-gym"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from integrations.fleet.entrypoints import main_fleet_tinker as mft  # noqa: E402

SRC = Path(__file__).resolve().parents[3] / "integrations/fleet/entrypoints/main_fleet_tinker.py"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeTokenizer:
    """encode: one int per char; decode: the inverse. Template concatenates."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        body = "\n".join(f"<{m['role']}>{m.get('content', '')}" for m in messages)
        if add_generation_prompt:
            body += "\n<assistant>"
        if tokenize:
            return self.encode(body)
        return body


class FakeSequence:
    def __init__(self, text: str):
        self.tokens = [ord(c) for c in text]
        self.logprobs = [0.0] * len(self.tokens)
        self.stop_reason = "stop"


class FakeSamplingClient:
    """Returns scripted assistant turns in order."""

    def __init__(self, turns):
        self._turns = list(turns)

    async def sample_async(self, prompt=None, num_samples=1, sampling_params=None):
        class Result:
            pass

        result = Result()
        result.sequences = [FakeSequence(self._turns.pop(0))]
        return result


class FakeEnv:
    """Two-turn scripted FleetTaskEnv stand-in.

    Step 1: one tool observation, reward 0.0, not done.
    Step 2: no observations, reward 1.0, done. tool_errors=1 on turn 2 so the
    counter-folding assertion can't pass by accident.
    """

    initial_history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    step1_obs = [{"role": "tool", "content": "obs1"}]

    def __init__(self, env_config=None, extras=None):
        self.extras = extras or {}
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.tools = []
        self.chat_history = []
        self.last_reward = None
        self.openenv_task_env = None
        self._verifier_stdout = ""
        self._verifier_error = ""

    async def init_async(self, prompt):
        self.chat_history = list(self.initial_history)
        return self.chat_history, {"env_key": "fira", "tools": None}

    async def step_async(self, action):
        self.turns += 1
        self.tool_calls += 1
        self.chat_history = self.chat_history + [{"role": "assistant", "content": action}]
        if self.turns == 1:
            self.chat_history = self.chat_history + list(self.step1_obs)
            return {"observations": list(self.step1_obs), "reward": 0.0, "done": False}
        self.tool_errors = 1
        return {"observations": [], "reward": 1.0, "done": True}

    async def close_async(self):
        pass

    def _capture_verifier_feedback(self):
        pass


class FakeTrace:
    def __init__(self):
        self.counters = {}
        self.metadata = {"producer_session_id": "run-1", "trace_id": "trace-1"}


class RecordingEmitter:
    def __init__(self):
        self.calls = []
        self.producer_session_id_calls = []
        self.requests = []
        self.trace = FakeTrace()

    def rollout_start(self, **kwargs):
        self.calls.append(("rollout_start", kwargs))
        return self.trace

    def producer_session_id(self, **kwargs):
        self.producer_session_id_calls.append(kwargs)
        return "11111111-1111-5111-8111-111111111111"

    def has_started_session(self, **_kwargs):
        return True

    def llm_request(self, trace, *, new_messages):
        request = {"messages": new_messages}
        self.requests.append(request)
        return request

    def env_step(self, **kwargs):
        self.calls.append(("env_step", kwargs))

    def rollout_end(self, **kwargs):
        self.calls.append(("rollout_end", kwargs))


class RaisingEmitter:
    def rollout_start(self, **kwargs):
        raise RuntimeError("boom")

    def llm_turn(self, **kwargs):
        raise RuntimeError("boom")

    def env_step(self, **kwargs):
        raise RuntimeError("boom")

    def rollout_end(self, **kwargs):
        raise RuntimeError("boom")


def run_rollout(monkeypatch, **kwargs):
    monkeypatch.setattr(mft, "FleetTaskEnv", FakeEnv)
    return asyncio.run(
        mft.collect_fleet_rollout(
            task_config={"task_key": "fira/t1"},
            tasks_file="tasks.json",
            sampling_client=FakeSamplingClient(["act1", "act2"]),
            tokenizer=FakeTokenizer(),
            max_turns=10,
            **kwargs,
        )
    )


# --------------------------------------------------------------------------- #
# Hook behavior in collect_fleet_rollout
# --------------------------------------------------------------------------- #


def test_provider_calls_are_orchestrated_with_rollout_metadata(monkeypatch):
    orchestrated_calls = []
    runtime = types.ModuleType("nemo_relay_runtime")

    async def orchestrated_llm_call_async(**kwargs):
        orchestrated_calls.append(kwargs)
        return await kwargs["invoke"](kwargs["request"])

    runtime.orchestrated_llm_call_async = orchestrated_llm_call_async
    monkeypatch.setitem(sys.modules, "nemo_relay_runtime", runtime)
    emitter = RecordingEmitter()
    rollout = run_rollout(monkeypatch, atof_emitter=emitter, global_step=7, phase="train_step_7", sample_idx=3)

    methods = [name for name, _ in emitter.calls]
    assert methods == [
        "rollout_start",
        "env_step",
        "env_step",
        "rollout_end",
    ]

    start = dict(emitter.calls[0][1])
    assert start == {
        "task_key": "fira/t1",
        "env_class": "fleet_task",
        "global_step": 7,
        "phase": "train_step_7",
        "sample_idx": 3,
        "job_id": None,
    }

    assert [call["request"]["messages"] for call in orchestrated_calls] == [
        FakeEnv.initial_history,
        FakeEnv.step1_obs,
    ]
    assert all(call["metadata"] is emitter.trace.metadata for call in orchestrated_calls)
    assert all(call["name"] == "tinker-policy" for call in orchestrated_calls)

    step1 = emitter.calls[1][1]
    assert step1["action"] == "act1"
    assert step1["observations"] == FakeEnv.step1_obs
    assert step1["reward"] == 0.0
    assert step1["done"] is False
    step2 = emitter.calls[2][1]
    assert step2["observations"] == []
    assert step2["reward"] == 1.0
    assert step2["done"] is True

    end = emitter.calls[3][1]
    assert end["reward"] == 1.0
    assert end["num_turns"] == 2
    # Tinker tool counters folded into the final mark's counters.
    assert emitter.trace.counters == {"tool_calls": 2, "tool_errors": 1}

    assert rollout.reward == 1.0
    assert rollout.turns == 2


def test_no_emitter_is_todays_behavior(monkeypatch):
    with_emitter = run_rollout(monkeypatch, atof_emitter=RecordingEmitter())
    without = run_rollout(monkeypatch)
    exclude = {"duration", "total_gen_time", "total_step_time"}  # wall-clock noise
    assert without.model_dump(exclude=exclude) == with_emitter.model_dump(exclude=exclude)
    assert without.reward == 1.0


def test_raising_emitter_never_breaks_the_rollout(monkeypatch):
    rollout = run_rollout(monkeypatch, atof_emitter=RaisingEmitter(), global_step=1, phase="p")
    assert rollout.reward == 1.0
    assert rollout.turns == 2
    assert rollout.stop_reason == "stop"


def test_rollout_scopes_session_to_trace_job(monkeypatch):
    monkeypatch.setattr(FakeEnv, "_trace_config", {"job_id": "job-1"}, raising=False)
    emitter = RecordingEmitter()

    run_rollout(monkeypatch, atof_emitter=emitter, global_step=7, phase="train_step_7", sample_idx=3)

    assert emitter.producer_session_id_calls == [
        {
            "task_key": "fira/t1",
            "global_step": 7,
            "phase": "train_step_7",
            "job_id": "job-1",
        }
    ]
    assert emitter.calls[0][1]["job_id"] == "job-1"


# --------------------------------------------------------------------------- #
# collect_batch_rollouts forwards emitter + metadata + per-rollout sample_idx
# --------------------------------------------------------------------------- #


def test_batch_forwards_emitter_metadata_and_sample_idx(monkeypatch):
    seen = []

    async def fake_collect(**kwargs):
        seen.append(kwargs)
        return mft.RolloutOutput(
            prompt_ids=[],
            response_ids=[],
            logprobs=[],
            loss_mask=[],
            reward=0.0,
            task_key=kwargs["task_config"]["task_key"],
            env_key="fira",
            turns=0,
            tool_calls=0,
            tool_errors=0,
            stop_reason="stop",
            duration=0.0,
        )

    monkeypatch.setattr(mft, "collect_fleet_rollout", fake_collect)
    emitter = object()
    asyncio.run(
        mft.collect_batch_rollouts(
            batch=[{"task_key": "fira/t1"}],
            tasks_file="tasks.json",
            sampling_client=None,
            tokenizer=None,
            n_samples_per_prompt=2,
            atof_emitter=emitter,
            global_step=7,
            phase="eval_step_7",
        )
    )

    assert len(seen) == 2
    assert sorted(k["sample_idx"] for k in seen) == [0, 1]
    for kwargs in seen:
        assert kwargs["atof_emitter"] is emitter
        assert kwargs["global_step"] == 7
        assert kwargs["phase"] == "eval_step_7"


def test_batch_uploads_one_group_session_per_task(monkeypatch):
    async def fake_collect(**kwargs):
        return mft.RolloutOutput(
            prompt_ids=[],
            response_ids=[1],
            logprobs=[],
            loss_mask=[],
            reward=float(kwargs["sample_idx"]),
            task_key=kwargs["task_config"]["task_key"],
            env_key="fira",
            turns=1,
            tool_calls=0,
            tool_errors=0,
            stop_reason="stop",
            duration=0.0,
        )

    uploads = []

    async def fake_upload(**kwargs):
        uploads.append(kwargs)
        return True

    monkeypatch.setattr(mft, "collect_fleet_rollout", fake_collect)
    monkeypatch.setattr(mft, "upload_group_session", fake_upload)
    monkeypatch.setattr(
        mft.FleetTaskEnv,
        "_trace_config",
        {"job_id": "job-1", "model": "model-1"},
    )
    monkeypatch.setenv("FLEET_API_KEY", "secret")

    emitter = RecordingEmitter()
    asyncio.run(
        mft.collect_batch_rollouts(
            batch=[{"task_key": "fira/t1", "env_key": "fira"}],
            tasks_file="tasks.json",
            sampling_client=None,
            tokenizer=None,
            n_samples_per_prompt=2,
            atof_emitter=emitter,
            global_step=7,
            phase="eval_step_7",
        )
    )

    assert len(uploads) == 1
    assert emitter.producer_session_id_calls == [
        {
            "task_key": "fira/t1",
            "global_step": 7,
            "phase": "eval_step_7",
            "job_id": "job-1",
        }
    ]
    assert uploads[0]["score"] == 1.0
    assert uploads[0]["metadata"] == {
        "skyrl_session_kind": "group",
        "skyrl_expected_rollouts": 2,
        "skyrl_completed_rollouts": 2,
        "env_key": "fira",
        "phase": "eval_step_7",
        "global_step": 7,
    }


# --------------------------------------------------------------------------- #
# main() wiring contract (AST, matching test_main_fleet_tinker_rollout_dump.py)
# --------------------------------------------------------------------------- #


def _parse() -> ast.Module:
    return ast.parse(SRC.read_text())


def _find_func(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _calls_named(fn, name: str):
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            call_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if call_name == name:
                yield node


def test_main_inits_emitter_with_run_identity():
    main_fn = _find_func(_parse(), "main")
    calls = list(_calls_named(main_fn, "init_atof"))
    assert len(calls) == 1, "main() must call init_atof exactly once"
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(kwargs["entrypoint"], ast.Constant)
    assert kwargs["entrypoint"].value == "main_fleet_tinker"
    assert {"run_name", "model"} <= set(kwargs)


def test_main_drains_on_both_exits():
    """eval_only returns early; the training path exits at the tail. Both must
    flush buffered events or the last seconds of a run are silently lost."""
    main_fn = _find_func(_parse(), "main")
    assert len(list(_calls_named(main_fn, "drain_atof"))) == 2


def test_every_run_eval_call_passes_atof_phase():
    main_fn = _find_func(_parse(), "main")
    sites = 0
    for call in _calls_named(main_fn, "_run_eval"):
        sites += 1
        kwargs = {kw.arg for kw in call.keywords}
        assert "atof_phase" in kwargs, f"_run_eval call missing atof_phase; keywords: {sorted(kwargs)}"
    assert sites >= 4, f"expected >=4 _run_eval call sites, found {sites}"


def test_rollout_collecting_call_sites_thread_atof_metadata():
    """The train loop and _run_eval must both pass emitter + step + phase into
    collect_batch_rollouts, or a whole phase's rollouts go dark."""
    main_fn = _find_func(_parse(), "main")
    sites = 0
    for call in _calls_named(main_fn, "collect_batch_rollouts"):
        sites += 1
        kwargs = {kw.arg for kw in call.keywords}
        assert {
            "atof_emitter",
            "global_step",
            "phase",
        } <= kwargs, f"collect_batch_rollouts call missing ATOF kwargs; has: {sorted(kwargs)}"
    assert sites == 2, f"expected the eval + train call sites, found {sites}"
