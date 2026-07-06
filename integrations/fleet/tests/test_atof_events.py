from __future__ import annotations

import base64
import hashlib
import json
import sys
import types

import pytest

from integrations.fleet import atof_events
from integrations.fleet.atof_events import (
    MAX_PAYLOAD_BYTES,
    MSK_PLUGIN_KIND,
    AtofEmitter,
    drain_atof,
    init_atof,
)


class FakeNemo(types.ModuleType):
    """Records every runtime call the emitter makes."""

    def __init__(self, diagnostics=None):
        super().__init__("nemo_relay")
        self.calls = []
        fake = self

        class ScopeType:
            Agent = "agent"

        class LLMRequest:
            def __init__(self, headers, content):
                self.headers = headers
                self.content = content

        class plugin:
            @staticmethod
            def initialize(config):
                fake.calls.append(("initialize", config))
                return {"diagnostics": diagnostics or []}

            @staticmethod
            def drain(timeout):
                fake.calls.append(("drain", timeout))
                return True

        class scope:
            @staticmethod
            def push(name, scope_type, **kwargs):
                fake.calls.append(("push", name, scope_type, kwargs))
                return "handle-1"

            @staticmethod
            def pop(handle, **kwargs):
                fake.calls.append(("pop", handle, kwargs))

            @staticmethod
            def event(name, **kwargs):
                fake.calls.append(("event", name, kwargs))

        class llm:
            @staticmethod
            def execute(name, request, func, **kwargs):
                fake.calls.append(("llm", name, request.content, func(request), kwargs))

        class tools:
            @staticmethod
            def execute(name, args, func, **kwargs):
                fake.calls.append(("tool", name, args, func(args), kwargs))

        self.ScopeType = ScopeType
        self.LLMRequest = LLMRequest
        self.plugin = plugin
        self.scope = scope
        self.llm = llm
        self.tools = tools

    def named(self, kind):
        return [call for call in self.calls if call[0] == kind]


class FakeS3:
    def __init__(self):
        self.uploads = []
        self.fail = False

    def put_object(self, *, Bucket, Key, Body):
        if self.fail:
            raise RuntimeError("s3 down")
        self.uploads.append((Bucket, Key, Body))


@pytest.fixture
def fake_nemo(monkeypatch):
    fake = FakeNemo()
    monkeypatch.setitem(sys.modules, "nemo_relay", fake)
    monkeypatch.setattr(atof_events, "_nemo_module", None)
    return fake


@pytest.fixture
def msk_env(monkeypatch):
    monkeypatch.setenv("SKYRL_ATOF_ENABLED", "1")
    monkeypatch.setenv("THESEUS_ATOF_MSK_BROKERS", "b-1:9198")
    monkeypatch.setenv("THESEUS_ATOF_TENANT_ID", "skyrl")


def make_emitter(fake_nemo):
    return AtofEmitter(fake_nemo, entrypoint="main_fleet", run_name="run-1", model="Qwen/Qwen3.5-9B")


def data_image_message(image_bytes=b"png-bytes"):
    url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}


class TestInit:
    def test_disabled_without_flag(self, fake_nemo, msk_env, monkeypatch):
        monkeypatch.delenv("SKYRL_ATOF_ENABLED")
        assert init_atof(entrypoint="e", run_name="r", model="m") is None
        assert fake_nemo.calls == []

    def test_disabled_without_msk_vars(self, fake_nemo, monkeypatch):
        monkeypatch.setenv("SKYRL_ATOF_ENABLED", "1")
        monkeypatch.delenv("THESEUS_ATOF_MSK_BROKERS", raising=False)
        assert init_atof(entrypoint="e", run_name="r", model="m") is None
        assert fake_nemo.calls == []

    def test_disabled_when_wheel_missing(self, msk_env, monkeypatch):
        monkeypatch.setitem(sys.modules, "nemo_relay", None)
        assert init_atof(entrypoint="e", run_name="r", model="m") is None

    def test_disabled_on_error_diagnostics(self, msk_env, monkeypatch):
        fake = FakeNemo(diagnostics=[{"level": "error", "message": "bad brokers"}])
        monkeypatch.setitem(sys.modules, "nemo_relay", fake)
        assert init_atof(entrypoint="e", run_name="r", model="m") is None

    def test_msk_component_config(self, fake_nemo, msk_env):
        emitter = init_atof(entrypoint="main_fleet", run_name="run-1", model="m")
        assert isinstance(emitter, AtofEmitter)
        ((_, config),) = fake_nemo.named("initialize")
        (component,) = config["components"]
        assert component["kind"] == MSK_PLUGIN_KIND
        assert component["config"]["brokers"] == "b-1:9198"
        assert component["config"]["tenant_id"] == "skyrl"
        assert component["config"]["topic"] == "atof.received"
        assert component["config"]["fail_open"] is True

    def test_file_exporter_mode(self, fake_nemo, monkeypatch, tmp_path):
        monkeypatch.setenv("SKYRL_ATOF_ENABLED", "1")
        monkeypatch.setenv("SKYRL_ATOF_FILE_DIR", str(tmp_path))
        assert init_atof(entrypoint="e", run_name="r", model="m") is not None
        ((_, config),) = fake_nemo.named("initialize")
        (component,) = config["components"]
        assert component["kind"] == "observability"
        assert component["config"]["atof"]["output_directory"] == str(tmp_path)

    def test_warning_diagnostics_do_not_disable(self, msk_env, monkeypatch):
        fake = FakeNemo(diagnostics=[{"level": "warning", "message": "meh"}])
        monkeypatch.setitem(sys.modules, "nemo_relay", fake)
        assert init_atof(entrypoint="e", run_name="r", model="m") is not None


class TestEmit:
    def test_rollout_start_stamps_metadata(self, fake_nemo):
        trace = make_emitter(fake_nemo).rollout_start(
            task_key="task-9", env_class="fleet_task", global_step=7, phase="train_step_7", sample_idx=2
        )
        ((_, name, scope_type, kwargs),) = fake_nemo.named("push")
        assert name == "rollout:task-9"
        assert scope_type == "agent"
        metadata = kwargs["metadata"]
        assert metadata["producer_session_id"] == "run-1"
        assert metadata["global_step"] == 7
        assert metadata["phase"] == "train_step_7"
        assert metadata["entrypoint"] == "main_fleet"
        assert metadata["model"] == "Qwen/Qwen3.5-9B"
        assert len(metadata["trace_id"]) == 32
        assert trace.metadata is metadata

    def test_llm_turn_payloads(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        messages = [{"role": "user", "content": "solve it"}]
        emitter.llm_turn(trace, new_messages=messages, response_text="done", stop_reason="stop")
        ((_, name, request, response, kwargs),) = fake_nemo.named("llm")
        assert name == "skyrl-policy"
        assert request == {"messages": messages}
        assert response == {"content": "done", "stop_reason": "stop"}
        assert kwargs["handle"] == "handle-1"
        assert kwargs["model_name"] == "Qwen/Qwen3.5-9B"
        assert kwargs["metadata"]["trace_id"] == trace.metadata["trace_id"]

    def test_env_step_payloads(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        obs = [{"role": "user", "content": "tool output"}]
        emitter.env_step(trace, action="<tool>ls</tool>", observations=obs, reward=0.5, done=True)
        ((_, name, args, result, kwargs),) = fake_nemo.named("tool")
        assert name == "env_step"
        assert args == {"action": "<tool>ls</tool>"}
        assert result == {"observations": obs, "reward": 0.5, "done": True}
        assert kwargs["handle"] == "handle-1"

    def test_rollout_end_marks_and_pops(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        emitter.rollout_end(trace, reward=1.0, stop_reason="stop", num_turns=3)
        ((_, name, kwargs),) = fake_nemo.named("event")
        assert name == "rollout_end"
        assert kwargs["data"]["reward"] == 1.0
        assert kwargs["data"]["num_turns"] == 3
        assert kwargs["data"]["counters"] == {"truncated": 0, "image_upload_failures": 0, "emit_errors": 0}
        ((_, handle, pop_kwargs),) = fake_nemo.named("pop")
        assert handle == "handle-1"
        assert pop_kwargs["output"] == {"reward": 1.0}

    def test_none_trace_is_noop(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        emitter.llm_turn(None, new_messages=[], response_text="", stop_reason=None)
        emitter.env_step(None, action="", observations=[], reward=0.0, done=False)
        emitter.rollout_end(None, reward=0.0, stop_reason=None, num_turns=0)
        assert fake_nemo.calls == []

    def test_emit_failure_is_swallowed_and_counted(self, fake_nemo, monkeypatch):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        monkeypatch.setattr(fake_nemo.llm, "execute", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        emitter.llm_turn(trace, new_messages=[], response_text="x", stop_reason=None)
        assert trace.counters["emit_errors"] == 1


class TestImages:
    def test_data_url_replaced_with_s3_url_and_uploaded_once(self, fake_nemo, monkeypatch):
        s3 = FakeS3()
        monkeypatch.setattr(atof_events, "_make_s3_client", lambda: s3)
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        image_bytes = b"png-bytes"
        sha = hashlib.sha256(image_bytes).hexdigest()
        messages = [data_image_message(image_bytes), data_image_message(image_bytes)]

        emitter.llm_turn(trace, new_messages=messages, response_text="ok", stop_reason="stop")
        emitter._image_pool.shutdown(wait=True)

        expected_url = f"s3://fleet-trajectory-artifacts/skyrl/run-1/{sha}"
        ((_, _, request, _, _),) = fake_nemo.named("llm")
        for message in request["messages"]:
            assert message["content"][0]["image_url"]["url"] == expected_url
        assert s3.uploads == [("fleet-trajectory-artifacts", f"skyrl/run-1/{sha}", image_bytes)]
        assert trace.image_urls == [
            {"url": expected_url, "sha256": sha, "bytes": len(image_bytes)},
            {"url": expected_url, "sha256": sha, "bytes": len(image_bytes)},
        ]
        # The caller's messages are untouched.
        assert messages[0]["content"][0]["image_url"]["url"].startswith("data:image")

    def test_event_emitted_even_when_upload_fails(self, fake_nemo, monkeypatch):
        s3 = FakeS3()
        s3.fail = True
        monkeypatch.setattr(atof_events, "_make_s3_client", lambda: s3)
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)

        emitter.llm_turn(trace, new_messages=[data_image_message()], response_text="ok", stop_reason="stop")
        emitter._image_pool.shutdown(wait=True)

        assert len(fake_nemo.named("llm")) == 1
        assert trace.counters["image_upload_failures"] == 1

    def test_plain_text_messages_not_copied(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        messages = [{"role": "user", "content": "plain"}]
        assert emitter._offload_images(trace, messages) is messages


class TestGuard:
    def test_normal_payload_untouched(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        payload = {"action": "small"}
        assert emitter._guard(trace, payload) is payload
        assert trace.counters["truncated"] == 0

    def test_oversized_payload_truncated_and_flagged(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        payload = {"action": "x" * (MAX_PAYLOAD_BYTES + 100)}
        guarded = emitter._guard(trace, payload)
        assert guarded["truncated"] is True
        assert guarded["original_bytes"] > MAX_PAYLOAD_BYTES
        assert len(json.dumps(guarded)) <= MAX_PAYLOAD_BYTES
        assert trace.counters["truncated"] == 1


class TestDrain:
    def test_drain_delegates(self, fake_nemo, msk_env):
        init_atof(entrypoint="e", run_name="r", model="m")
        drain_atof(timeout=2.5)
        assert ("drain", 2.5) in fake_nemo.calls

    def test_drain_noop_when_disabled(self, fake_nemo):
        drain_atof()
        assert fake_nemo.named("drain") == []
