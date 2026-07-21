"""NeMo orchestration for hint-synthesis provider calls."""

from __future__ import annotations

import asyncio
import sys
import types

from skyrl_gym.envs.fleet_task import hint_synthesizer


def _response(text: str):
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def _hint_kwargs():
    return {
        "task_prompt": "book a flight",
        "chat_history": [],
        "verifier_stdout": "0/3 checks",
        "verifier_error": None,
        "tool_error_messages": ["timeout"],
        "model": "openrouter/anthropic/claude-sonnet-4",
        "nemo_metadata": {"instance_id": "i-1", "global_step": 7},
    }


def test_hint_provider_call_runs_through_shared_runtime(monkeypatch):
    provider_requests = []
    litellm = types.ModuleType("litellm")

    async def acompletion(**request):
        provider_requests.append(request)
        return _response("Use the date picker.")

    litellm.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", litellm)

    orchestrated_calls = []
    runtime = types.ModuleType("nemo_relay_runtime")

    async def orchestrated_openai_chat_call_async(**kwargs):
        orchestrated_calls.append(kwargs)
        return await kwargs["invoke"](kwargs["request"])

    runtime.orchestrated_openai_chat_call_async = orchestrated_openai_chat_call_async
    monkeypatch.setitem(sys.modules, "nemo_relay_runtime", runtime)
    monkeypatch.setenv("SKYRL_ATOF_RUN_NAME", "run-1")

    result = asyncio.run(hint_synthesizer.synthesize_hint(**_hint_kwargs()))

    assert result == ("Use the date picker.", hint_synthesizer.CATEGORY_LLM)
    assert len(provider_requests) == 1
    assert len(orchestrated_calls) == 1
    call = orchestrated_calls[0]
    assert call["call_site"] == "skyrl_gym.fleet_task.hint_synthesis"
    assert call["metadata"] == {
        "run_name": "run-1",
        "instance_id": "i-1",
        "global_step": 7,
    }


def test_hint_call_falls_back_when_runtime_wheel_is_missing(monkeypatch):
    calls = []
    litellm = types.ModuleType("litellm")

    async def acompletion(**request):
        calls.append(request)
        return _response("Try another route.")

    litellm.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setitem(sys.modules, "nemo_relay_runtime", None)

    result = asyncio.run(hint_synthesizer.synthesize_hint(**_hint_kwargs()))

    assert result == ("Try another route.", hint_synthesizer.CATEGORY_LLM)
    assert len(calls) == 1
