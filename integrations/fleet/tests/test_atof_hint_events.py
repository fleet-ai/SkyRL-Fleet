"""Hint-synthesis events in the shared ATOF module (phase 2).

The generator's _run_hint_augmentation calls
emitter.hint_synthesis_events(...) through _atof_emit; all shaping lives
here in atof_events so the core generator file carries one call only.
"""

from __future__ import annotations

import pytest

from integrations.fleet.tests.test_atof_events import FakeNemo, make_emitter

HINT_MODEL = "openrouter/anthropic/claude-sonnet-4"


@pytest.fixture
def fake_nemo():
    return FakeNemo()


def hint_fixtures():
    hint_requests = [
        {
            "task_prompt": "book a flight",
            "verifier_stdout": "0/3 checks",
            "verifier_error": None,
            "tool_error_messages": ["timeout"],
            "instance_id": "i-1",
        },
        {
            "task_prompt": "send an email",
            "verifier_stdout": None,
            "verifier_error": "crash",
            "tool_error_messages": None,
            "instance_id": "i-2",
        },
    ]
    hint_results = [
        ("focus on the date picker", "llm_synthesized"),
        ("The previous attempt failed.", "llm_failed_static_fallback"),
    ]
    return hint_requests, hint_results


def test_emits_one_event_per_llm_synthesized_hint(fake_nemo):
    emitter = make_emitter(fake_nemo)
    hint_requests, hint_results = hint_fixtures()

    emitter.hint_synthesis_events(
        hint_requests=hint_requests,
        hint_results=hint_results,
        model=HINT_MODEL,
        global_step=7,
        phase="train_step_7",
    )

    # Only the llm_synthesized hint emits; fallbacks made no LLM call.
    ((_, name, _scope_type, push_kwargs),) = fake_nemo.named("push")
    assert name == "helper:hint_synthesis"
    meta = push_kwargs["metadata"]
    assert meta["call_site"] == "hint_synthesis"
    assert meta["instance_id"] == "i-1"
    assert meta["global_step"] == 7
    assert meta["phase"] == "train_step_7"
    ((_, _, request, response, llm_kwargs),) = fake_nemo.named("llm")
    assert request == {
        "task_prompt": "book a flight",
        "verifier_stdout": "0/3 checks",
        "verifier_error": None,
        "tool_error_messages": ["timeout"],
    }
    assert response == {"hint": "focus on the date picker", "category": "llm_synthesized"}
    assert llm_kwargs["model_name"] == HINT_MODEL
    assert len(fake_nemo.named("pop")) == 1


def test_none_step_phase_dropped_from_metadata(fake_nemo):
    emitter = make_emitter(fake_nemo)
    hint_requests, hint_results = hint_fixtures()

    emitter.hint_synthesis_events(hint_requests=hint_requests, hint_results=hint_results, model=HINT_MODEL)

    ((_, _, _, push_kwargs),) = fake_nemo.named("push")
    meta = push_kwargs["metadata"]
    assert "global_step" not in meta
    assert "phase" not in meta
    assert meta["instance_id"] == "i-1"


def test_malformed_results_fail_open(fake_nemo):
    emitter = make_emitter(fake_nemo)
    emitter.hint_synthesis_events(
        hint_requests=[{"task_prompt": "x"}],
        hint_results=["not-a-tuple"],
        model=HINT_MODEL,
    )
    assert fake_nemo.named("llm") == []
