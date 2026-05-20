import argparse
import asyncio
import json

import pytest

from integrations.fleet.task_gen_baseline import cli


def test_extract_task_from_json_rebuilds_task_xml_from_prompt_and_verifier():
    fresh_verifier = "async def verify(env):\n    return 0.0"
    data = {
        "prompt": "Fresh prompt",
        "verifier": fresh_verifier,
        "task_xml": "<task><prompt>Stale prompt</prompt><verifier>stale</verifier></task>",
    }

    prompt, verifier, task_xml = cli.extract_task_from_text(json.dumps(data))

    assert prompt == "Fresh prompt"
    assert verifier == fresh_verifier
    assert "Fresh prompt" in task_xml
    assert "Stale prompt" not in task_xml


def test_validate_context_requires_controlled_generate_source():
    parser = cli.generate_cli()
    args = parser.parse_args(["generate", "--model", "openai/gpt-5.5"])

    with pytest.raises(ValueError, match="Pass --fleet-task-key"):
        cli.validate_context_arguments(args)


def test_validate_context_accepts_task_key_file_generate_source(tmp_path):
    task_key_file = tmp_path / "keys.json"
    task_key_file.write_text(json.dumps(["task_a"]))
    parser = cli.generate_cli()
    args = parser.parse_args(
        ["generate", "--model", "openai/gpt-5.5", "--fleet-task-key-file", str(task_key_file)]
    )

    cli.validate_context_arguments(args)


def test_validate_context_rejects_partial_snapshot():
    parser = cli.generate_cli()
    args = parser.parse_args(
        [
            "generate",
            "--model",
            "openai/gpt-5.5",
            "--env-key",
            "zillow",
            "--data-key",
            "seed_123",
            "--allow-live-task-list",
        ]
    )

    with pytest.raises(ValueError, match="--env-key, --data-key, and --data-version"):
        cli.validate_context_arguments(args)


def test_attach_rollout_metrics_uses_solver_pass_rate_as_primary_signal():
    result = {"training_phase": "eval", "eval_k_rollouts": 3, "k_rollouts": 4}
    rollout_record = {
        "raw_scores": [0.0, 1.0, 1.0],
        "hinted_scores": [],
        "raw_job_id": "job_123",
        "hinted_job_id": None,
    }

    cli.attach_rollout_metrics(result, rollout_record)

    assert result["solver_scores"] == [0.0, 1.0, 1.0]
    assert result["solver_rollouts"] == 3
    assert result["solver_pass_count"] == 2
    assert result["solver_pass_rate"] == pytest.approx(2 / 3)
    assert result["solver_pass_at_k"] is True


def test_attach_rollout_metrics_rejects_missing_raw_job_id():
    result = {"training_phase": "eval", "eval_k_rollouts": 1, "k_rollouts": 4}
    rollout_record = {
        "raw_scores": [0.0],
        "hinted_scores": [],
        "raw_job_id": None,
        "hinted_job_id": None,
    }

    with pytest.raises(RuntimeError, match="raw solver job"):
        cli.attach_rollout_metrics(result, rollout_record)


def test_execute_native_tool_calls_keeps_tool_outputs_separate():
    actions = []

    class FakeEnv:
        callable_tools = {"query_db"}

        async def step_async(self, action):
            actions.append(action)
            return {
                "observations": [{"role": "user", "content": f"result {len(actions)}"}],
                "reward": 0.0,
                "done": False,
                "metadata": {"tool_calls": ["query_db"]},
            }

    native_tool_calls = [
        {"id": "call_1", "name": "query_db", "arguments": {"sql": "select 1"}, "argument_error": ""},
        {"id": "call_2", "name": "query_db", "arguments": {"sql": "select 2"}, "argument_error": ""},
    ]

    step_output, tool_messages = asyncio.run(cli.execute_native_tool_calls_async(FakeEnv(), native_tool_calls))

    assert step_output["done"] is False
    assert [message["tool_call_id"] for message in tool_messages] == ["call_1", "call_2"]
    assert [message["content"] for message in tool_messages] == ["result 1", "result 2"]
