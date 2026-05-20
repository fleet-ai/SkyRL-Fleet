import json

from integrations.fleet.task_gen_baseline.cli import (
    build_unscored_verifier_row,
    parse_verifier_judge_response,
    source_files_for_verifier_score,
    verifier_score_messages,
)


def test_parse_verifier_judge_response_extracts_fenced_json_and_clamps_score():
    raw = '```json\n{"score": 1.7, "issues": ["too broad"], "rationale": "Mostly okay."}\n```'

    score, issues, rationale, parsed, error = parse_verifier_judge_response(raw)

    assert score == 1.0
    assert issues == ["too broad"]
    assert rationale == "Mostly okay."
    assert parsed["score"] == 1.7
    assert error == ""


def test_parse_verifier_judge_response_handles_non_json():
    score, issues, rationale, parsed, error = parse_verifier_judge_response("not json")

    assert score == 0.0
    assert issues == ["judge_response_not_json"]
    assert rationale == "not json"
    assert parsed == {}
    assert error == "parse_error"


def test_build_unscored_verifier_row_extracts_generated_artifact(tmp_path):
    path = (
        tmp_path
        / "generated"
        / "booking"
        / "anthropic__claude-opus-4.7"
        / "01_task_abc.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "env_key": "booking",
                "fleet_task_key": "task_abc",
                "model": "anthropic/claude-opus-4.7",
                "prompt": "Update the booking.",
                "verifier": "def validate_task(env, final_answer=None):\n    return 0",
            }
        )
    )

    row = build_unscored_verifier_row(path)

    assert row["env_key"] == "booking"
    assert row["fleet_task_key"] == "task_abc"
    assert row["generator_model"] == "anthropic/claude-opus-4.7"
    assert row["artifact_kind"] == "generated"
    assert row["solver_prompt"] == "Update the booking."
    assert row["prompt_chars"] == len("Update the booking.")
    assert row["solver_prompt_chars"] == len("Update the booking.")
    assert row["verifier_chars"] > 0
    assert row["status"] == "pending"


def test_failed_generated_artifact_scores_zero_without_judge(tmp_path):
    path = tmp_path / "generated" / "booking" / "openai__gpt-5.5" / "01_task_abc.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "env_key": "booking",
                "fleet_task_key": "task_abc",
                "model": "openai/gpt-5.5",
                "done_reason": "validation_failed",
                "validation": {"valid": False, "failed": ["env_usage"], "error": "Verifier does not use env"},
                "prompt": "Update the booking.",
                "verifier": "def validate_task(env, final_answer=None):\n    return 1",
            }
        )
    )

    row = build_unscored_verifier_row(path)

    assert row["status"] == "invalid_input"
    assert row["score"] == 0.0
    assert row["issues"][0] == "known_generation_failure"
    assert "validation_failed" in row["error"]


def test_source_files_for_verifier_score_run_root_uses_generated_only(tmp_path):
    generated = tmp_path / "generated" / "booking" / "openai__gpt-5.5"
    solves = tmp_path / "solves" / "booking" / "openai__gpt-5.5" / "openai__gpt-4o-mini"
    generated.mkdir(parents=True)
    solves.mkdir(parents=True)
    generated_file = generated / "01_task_abc.json"
    solve_file = solves / "01_task_abc__k3__steps20.json"
    manifest_file = tmp_path / "manifest.json"
    generated_file.write_text(
        json.dumps(
            {
                "done_reason": "task_generated",
                "prompt": "Update the booking.",
                "verifier": "def validate_task(env, final_answer=None):\n    return 0",
            }
        )
    )
    solve_file.write_text(
        json.dumps(
            {
                "mode": "gates_and_solve",
                "evaluator_model": "openai/gpt-4o-mini",
                "prompt": "Update the booking.",
                "verifier": "def validate_task(env, final_answer=None):\n    return 0",
            }
        )
    )
    manifest_file.write_text(json.dumps({"counts": {"generation_jobs": 1}}))

    files = source_files_for_verifier_score(tmp_path, "*.json")

    assert files == [generated_file]


def test_solve_artifact_is_not_scored(tmp_path):
    path = tmp_path / "solves" / "booking" / "openai__gpt-5.5" / "openai__gpt-4o-mini" / "01_task.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "mode": "gates_and_solve",
                "env_key": "booking",
                "fleet_task_key": "task_abc",
                "evaluator_model": "openai/gpt-4o-mini",
                "k_rollouts": 3,
                "max_eval_steps": 20,
                "solver_scores": [1.0, 0.0, 1.0],
                "solver_pass_rate": 2 / 3,
                "prompt": "Update the booking.",
                "verifier": "def validate_task(env, final_answer=None):\n    return 0",
            }
        )
    )

    row = build_unscored_verifier_row(path)

    assert row["artifact_kind"] == "solve"
    assert row["status"] == "invalid_input"
    assert row["score"] == 0.0
    assert row["issues"] == ["not_generated_artifact"]
    assert source_files_for_verifier_score(path, "*.json") == []


def test_verifier_score_message_uses_generated_prompt_as_solver_facing_prompt(tmp_path):
    path = tmp_path / "generated" / "booking" / "openai__gpt-5.5" / "01_task_abc.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "env_key": "booking",
                "fleet_task_key": "task_abc",
                "model": "openai/gpt-5.5",
                "done_reason": "task_generated",
                "prompt": "Update the booking.",
                "verifier": "def validate_task(env, final_answer=None):\n    return 0",
            }
        )
    )

    row = build_unscored_verifier_row(path)
    message_text = verifier_score_messages(row)[1]["content"]

    assert "Solver-facing prompt:\nUpdate the booking." in message_text
    assert "Solver/run evidence:" not in message_text
