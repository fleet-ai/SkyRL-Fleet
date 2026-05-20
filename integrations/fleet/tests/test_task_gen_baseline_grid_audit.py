import json
from pathlib import Path

from scripts.fleet_task_gen_baseline_grid import (
    SolveJob,
    solve_rollout_artifacts_complete,
    solve_row_from_manifest_job,
)


def make_solve_job(tmp_path, eval_k_rollouts=1):
    return SolveJob(
        job_id="solve-1",
        generation_job_id="gen-1",
        env="booking",
        task_index=1,
        task_key="task-key",
        generator_model="generator",
        solver_model="solver",
        eval_k_rollouts=eval_k_rollouts,
        max_eval_steps=20,
        generated_file=str(tmp_path / "generated.json"),
        rollout_dir=str(tmp_path / "rollouts"),
        output=str(tmp_path / "solve.json"),
        command=[],
    )


def write_rollout_artifacts(job, raw_sessions):
    rollout_dir = Path(job.rollout_dir)
    rollout_dir.mkdir(parents=True)
    raw_job_id = "fleet-job-1"
    rollout_file = rollout_dir / "run.jsonl"
    rollout_file.write_text(
        json.dumps(
            {
                "raw_job_id": raw_job_id,
                "raw_scores": [1.0] * job.eval_k_rollouts,
                "raw_sessions": raw_sessions,
            }
        )
        + "\n"
    )
    status_file = rollout_dir / "fleet_job_status.jsonl"
    status_file.write_text(json.dumps({"job_id": raw_job_id, "event": "terminal", "status": "completed"}) + "\n")
    jobs_dir = rollout_dir / "fleet_jobs"
    jobs_dir.mkdir()
    (jobs_dir / f"{raw_job_id}.json").write_text(json.dumps({"job_id": raw_job_id}) + "\n")
    return {"raw_job_id": raw_job_id, "rollout_file": str(rollout_file)}


def test_solve_rollout_audit_accepts_session_transcript(tmp_path):
    job = make_solve_job(tmp_path)
    data = write_rollout_artifacts(
        job,
        [{"session_id": "session-1", "session_transcript": {"messages": [{"role": "user", "content": "hi"}]}}],
    )

    assert solve_rollout_artifacts_complete(data, job)


def test_solve_rollout_audit_rejects_empty_session_transcript(tmp_path):
    job = make_solve_job(tmp_path)
    data = write_rollout_artifacts(
        job,
        [{"session_id": "session-1", "session_transcript": {"task": {"key": "task"}, "transcript": []}}],
    )

    assert not solve_rollout_artifacts_complete(data, job)


def test_solve_rollout_audit_accepts_nested_nonempty_transcript(tmp_path):
    job = make_solve_job(tmp_path)
    data = write_rollout_artifacts(
        job,
        [
            {
                "session_id": "session-1",
                "session_transcript": {
                    "task": {"key": "task"},
                    "transcript": [{"role": "user", "content": "do it"}],
                },
            }
        ],
    )

    assert solve_rollout_artifacts_complete(data, job)


def test_solve_rollout_audit_rejects_raw_session_without_trajectory_payload(tmp_path):
    job = make_solve_job(tmp_path)
    data = write_rollout_artifacts(job, [{"session_id": "session-1", "raw_session": {"id": "session-1"}}])

    assert not solve_rollout_artifacts_complete(data, job)


def test_solve_rollout_audit_rejects_zero_expected_rollouts(tmp_path):
    job = make_solve_job(tmp_path, eval_k_rollouts=0)
    data = write_rollout_artifacts(job, [])

    assert not solve_rollout_artifacts_complete(data, job)


def test_solve_row_allows_zero_rollout_when_gates_stop_before_solver(tmp_path):
    output = tmp_path / "solve.json"
    output.write_text(
        json.dumps(
            {
                "mode": "gates_and_solve",
                "task_gen_reward": 0.0,
                "metadata": {"reward_breakdown": {"sandbox": 0.0, "dryrun": 0.0, "judge": 0.0}},
                "solver_rollouts": 0,
            }
        )
        + "\n"
    )
    job = make_solve_job(tmp_path).__dict__ | {"output": str(output)}

    row = solve_row_from_manifest_job(tmp_path, {}, job)

    assert row["successful_solve"] is True
    assert row["audit_complete"] is True
    assert row["error"] == ""
    assert row["solver_rollouts"] == 0
