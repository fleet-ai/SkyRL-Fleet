from pathlib import Path
import sys
from types import SimpleNamespace


SKYRL_GYM_SRC = Path(__file__).resolve().parents[1]
if str(SKYRL_GYM_SRC) not in sys.path:
    sys.path.insert(0, str(SKYRL_GYM_SRC))

from skyrl_gym.envs.task_gen.task_gen_env import TaskGenEnv


def make_env(tmp_path):
    env = object.__new__(TaskGenEnv)
    env._rollout_dir = str(tmp_path)
    env.max_eval_steps = 10
    env.env_key = "booking"
    env.data_key = "data"
    env.data_version = "v1"
    env.evaluator_model = "solver"
    return env


class FakeFleet:
    def __init__(self, sessions, transcripts=None, error=None):
        self.sessions = sessions
        self.transcripts = transcripts or {}
        self.error = error
        self.transcript_calls = []

    def list_job_sessions(self, job_id):
        return SimpleNamespace(tasks=[SimpleNamespace(task_key="task-key", sessions=self.sessions)])

    def get_session_transcript(self, session_id):
        self.transcript_calls.append(session_id)
        if self.error:
            raise RuntimeError(self.error)
        return self.transcripts[session_id]


def verifier_execution(score=1.0):
    return SimpleNamespace(
        id="ve-1",
        verifier_execution_id="",
        score=score,
        success=score >= 1.0,
        stdout="ok",
        result=None,
    )


def test_extract_job_results_fetches_and_projects_session_transcript(tmp_path):
    session = SimpleNamespace(session_id="session-1", id="", verifier_execution=verifier_execution(0.5))
    transcript = {
        "messages": [{"role": "user", "content": "solve it"}],
        "actions": [{"tool": "search"}],
    }
    fleet = FakeFleet([session], transcripts={"session-1": transcript})

    results = make_env(tmp_path)._extract_job_results(
        fleet,
        "job-1",
        task_key="task-key",
        rollout_label="raw",
        pass_k=1,
    )

    assert fleet.transcript_calls == ["session-1"]
    assert results[0]["session_transcript"] == transcript
    assert results[0]["messages"] == transcript["messages"]
    assert results[0]["trajectory"] == transcript["messages"]
    assert results[0]["actions"] == transcript["actions"]


def test_extract_job_results_skips_transcript_fetch_when_session_has_messages(tmp_path):
    messages = [{"role": "assistant", "content": "done"}]
    session = SimpleNamespace(
        session_id="session-1",
        id="",
        verifier_execution=verifier_execution(1.0),
        messages=messages,
    )
    fleet = FakeFleet([session], error="should not fetch")

    results = make_env(tmp_path)._extract_job_results(
        fleet,
        "job-1",
        task_key="task-key",
        rollout_label="raw",
        pass_k=1,
    )

    assert fleet.transcript_calls == []
    assert results[0]["messages"] == messages
    assert results[0]["session_transcript"] == messages


def test_extract_job_results_records_transcript_fetch_error_per_session(tmp_path):
    session = SimpleNamespace(session_id="session-1", id="", verifier_execution=verifier_execution(1.0))
    fleet = FakeFleet([session], error="transcript unavailable")

    results = make_env(tmp_path)._extract_job_results(
        fleet,
        "job-1",
        task_key="task-key",
        rollout_label="raw",
        pass_k=1,
    )

    assert fleet.transcript_calls == ["session-1"]
    assert results[0]["score"] == 1.0
    assert results[0]["session_transcript_error"] == "transcript unavailable"
