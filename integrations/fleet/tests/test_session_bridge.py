from __future__ import annotations

import sys
import types

import pytest

from integrations.fleet.session_bridge import upload_group_session


class _Response:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error


class _Client:
    def __init__(self, response: _Response, calls: list[dict]):
        self.response = response
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url, *, json, headers):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.response


def _install_httpx(monkeypatch, response: _Response, calls: list[dict]):
    module = types.ModuleType("httpx")
    module.AsyncClient = lambda **_kwargs: _Client(response, calls)
    monkeypatch.setitem(sys.modules, "httpx", module)


@pytest.mark.asyncio
async def test_group_upload_sends_metadata_only_session(monkeypatch):
    calls: list[dict] = []
    _install_httpx(monkeypatch, _Response(), calls)

    uploaded = await upload_group_session(
        api_key="secret",
        session_id="11111111-1111-5111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        task_key="task-1",
        model="model-1",
        score=1.0,
        metadata={
            "skyrl_session_kind": "group",
            "skyrl_expected_rollouts": 8,
            "skyrl_completed_rollouts": 8,
        },
        status="completed",
    )

    assert uploaded is True
    assert calls[0]["json"] == {
        "session_id": "11111111-1111-5111-8111-111111111111",
        "create_if_missing": True,
        "history": [],
        "job_id": "22222222-2222-4222-8222-222222222222",
        "task_key": "task-1",
        "model": "model-1",
        "status": "completed",
        "score": 1.0,
        "metadata": {
            "skyrl_session_kind": "group",
            "skyrl_expected_rollouts": 8,
            "skyrl_completed_rollouts": 8,
        },
    }


@pytest.mark.asyncio
async def test_group_upload_is_a_soft_failure(monkeypatch):
    calls: list[dict] = []
    _install_httpx(monkeypatch, _Response(RuntimeError("409 conflict")), calls)

    uploaded = await upload_group_session(
        api_key="secret",
        session_id="11111111-1111-5111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        task_key="task-1",
        model="model-1",
        score=None,
        metadata={"skyrl_session_kind": "group"},
    )

    assert uploaded is False


@pytest.mark.asyncio
async def test_group_upload_can_create_an_active_session(monkeypatch):
    calls: list[dict] = []
    _install_httpx(monkeypatch, _Response(), calls)

    uploaded = await upload_group_session(
        api_key="secret",
        session_id="11111111-1111-5111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        task_key="task-1",
        model="model-1",
        score=None,
        metadata={
            "skyrl_session_kind": "group",
            "skyrl_expected_rollouts": 8,
        },
    )

    assert uploaded is True
    assert "status" not in calls[0]["json"]
    assert "score" not in calls[0]["json"]
