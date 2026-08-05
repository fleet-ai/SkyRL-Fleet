from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

FLEET_TRACE_HOST = "https://api.internal.fleet-platform.fleetai.com"


async def upload_group_session(
    *,
    api_key: str,
    session_id: str,
    job_id: str,
    task_key: str,
    model: str,
    score: Optional[float],
    metadata: dict[str, Any],
) -> bool:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "create_if_missing": True,
        "history": [],
        "job_id": job_id,
        "task_key": task_key,
        "model": model,
        "status": "completed",
        "metadata": metadata,
    }
    if score is not None:
        payload["score"] = score

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{FLEET_TRACE_HOST}/v1/traces/logs",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning(
            "SkyRL group session upload failed for %s; training will continue: %s",
            session_id,
            exc,
        )
        return False
