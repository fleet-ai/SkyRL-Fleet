"""Slack notifications for auto-train events.

Uses the Slack Web API via SLACK_BOT_TOKEN. Channel from
SLACK_AUTO_TRAIN_CHANNEL (defaults to #fleet-training-runs).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from .config import SLACK_CHANNEL_DEFAULT

logger = logging.getLogger(__name__)


def _channel() -> str:
    # SLACK_TINKER_AUTO_TRAIN_CHANNEL routes Tinker pipeline notifications to a
    # separate channel so the two pipelines' signal stays distinguishable.
    tinker_chan = os.environ.get("SLACK_TINKER_AUTO_TRAIN_CHANNEL")
    if tinker_chan and os.environ.get("AUTO_TRAIN_BACKEND") == "tinker":
        return tinker_chan
    return os.environ.get("SLACK_AUTO_TRAIN_CHANNEL", SLACK_CHANNEL_DEFAULT)


def notify_tinker_eval(
    dataset_key: str,
    modality: str,
    base_model: str,
    pre_pass_rate: float,
    post_pass_rate: float,
    n_holdout: int,
    num_steps: int,
    wandb_url: str | None = None,
) -> None:
    delta = post_pass_rate - pre_pass_rate
    arrow = (
        ":chart_with_upwards_trend:" if delta > 0 else (":chart_with_downwards_trend:" if delta < 0 else ":bar_chart:")
    )
    wandb_line = f"\n• wandb: {wandb_url}" if wandb_url else ""
    slack_notify(
        f"{arrow} Tinker training finished\n"
        f"• dataset: `{dataset_key}`\n"
        f"• modality: `{modality}`\n"
        f"• base model: `{base_model}`\n"
        f"• steps: {num_steps}, holdout: {n_holdout}\n"
        f"• pre pass_at_1: {pre_pass_rate:.3f}\n"
        f"• post pass_at_1: {post_pass_rate:.3f}\n"
        f"• delta: {delta:+.3f}"
        f"{wandb_line}"
    )


def slack_notify(text: str, channel: Optional[str] = None) -> bool:
    """Post a message to Slack. Returns True on success, False otherwise.

    No-op (logs only) if SLACK_BOT_TOKEN is missing.
    """
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        logger.warning("SLACK_BOT_TOKEN not set; skipping Slack notify: %s", text)
        return False
    target = channel or _channel()
    try:
        r = httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": target, "text": text},
            timeout=10,
        )
        data = r.json()
        if not data.get("ok"):
            logger.error("Slack post failed: %s", data)
            return False
        return True
    except Exception as e:
        logger.error("Slack post error: %s", e)
        return False


def notify_launch(dataset_key: str, modality: str, task_count: int, s3_uri: str) -> None:
    slack_notify(
        f":rocket: Auto-launched training\n"
        f"• dataset: `{dataset_key}`\n"
        f"• modality: `{modality}`\n"
        f"• tasks: {task_count}\n"
        f"• s3: `{s3_uri}`"
    )


def notify_smoke_failure(dataset_key: str, modality: str, report) -> None:
    failures = report.failures()
    lines = [f"• `{r.env_key}`: {r.error}" for r in failures[:5]]
    if len(failures) > 5:
        lines.append(f"• ...and {len(failures) - 5} more")
    body = "\n".join(lines) if lines else "(no failure details)"
    slack_notify(
        f":warning: Smoke test FAILED, NOT launching training\n"
        f"• dataset: `{dataset_key}`\n"
        f"• modality: `{modality}`\n"
        f"• {report.summary()}\n"
        f"Failed envs:\n{body}"
    )


def notify_launch_failure(dataset_key: str, modality: str, reason: str) -> None:
    slack_notify(
        f":x: Training launch FAILED\n"
        f"• dataset: `{dataset_key}`\n"
        f"• modality: `{modality}`\n"
        f"• reason: {reason}"
    )


def notify_not_implemented(dataset_key: str, modality: str, env_count: int) -> None:
    slack_notify(
        f":construction: Skipped unsupported modality\n"
        f"• dataset: `{dataset_key}`\n"
        f"• modality: `{modality}` (no training YAML; {env_count} envs)\n"
        "Marked as processed; will not re-alert."
    )


def notify_below_threshold(dataset_key: str, modality: str, task_count: int, threshold: int) -> None:
    slack_notify(
        f":no_entry: Skipped: too few tasks to train\n"
        f"• dataset: `{dataset_key}`\n"
        f"• modality: `{modality}`\n"
        f"• tasks: {task_count} (min: {threshold})\n"
        "Marked as processed; will not re-alert."
    )
