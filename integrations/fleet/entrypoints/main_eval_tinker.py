"""Eval-only trace collection for Tinker checkpoints over Fleet tasks.

``main_eval.py`` resumes SkyRL FSDP checkpoints; ``main_fleet_tinker.py`` runs the
Tinker training loop. This module fills the gap between them: it rolls one or
more Tinker *sampler* checkpoints over a fixed set of Fleet tasks with **no
training**, capturing each rollout's full transcript (messages, tool calls,
reward, turns, stop reason) to JSON. The intended use is diffing a model's
behavior across checkpoints — e.g. ``step_pretrain`` vs ``step_final`` on the
same task set — for skill/regression analysis.

The rollout loop mirrors ``main_fleet_tinker.collect_fleet_rollout`` (same
``FleetTaskEnv`` + Tinker sampling), but returns the conversation transcript
rather than tokenized training tensors.

Scoring matches training: ``--partial-reward`` (on by default, as in
``main_fleet_tinker``) asks the verifier for fractional credit from its
accumulator counts instead of a binary 0/1, so a checkpoint that solves *part*
of a task is distinguishable from one that solves none. A full pass is
``reward >= 1.0``; each trace records a ``passed`` flag for that threshold.

Outputs per checkpoint label, under ``<output-dir>/<label>/``:
  - ``<task_key>__s<sample>.json``: one file per rollout (full transcript).
  - ``summary.json``: the aggregate (overall + per-task ``pass_rate`` and
    ``mean_reward``) — the structured contract an external orchestrator reads.
A single ``RESULT: {...}`` line is also logged per checkpoint for log tailing.

Example:
    python -m integrations.fleet.entrypoints.main_eval_tinker \\
        --checkpoint step_pretrain=tinker://<run>:train:0/sampler_weights/step_pretrain \\
        --checkpoint step_final=tinker://<run>:train:0/sampler_weights/step_final \\
        --tasks-file ~/data/tasks.json \\
        --base-model moonshotai/Kimi-K2.6 \\
        --output-dir ~/eval_traces --n-samples 1 --concurrency 10 --partial-reward

Environment variables:
    TINKER_API_KEY: required. Authenticates Tinker sampling.
    FLEET_API_KEY:  required. Authenticates Fleet environment provisioning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import tinker
from omegaconf import OmegaConf
from tinker import types
from transformers import AutoTokenizer

from skyrl_gym.envs.fleet_task.env import FleetTaskEnv

logger = logging.getLogger(__name__)


@dataclass
class RolloutTrace:
    """A single checkpoint's rollout on a single task.

    Attributes:
        checkpoint: Caller-supplied label for the checkpoint (e.g. ``step_final``).
        task_key: The Fleet task key that was rolled out.
        sample: Sample index, for runs with ``n_samples`` > 1.
        env_key: The Fleet environment key reported at reset (``None`` on error).
        reward: Final verifier reward in ``[0, 1]``. Fractional when
            ``partial_reward`` is set (partial credit from the verifier's
            accumulator counts); otherwise binary ``0.0``/``1.0``.
        passed: Whether the rollout fully passed (``reward >= 1.0``). Lets
            consumers separate a true pass from partial credit without
            re-deriving the threshold.
        partial_reward: Whether partial-credit scoring was requested for this
            rollout (recorded for provenance).
        turns: Number of agent turns taken.
        tool_calls: Count of successfully parsed/executed tool calls.
        tool_errors: Count of turns whose tool call failed to parse/execute.
        stop_reason: ``"stop"`` (env signalled done) or ``"length"`` (context cap).
        duration_s: Wall-clock seconds for the rollout.
        messages: The full conversation transcript (role/content per turn).
        error: Populated with the exception repr if the rollout failed.
    """

    checkpoint: str
    task_key: str
    sample: int
    env_key: str | None
    reward: float
    passed: bool
    partial_reward: bool
    turns: int
    tool_calls: int
    tool_errors: int
    stop_reason: str
    duration_s: float
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _tokenize_chat(
    tokenizer: AutoTokenizer,
    chat_history: list[dict[str, Any]],
    add_generation_prompt: bool = True,
) -> list[int]:
    """Apply the chat template and return a plain list of token ids.

    ``apply_chat_template`` returns either a list, a ``BatchEncoding``, or a dict
    depending on the tokenizer; Tinker's ``ModelInput.from_ints`` needs a plain
    list of ints.

    Args:
        tokenizer: The base-model tokenizer.
        chat_history: The running conversation to tokenize.
        add_generation_prompt: Whether to append the assistant generation prefix.

    Returns:
        The prompt token ids.
    """
    result = tokenizer.apply_chat_template(
        chat_history, add_generation_prompt=add_generation_prompt, tokenize=True
    )
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    if isinstance(result, dict) and "input_ids" in result:
        return list(result["input_ids"])
    return list(result)


async def _run_in_executor(func: Callable[..., Any], *args: Any) -> Any:
    """Run a blocking callable in the default thread pool.

    ``FleetTaskEnv``'s ``init``/``step`` are synchronous and open MCP connections;
    running them off the event loop keeps concurrent rollouts from blocking.
    """
    return await asyncio.get_event_loop().run_in_executor(None, func, *args)


async def collect_eval_trace(
    task_config: dict[str, Any],
    tasks_file: str,
    sampling_client: "tinker.SamplingClient",
    tokenizer: AutoTokenizer,
    *,
    checkpoint: str,
    sample: int,
    partial_reward: bool = True,
    max_turns: int = 50,
    max_generate_length: int = 3000,
    max_input_length: int = 128000,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> RolloutTrace:
    """Roll one task against one checkpoint and capture the transcript.

    Args:
        task_config: The task entry from ``tasks_file`` (``key``, ``prompt``,
            ``env_id``, ``version``, ``data_id``, ``data_version``, ...).
        tasks_file: Path to the ``FleetTaskEnv`` tasks JSON; the env loads the
            task config from it by key.
        sampling_client: A Tinker sampling client bound to the checkpoint.
        tokenizer: Tokenizer for the base model (chat template + decode).
        checkpoint: Label recorded on the returned trace.
        sample: Sample index recorded on the returned trace.
        partial_reward: When ``True`` (default, matching training), score with
            fractional verifier credit instead of binary 0/1. A full pass is
            still ``reward >= 1.0``.
        max_turns: Hard cap on agent turns.
        max_generate_length: Max tokens sampled per turn.
        max_input_length: Context-length cap; the rollout ends with
            ``stop_reason="length"`` once exceeded.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling threshold.

    Returns:
        A populated :class:`RolloutTrace`. Failures are captured in ``error``
        rather than raised, so a single bad task never aborts a batch.
    """
    env = FleetTaskEnv(
        env_config=OmegaConf.create(
            {"tasks_file": tasks_file, "ttl_seconds": 7200, "partial_reward": partial_reward}
        ),
        extras={"task_key": task_config["key"], "max_turns": max_turns},
    )
    started = time.time()
    env_key: str | None = None
    reward, stop_reason, done = 0.0, "stop", False
    try:
        _, metadata = await _run_in_executor(env.init, [])
        env_key = metadata.get("env_key")
        while not done and env.turns < max_turns:
            input_ids = _tokenize_chat(tokenizer, env.chat_history, add_generation_prompt=True)
            if len(input_ids) > max_input_length:
                stop_reason = "length"
                break
            result = await sampling_client.sample_async(
                prompt=types.ModelInput.from_ints(tokens=input_ids),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=max_generate_length, temperature=temperature, top_p=top_p
                ),
            )
            if not result.sequences:
                break
            output_text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
            step_output = await _run_in_executor(env.step, output_text)
            reward = step_output["reward"]
            done = step_output["done"]
        return RolloutTrace(
            checkpoint=checkpoint, task_key=task_config["key"], sample=sample,
            env_key=env_key, reward=reward, passed=reward >= 1.0, partial_reward=partial_reward, turns=env.turns,
            tool_calls=env.tool_calls, tool_errors=env.tool_errors,
            stop_reason=stop_reason, duration_s=round(time.time() - started, 1),
            messages=list(env.chat_history),
        )
    except Exception as exc:  # noqa: BLE001 — one bad task must not kill the batch
        logger.exception("rollout failed: %s / %s", checkpoint, task_config["key"])
        return RolloutTrace(
            checkpoint=checkpoint, task_key=task_config["key"], sample=sample,
            env_key=env_key, reward=reward, passed=reward >= 1.0, partial_reward=partial_reward, turns=env.turns,
            tool_calls=env.tool_calls, tool_errors=env.tool_errors,
            stop_reason="error", duration_s=round(time.time() - started, 1),
            messages=list(env.chat_history), error=repr(exc),
        )
    finally:
        env.close()


def summarize_checkpoint(
    checkpoint: str, traces: list[RolloutTrace], *, partial_reward: bool
) -> dict[str, Any]:
    """Aggregate one checkpoint's rollout traces into pass-rate + mean reward.

    A rollout counts as a pass when ``trace.passed`` (``reward >= 1.0``); the
    mean reward additionally credits partial solves when ``partial_reward`` is
    set. Error traces (``trace.error`` populated) carry ``reward=0.0`` and so
    fold in as failures, which is the intended scoring.

    Args:
        checkpoint: The checkpoint label these traces came from.
        traces: Every rollout (all tasks x all samples) for the checkpoint.
        partial_reward: Whether the rollouts were scored with partial credit;
            recorded on the summary for provenance.

    Returns:
        A JSON-serializable summary with overall ``pass_rate``/``mean_reward``
        and a ``per_task`` breakdown. This is the structured contract a caller
        (e.g. the Fleet research API's eval controller) reads instead of
        scraping the per-rollout logs.
    """
    per_task: dict[str, dict[str, Any]] = {}
    for t in traces:
        acc = per_task.setdefault(t.task_key, {"rewards": [], "n_passed": 0})
        acc["rewards"].append(t.reward)
        acc["n_passed"] += int(t.passed)
    per_task_summary: dict[str, dict[str, Any]] = {}
    for key, acc in per_task.items():
        rewards = acc["rewards"]
        n = len(rewards)
        per_task_summary[key] = {
            "n": n,
            "n_passed": acc["n_passed"],
            "pass_rate": acc["n_passed"] / n if n else 0.0,
            "mean_reward": sum(rewards) / n if n else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }
    n_rollouts = len(traces)
    n_passed = sum(int(t.passed) for t in traces)
    return {
        "checkpoint": checkpoint,
        "partial_reward": partial_reward,
        "n_tasks": len(per_task),
        "n_rollouts": n_rollouts,
        "n_passed": n_passed,
        "pass_rate": n_passed / n_rollouts if n_rollouts else 0.0,
        "mean_reward": sum(t.reward for t in traces) / n_rollouts if n_rollouts else 0.0,
        "per_task": per_task_summary,
    }


async def run_checkpoint(
    label: str,
    checkpoint_uri: str,
    tasks: list[dict[str, Any]],
    tasks_file: str,
    tokenizer: AutoTokenizer,
    service_client: "tinker.ServiceClient",
    output_dir: Path,
    *,
    n_samples: int,
    concurrency: asyncio.Semaphore,
    rollout_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Roll every task (``n_samples`` each) against one checkpoint, writing JSON.

    Each completed rollout is written immediately to
    ``<output_dir>/<label>/<task_key>__s<sample>.json`` so partial progress
    survives an interrupted run. After all rollouts finish, an aggregate
    ``<output_dir>/<label>/summary.json`` is written and a single ``RESULT:``
    line is logged, both consumable by an external orchestrator.

    Returns:
        The :func:`summarize_checkpoint` aggregate for this checkpoint.
    """
    sampling_client = service_client.create_sampling_client(model_path=checkpoint_uri)
    dest = output_dir / label
    dest.mkdir(parents=True, exist_ok=True)

    async def _one(task_config: dict[str, Any], sample: int) -> RolloutTrace:
        async with concurrency:
            trace = await collect_eval_trace(
                task_config, tasks_file, sampling_client, tokenizer,
                checkpoint=label, sample=sample, **rollout_kwargs,
            )
            (dest / f"{task_config['key']}__s{sample}.json").write_text(
                json.dumps(asdict(trace), indent=2, default=str)
            )
            logger.info(
                "[%s] %s s%d: reward=%s pass=%s turns=%d stop=%s dur=%ss%s",
                label, trace.task_key, sample, trace.reward, trace.passed, trace.turns,
                trace.stop_reason, trace.duration_s,
                f" error={trace.error}" if trace.error else "",
            )
            return trace

    traces = await asyncio.gather(*[_one(tc, i) for tc in tasks for i in range(n_samples)])
    summary = summarize_checkpoint(
        label, list(traces), partial_reward=rollout_kwargs.get("partial_reward", True)
    )
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    # Single machine-parseable line: orchestrators tail stdout for this.
    logger.info("RESULT: %s", json.dumps({
        "benchmark": "fleet-internal", "checkpoint": label, "status": "success",
        "pass_rate": summary["pass_rate"], "mean_reward": summary["mean_reward"],
        "n": summary["n_rollouts"],
    }))
    return summary


def _parse_checkpoint(spec: str) -> tuple[str, str]:
    """Parse a ``label=uri`` checkpoint spec into ``(label, uri)``."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--checkpoint must be label=uri, got {spec!r}")
    label, uri = spec.split("=", 1)
    return label, uri


async def _main_async(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    service_client = tinker.ServiceClient(api_key=os.environ["TINKER_API_KEY"])
    tasks = json.loads(Path(args.tasks_file).expanduser().read_text())
    # Accept either a bare array or the {"tasks": [...]} wrapper that the
    # training exporter and FleetTaskEnv's own loader both emit/accept.
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]
    output_dir = Path(args.output_dir).expanduser()
    concurrency = asyncio.Semaphore(args.concurrency)
    rollout_kwargs = {
        "partial_reward": args.partial_reward,
        "max_turns": args.max_turns,
        "max_generate_length": args.max_generate_length,
        "max_input_length": args.max_input_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    for label, uri in args.checkpoint:
        logger.info("=== %s: %d tasks x %d samples ===", label, len(tasks), args.n_samples)
        summary = await run_checkpoint(
            label, uri, tasks, args.tasks_file, tokenizer, service_client, output_dir,
            n_samples=args.n_samples, concurrency=concurrency, rollout_kwargs=rollout_kwargs,
        )
        logger.info(
            "=== %s summary: pass_rate=%.3f mean_reward=%.3f (%d/%d rollouts passed) ===",
            label, summary["pass_rate"], summary["mean_reward"],
            summary["n_passed"], summary["n_rollouts"],
        )
    logger.info("done; traces under %s", output_dir)


def main() -> None:
    """CLI entry point. See module docstring for usage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Eval-only Tinker trace collection over Fleet tasks")
    parser.add_argument(
        "--checkpoint", type=_parse_checkpoint, action="append", required=True, dest="checkpoint",
        metavar="LABEL=URI", help="Checkpoint to roll out, e.g. step_final=tinker://...; repeatable.",
    )
    parser.add_argument("--tasks-file", required=True, help="FleetTaskEnv tasks JSON (array of task configs).")
    parser.add_argument("--base-model", required=True, help="HF model id for the tokenizer/chat template.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-rollout JSON traces.")
    parser.add_argument("--n-samples", type=int, default=1, help="Samples per task per checkpoint.")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent rollouts.")
    parser.add_argument(
        "--partial-reward", action=argparse.BooleanOptionalAction, default=True,
        help="Score with fractional verifier credit (matches training, default on). "
             "Pass --no-partial-reward for binary 0/1.",
    )
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-generate-length", type=int, default=3000)
    parser.add_argument("--max-input-length", type=int, default=128000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
