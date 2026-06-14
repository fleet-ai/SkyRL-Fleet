"""Guarded rollout loop that emits auto-labeled (context, action) -> outcome examples.

Modality- and model-agnostic by design: the policy (the model) and the driver
(how an action is actuated + what the agent observes) are injected. The loop
owns only the parts that are the same for every route:

  - cursor-diff sense attribution around each action (the free ground-truth label),
  - the sensory on/off switch (whether the env's rendered `text` enters the
    observation the agent sees — both arms still record the label),
  - doom-loop guardrails keyed off the SENSE delta and action repetition, NOT
    pixels (robust to falmart's auto-advancing homepage carousel), and
  - JSONL example serialization.

Plug in:
  * a `Policy` (e.g. Qwen3.5-9B over OpenRouter for tool_use, or vLLM for VL), and
  * a `Driver` (e.g. a local Playwright browser against `pnpm dev`, or a Fleet
    computer-use instance) that exposes the env origin for the SenseClient.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .sense import Outcome, SenseClient, SenseDelta


@dataclass
class Observation:
    """What the agent sees at one step.

    `text` is any textual observation from the env (tool result, page summary).
    `image_ref` is a lightweight pointer (path/url/id) to a screenshot for VL
    runs — kept out of the example payload so JSONL stays small. `image` holds
    the in-memory bytes/base64 the policy needs but we never serialize.
    """

    text: Optional[str] = None
    image_ref: Optional[str] = None
    image: Optional[Any] = field(default=None, repr=False)


class Policy(Protocol):
    def act(self, observation: Observation, history: List[Dict[str, Any]]) -> str:
        """Return the assistant action string (tool call, click, or <done>)."""
        ...


class Driver(Protocol):
    base_url: str  # env origin for the SenseClient (e.g. http://localhost:5173)

    def reset(self) -> Observation:
        """Start a fresh episode (load the storefront) and return obs."""
        ...

    def execute(self, action: str) -> Observation:
        """Actuate an action and return the raw env observation (no sense text)."""
        ...

    def close(self) -> None:
        ...


@dataclass
class RolloutConfig:
    sensory_on: bool = True
    # Guardrails. Low max_steps because exploration episodes don't need 50/80,
    # and a doom-loop must not be allowed to burn the budget (wasting tokens on
    # a doom-loop is bad signal).
    max_steps: int = 20
    loop_break_consecutive_empty: int = 4   # N no-op clicks in a row -> stop
    loop_break_repeat_action: int = 3       # same action N times in a row -> stop
    done_markers: tuple = ("<done>", "[done]")


@dataclass
class TrainingExample:
    """One self-supervised (context, action) -> outcome datum."""

    episode_id: str
    task_key: str
    step: int
    sensory_on: bool
    # Context the policy had BEFORE acting (the prediction is conditioned on this).
    context_text: Optional[str]
    context_image_ref: Optional[str]
    action: str
    # Label (the env's free ground truth).
    outcome: str             # coarse 4-class target
    outcome_fine: str        # 5-class (telemetry-write kept distinct)
    effects: List[str]
    procedures: List[str]
    routes: List[Dict[str, str]]
    flags: List[str]
    statuses: List[int]
    sense_text: Optional[str]   # the env-rendered gloss for this delta
    has_unknown: bool           # vocabulary miss in this delta
    reliable: bool              # False if the cursor dropped records (skip these)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class EpisodeResult:
    episode_id: str
    task_key: str
    sensory_on: bool
    examples: List[TrainingExample]
    steps: int
    stop_reason: str
    outcome_counts: Dict[str, int]


def _is_done(action: str, markers: tuple) -> bool:
    low = action.lower()
    return any(m in low for m in markers)


def _compose_observation(
    raw: Observation, delta_text: Optional[str], sensory_on: bool
) -> Observation:
    """Build the next observation, injecting the env gloss iff sensory_on."""
    if not sensory_on or not delta_text:
        return raw
    text = (
        f"{raw.text}\n\n{delta_text}" if raw.text else delta_text
    )
    return Observation(text=text, image_ref=raw.image_ref, image=raw.image)


def run_episode(
    policy: Policy,
    driver: Driver,
    sense: SenseClient,
    task: Dict[str, Any],
    config: RolloutConfig,
    episode_id: str,
) -> EpisodeResult:
    """Run one episode and return the labeled examples + episode metadata."""
    task_key = task.get("key", "task")
    prompt = task["prompt"]

    obs = driver.reset()
    # Consume the page-load rpcs (getBanners, products.list, ...) so step 0's
    # action gets a clean cursor diff attributed only to that action.
    sense.snapshot()

    # Seed the observation with the task prompt.
    obs = Observation(
        text=(f"{prompt}\n\n{obs.text}" if obs.text else prompt),
        image_ref=obs.image_ref,
        image=obs.image,
    )

    history: List[Dict[str, Any]] = []
    examples: List[TrainingExample] = []
    outcome_counts: Dict[str, int] = {}
    recent_actions: deque = deque(maxlen=config.loop_break_repeat_action)
    consecutive_empty = 0
    stop_reason = "max_steps"

    for step in range(config.max_steps):
        action = policy.act(obs, history)
        history.append({"role": "assistant", "content": action})

        if _is_done(action, config.done_markers):
            stop_reason = "agent_done"
            break

        context_text = obs.text
        context_image_ref = obs.image_ref

        raw_obs = driver.execute(action)
        delta: SenseDelta = sense.read_delta()

        coarse = delta.coarse_4class()
        outcome_counts[coarse.value] = outcome_counts.get(coarse.value, 0) + 1

        examples.append(TrainingExample(
            episode_id=episode_id,
            task_key=task_key,
            step=step,
            sensory_on=config.sensory_on,
            context_text=context_text,
            context_image_ref=context_image_ref,
            action=action,
            outcome=coarse.value,
            outcome_fine=delta.outcome.value,
            effects=delta.effects,
            procedures=delta.procedures,
            routes=delta.routes,
            flags=delta.flags,
            statuses=delta.statuses,
            sense_text=delta.text,
            has_unknown=delta.has_unknown,
            reliable=delta.reliable,
        ))

        obs = _compose_observation(raw_obs, delta.text, config.sensory_on)
        history.append({"role": "user", "content": obs.text or "(no text observation)"})

        # --- Guardrails (keyed off the sense delta + action repetition) ---
        if coarse is Outcome.EMPTY_DELTA:
            consecutive_empty += 1
        else:
            consecutive_empty = 0
        if consecutive_empty >= config.loop_break_consecutive_empty:
            stop_reason = "loop_break_empty"
            break

        norm = " ".join(action.split())
        recent_actions.append(norm)
        if (
            len(recent_actions) == recent_actions.maxlen
            and len(set(recent_actions)) == 1
        ):
            stop_reason = "loop_break_repeat"
            break

    return EpisodeResult(
        episode_id=episode_id,
        task_key=task_key,
        sensory_on=config.sensory_on,
        examples=examples,
        steps=len(examples),
        stop_reason=stop_reason,
        outcome_counts=outcome_counts,
    )


class JsonlWriter:
    """Append-only JSONL sink for TrainingExample rows."""

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def __enter__(self) -> "JsonlWriter":
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def write_episode(self, result: EpisodeResult) -> None:
        assert self._fh is not None
        for ex in result.examples:
            self._fh.write(ex.to_json() + "\n")
        self._fh.flush()

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
