"""Pluggable Policy + Driver for the VL / computer_use route — IMPLEMENT THESE.

The data-engine core (`rollout.run_episode`) is complete and tested. To collect
real traces on the cluster you implement exactly two seams against the Protocols
in `rollout.py`:

  * Policy.act(observation, history) -> action string
  * Driver.reset()/execute(action)/close()  (+ a `base_url` attribute)

These stubs document the contract. Fill in the bodies marked `TODO(cluster)`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .rollout import Observation


class VLLMQwenPolicy:
    """Qwen3.5-9B (natively multimodal) served by vLLM, OpenAI-compatible API.

    For the VL arms the observation carries a screenshot (`observation.image`)
    and `observation.text`. Send both as a chat message with an `image_url`
    content block (base64 data URL) so the model sees the pixels. Return the
    assistant's raw text — it must end in a `<tool_call>{...}</tool_call>` (the
    `computer` tool) or `<done>`, matching the env's system-prompt contract.

    Coordinate space: Qwen3-VL/3.5 emit coordinates in a normalized [0,1000]
    grid. The Driver is responsible for converting [0,1000] -> page pixels
    before clicking (see `FleetTaskEnv._convert_qwen_coordinates` in
    skyrl-gym/.../fleet_task/env.py for the exact formula).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3.5-9B",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        # TODO(cluster): from openai import OpenAI; self.client = OpenAI(base_url=..., api_key="EMPTY")

    def act(self, observation: Observation, history: List[Dict[str, Any]]) -> str:
        # TODO(cluster):
        #   1. Build messages: system (tool defs + the computer-use contract),
        #      then `history`, then the current observation. For VL, the current
        #      user message content is a list:
        #        [{"type": "text", "text": observation.text},
        #         {"type": "image_url",
        #          "image_url": {"url": f"data:image/jpeg;base64,{observation.image}"}}]
        #   2. self.client.chat.completions.create(model=..., messages=..., ...)
        #   3. return choice.message.content
        raise NotImplementedError("Implement VLLMQwenPolicy.act (see docstring)")


class PlaywrightDriver:
    """Local browser driver against `pnpm dev` (falmart) with SENSE_LOG=true.

    `base_url` MUST be the env origin the SenseClient also reads from
    (http://localhost:5173 — sense routes are dual-mounted at /api/sense/*).

    `execute` parses the model's `computer` tool action (click / type / scroll /
    key / wait), converts [0,1000] coords to page pixels, performs it via
    Playwright, waits for network to settle, and returns a fresh screenshot.
    """

    def __init__(self, base_url: str = "http://localhost:5173",
                 viewport=(1366, 768), screenshot_quality: int = 60):
        self.base_url = base_url
        self.viewport = viewport
        self.screenshot_quality = screenshot_quality
        self._page = None
        self._browser = None

    def _screenshot_b64(self) -> str:
        # TODO(cluster): return base64 JPEG of self._page.screenshot(...)
        raise NotImplementedError

    def reset(self) -> Observation:
        # TODO(cluster):
        #   launch chromium, new page at self.viewport, goto self.base_url,
        #   wait_for_load_state("networkidle"), return Observation(
        #     text="(homepage loaded)", image=self._screenshot_b64()).
        raise NotImplementedError("Implement PlaywrightDriver.reset (see docstring)")

    def execute(self, action: str) -> Observation:
        # TODO(cluster):
        #   parse <tool_call>{"name":"computer","arguments":{...}}</tool_call>,
        #   convert [0,1000] coords -> pixels, dispatch the action on self._page,
        #   wait_for_load_state("networkidle") (so the sense cursor-diff captures
        #   the consequence before rollout.run_episode reads it), then
        #   return Observation(text=None, image=self._screenshot_b64()).
        raise NotImplementedError("Implement PlaywrightDriver.execute (see docstring)")

    def close(self) -> None:
        # TODO(cluster): close page/browser if open.
        pass
