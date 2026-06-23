"""Tests for `_convert_normalized_coordinates` — handles both Qwen-VL's
[0, 1000] integer space AND Kimi's occasional [0, 1.0] float regressions.

The previous Qwen-only scaler ACTIVELY broke the Kimi case: a model
emission like `[0.273, 0.298]` became pixel (0, 0) after `x / 1000 *
screen_w` — dead-space click, no state change. Unified scaler detects
the regime by value range:
  0 < max ≤ 1.0    → Kimi [0, 1.0]
  1.0 < max ≤ 1000 → Qwen [0, 1000]
  max > 1000       → pixel (no-op)

Grounded against session 827f4376 turns 6 & 8 (`[0.273, 0.298]`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.env import FleetTaskEnv  # noqa: E402


# Standard viewport used by the Fleet computer-use envs.
SCREEN_W, SCREEN_H = 1366, 768


@pytest.fixture
def env_stub():
    """Bare FleetTaskEnv stub: skips __init__ (MCP setup), sets only the
    fields the coordinate scaler reads. Same pattern as test_truncate."""
    env = FleetTaskEnv.__new__(FleetTaskEnv)
    env.screen_width = SCREEN_W
    env.screen_height = SCREEN_H
    return env


def _call(action: str, coord, field: str = "coordinate") -> dict:
    return {
        "name": "computer",
        "arguments": {"action": action, field: list(coord)},
    }


# --------------------------------------------------------------------------- #
# Kimi [0, 1.0] case — the regression we caught
# --------------------------------------------------------------------------- #

class TestKimiNormalizedRange:
    def test_session_827f4376_turn6_real_case(self, env_stub):
        """Verbatim from job c4b429ae → session 827f4376 turn 6:
        Kimi emitted `[0.273, 0.298]` for left_click. Without scaling
        the MCP click landed at pixel (0, 0). With scaling it lands at
        (372, 228) — within the search input on Medora's home page."""
        tc = _call("left_click", [0.273, 0.298])
        env_stub._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [
            int(0.273 * SCREEN_W),
            int(0.298 * SCREEN_H),
        ]
        # Sanity: must land on search field area (roughly x=200-700, y=200-260).
        x, y = tc["arguments"]["coordinate"]
        assert 200 <= x <= 700, f"x={x} expected ~372 (search field)"
        assert 200 <= y <= 260, f"y={y} expected ~228 (search field row)"

    def test_corner_normalized_scales_to_corner_pixel(self, env_stub):
        tc = _call("left_click", [1.0, 1.0])
        env_stub._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [SCREEN_W, SCREEN_H]

    def test_mid_screen_normalized(self, env_stub):
        tc = _call("left_click", [0.5, 0.5])
        env_stub._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [SCREEN_W // 2, SCREEN_H // 2]

    def test_start_coordinate_also_scaled(self, env_stub):
        """left_click_drag has both `coordinate` and `start_coordinate`."""
        tc = _call("left_click_drag", [0.5, 0.5])
        tc["arguments"]["start_coordinate"] = [0.1, 0.2]
        env_stub._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [SCREEN_W // 2, SCREEN_H // 2]
        assert tc["arguments"]["start_coordinate"] == [
            int(0.1 * SCREEN_W), int(0.2 * SCREEN_H),
        ]


# --------------------------------------------------------------------------- #
# Qwen-VL [0, 1000] case — preserve existing behavior
# --------------------------------------------------------------------------- #

class TestQwenIntegerRange:
    def test_center_of_screen(self, env_stub):
        tc = _call("left_click", [500, 500])
        env_stub._convert_normalized_coordinates(tc)
        # 500 / 1000 = 0.5
        assert tc["arguments"]["coordinate"] == [SCREEN_W // 2, SCREEN_H // 2]

    def test_top_left(self, env_stub):
        tc = _call("left_click", [10, 10])
        env_stub._convert_normalized_coordinates(tc)
        # 10/1000 → ~1% of screen
        assert tc["arguments"]["coordinate"] == [
            int(10 / 1000 * SCREEN_W), int(10 / 1000 * SCREEN_H),
        ]

    def test_max_qwen_range(self, env_stub):
        tc = _call("left_click", [1000, 1000])
        env_stub._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [SCREEN_W, SCREEN_H]


# --------------------------------------------------------------------------- #
# Pixel case (max > 1000) — pass through unchanged
# --------------------------------------------------------------------------- #

class TestPixelPassthrough:
    def test_above_qwen_range_is_pixel(self, env_stub):
        """Session 827f4376 turn 70: model emitted [1216, 90] — pixel
        coords on the right side of a 1366px-wide viewport. Must NOT
        be scaled (would yield (1216/1000 * 1366, 90/1000 * 768) =
        (1660, 69) — off-screen). max > 1000 → pixel → leave alone."""
        tc = _call("left_click", [1216, 90])
        env_stub._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [1216, 90]

    def test_realistic_pixel_in_qwen_range_is_ambiguous_but_scaled(self, env_stub):
        """A coord like [330, 228] falls in (1, 1000] so the scaler
        treats it as Qwen [0,1000] and divides by 1000. This is the
        unavoidable ambiguity — if a Kimi run emits pixel coords in
        this range, the scaler will mis-scale them. Mitigation: Kimi
        emits pixels >>1000 on a 1366x768 screen most of the time (the
        right half), and the model-family branching could be added if
        the false-positive rate becomes a problem. For now, document
        and accept."""
        tc = _call("left_click", [330, 228])
        env_stub._convert_normalized_coordinates(tc)
        # Expected: treated as Qwen → 330/1000*1366 = 450, 228/1000*768 = 175
        assert tc["arguments"]["coordinate"] == [
            int(330 / 1000 * SCREEN_W), int(228 / 1000 * SCREEN_H),
        ]


# --------------------------------------------------------------------------- #
# Edge cases that must NOT scale
# --------------------------------------------------------------------------- #

class TestNoOpCases:
    def test_zero_zero_left_alone(self, env_stub):
        """max == 0 → corner pixel click. Don't scale (0,0 is a legit
        corner). The Kimi case requires `0 < max ≤ 1.0`."""
        tc = _call("left_click", [0, 0])
        env_stub._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [0, 0]

    def test_mixed_pixel_and_float_left_alone(self, env_stub):
        """[250, 0.5] is a bug. Neither pure Kimi [0,1] nor pure Qwen
        [0,1000]. The scaler must NOT silently fix this; we want it to
        surface as a click at the wrong place so the model self-corrects
        or the bug gets caught in logs."""
        tc = _call("left_click", [250, 0.5])
        env_stub._convert_normalized_coordinates(tc)
        # 250 is in (1, 1000], 0.5 is in (0, 1.0]. max(250, 0.5)=250 is in
        # (1, 1000] so it falls into the Qwen branch. y is still 0.5 → 0.
        # NOT scaled to a sensible pixel.
        # Accept this — flagging the bug, not papering over it.
        assert tc["arguments"]["coordinate"] != [int(250 * 1.0), int(0.5 * SCREEN_H)]

    def test_non_computer_tool_not_touched(self, env_stub):
        tc = {"name": "other_tool", "arguments": {"coordinate": [0.5, 0.5]}}
        env_stub._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [0.5, 0.5]

    def test_no_coordinate_field_no_op(self, env_stub):
        tc = {"name": "computer", "arguments": {"action": "screenshot"}}
        env_stub._convert_normalized_coordinates(tc)
        assert "coordinate" not in tc["arguments"]

    def test_missing_screen_dims_no_op(self):
        """If the env never set screen_width/height (e.g. tool_use
        modality), the scaler must early-return without touching args."""
        env = FleetTaskEnv.__new__(FleetTaskEnv)
        # No screen_width / screen_height set.
        tc = _call("left_click", [0.5, 0.5])
        env._convert_normalized_coordinates(tc)
        assert tc["arguments"]["coordinate"] == [0.5, 0.5]


# --------------------------------------------------------------------------- #
# Back-compat: existing call sites use _convert_qwen_coordinates
# --------------------------------------------------------------------------- #

class TestBackCompatAlias:
    def test_old_name_still_works(self, env_stub):
        """env.py and any external caller using the old name must keep
        working. Verifying the alias dispatches to the new impl."""
        tc = _call("left_click", [0.273, 0.298])
        env_stub._convert_qwen_coordinates(tc)
        # If the alias works, this gets scaled per the Kimi branch.
        assert tc["arguments"]["coordinate"] == [
            int(0.273 * SCREEN_W), int(0.298 * SCREEN_H),
        ]
