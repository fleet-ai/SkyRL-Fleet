"""Tests for `truncate_tool_result` in FleetTaskEnv.

Why this file exists: a regression where multimodal tool results
(`[{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]`)
were being JSON-dumped for length measurement. The base64-encoded screenshot
inflated the JSON well past the 16K char budget, which triggered the
truncated-string fallback. That collapsed the multimodal list into

    'Tool result:\\n[{"type": "image_url", "image_url": {"url": "data:..."]'

leaking the entire base64 string into chat_history. The VL pipeline never
saw the screenshot as image content — the tokenizer ingested base64 garbage
instead, padding out the prompt and wasting context.

These tests pin the fixed behavior:
  - multimodal lists pass through with `image_url` blocks untouched,
  - only `text` blocks inside multimodal lists are length-checked,
  - non-multimodal shapes still get the old serialize-and-truncate path,
  - `None` still passes through.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.env import (  # noqa: E402  (import after sys.path setup)
    MAX_TOOL_OUTPUT_CHARS,
    truncate_tool_result,
)


def _huge_base64(approx_chars: int = 200_000) -> str:
    """A base64-ish string that's deliberately larger than MAX_TOOL_OUTPUT_CHARS
    so the bug's serialization path would fire. We don't need real PNG bytes —
    just length sufficient to exceed the budget."""
    return "A" * approx_chars


# --------------------------------------------------------------------------- #
# Multimodal lists
# --------------------------------------------------------------------------- #


class TestMultimodalPreserved:
    def test_image_only_passes_through_unchanged(self):
        """A screenshot-only multimodal result must stay a list, not become a
        truncated `Tool result:\\n[{...}]` string."""
        url = f"data:image/jpeg;base64,{_huge_base64()}"
        blocks = [{"type": "image_url", "image_url": {"url": url}}]
        out = truncate_tool_result(blocks)
        assert isinstance(out, list), "must remain a list, not be stringified"
        assert len(out) == 1
        assert out[0]["type"] == "image_url"
        assert out[0]["image_url"]["url"] == url, "image_url must be byte-identical"
        # The truncated-string marker MUST NOT appear anywhere.
        flat = repr(out)
        assert "TRUNCATED" not in flat
        assert "Tool result:" not in flat

    def test_text_plus_image_keeps_image_strips_long_text(self):
        long_text = "x" * (MAX_TOOL_OUTPUT_CHARS * 2)
        url = f"data:image/jpeg;base64,{_huge_base64()}"
        blocks = [
            {"type": "text", "text": long_text},
            {"type": "image_url", "image_url": {"url": url}},
        ]
        out = truncate_tool_result(blocks)
        assert isinstance(out, list) and len(out) == 2

        text_block, image_block = out[0], out[1]
        # Text was over budget -> truncated string with marker.
        assert text_block["type"] == "text"
        assert "TRUNCATED" in text_block["text"]
        assert len(text_block["text"]) <= MAX_TOOL_OUTPUT_CHARS + 200  # +marker
        # Image is byte-identical regardless of how big it is.
        assert image_block["type"] == "image_url"
        assert image_block["image_url"]["url"] == url

    def test_short_text_block_unchanged(self):
        url = f"data:image/jpeg;base64,{_huge_base64()}"
        blocks = [
            {"type": "text", "text": "Screenshot at step 5"},
            {"type": "image_url", "image_url": {"url": url}},
        ]
        out = truncate_tool_result(blocks)
        assert out[0]["text"] == "Screenshot at step 5"
        assert out[1]["image_url"]["url"] == url

    def test_image_only_under_budget_also_passes_through(self):
        """A tiny screenshot — the JSON-dump path would have passed it too,
        but verify we don't accidentally regress the small-image case."""
        url = "data:image/png;base64,iVBORw0KGgoAAAA"  # tiny stub
        blocks = [{"type": "image_url", "image_url": {"url": url}}]
        out = truncate_tool_result(blocks)
        assert out == blocks  # exact passthrough, not just shape-equivalent


# --------------------------------------------------------------------------- #
# Non-multimodal shapes — the existing behavior must be preserved
# --------------------------------------------------------------------------- #


class TestNonMultimodalUnchanged:
    def test_none_passes_through(self):
        assert truncate_tool_result(None) is None

    def test_short_string_passes_through(self):
        s = "small output\nline 2"
        assert truncate_tool_result(s) == s

    def test_long_string_truncated(self):
        s = "y" * (MAX_TOOL_OUTPUT_CHARS * 3)
        out = truncate_tool_result(s)
        assert isinstance(out, str)
        assert "TRUNCATED" in out
        assert "chars elided" in out
        assert len(out) <= MAX_TOOL_OUTPUT_CHARS + 200

    def test_short_dict_passes_through(self):
        d = {"rows": 3, "columns": ["id", "name"]}
        assert truncate_tool_result(d) == d

    def test_long_dict_truncated_to_string(self):
        d = {"rows": [{"name": "x" * 100} for _ in range(1000)]}
        out = truncate_tool_result(d)
        assert isinstance(out, str)
        assert "TRUNCATED" in out

    def test_list_without_type_keys_is_not_treated_as_multimodal(self):
        """A list of plain dicts (no 'type' key) is NOT multimodal content —
        keep the serialize-and-truncate semantics."""
        raw = [{"k": "v"} for _ in range(100_000)]  # huge
        out = truncate_tool_result(raw)
        assert isinstance(out, str)  # serialized + truncated
        assert "TRUNCATED" in out

    def test_empty_list_is_not_multimodal(self):
        """`[]` is technically a list with no blocks but must not be treated
        as multimodal (no shape to preserve)."""
        out = truncate_tool_result([])
        # Either a passthrough empty list OR an empty serialization is fine,
        # so long as no TRUNCATED marker is added (empty fits under budget).
        assert "TRUNCATED" not in str(out)


# --------------------------------------------------------------------------- #
# Regression: the specific bug shape from the canonical hero run
# --------------------------------------------------------------------------- #


class TestRegressionMultimodalBug:
    def test_multimodal_list_never_becomes_tool_result_string(self):
        """The bug user-observed: `Tool result:\\n[{type:image_url,...}]` text
        with base64 inline. After fix, this exact shape can never appear in
        the output of truncate_tool_result."""
        url = f"data:image/jpeg;base64,{_huge_base64()}"
        blocks = [{"type": "image_url", "image_url": {"url": url}}]
        out = truncate_tool_result(blocks)
        # If `out` is a list, no string-fallback fired.
        assert isinstance(out, list)
        # If it's somehow a string, the leak signature must not be there.
        if isinstance(out, str):
            assert "Tool result:" not in out
            assert "image_url" not in out
            assert "data:image/jpeg;base64" not in out
