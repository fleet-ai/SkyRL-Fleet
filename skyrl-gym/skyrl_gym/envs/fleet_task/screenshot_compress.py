"""Screenshot compression for browser_use / computer_use multimodal observations.

Why
---
Kimi-K2.6:peft:131072 has a 128 K context window. browser_use rollouts on
PSI tasks accumulate ~2 K tokens per turn of which ~1 K is the screenshot
image, so 64 turns × 2 K ≈ 128 K — we hit the ceiling exactly at
max_turns=64. Halving image dimensions roughly quarters image-token cost
(image tokens scale with patch area), buying back enough headroom to
either raise max_turns or just leave more room for thinking + tool output.

What
----
Pure functions that take an OpenAI multimodal content list (or just a
single `data:image/jpeg;base64,...` URL) and return one with the image
downscaled + re-encoded at a configurable JPEG quality. URLs that are
NOT base64 data URLs (e.g. https://… S3 links) pass through unchanged
because there's nothing to compress locally.

How it gets toggled
-------------------
FleetTaskEnv reads `screenshot_max_dim` from its `extras` dict (set by
main_fleet_tinker / fleet-research-api). 0 / None disables compression
— byte-identical to today's behavior. JPEG re-encode quality is fixed
at 85 inside this module; not exposed because it doesn't affect token
count (only file bytes and visual fidelity).
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Match `data:image/<fmt>;base64,<payload>` where fmt is jpeg/png/webp
_DATA_URL_RE = re.compile(
    r"^data:image/(?P<fmt>jpeg|jpg|png|webp);base64,(?P<b64>.+)$",
    re.IGNORECASE | re.DOTALL,
)


_JPEG_QUALITY = 85  # Pillow needs a value; not exposed as a knob because
#                    it doesn't affect context-token count (only file size
#                    and visual fidelity, neither of which is the bottleneck).


def compress_image_url(
    data_url: str,
    *,
    max_dim: int,
) -> str:
    """Downscale + recompress a base64-encoded image URL.

    Args:
      data_url: a `data:image/<fmt>;base64,<payload>` string.
      max_dim: cap on max(width, height). If 0 or negative, no compression.
        If the image is already smaller than max_dim in both dims, pass
        through unchanged.

    Returns:
      A new `data:image/jpeg;base64,...` URL if recompressed, or the
      original URL unchanged if compression is disabled / unnecessary /
      not applicable.
    """
    if not isinstance(data_url, str) or max_dim <= 0:
        return data_url
    m = _DATA_URL_RE.match(data_url)
    if not m:
        # Not a base64 image (e.g. https:// link) — nothing to compress.
        return data_url
    try:
        from PIL import Image  # local import — keeps PIL optional at import time
    except ImportError:
        logger.warning("PIL not available; screenshot compression disabled")
        return data_url

    try:
        raw = base64.b64decode(m.group("b64"))
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if max(w, h) <= max_dim:
            return data_url  # already small enough
        # Preserve aspect ratio
        scale = max_dim / max(w, h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.convert("RGB").resize(new_size, Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        b64 = base64.b64encode(out.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:  # noqa: BLE001
        logger.warning("screenshot compress failed (%s); passing through", e)
        return data_url


def compress_content_blocks(
    content: Any,
    *,
    max_dim: int,
) -> Any:
    """Walk an OpenAI multimodal content value and compress any
    `image_url` blocks in place. Non-list content (plain strings) and
    blocks of other types are returned untouched.

    Returns the same object if max_dim <= 0 or content is not a list of
    blocks — caller can swap unconditionally.
    """
    if max_dim <= 0 or not isinstance(content, list):
        return content
    out = []
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "image_url"
            and isinstance(block.get("image_url"), dict)
            and "url" in block["image_url"]
        ):
            new_url = compress_image_url(
                block["image_url"]["url"],
                max_dim=max_dim,
            )
            new_block = dict(block)
            new_block["image_url"] = dict(block["image_url"])
            new_block["image_url"]["url"] = new_url
            out.append(new_block)
        else:
            out.append(block)
    return out
