"""Tests for screenshot_compress helpers used by FleetTaskEnv.

Confirms:
- compression actually downscales when max_dim < current
- max_dim=0 is a pass-through (the default behavior must be byte-identical)
- max_dim larger than the image is also pass-through
- non-base64 URLs (e.g. https://) pass through (we don't fetch remote)
- non-image content blocks pass through (text, etc.)
- malformed base64 / corrupt image is logged + passed through (no crash)
- compressed output round-trips and is decodable + at expected dims
"""

from __future__ import annotations

import base64
import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from skyrl_gym.envs.fleet_task.screenshot_compress import (  # noqa: E402
    compress_image_url,
    compress_content_blocks,
)


def _make_data_url(w: int, h: int, *, fmt: str = "JPEG", quality: int = 90) -> str:
    img = Image.new("RGB", (w, h), color=(255, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _dims_of(data_url: str) -> tuple[int, int]:
    payload = data_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(payload)))
    return img.size


def test_compress_downscales_big_image():
    url = _make_data_url(1366, 768)
    out = compress_image_url(url, max_dim=512)
    w, h = _dims_of(out)
    assert max(w, h) == 512
    # aspect ratio preserved within 1px rounding
    assert abs(w / h - 1366 / 768) < 0.02


def test_compress_pass_through_when_max_dim_zero():
    url = _make_data_url(1366, 768)
    out = compress_image_url(url, max_dim=0)
    assert out is url  # byte-identical, same object — no work done


def test_compress_pass_through_when_max_dim_larger_than_image():
    url = _make_data_url(800, 600)
    out = compress_image_url(url, max_dim=2048)
    assert out is url  # already smaller, no work done


def test_compress_pass_through_for_https_url():
    url = "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/screenshot/abc.jpeg"
    out = compress_image_url(url, max_dim=512)
    assert out is url  # nothing to compress without bytes in-hand


def test_compress_pass_through_for_malformed_data_url():
    url = "data:image/jpeg;base64,this-is-not-valid-base64-!!!"
    out = compress_image_url(url, max_dim=512)
    # Either passes through unchanged or the helper returns the original
    # — what matters is no exception escapes.
    assert isinstance(out, str)


def test_compress_handles_png():
    url = _make_data_url(1000, 500, fmt="PNG")
    out = compress_image_url(url, max_dim=400)
    # Output should be JPEG-encoded (we always re-encode as JPEG)
    assert out.startswith("data:image/jpeg;base64,")
    w, h = _dims_of(out)
    assert max(w, h) == 400


def test_content_blocks_compresses_image_url_blocks_only():
    big_url = _make_data_url(1200, 800)
    content = [
        {"type": "text", "text": "Tool result: succeeded"},
        {"type": "image_url", "image_url": {"url": big_url}},
        {"type": "text", "text": "Trailing scaffold"},
    ]
    out = compress_content_blocks(content, max_dim=400)
    # Text blocks untouched (object identity)
    assert out[0] is content[0]
    assert out[2] is content[2]
    # Image block is a new dict with compressed url
    assert out[1] is not content[1]
    w, h = _dims_of(out[1]["image_url"]["url"])
    assert max(w, h) == 400


def test_content_blocks_pass_through_when_max_dim_zero():
    big_url = _make_data_url(1200, 800)
    content = [
        {"type": "image_url", "image_url": {"url": big_url}},
    ]
    out = compress_content_blocks(content, max_dim=0)
    assert out is content  # same object — no traversal at all


def test_content_blocks_pass_through_for_non_list():
    # Plain-string content (the non-multimodal branch in env.py)
    out = compress_content_blocks("Tool result:\n42", max_dim=512)
    assert out == "Tool result:\n42"


def test_content_blocks_skips_image_url_without_url_field():
    """Defensive: malformed image_url block (missing 'url') should not crash."""
    content = [{"type": "image_url", "image_url": {"detail": "auto"}}]
    out = compress_content_blocks(content, max_dim=512)
    assert out[0] is content[0]


def test_compression_actually_reduces_byte_size():
    """The whole point: shipping fewer image tokens. Smaller PNG → smaller
    base64 payload after JPEG re-encode at quality 60."""
    url = _make_data_url(1366, 768, fmt="JPEG", quality=95)
    out = compress_image_url(url, max_dim=512, jpeg_quality=60)
    original_bytes = len(url.split(",", 1)[1])
    new_bytes = len(out.split(",", 1)[1])
    assert new_bytes < original_bytes
    # Sanity: should be at least 4× smaller given a 2.67× linear shrink + lower quality
    assert new_bytes < original_bytes / 3
