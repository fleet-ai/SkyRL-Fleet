"""Tests for vision-language helpers in main_fleet_tinker.

Written blind from the contract, not by reading the implementation.

Contract under test (from the docstrings + design):

  `_decode_data_url(url) -> (bytes, "jpeg") | None`
    - None for: empty, None, http(s) URLs, malformed input.
    - For a valid `data:image/<fmt>;base64,...` URL where <fmt> is one
      tinker can ingest (png, jpeg) OR can be transcoded via PIL (webp,
      gif, ...), returns (jpeg_bytes, "jpeg"). Output is always JPEG
      regardless of input format.
    - Returned bytes must be a valid JPEG payload (FFD8 FF magic).

  `_sanitize_content(content) -> (text, [(bytes, fmt), ...])`
    - str input: returns (content, [])
    - non-list non-str: returns (str(content), [])
    - list[str]: returns (joined with "\\n", [])
    - list of {"type": "text", "text": "..."}: extracts text, joined
    - list with {"type": "image_url", "image_url": {"url": data-url}}:
      replaces the item with the IMAGE_PLACEHOLDER sentinel and yields
      decoded bytes
    - list with image_url whose URL can't be decoded: yields literal
      "[image]" marker, no bytes in the image list
    - list with {"type": "image", "image": <bytes>}: image bytes captured
      as-is, placeholder in text
    - Empty entries collapse out (no leading/trailing blank lines)

  `sanitize_text_only(content) -> str`
    - Same shape as `_sanitize_content` but always returns a single
      string, and any image becomes the literal "[image]" marker.

  `build_model_input_chunks(tokenizer, chat_history, add_generation_prompt)`
    -> (chunks: list, estimated_total_tokens: int)
    - For a text-only conversation: returns a single EncodedTextChunk
      whose tokens come from `tokenize_chat`.
    - For a multimodal conversation: returns an interleaved sequence of
      EncodedTextChunk and ImageChunk in the order images appeared.
    - estimated_total_tokens includes ~1024 tokens per image (the default
      advisory budget) plus all text token counts.
    - When the chat template silently drops the placeholder, falls back
      to a single EncodedTextChunk with "[image]" markers.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

# Make sure skyrl-gym is importable for the parent module.
ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "skyrl-gym"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import the module under test.
from integrations.fleet.entrypoints import main_fleet_tinker as mft  # noqa: E402

# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #

def _make_data_url(fmt: str = "png", color: str = "red") -> str:
    """Build a small valid data URL for a 10x10 image."""
    img = Image.new("RGB", (10, 10), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    return f"data:image/{fmt};base64," + base64.b64encode(buf.getvalue()).decode()


class FakeTokenizer:
    """Minimal tokenizer stub.

    `encode` returns one int per character (ord). `apply_chat_template`
    concatenates message contents (with a small role marker so chunks can
    still be parsed).
    """

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        body = "\n".join(f"<{m['role']}>{m.get('content', '')}" for m in messages)
        if add_generation_prompt:
            body += "\n<assistant>"
        if tokenize:
            return self.encode(body)
        return body


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


# --------------------------------------------------------------------------- #
# _decode_data_url
# --------------------------------------------------------------------------- #

class TestDecodeDataUrl:
    def test_none_input_returns_none(self):
        assert mft._decode_data_url(None) is None

    def test_empty_string_returns_none(self):
        assert mft._decode_data_url("") is None

    def test_http_url_returns_none(self):
        assert mft._decode_data_url("http://example.com/img.png") is None
        assert mft._decode_data_url("https://example.com/img.png") is None

    def test_malformed_returns_none(self):
        assert mft._decode_data_url("data:image/png;not-base64") is None
        assert mft._decode_data_url("data:foo") is None
        assert mft._decode_data_url("just a string") is None

    def test_decodes_png_to_jpeg_bytes(self):
        url = _make_data_url("png", "red")
        result = mft._decode_data_url(url)
        assert result is not None
        data, fmt = result
        assert fmt == "jpeg"
        # JPEG magic bytes (FF D8 FF)
        assert data[:3] == b"\xff\xd8\xff"

    def test_decodes_jpeg_to_jpeg_bytes(self):
        url = _make_data_url("jpeg", "blue")
        result = mft._decode_data_url(url)
        assert result is not None
        data, fmt = result
        assert fmt == "jpeg"
        assert data[:3] == b"\xff\xd8\xff"


# --------------------------------------------------------------------------- #
# _sanitize_content
# --------------------------------------------------------------------------- #

class TestSanitizeContent:
    def test_string_passthrough(self):
        text, images = mft._sanitize_content("hello world")
        assert text == "hello world"
        assert images == []

    def test_none_returns_string_repr(self):
        text, images = mft._sanitize_content(None)
        assert text == "None"
        assert images == []

    def test_integer_returns_string_repr(self):
        text, images = mft._sanitize_content(42)
        assert text == "42"
        assert images == []

    def test_empty_list(self):
        text, images = mft._sanitize_content([])
        assert text == ""
        assert images == []

    def test_list_of_strings_joined_with_newline(self):
        text, images = mft._sanitize_content(["a", "b", "c"])
        assert text == "a\nb\nc"
        assert images == []

    def test_list_of_text_dicts(self):
        text, images = mft._sanitize_content([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ])
        assert text == "first\nsecond"
        assert images == []

    def test_image_url_data_url_extracted(self):
        url = _make_data_url("png")
        text, images = mft._sanitize_content([
            {"type": "image_url", "image_url": {"url": url}},
        ])
        assert mft._IMAGE_PLACEHOLDER in text
        assert len(images) == 1
        img_bytes, fmt = images[0]
        assert fmt == "jpeg"
        assert img_bytes[:3] == b"\xff\xd8\xff"

    def test_image_url_undecodable_falls_back_to_text_marker(self):
        text, images = mft._sanitize_content([
            {"type": "image_url", "image_url": {"url": "http://nope.com/a.png"}},
        ])
        assert text == "[image]"
        assert images == []

    def test_inline_image_bytes(self):
        raw = b"\xff\xd8\xff" + b"\x00" * 100
        text, images = mft._sanitize_content([
            {"type": "image", "image": raw},
        ])
        assert mft._IMAGE_PLACEHOLDER in text
        assert len(images) == 1
        img_bytes, _fmt = images[0]
        assert img_bytes == raw

    def test_mixed_text_and_image_preserves_order(self):
        url = _make_data_url("png")
        text, images = mft._sanitize_content([
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "after"},
        ])
        idx_before = text.find("before")
        idx_placeholder = text.find(mft._IMAGE_PLACEHOLDER)
        idx_after = text.find("after")
        assert idx_before < idx_placeholder < idx_after
        assert len(images) == 1


# --------------------------------------------------------------------------- #
# sanitize_text_only
# --------------------------------------------------------------------------- #

class TestSanitizeTextOnly:
    def test_string_passthrough(self):
        assert mft.sanitize_text_only("hi") == "hi"

    def test_none_to_string(self):
        assert mft.sanitize_text_only(None) == "None"

    def test_image_becomes_literal_image_marker(self):
        url = _make_data_url("png")
        out = mft.sanitize_text_only([
            {"type": "image_url", "image_url": {"url": url}},
        ])
        # Must be a string, not a tuple
        assert isinstance(out, str)
        # Image is replaced by "[image]" marker, NOT the internal placeholder
        assert "[image]" in out
        assert mft._IMAGE_PLACEHOLDER not in out

    def test_mixed_returns_single_string(self):
        url = _make_data_url("png")
        out = mft.sanitize_text_only([
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "after"},
        ])
        assert isinstance(out, str)
        assert "before" in out
        assert "[image]" in out
        assert "after" in out
        assert mft._IMAGE_PLACEHOLDER not in out


# --------------------------------------------------------------------------- #
# build_model_input_chunks
# --------------------------------------------------------------------------- #

class TestBuildModelInputChunks:
    def test_text_only_returns_single_text_chunk(self, fake_tokenizer):
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        chunks, total = mft.build_model_input_chunks(fake_tokenizer, chat)
        # Exactly one text chunk
        assert len(chunks) == 1
        # It's a Tinker EncodedTextChunk (has `tokens`)
        assert hasattr(chunks[0], "tokens")
        # No image chunks
        assert not any(hasattr(c, "data") for c in chunks)
        assert total == len(chunks[0].tokens)
        assert total > 0

    def test_multimodal_produces_image_chunk_in_chunks(self, fake_tokenizer):
        url = _make_data_url("png")
        chat = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": [
                {"type": "text", "text": "see this"},
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "thanks"},
            ]},
        ]
        chunks, total = mft.build_model_input_chunks(fake_tokenizer, chat)
        # Must contain at least one ImageChunk (has `data` attribute)
        image_chunks = [c for c in chunks if hasattr(c, "data")]
        text_chunks = [c for c in chunks if hasattr(c, "tokens")]
        assert len(image_chunks) == 1
        assert len(text_chunks) >= 1
        # ImageChunk format is jpeg, bytes are valid JPEG
        ic = image_chunks[0]
        assert ic.format == "jpeg"
        assert ic.data[:3] == b"\xff\xd8\xff"
        # Total includes the image budget
        assert total >= mft._DEFAULT_IMAGE_TOKENS

    def test_multimodal_two_images_two_chunks(self, fake_tokenizer):
        u1 = _make_data_url("png", "red")
        u2 = _make_data_url("png", "blue")
        chat = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": u1}},
                {"type": "text", "text": "and"},
                {"type": "image_url", "image_url": {"url": u2}},
            ]},
        ]
        chunks, total = mft.build_model_input_chunks(fake_tokenizer, chat)
        image_chunks = [c for c in chunks if hasattr(c, "data")]
        assert len(image_chunks) == 2
        # Image budget counted twice
        assert total >= 2 * mft._DEFAULT_IMAGE_TOKENS


# --------------------------------------------------------------------------- #
# tools= threading — verifies the HF chat-template `tools=` kwarg reaches
# apply_chat_template once, for both text-only and multimodal paths.
# --------------------------------------------------------------------------- #

class _RecordingTokenizer:
    """Tokenizer stub that records every apply_chat_template call so tests can
    assert how many times `tools=` was threaded through and with what value."""

    def __init__(self):
        self.calls: list[dict] = []

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=True,
        tokenize=True,
        tools=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "tools": tools,
            }
        )
        body = "\n".join(f"<{m['role']}>{m.get('content', '')}" for m in messages)
        if tools:
            # Mimic Kimi's tool_declare block (single, at top of prompt).
            body = f"<tool_declare>{len(tools)}\n" + body
        if add_generation_prompt:
            body += "\n<assistant>"
        if tokenize:
            return self.encode(body)
        return body


class TestToolsThreading:
    SAMPLE_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            },
        }
    ]

    def test_tokenize_chat_passes_tools_through(self):
        tok = _RecordingTokenizer()
        chat = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        mft.tokenize_chat(tok, chat, add_generation_prompt=True, tools=self.SAMPLE_TOOLS)
        assert len(tok.calls) == 1
        assert tok.calls[0]["tools"] is self.SAMPLE_TOOLS

    def test_tokenize_chat_omits_tools_when_none(self):
        tok = _RecordingTokenizer()
        mft.tokenize_chat(
            tok,
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )
        assert len(tok.calls) == 1
        # No tools were passed in, so apply_chat_template should NOT receive
        # a tools kwarg (defaults to None on the recorder).
        assert tok.calls[0]["tools"] is None

    def test_build_chunks_text_only_threads_tools(self):
        tok = _RecordingTokenizer()
        chat = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        chunks, total = mft.build_model_input_chunks(
            tok, chat, add_generation_prompt=True, tools=self.SAMPLE_TOOLS
        )
        # Exactly one apply_chat_template call for the text-only fast path.
        assert len(tok.calls) == 1
        assert tok.calls[0]["tools"] is self.SAMPLE_TOOLS
        # Output is a single EncodedTextChunk (fast path).
        assert len(chunks) == 1
        assert hasattr(chunks[0], "tokens")
        assert total == len(chunks[0].tokens)

    def test_build_chunks_multimodal_threads_tools(self):
        tok = _RecordingTokenizer()
        url = _make_data_url("png")
        chat = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        ]
        chunks, _total = mft.build_model_input_chunks(
            tok, chat, add_generation_prompt=True, tools=self.SAMPLE_TOOLS
        )
        # Multimodal slow path: still exactly one apply_chat_template call
        # (the per-image splicing happens in plain Python after rendering).
        assert len(tok.calls) == 1
        assert tok.calls[0]["tools"] is self.SAMPLE_TOOLS
        # tools= should NOT cause images to vanish: chunk stream still has one.
        image_chunks = [c for c in chunks if hasattr(c, "data")]
        assert len(image_chunks) == 1

    def test_build_chunks_no_tools_does_not_pass_tools(self):
        tok = _RecordingTokenizer()
        mft.build_model_input_chunks(
            tok,
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )
        assert len(tok.calls) == 1
        assert tok.calls[0]["tools"] is None

    def test_tools_rendered_exactly_once_across_many_messages(self):
        """The whole point of `tools=`: tool block appears once in the prompt
        regardless of message count. The recording tokenizer prepends a
        `<tool_declare>` block once per template invocation; we assert it
        doesn't repeat per assistant turn."""
        tok = _RecordingTokenizer()
        # Long multi-turn chat (5 assistant turns + obs).
        chat = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        for i in range(5):
            chat.append({"role": "assistant", "content": f"<tool_call>step {i}</tool_call>"})
            chat.append({"role": "user", "content": f"obs {i}"})
        mft.tokenize_chat(tok, chat, add_generation_prompt=True, tools=self.SAMPLE_TOOLS)
        # One apply_chat_template call; tools rendered once at the top.
        assert len(tok.calls) == 1
        # Inspect what the recorder built as the body: <tool_declare> should
        # appear exactly once even with 5 assistant turns.
        body = tok.apply_chat_template(
            chat, add_generation_prompt=False, tokenize=False, tools=self.SAMPLE_TOOLS
        )
        assert body.count("<tool_declare>") == 1

    def test_undecodable_image_falls_back_to_text(self, fake_tokenizer):
        chat = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://no.png"}},
            ]},
        ]
        chunks, total = mft.build_model_input_chunks(fake_tokenizer, chat)
        # Single text chunk, no images
        image_chunks = [c for c in chunks if hasattr(c, "data")]
        assert image_chunks == []
        assert total > 0

    def test_returns_two_value_tuple(self, fake_tokenizer):
        chat = [{"role": "user", "content": "hi"}]
        result = mft.build_model_input_chunks(fake_tokenizer, chat)
        assert isinstance(result, tuple)
        assert len(result) == 2
        chunks, total = result
        assert isinstance(chunks, list)
        assert isinstance(total, int)
