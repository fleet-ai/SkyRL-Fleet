"""Constrained-decoding "think gate" for the negotiation thinking arm.

Background (research_logs/think-close-debugging-0618.md): with the Qwen3 thinking
arm, ~31% of assistant turns opened ``<think>`` and then emitted their visible
message + action tag *inside* the still-open block, never emitting ``</think>``.
That breaks the reasoning<->message boundary that both the opponent-view strip
(``_strip_think``) and the ``qwen3_without_thinking`` training-context strip rely
on, and the boundary is unrecoverable post-hoc.

This module fixes it at the source with a vLLM v1 logits processor that forces every
turn into the well-formed shape::

    <think> (>= min_think_tokens) </think> <message> <action>

Anti-gaming (the prior "force-open-<think>" attempt collapsed to empty
``<think></think>`` with reasoning dumped into the visible channel):

  * First token is forced to ``<think>``                  -> the channel always opens.
  * ``</think>`` is masked until >= ``min_think_tokens`` have been generated
    -> an *empty / vestigial* think is impossible, so real reasoning must land
       in the private (stripped) channel.
  * The action-tag start token(s) and the turn-end token(s) are masked until
    ``</think>`` has been emitted -> no action-inside-think, and the close is
    forced before the turn can end.

Once ``</think>`` is emitted the gate stops masking and the model freely writes its
message + action (ended by the env's normal stop strings). The remaining "reason in
a long open message" escape is left to the existing value_leak penalty + adversarial
opponent + per-step metrics (see the research log; "soft" anti-gaming was chosen).

The processor is model-agnostic: all token ids + the floor are supplied per request
via ``SamplingParams.extra_args[EXTRA_ARGS_KEY]`` (resolved from the tokenizer by
``build_think_gate_extra_args``), so the core masking logic (``apply_think_gate``) is
a pure function that can be unit-tested without vLLM or a GPU.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

# Key under which the per-request gate config is stashed in SamplingParams.extra_args.
EXTRA_ARGS_KEY = "negotiation_think_gate"

_NEG_INF = float("-inf")


def apply_think_gate(
    output_ids: Sequence[int],
    logits: torch.Tensor,
    *,
    think_close_id: int,
    block_ids: Sequence[int],
    eos_ids: Sequence[int],
    min_think_tokens: int,
    closed_flag: List[bool],
) -> torch.Tensor:
    """Mask ``logits`` (a 1-D ``[vocab]`` tensor, modified in place) to enforce the
    ``<think> ... </think> <message> <action>`` structure for one request.

    Args:
        output_ids: tokens generated so far this turn (a live reference; empty on the
            first decode step).
        logits: this request's next-token logits, shape ``[vocab_size]``.
        think_open_id / think_close_id: single-token ids for ``<think>`` / ``</think>``.
        block_ids: token ids that begin an action tag (e.g. the ``<`` that starts
            ``<propose>`` / ``<accept>`` / ``<deal>``); masked while think is open.
        eos_ids: turn-ending token ids (eos + ``<|im_end|>``); masked while think is open.
        min_think_tokens: ``</think>`` is masked until this many tokens follow ``<think>``.
        closed_flag: 1-element ``[bool]`` cache; set True once ``</think>`` is seen so the
            gate becomes a no-op for the rest of the turn (avoids rescanning ``output_ids``).

    Returns:
        The (in-place modified) ``logits`` tensor.
    """
    n = len(output_ids)

    # Already closed earlier this turn -> gate is inert.
    if closed_flag[0]:
        return logits
    # The opening <think> is injected by the chat template (qwen35_strip_thinking) into the
    # PROMPT, so generation starts INSIDE the think block -- we no longer force <think>, only
    # enforce the close. The close token shows up as the most recent output token exactly once.
    if n > 0 and output_ids[-1] == think_close_id:
        closed_flag[0] = True
        return logits

    # Think still open: forbid ending the turn and forbid starting any action tag.
    for tid in eos_ids:
        logits[tid] = _NEG_INF
    for tid in block_ids:
        logits[tid] = _NEG_INF

    # Forbid closing think until the content floor is met. Generation starts inside the
    # prompt-injected think block, so all n output tokens so far are think content.
    if n < min_think_tokens:
        logits[think_close_id] = _NEG_INF

    return logits


def _encode_first_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"think-gate: {text!r} encoded to no tokens")
    return ids[0]


def _encode_single_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"think-gate: expected {text!r} to be a single token, got {ids}. "
            "The gate assumes <think>/</think>/<|im_end|> are atomic special tokens."
        )
    return ids[0]


def _action_start_block_ids(tokenizer, exclude_ids: Sequence[int]) -> set:
    """Every token id whose decoded form ENDS WITH ``<`` — the set of tokens that can
    begin an action tag (``<propose>`` / ``<accept>`` / ``<deal>``) in any context.

    Why not just ``encode("<propose>")[0]`` (the original approach, which was the bug):
    an action tag is always a ``<`` followed by a *separate* content token (``prop`` /
    ``accept`` / ``deal``), but BPE merges that ``<`` with whatever character precedes
    it. So the real leading token is context-dependent — ``<`` (id 27) after a newline,
    but `` <``, ``。<``, ``><`` etc. after a space / CJK char / ``>``. Encoding the tag
    in isolation only yields the bare ``<`` and misses every merged variant, so actions
    leaked inside an open ``<think>`` (think-close-debugging-0618.md). Every variant ends
    with ``<``, so masking all such tokens blocks the action start in all contexts.
    ``</think>``/``<think>`` end with ``>`` so they are never in this set, but we exclude
    them defensively (the close must stay sampleable).
    """
    excl = {int(x) for x in exclude_ids}
    block = set()
    try:
        vocab = tokenizer.get_vocab()  # {raw_token: id}; byte-level BPE maps 0x3C -> '<'
    except Exception:
        vocab = None
    # Fast path: prefilter on the raw token string (dict scan, no per-id decode), then
    # decode-verify the few candidates. Fall back to a full decode scan if get_vocab is
    # unavailable or the raw keys don't surface '<' as expected.
    candidates = [int(tid) for raw, tid in vocab.items() if isinstance(raw, str) and raw.endswith("<")] if vocab else []
    if not candidates and vocab:
        candidates = [int(tid) for tid in vocab.values()]
    if not candidates:
        candidates = list(range(int(getattr(tokenizer, "vocab_size", 0)) or len(tokenizer)))
    for tid in candidates:
        if tid in excl:
            continue
        try:
            if tokenizer.decode([tid]).endswith("<"):
                block.add(tid)
        except Exception:
            continue
    return block


def build_think_gate_extra_args(tokenizer, min_think_tokens: int = 16) -> Dict[str, object]:
    """Resolve the token ids the gate needs from ``tokenizer`` and package them for
    ``SamplingParams.extra_args[EXTRA_ARGS_KEY]``.

    Validates the load-bearing assumption that ``</think>`` is a standalone special
    token (so masking the ``<`` that starts action tags does not also block the close).
    """
    think_open_id = _encode_single_id(tokenizer, "<think>")
    think_close_id = _encode_single_id(tokenizer, "</think>")

    # Mask every token that can *start* an action tag in any context (all tokens whose
    # decoded form ends with "<"), not just the isolated "<" — see _action_start_block_ids.
    # Union the isolated-encoding leading tokens as a floor in case a tokenizer surfaces a
    # start token that doesn't end with "<".
    block_ids = _action_start_block_ids(tokenizer, exclude_ids=(think_open_id, think_close_id))
    block_ids |= {_encode_first_id(tokenizer, t) for t in ("<propose>", "<accept>", "<deal>")}
    block_ids.discard(think_open_id)
    block_ids.discard(think_close_id)
    block_ids = sorted(block_ids)

    eos_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    try:
        eos_ids.add(_encode_single_id(tokenizer, "<|im_end|>"))
    except ValueError:
        pass  # non-Qwen tokenizer; eos_token_id alone is enough

    if think_close_id in block_ids:
        raise ValueError(
            "think-gate: </think> shares its leading token with an action tag; masking "
            "action starts would also block the close. Gate not safe for this tokenizer."
        )

    return {
        "min_think_tokens": int(min_think_tokens),
        "think_open_id": think_open_id,
        "think_close_id": think_close_id,
        "block_ids": block_ids,
        "eos_ids": sorted(eos_ids),
    }


# --- vLLM v1 integration -----------------------------------------------------------
# Imported lazily inside the class so that `apply_think_gate` / `build_think_gate_extra_args`
# can be imported (and unit-tested) without pulling in vLLM.
try:  # pragma: no cover - exercised only inside the vLLM engine process
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

    class ThinkGateLogitsProcessor(AdapterLogitsProcessor):
        """vLLM v1 logits processor enforcing the negotiation think structure.

        Registered unconditionally on the engine (via ``logits_processors=[...]``); it is
        inert for any request whose ``SamplingParams.extra_args`` lacks ``EXTRA_ARGS_KEY``,
        so non-thinking / eval requests are unaffected.
        """

        def is_argmax_invariant(self) -> bool:
            # We censor tokens, so we can change the greedy argmax.
            return False

        def new_req_logits_processor(self, params) -> Optional[Callable]:
            cfg = (params.extra_args or {}).get(EXTRA_ARGS_KEY) if params.extra_args else None
            if not cfg:
                return None
            think_close_id = int(cfg["think_close_id"])
            block_ids = [int(x) for x in cfg["block_ids"]]
            eos_ids = [int(x) for x in cfg["eos_ids"]]
            min_think_tokens = int(cfg.get("min_think_tokens", 16))
            closed_flag = [False]  # per-request cache, persists across decode steps

            def _req_lp(output_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
                return apply_think_gate(
                    output_ids,
                    logits,
                    think_close_id=think_close_id,
                    block_ids=block_ids,
                    eos_ids=eos_ids,
                    min_think_tokens=min_think_tokens,
                    closed_flag=closed_flag,
                )

            return _req_lp

except ImportError:  # pragma: no cover
    ThinkGateLogitsProcessor = None  # type: ignore
