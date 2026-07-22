"""Unit tests for the negotiation think-gate masking logic (research_logs/
think-close-debugging-0618.md + debug_logs/think-template-mismatch-0619.md). Pure-function
tests: no vLLM / no GPU.

SLIMMED gate: the opening <think> is injected by the qwen35_strip_thinking chat template
(prompt side), so generation starts INSIDE the think block. The gate no longer forces
<think>; it only enforces the close (min-think floor + no action/EOS until </think>)."""

import torch

from skyrl.backends.skyrl_train.inference_engines.vllm.think_gate_logits_processor import (
    apply_think_gate,
    build_think_gate_extra_args,
)

THINK_CLOSE = 2
LT = 3  # leading token of <propose>/<accept>/<deal>
EOS = 4
IM_END = 5
VOCAB = 10
MIN_THINK = 16


def _fresh_logits():
    return torch.arange(VOCAB, dtype=torch.float32)


def _gate(output_ids, logits, closed_flag):
    return apply_think_gate(
        output_ids,
        logits,
        think_close_id=THINK_CLOSE,
        block_ids=[LT],
        eos_ids=[EOS, IM_END],
        min_think_tokens=MIN_THINK,
        closed_flag=closed_flag,
    )


def test_first_step_no_force_open():
    # n==0: the gate no longer forces <think> (the template injects it). Think is open, so
    # the close is masked by the floor and action/eos are masked, but reasoning may start.
    out = _gate([], _fresh_logits(), [False])
    assert out[THINK_CLOSE].item() == float("-inf")  # floor: 0 < MIN_THINK
    assert out[LT].item() == float("-inf")
    assert out[EOS].item() == float("-inf")
    assert out[9].item() != float("-inf")  # free to begin reasoning


def test_close_blocked_before_min_think_floor():
    # 5 content tokens < 16 -> close masked; action + turn-end masked while open.
    out = _gate([9] * 5, _fresh_logits(), [False])
    assert out[THINK_CLOSE].item() == float("-inf")
    assert out[LT].item() == float("-inf")
    assert out[EOS].item() == float("-inf")
    assert out[IM_END].item() == float("-inf")
    assert out[9].item() != float("-inf")


def test_close_allowed_after_min_think_floor():
    # 16 content tokens -> close allowed; actions/eos still blocked until close is emitted.
    out = _gate([9] * MIN_THINK, _fresh_logits(), [False])
    assert out[THINK_CLOSE].item() != float("-inf")
    assert out[LT].item() == float("-inf")
    assert out[EOS].item() == float("-inf")


def test_action_cannot_start_inside_open_think():
    out = _gate([9] * (MIN_THINK + 3), _fresh_logits(), [False])
    assert out[LT].item() == float("-inf")  # cannot start <propose>/<accept>/<deal>
    assert out[THINK_CLOSE].item() != float("-inf")  # must close first


def test_gate_inert_after_close():
    closed = [False]
    out = _gate([9] * MIN_THINK + [THINK_CLOSE], _fresh_logits(), closed)
    assert closed[0] is True
    assert torch.equal(out, _fresh_logits())  # nothing masked
    out2 = _gate([9] * MIN_THINK + [THINK_CLOSE, 9, LT], _fresh_logits(), closed)
    assert out2[LT].item() != float("-inf")
    assert out2[EOS].item() != float("-inf")


def test_full_turn_trajectory_is_well_formed():
    closed = [False]
    seq = []
    for _ in range(MIN_THINK):
        out = _gate(seq, _fresh_logits(), closed)
        assert out[THINK_CLOSE].item() == float("-inf")
        seq.append(9)
    out = _gate(seq, _fresh_logits(), closed)
    assert out[THINK_CLOSE].item() != float("-inf")
    seq.append(THINK_CLOSE)
    out = _gate(seq, _fresh_logits(), closed)
    assert torch.equal(out, _fresh_logits())


# Regression: build_think_gate_extra_args must mask context-merged action-start tokens
# (the 2026-06-19 bug — see debug_logs/think-template-mismatch-0619.md).
class _FakeTokenizer:
    eos_token_id = 4
    _vocab = {
        "<think>": 1, "</think>": 2, "<|im_end|>": 5, "<": 3,
        " <": 30, "。<": 31, "><": 32, "abc": 10, "prop": 11,
    }
    _dec = {v: k for k, v in _vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids):
        return "".join(self._dec.get(int(i), "") for i in ids)

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        if text in ("<propose>", "<accept>", "<deal>"):
            return [3, 11]
        raise AssertionError(f"unexpected encode({text!r})")


def test_build_extra_args_masks_context_merged_lt_tokens():
    cfg = build_think_gate_extra_args(_FakeTokenizer(), min_think_tokens=16)
    block = set(cfg["block_ids"])
    assert 3 in block, "isolated '<' must be blocked"
    assert {30, 31, 32} <= block, "context-merged '<' tokens must be blocked"
    assert cfg["think_close_id"] not in block
    assert 10 not in block and 11 not in block
