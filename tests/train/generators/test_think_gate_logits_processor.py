"""Unit tests for the negotiation think-gate masking logic (research_logs/
think-close-debugging-0618.md, Fix 3). Pure-function tests: no vLLM / no GPU."""

import torch

from skyrl.backends.skyrl_train.inference_engines.vllm.think_gate_logits_processor import (
    apply_think_gate,
)

# Toy vocab id assignments for the test.
THINK_OPEN = 1
THINK_CLOSE = 2
LT = 3  # leading token of <propose>/<accept>/<deal>
EOS = 4
IM_END = 5
VOCAB = 10
MIN_THINK = 16


def _fresh_logits():
    # Distinct finite values so we can tell which entries got masked to -inf.
    return torch.arange(VOCAB, dtype=torch.float32)


def _gate(output_ids, logits, closed_flag):
    return apply_think_gate(
        output_ids,
        logits,
        think_open_id=THINK_OPEN,
        think_close_id=THINK_CLOSE,
        block_ids=[LT],
        eos_ids=[EOS, IM_END],
        min_think_tokens=MIN_THINK,
        closed_flag=closed_flag,
    )


def test_first_token_forced_to_think_open():
    logits = _fresh_logits()
    out = _gate([], logits, [False])
    # Only <think> is sampleable.
    assert out[THINK_OPEN].item() == 0.0
    for tid in range(VOCAB):
        if tid != THINK_OPEN:
            assert out[tid].item() == float("-inf")
    assert int(torch.argmax(out).item()) == THINK_OPEN


def test_close_blocked_before_min_think_floor():
    # 1 forced <think> + 5 content tokens => 5 < 16, close must be masked.
    output_ids = [THINK_OPEN] + [9] * 5
    out = _gate(output_ids, _fresh_logits(), [False])
    assert out[THINK_CLOSE].item() == float("-inf")
    # action-start and turn-end are masked while open
    assert out[LT].item() == float("-inf")
    assert out[EOS].item() == float("-inf")
    assert out[IM_END].item() == float("-inf")
    # ordinary content tokens stay sampleable
    assert out[9].item() != float("-inf")


def test_close_allowed_after_min_think_floor():
    # 1 forced <think> + 16 content tokens => think_len == 16 >= 16, close allowed.
    output_ids = [THINK_OPEN] + [9] * MIN_THINK
    out = _gate(output_ids, _fresh_logits(), [False])
    assert out[THINK_CLOSE].item() != float("-inf")
    # but actions/eos still blocked until the close is actually emitted
    assert out[LT].item() == float("-inf")
    assert out[EOS].item() == float("-inf")


def test_gate_inert_after_close():
    closed = [False]
    # The step where </think> is the newest token flips the flag and stops masking.
    out = _gate([THINK_OPEN] + [9] * MIN_THINK + [THINK_CLOSE], _fresh_logits(), closed)
    assert closed[0] is True
    assert torch.equal(out, _fresh_logits())  # nothing masked
    # Subsequent steps (message + action) are fully unconstrained.
    out2 = _gate([THINK_OPEN] + [9] * MIN_THINK + [THINK_CLOSE, 9, LT], _fresh_logits(), closed)
    assert out2[LT].item() != float("-inf")
    assert out2[EOS].item() != float("-inf")


def test_action_cannot_start_inside_open_think():
    # The core bug: <propose> emitted inside an unclosed <think>. The gate forbids the
    # action's leading token until </think> is emitted.
    output_ids = [THINK_OPEN] + [9] * (MIN_THINK + 3)  # past the floor, still open
    out = _gate(output_ids, _fresh_logits(), [False])
    assert out[LT].item() == float("-inf")  # cannot start <propose>/<accept>/<deal>
    assert out[THINK_CLOSE].item() != float("-inf")  # must close first


def test_full_turn_trajectory_is_well_formed():
    # Walk a turn step-by-step; assert the only reachable structure is
    # <think> (>=16) </think> ... <action>.
    closed = [False]
    # step 0 -> forced think
    assert int(torch.argmax(_gate([], _fresh_logits(), closed)).item()) == THINK_OPEN
    seq = [THINK_OPEN]
    # below floor: close masked until MIN_THINK content tokens follow <think>.
    for _ in range(MIN_THINK):
        out = _gate(seq, _fresh_logits(), closed)
        assert out[THINK_CLOSE].item() == float("-inf")
        seq.append(9)
    # at floor (MIN_THINK content tokens): close now permitted
    out = _gate(seq, _fresh_logits(), closed)
    assert out[THINK_CLOSE].item() != float("-inf")
    seq.append(THINK_CLOSE)
    # after close: free
    out = _gate(seq, _fresh_logits(), closed)
    assert torch.equal(out, _fresh_logits())
