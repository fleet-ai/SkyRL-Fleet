"""Tests for the negotiation thinking-channel detectors.

Focus: ``_action_before_think_close`` — the soft, reward-shaped detector used by the
penalty arm (the constrained-decoding think gate's alternative). It flags a policy turn
that opens an action tag (<propose>/<accept>/<deal>) before closing <think>.
"""

import pytest

from skyrl_gym.envs.negotiation.env import _action_before_think_close


@pytest.mark.parametrize(
    "action, expected",
    [
        # Well-formed: reasoning, close, THEN the action -> not flagged.
        ("weighing the split here </think> Here is my offer <propose>{\"a\": 1}</propose>", False),
        # Action opened while think is still open (close comes later) -> flagged.
        ("planning <propose>{\"a\": 1}</propose> so that </think> take it", True),
        # No </think> at all (stop string cut the turn) but an action is present -> flagged.
        ("let me just grab it <propose>{\"a\": 2}</propose>", True),
        # Accept emitted before the close -> flagged.
        ("this looks fine <accept> </think>", True),
        # Dual-protocol deal before the close -> flagged.
        ("I'll claim <deal>{\"a\": 1}</deal> </think>", True),
        # Reasoning only, properly closed, no action this turn -> not flagged.
        ("still thinking about the tradeoffs </think> what do you value most?", False),
        # Reasoning only, never closed, no action -> not flagged (that's a different
        # pathology, caught by think_closed_rate, not this detector).
        ("hmm let me consider the options carefully", False),
        # Closing action tags must NOT trip it (only opening tags count).
        ("done </think> ok <propose>{\"a\": 1}</propose> and </accept>", False),
    ],
)
def test_action_before_think_close(action, expected):
    assert _action_before_think_close(action) is expected
