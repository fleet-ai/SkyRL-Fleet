"""Shared configuration, constants, and data model for the NegotiationArena harness.

Faithful reconstruction of NegotiationArena (Bianchi et al., ICML 2024,
arXiv:2402.05863; code github.com/vinid/NegotiationArena). This file is the single
source of truth: constants (XML-like tags, player names, money tokens), the three
game presets, scenario-generation defaults, and the dataclasses every other module
imports. See SPEC.md for the protocol/equation references.

Values flagged "CALIBRATION DEFAULT" are not numerically pinned by the paper and were
chosen to match the described setup at a lower per-cell episode count; override via
NegArenaConfig / the CLI if needed.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ----------------------------------------------------------------------------------
# Players & structured-protocol tags (Appendix F, Fig. 21; repo constants.py).
# ----------------------------------------------------------------------------------
PLAYER_RED = "Player RED"
PLAYER_BLUE = "Player BLUE"
SEAT_NAMES: tuple[str, str] = (PLAYER_RED, PLAYER_BLUE)  # seat 0 = RED, seat 1 = BLUE

MY_NAME_TAG = "my name"
RESOURCES_TAG = "resources in hand"
GOALS_TAG = "goal"
REASONING_TAG = "reason"
PLAYER_ANSWER_TAG = "player answer"
PROPOSED_TRADE_TAG = "newly proposed trade"
MESSAGE_TAG = "message"

# player-answer / proposed-trade sentinel tokens.
ACCEPTING_TAG = "ACCEPT"
REFUSING_OR_WAIT_TAG = "WAIT"

# Tags whose content is stripped before a message is forwarded to the other agent
# (kept secret; §2.2 / Appendix D.3). name/resources/goal are public echoes.
PRIVATE_TAGS: tuple[str, ...] = (REASONING_TAG,)

# Order in which all tags must appear in an agent response (Fig. 21).
RESPONSE_TAG_ORDER: tuple[str, ...] = (
    MY_NAME_TAG,
    RESOURCES_TAG,
    GOALS_TAG,
    REASONING_TAG,
    PLAYER_ANSWER_TAG,
    PROPOSED_TRADE_TAG,
    MESSAGE_TAG,
)

# Social-behaviour personas (Appendix F.2). Keyed by (persona, game).
SOCIAL_BEHAVIOURS: dict[str, dict[str, str]] = {
    "cunning": {
        "resource_exchange": "You are cunning and sly in your plan to get more resources. "
        "Humiliate your opponent as much as possible. Insult them.",
        "ultimatum": "You are cunning and sly in your plan to get more than your opponent. "
        "Humiliate your opponent as much as possible. Insult them.",
        "sell_buy": "You are cunning and sly in your plan to spend less ZUP. "
        "Humiliate your opponent as much as possible. Insult them.",
    },
    "desperate": {
        "resource_exchange": "You must fake being desperate. Supplicate and beg to get more resources.",
        "ultimatum": "You must fake being desperate. Supplicate and beg to get more dollars.",
        "sell_buy": "You must fake being desperate. Supplicate and beg to spend less ZUP.",
    },
}

# ----------------------------------------------------------------------------------
# Games.
# ----------------------------------------------------------------------------------
GameType = Literal["resource_exchange", "ultimatum", "sell_buy"]
GAMES_ORDER: tuple[GameType, ...] = ("resource_exchange", "ultimatum", "sell_buy")

# Role labels per seat, by game (used for prompts + metric breakdowns).
ROLE_LABELS: dict[str, tuple[str, str]] = {
    "resource_exchange": ("player_1", "player_2"),
    "ultimatum": ("proposer", "responder"),
    "sell_buy": ("seller", "buyer"),
}


@dataclass(frozen=True)
class GamePreset:
    """Static structure of a game (Tables 1–3)."""

    name: GameType
    money_token: Optional[str]  # currency token, or None (resource_exchange)
    resource_tokens: tuple[str, ...]  # all tradeable tokens in this game
    max_turns: int  # turn budget (ACCEPT or budget exhaustion ends it)
    number_of_proposals: int  # per-seat cap on own proposals (Fig. 21 rule 2)
    integer_only: bool = True


GAME_PRESETS: dict[str, GamePreset] = {
    "resource_exchange": GamePreset(
        name="resource_exchange",
        money_token=None,
        resource_tokens=("X", "Y"),
        max_turns=8,  # Table 1
        number_of_proposals=4,  # max_turns // 2
    ),
    "ultimatum": GamePreset(
        name="ultimatum",
        money_token="Dollars",
        resource_tokens=("Dollars",),
        max_turns=8,  # Table 2
        number_of_proposals=4,
    ),
    "sell_buy": GamePreset(
        name="sell_buy",
        money_token="ZUP",
        resource_tokens=("X", "ZUP"),
        max_turns=10,  # Table 3 / README BuySellGame(iterations=10)
        number_of_proposals=5,
    ),
}

# ----------------------------------------------------------------------------------
# Scenario-generation defaults (Tables 1–3; §5.1 for the sell/buy draws).
# ----------------------------------------------------------------------------------
RESOURCE_EXCHANGE_ENDOWMENTS: tuple[dict[str, int], dict[str, int]] = (
    {"X": 25, "Y": 5},
    {"X": 5, "Y": 25},
)

ULTIMATUM_DEFAULT_AMOUNT = 100  # pot held by the proposer (seat 0)

SELL_BUY_BUYER_BUDGET = 100  # ZUP the buyer starts with
SELL_BUY_DEFAULT_COST = 40  # seller cost of production
SELL_BUY_DEFAULT_WILLINGNESS = 60  # buyer willingness to pay
SELL_BUY_COST_RANGE: tuple[int, int] = (20, 40)  # U{20,40} (§5.1)
SELL_BUY_WILLINGNESS_RANGE: tuple[int, int] = (60, 80)  # U{60,80} (§5.1)

N_PER_CELL_FULL_SUITE = 20  # CALIBRATION DEFAULT (paper ran 60 per ordered pair)


@dataclass(frozen=True)
class NegArenaConfig:
    """Tunable global defaults (override per-run)."""

    ultimatum_amount: int = ULTIMATUM_DEFAULT_AMOUNT
    buyer_budget: int = SELL_BUY_BUYER_BUDGET
    sell_buy_cost: int = SELL_BUY_DEFAULT_COST
    sell_buy_willingness: int = SELL_BUY_DEFAULT_WILLINGNESS
    vary_sell_buy: bool = True  # draw cost/willingness from the §5.1 ranges
    vary_amount: bool = False  # draw ultimatum amount (numerosity probe, §5.2)


DEFAULT_CONFIG = NegArenaConfig()

# ----------------------------------------------------------------------------------
# Data model.
# ----------------------------------------------------------------------------------
SeatResources = dict  # alias for {token: int}


@dataclass
class Trade:
    """A bilateral exchange. ``gives[seat]`` is the bundle that ``seat`` hands over.

    Keyed by seat index (0 = RED, 1 = BLUE). Both seats always present (a seat that
    gives nothing has an empty / all-zero bundle).
    """

    gives: dict[int, dict[str, int]]

    def give(self, seat: int) -> dict[str, int]:
        return dict(self.gives.get(seat, {}))

    def to_string(self, seat_names: tuple[str, str] = SEAT_NAMES) -> str:
        """Render as the canonical NegotiationArena trade string."""
        parts = []
        for seat in (0, 1):
            bundle = self.gives.get(seat, {}) or {}
            if bundle:
                items = ", ".join(f"{tok}: {int(amt)}" for tok, amt in bundle.items())
            else:
                items = "nothing"
            parts.append(f"{seat_names[seat]} Gives {items}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {str(seat): dict(self.gives.get(seat, {})) for seat in (0, 1)}


@dataclass
class AgentAction:
    """Parsed structured response from an agent for one turn."""

    answer: Optional[str]  # ACCEPTING_TAG / REFUSING_OR_WAIT_TAG / None
    proposed_trade: Optional[Trade]
    message: str = ""
    reasoning: str = ""
    raw: str = ""
    parse_error: Optional[str] = None

    @property
    def is_accept(self) -> bool:
        return (self.answer or "").strip().upper() == ACCEPTING_TAG

    @property
    def has_proposal(self) -> bool:
        return self.proposed_trade is not None


@dataclass
class TurnLog:
    turn: int
    seat: int
    answer: Optional[str]
    proposed_trade: Optional[dict]  # serialized Trade.to_dict() or None
    message: str = ""
    has_reasoning: bool = False
    violations: list[str] = field(default_factory=list)


@dataclass
class GameResult:
    scenario: "Scenario"
    turns: list[TurnLog]
    deal: bool
    accepted_trade: Optional[Trade]
    termination: str  # Agreement / Timeout / FormatError / Error
    payoffs: dict[int, float]  # {seat: payoff}
    focal_payoff: float
    opp_payoff: float
    decisive: bool  # focal_payoff != opp_payoff
    focal_win: Optional[bool]  # None when not decisive (tie)
    sale_price: Optional[float] = None  # sell_buy deal price (else None)
    proposer_give: Optional[float] = None  # ultimatum x = amount to responder
    focal_opening_price: Optional[float] = None  # focal's first numeric offer (sell_buy)
    format_violation: bool = False  # any focal format/illegal-action violation
    violation_tags: list[str] = field(default_factory=list)
    n_turns: int = 0
    error: Optional[str] = None


@dataclass
class Scenario:
    """A fully-specified game instance before any interaction."""

    episode_id: str
    game: GameType
    focal_seat: int  # which seat the evaluated policy occupies (0/1)
    first_mover: int  # which seat speaks first
    initial_resources: tuple[dict[str, int], dict[str, int]]
    money_token: Optional[str]
    resource_tokens: tuple[str, ...]
    max_turns: int
    number_of_proposals: int
    seed: int
    # game-specific valuations (None when not applicable):
    amount_to_split: Optional[int] = None  # ultimatum
    seller_cost: Optional[int] = None  # sell_buy (seat 0)
    buyer_willingness: Optional[int] = None  # sell_buy (seat 1)
    social_behaviour: tuple[str, str] = ("", "")

    # ----- convenience -----
    def seat_name(self, seat: int) -> str:
        return SEAT_NAMES[seat]

    def other_seat(self, seat: int) -> int:
        return 1 - seat

    def role_label(self, seat: int) -> str:
        return ROLE_LABELS[self.game][seat]

    @property
    def opponent_seat(self) -> int:
        return 1 - self.focal_seat

    @property
    def focal_role(self) -> str:
        return self.role_label(self.focal_seat)

    def valuation(self, seat: int) -> Optional[int]:
        """The private valuation for ``seat`` in incomplete-info games (else None)."""
        if self.game == "sell_buy":
            return self.seller_cost if seat == 0 else self.buyer_willingness
        return None


__all__ = [
    "PLAYER_RED",
    "PLAYER_BLUE",
    "SEAT_NAMES",
    "MY_NAME_TAG",
    "RESOURCES_TAG",
    "GOALS_TAG",
    "REASONING_TAG",
    "PLAYER_ANSWER_TAG",
    "PROPOSED_TRADE_TAG",
    "MESSAGE_TAG",
    "ACCEPTING_TAG",
    "REFUSING_OR_WAIT_TAG",
    "PRIVATE_TAGS",
    "RESPONSE_TAG_ORDER",
    "SOCIAL_BEHAVIOURS",
    "GameType",
    "GAMES_ORDER",
    "ROLE_LABELS",
    "GamePreset",
    "GAME_PRESETS",
    "RESOURCE_EXCHANGE_ENDOWMENTS",
    "ULTIMATUM_DEFAULT_AMOUNT",
    "SELL_BUY_BUYER_BUDGET",
    "SELL_BUY_DEFAULT_COST",
    "SELL_BUY_DEFAULT_WILLINGNESS",
    "SELL_BUY_COST_RANGE",
    "SELL_BUY_WILLINGNESS_RANGE",
    "N_PER_CELL_FULL_SUITE",
    "NegArenaConfig",
    "DEFAULT_CONFIG",
    "Trade",
    "AgentAction",
    "TurnLog",
    "GameResult",
    "Scenario",
]


# ----------------------------------------------------------------------------------
# Smoke test: python3 config.py
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    # Trade string round-trip shape.
    t = Trade(gives={0: {"X": 10, "Y": 0}, 1: {"X": 0, "Y": 3}})
    s = t.to_string()
    assert s == "Player RED Gives X: 10, Y: 0 | Player BLUE Gives X: 0, Y: 3", s
    assert t.to_dict() == {"0": {"X": 10, "Y": 0}, "1": {"X": 0, "Y": 3}}

    # Presets cover every game in GAMES_ORDER.
    for g in GAMES_ORDER:
        assert g in GAME_PRESETS and GAME_PRESETS[g].name == g
        assert g in ROLE_LABELS and len(ROLE_LABELS[g]) == 2
        for persona in SOCIAL_BEHAVIOURS:
            assert g in SOCIAL_BEHAVIOURS[persona]

    sc = Scenario(
        episode_id="sell_buy-buyer-0000",
        game="sell_buy",
        focal_seat=1,
        first_mover=0,
        initial_resources=({"X": 1}, {"ZUP": 100}),
        money_token="ZUP",
        resource_tokens=("X", "ZUP"),
        max_turns=10,
        number_of_proposals=5,
        seed=123,
        seller_cost=40,
        buyer_willingness=60,
    )
    assert sc.role_label(0) == "seller" and sc.role_label(1) == "buyer"
    assert sc.focal_role == "buyer"
    assert sc.opponent_seat == 0
    assert sc.valuation(1) == 60 and sc.valuation(0) == 40
    print("config.py smoke test passed.")
