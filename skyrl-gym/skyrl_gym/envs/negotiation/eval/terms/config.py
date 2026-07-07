"""Shared configuration, hyperparameters, and data model for the TERMS-Bench harness.

This is a faithful reconstruction of the bilateral price-negotiation instantiation of
TERMS-Bench (Zhang et al., 2026, arXiv:2605.13909). The benchmark's official code
(github.com/zou-group/terms-bench) was NOT public at the time this harness was built,
so the simulator kernel, scenario generator, and metrics are reimplemented from the
paper's appendices. Equation/table references in comments point at that paper.

All other modules in this package import their constants and dataclasses from here, so
this file is the single source of truth for fidelity. Values flagged "CALIBRATION
DEFAULT" are not numerically specified in the paper text and were chosen to match the
qualitative behavior the paper describes; override them via TermsConfig if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

# ----------------------------------------------------------------------------------
# Stance ordering (Appendix C.1): (C, N, A) = (conciliatory, neutral, aggressive).
# ----------------------------------------------------------------------------------
STANCES: tuple[str, str, str] = ("conciliatory", "neutral", "aggressive")
STANCE_IDX = {s: i for i, s in enumerate(STANCES)}

Role = Literal["buyer", "seller"]
Opener = Literal["AgentOpens", "CounterpartOpens"]
Decision = Literal["Offer", "Accept", "Reject"]
Regime = Literal["overlap", "urgency_shift", "no_deal"]
Sentiment = Literal["positive", "neutral", "negative"]
StrategyCue = Literal["Concede", "Hold", "Pressure"]

REGIMES: tuple[Regime, ...] = ("overlap", "urgency_shift", "no_deal")
FAMILIES_ORDER: tuple[str, ...] = (
    "candid",
    "taciturn",
    "expressive",
    "strategic",
    "stochastic",
    "adversarial",
)


# ----------------------------------------------------------------------------------
# Counterpart behavior families (Tables 3 & 4, Appendix C.1 / C.5.3).
# rho/xi/lambda2 are stance-ordered (conciliatory, neutral, aggressive).
# price_noise is the normalized sigma_p-bar applied to counter-offers.
# cue_mode: "base" (accurate), "uninformative" (neutral/Hold),
#           "noisy" (Stochastic), "pressuring" (Adversarial: negative/Pressure).
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilyPreset:
    name: str
    rho: tuple[float, float, float]
    xi: tuple[float, float, float]
    lambda2: tuple[float, float, float]
    price_noise: float
    stance_prior: tuple[float, float, float]
    cue_mode: Literal["base", "uninformative", "noisy", "pressuring"]
    role: str


_UNIFORM_PRIOR = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

FAMILIES: dict[str, FamilyPreset] = {
    "candid": FamilyPreset(
        "candid",
        (0.0, -0.25, -0.75),
        (0.40, 0.0, -0.50),
        (0.30, 0.50, 1.00),
        0.01,
        _UNIFORM_PRIOR,
        "base",
        "core",
    ),
    "taciturn": FamilyPreset(
        "taciturn",
        (0.0, -0.25, -0.75),
        (0.40, 0.0, -0.50),
        (0.30, 0.50, 1.00),
        0.01,
        _UNIFORM_PRIOR,
        "uninformative",
        "core",
    ),
    "expressive": FamilyPreset(
        "expressive",
        (0.0, -0.75, -1.50),
        (0.40, 0.0, -0.75),
        (0.45, 0.90, 1.80),
        0.03,
        _UNIFORM_PRIOR,
        "base",
        "core",
    ),
    "strategic": FamilyPreset(
        "strategic",
        (0.0, -0.75, -1.50),
        (0.40, 0.0, -0.75),
        (0.45, 0.90, 1.80),
        0.03,
        _UNIFORM_PRIOR,
        "uninformative",
        "core",
    ),
    "stochastic": FamilyPreset(
        "stochastic",
        (0.0, -0.50, -1.10),
        (0.35, 0.0, -0.60),
        (0.35, 0.70, 1.40),
        0.08,
        _UNIFORM_PRIOR,
        "noisy",
        "noise_floor",
    ),
    "adversarial": FamilyPreset(
        "adversarial",
        (-0.25, -1.25, -2.25),
        (0.0, -0.50, -1.20),
        (0.60, 1.40, 2.60),
        0.01,
        (0.05, 0.15, 0.80),
        "pressuring",
        "stress",
    ),
}


# ----------------------------------------------------------------------------------
# Default simulator hyperparameters (Tables 5 & 6). Everything here is specified
# in the paper unless marked CALIBRATION DEFAULT.
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class TermsConfig:
    # Public price bounds for synthetic runs (Appendix K: "fixed at [0,100]").
    p_min: float = 0.0
    p_max: float = 100.0
    # Horizon K ("typically K=10"; Appendix I, line "at most K rounds").
    K: int = 10

    # --- Acceptance model (Table 5; eqs. 5-6) ---
    alpha: float = 6.0  # sensitivity to normalized favorability Delta-bar_k
    beta: float = 1.0  # sensitivity to counterpart urgency kappa_B
    gamma: float = 2.0  # sensitivity to transformed remaining time (1 - sqrt(k/K))

    # --- Walk-away hazard (Table 5; eqs. 7/21) ---
    phi0: float = -4.5
    phi_delta: float = 30.0
    phi_T: float = 1.5
    # k_walk = ceil(K/2) computed from K.

    # --- Counter-offer model (Table 5; concession score lambda_B) ---
    lambda0: float = 0.12  # baseline latent concession tendency
    lambda1: float = 0.28  # urgency sensitivity
    lambda3: float = 0.10  # slows aggressive counterparts
    lambda4: float = 0.10  # accelerates conciliatory counterparts
    # lambda2 is family/stance-specific (FamilyPreset.lambda2).
    eps_c: float = 1e-6  # numerical-stability constant for C_k^B

    # --- History features (Appendix C.3) ---
    tau_rigid: float = 0.10

    # --- Opening-offer model (Table 6; eqs. 15-16) ---
    d0_min: float = 0.20
    d0_max: float = 0.80
    omega_kappa: float = 0.30
    omega_eta: float = 0.15
    omega_eta_prime: float = 0.15
    phi_min: float = 0.5
    phi_max: float = 1.5
    sigma0_bar: float = 0.02  # normalized opening-offer noise scale

    # --- Strategic cue model (Table 6; Appendix C.5.1) ---
    tau_dead: float = 0.80
    b_C: float = 1.0
    b_H: float = 0.5
    b_P: float = 1.0
    alpha_C: float = 2.0
    alpha_P: float = 2.0
    beta_C: float = 1.0
    tau_conc: float = 0.0  # CALIBRATION DEFAULT: concession threshold in Concede logit
    # (not given numerically in the paper text; 0.0 means any
    #  realized concession raises the Concede logit).

    # --- Sentiment cue model (Table 6; Appendix C.5.2) ---
    mu_s: float = 1.0
    tau_s: float = 0.5
    sigma_s: float = 0.75

    # --- Stochastic-family cue noise (Table 6; Appendix C.5.3) ---
    sigma_s_stoch: float = 2.0
    T_stoch: float = 2.5

    # ------------------------------------------------------------------------------
    # Regime geometry. The paper defers these exact numbers to "Appendix C.6" but
    # the released text does not tabulate them, so the following are CALIBRATION
    # DEFAULTS chosen to reproduce the described regimes on a [0,100] scale:
    #   - overlap/urgency-shift: positive ZOPA width Delta in [zopa_min, zopa_max]
    #   - no-deal: infeasible gap g in [gap_min, gap_max]
    #   - baseline urgency ~ Beta(2,2) (mean 0.5); shifted urgency ~ Beta(5,2)
    #     (mean ~0.71, the "counterpart-more-urgent" direction used in the main suite).
    # ------------------------------------------------------------------------------
    zopa_min: float = 10.0  # CALIBRATION DEFAULT
    zopa_max: float = 40.0  # CALIBRATION DEFAULT
    gap_min: float = 5.0  # CALIBRATION DEFAULT
    gap_max: float = 30.0  # CALIBRATION DEFAULT
    urgency_alpha: float = 2.0  # CALIBRATION DEFAULT (baseline Beta alpha_kappa)
    urgency_beta: float = 2.0  # CALIBRATION DEFAULT (baseline Beta beta_kappa)
    urgency_shift_alpha: float = 5.0  # CALIBRATION DEFAULT (shifted Beta)
    urgency_shift_beta: float = 2.0  # CALIBRATION DEFAULT (shifted Beta)

    @property
    def R(self) -> float:
        return self.p_max - self.p_min

    @property
    def k_walk(self) -> int:
        return math.ceil(self.K / 2)


DEFAULT_CONFIG = TermsConfig()


# ----------------------------------------------------------------------------------
# Data model shared across modules.
# ----------------------------------------------------------------------------------
@dataclass
class Scenario:
    """A single fully-specified episode instance (before any interaction)."""

    episode_id: str
    regime: Regime
    family: str
    agent_role: Role  # role of the evaluated agent
    counterpart_role: Role  # always the opposite of agent_role
    opener: Opener
    p_min: float
    p_max: float
    r_agent: float  # r_A (private to agent, given in its prompt)
    r_counterpart: float  # r_B (hidden; the type's reservation value)
    kappa_B: float  # counterpart urgency in [0,1] (hidden)
    eta_B: str  # counterpart stance in STANCES (hidden)
    d0e: float  # opening harshness (hidden)
    seed: int  # disjoint per-episode seed stream base

    @property
    def r_buyer(self) -> float:
        return self.r_agent if self.agent_role == "buyer" else self.r_counterpart

    @property
    def r_seller(self) -> float:
        return self.r_agent if self.agent_role == "seller" else self.r_counterpart

    @property
    def delta(self) -> float:
        """ZOPA signed width: >0 feasible, <0 no-deal."""
        return self.r_buyer - self.r_seller

    @property
    def feasible(self) -> bool:
        return self.delta > 0


@dataclass
class Belief:
    """Agent's self-reported type estimate (scored, never shown to counterpart)."""

    r_hat: float | None = None
    kappa_hat: float | None = None
    stance_probs: dict[str, float] | None = None  # keys: conciliatory/neutral/aggressive


@dataclass
class AgentAction:
    decision: Decision
    price: float | None
    message: str
    belief: Belief = field(default_factory=Belief)
    raw: str = ""  # raw model text, for logging/debugging
    parse_error: str | None = None


@dataclass
class CounterpartMove:
    decision: Decision  # Offer / Accept / Reject (Reject == walk-away)
    price: float | None
    sentiment: Sentiment
    strategy_cue: StrategyCue
    message: str = ""


@dataclass
class RoundLog:
    k: int
    actor: Literal["agent", "counterpart"]
    decision: Decision
    price: float | None
    message: str = ""
    sentiment: str | None = None
    strategy_cue: str | None = None
    belief: Belief | None = None
    violations: list[str] = field(default_factory=list)  # critical-violation tags this round


@dataclass
class EpisodeResult:
    scenario: Scenario
    rounds: list[RoundLog]
    agreed: bool
    terminal_price: float | None  # f_i (deal price) or None
    termination: str  # Agreement / AgentReject / CounterpartWalkAway / Timeout / Error
    agent_utility: float  # u_A(f_i); 0 if no deal
    critical_violation: bool  # any critical violation in the episode
    violation_tags: list[str] = field(default_factory=list)
    agent_opening_price: float | None = None
    belief_samples: list[dict] = field(default_factory=list)  # per-round belief vs truth
    error: str | None = None


__all__ = [
    "STANCES",
    "STANCE_IDX",
    "REGIMES",
    "FAMILIES_ORDER",
    "FamilyPreset",
    "FAMILIES",
    "TermsConfig",
    "DEFAULT_CONFIG",
    "Scenario",
    "Belief",
    "AgentAction",
    "CounterpartMove",
    "RoundLog",
    "EpisodeResult",
    "Role",
    "Opener",
    "Decision",
    "Regime",
    "Sentiment",
    "StrategyCue",
]
