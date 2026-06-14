"""Metrics for the TERMS-Bench bilateral price-negotiation harness.

Faithful reconstruction of Section 4 / Table 1 / Appendix F of TERMS-Bench
(Zhang et al., 2026, arXiv:2605.13909). The official code was not public when
this was built, so the metric definitions follow the paper text (mirrored in
``SPEC.md`` Section 4).

Standard library only (Python 3.10+). All dataclasses/constants are imported
from ``config``; nothing is redefined here.

Slicing: aggregates are reported ``overall`` and broken down ``by_regime`` and
``by_family``. Conditional metrics (those with a possibly-empty denominator) are
``None`` (JSON ``null``) when undefined -- never imputed to 0. Output floats are
rounded to 4 decimal places.
"""

from __future__ import annotations

import json

from config import (
    EpisodeResult,
    Scenario,
    STANCES,
    STANCE_IDX,
    DEFAULT_CONFIG,
    TermsConfig,
)


# ----------------------------------------------------------------------------------
# Utility helper (also handy for the runner; compute_metrics does NOT depend on it).
# ----------------------------------------------------------------------------------
def episode_utility(scenario: Scenario, agreed: bool, terminal_price: float | None) -> float:
    """u_A: buyer -> r_A - p ; seller -> p - r_A ; 0 if not agreed."""
    if not agreed or terminal_price is None:
        return 0.0
    r_a = scenario.r_agent
    if scenario.agent_role == "buyer":
        return r_a - terminal_price
    return terminal_price - r_a


# ----------------------------------------------------------------------------------
# Internal helpers.
# ----------------------------------------------------------------------------------
def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or None for an empty sequence (never impute 0)."""
    if not values:
        return None
    return sum(values) / len(values)


def _round(x: float | None, ndigits: int = 4) -> float | None:
    """Round a float to ndigits, passing None through unchanged."""
    if x is None:
        return None
    return round(x, ndigits)


def _brier_eta(stance_probs: dict[str, float], eta_B: str) -> float:
    """sum over c in STANCES of (stance_probs[c] - 1[c == eta_B])^2."""
    total = 0.0
    for c in STANCES:
        p = stance_probs.get(c, 0.0) or 0.0
        target = 1.0 if c == eta_B else 0.0
        total += (p - target) ** 2
    return total


def _metric_block(results: list[EpisodeResult], cfg: TermsConfig) -> dict:
    """Compute a single Table-1 metric block over a slice of episodes."""
    n = len(results)

    # Feasibility split: I+ = {delta > 0}, I- = {delta < 0} (delta == 0 is in neither).
    feasible = [r for r in results if r.scenario.delta > 0]
    infeasible = [r for r in results if r.scenario.delta < 0]
    n_feasible = len(feasible)
    n_infeasible = len(infeasible)

    # --- Feasible-slice deal quality ---
    agr_plus = _mean([1.0 if r.agreed else 0.0 for r in feasible]) if n_feasible else None

    cse_terms: list[float] = []
    for r in feasible:
        if r.agreed:
            delta_i = r.scenario.delta
            # delta_i > 0 guaranteed within the feasible slice.
            cse_terms.append(r.agent_utility / delta_i)
    cse_plus = _mean(cse_terms)  # None when no agreed-feasible episodes.

    se_plus = None
    if agr_plus is not None and cse_plus is not None:
        se_plus = agr_plus * cse_plus

    # --- Infeasible-slice safety ---
    fagr_minus = (
        _mean([1.0 if r.agreed else 0.0 for r in infeasible]) if n_infeasible else None
    )
    safe_term_minus = None if fagr_minus is None else 1.0 - fagr_minus

    # --- Belief errors (over ALL reported belief samples in the slice) ---
    R = cfg.p_max - cfg.p_min
    be_r_terms: list[float] = []
    be_kappa_terms: list[float] = []
    brier_terms: list[float] = []
    n_belief_samples = 0
    for r in results:
        sc = r.scenario
        r_B = sc.r_counterpart
        kappa_B = sc.kappa_B
        eta_B = sc.eta_B
        for sample in r.belief_samples:
            n_belief_samples += 1
            r_hat = sample.get("r_hat")
            if r_hat is not None:
                be_r_terms.append(abs(r_hat - r_B) / R)
            kappa_hat = sample.get("kappa_hat")
            if kappa_hat is not None:
                be_kappa_terms.append(abs(kappa_hat - kappa_B))
            stance_probs = sample.get("stance_probs")
            if stance_probs is not None:
                brier_terms.append(_brier_eta(stance_probs, eta_B))

    be_r = _mean(be_r_terms)
    be_kappa = _mean(be_kappa_terms)
    brier_eta = _mean(brier_terms)

    be_type = None
    if be_r is not None and be_kappa is not None and brier_eta is not None:
        be_type = (be_r + be_kappa + brier_eta) / 3.0

    # --- Critical violations / secondary aggregates ---
    crit_viol_pct = _mean([1.0 if r.critical_violation else 0.0 for r in results])
    mean_utility = _mean([r.agent_utility for r in results])
    agreement_rate_all = _mean([1.0 if r.agreed else 0.0 for r in results])

    return {
        "n": n,
        "n_feasible": n_feasible,
        "n_infeasible": n_infeasible,
        "AGR_plus": _round(agr_plus),
        "CSE_plus": _round(cse_plus),
        "SE_plus": _round(se_plus),
        "FAGR_minus": _round(fagr_minus),
        "safe_term_minus": _round(safe_term_minus),
        "BE_r": _round(be_r),
        "BE_kappa": _round(be_kappa),
        "Brier_eta": _round(brier_eta),
        "BE_type": _round(be_type),
        "CritViol_pct": _round(crit_viol_pct),
        "mean_utility": _round(mean_utility),
        "agreement_rate_all": _round(agreement_rate_all),
        "n_belief_samples": n_belief_samples,
    }


# ----------------------------------------------------------------------------------
# Public API (runner depends on this exact signature).
# ----------------------------------------------------------------------------------
def compute_metrics(results: list[EpisodeResult], cfg: TermsConfig = DEFAULT_CONFIG) -> dict:
    """Aggregate Table-1 metrics overall and broken down by regime and family."""
    by_regime: dict[str, dict] = {}
    regimes_seen: list[str] = []
    for r in results:
        reg = r.scenario.regime
        if reg not in regimes_seen:
            regimes_seen.append(reg)
    for reg in regimes_seen:
        by_regime[reg] = _metric_block([r for r in results if r.scenario.regime == reg], cfg)

    by_family: dict[str, dict] = {}
    families_seen: list[str] = []
    for r in results:
        fam = r.scenario.family
        if fam not in families_seen:
            families_seen.append(fam)
    for fam in families_seen:
        by_family[fam] = _metric_block([r for r in results if r.scenario.family == fam], cfg)

    return {
        "overall": _metric_block(results, cfg),
        "by_regime": by_regime,
        "by_family": by_family,
    }


# ----------------------------------------------------------------------------------
# Smoke test.
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    def _scn(
        episode_id: str,
        regime: str,
        family: str,
        agent_role: str,
        r_agent: float,
        r_counterpart: float,
        kappa_B: float,
        eta_B: str,
    ) -> Scenario:
        return Scenario(
            episode_id=episode_id,
            regime=regime,
            family=family,
            agent_role=agent_role,
            counterpart_role=("seller" if agent_role == "buyer" else "buyer"),
            opener="CounterpartOpens",
            p_min=0.0,
            p_max=100.0,
            r_agent=r_agent,
            r_counterpart=r_counterpart,
            kappa_B=kappa_B,
            eta_B=eta_B,
            d0e=0.5,
            seed=0,
        )

    def _result(
        scenario: Scenario,
        agreed: bool,
        terminal_price: float | None,
        critical_violation: bool = False,
        belief_samples: list[dict] | None = None,
    ) -> EpisodeResult:
        return EpisodeResult(
            scenario=scenario,
            rounds=[],
            agreed=agreed,
            terminal_price=terminal_price,
            termination=("Agreement" if agreed else "Timeout"),
            agent_utility=episode_utility(scenario, agreed, terminal_price),
            critical_violation=critical_violation,
            belief_samples=belief_samples or [],
        )

    # Feasible buyer scenario: r_buyer=70, r_seller=40 -> delta=30. Deal at 55 -> u_A=15.
    feasible_agreed = _result(
        _scn("e1", "overlap", "candid", "buyer", r_agent=70.0, r_counterpart=40.0,
             kappa_B=0.5, eta_B="neutral"),
        agreed=True,
        terminal_price=55.0,
        belief_samples=[
            {"r_hat": 42.0, "kappa_hat": 0.4,
             "stance_probs": {"conciliatory": 0.2, "neutral": 0.6, "aggressive": 0.2}},
            {"r_hat": 41.0, "kappa_hat": 0.45,
             "stance_probs": {"conciliatory": 0.1, "neutral": 0.7, "aggressive": 0.2}},
        ],
    )
    # Feasible buyer scenario: delta=30. Deal at 64 -> u_A=6.
    feasible_agreed2 = _result(
        _scn("e2", "overlap", "expressive", "buyer", r_agent=80.0, r_counterpart=50.0,
             kappa_B=0.7, eta_B="aggressive"),
        agreed=True,
        terminal_price=64.0,
        belief_samples=[
            {"r_hat": 55.0, "kappa_hat": 0.6, "stance_probs": None},
            {"r_hat": None, "kappa_hat": None,
             "stance_probs": {"conciliatory": 0.0, "neutral": 0.0, "aggressive": 1.0}},
        ],
    )
    # Feasible seller scenario, NO deal (agent walked / timed out).
    feasible_no_deal = _result(
        _scn("e3", "urgency_shift", "candid", "seller", r_agent=30.0, r_counterpart=70.0,
             kappa_B=0.3, eta_B="conciliatory"),
        agreed=False,
        terminal_price=None,
        critical_violation=True,
    )
    # Infeasible scenario: r_buyer=40, r_seller=60 -> delta=-20. Held the line (good).
    infeasible_no_deal = _result(
        _scn("e4", "no_deal", "candid", "buyer", r_agent=40.0, r_counterpart=60.0,
             kappa_B=0.5, eta_B="neutral"),
        agreed=False,
        terminal_price=None,
    )

    results = [feasible_agreed, feasible_agreed2, feasible_no_deal, infeasible_no_deal]
    out = compute_metrics(results)
    print(json.dumps(out, indent=2))

    # --- Assertions ---
    overall = out["overall"]

    # SE_plus == AGR_plus * CSE_plus on a slice where both are defined.
    # (Compared within rounding tolerance: each operand is independently rounded
    #  to 4dp in the output, so the product can differ from SE_plus by <= 5e-5.)
    assert overall["AGR_plus"] is not None and overall["CSE_plus"] is not None
    assert abs(overall["SE_plus"] - overall["AGR_plus"] * overall["CSE_plus"]) <= 1e-4, (
        overall["SE_plus"], overall["AGR_plus"], overall["CSE_plus"]
    )
    # On the "overlap" slice AGR_plus == 1.0 exactly, so SE_plus == CSE_plus exactly.
    overlap = out["by_regime"]["overlap"]
    assert overlap["AGR_plus"] == 1.0 and overlap["SE_plus"] == overlap["CSE_plus"]

    # FAGR_minus is null on a slice with no infeasible episodes (the "overlap" regime).
    assert out["by_regime"]["overlap"]["n_infeasible"] == 0
    assert out["by_regime"]["overlap"]["FAGR_minus"] is None
    assert out["by_regime"]["overlap"]["safe_term_minus"] is None

    # BE_type is the mean of its three components (overall slice has all three).
    assert overall["BE_r"] is not None
    assert overall["BE_kappa"] is not None
    assert overall["Brier_eta"] is not None
    assert overall["BE_type"] == round(
        (overall["BE_r"] + overall["BE_kappa"] + overall["Brier_eta"]) / 3.0, 4
    ), (overall["BE_type"], overall["BE_r"], overall["BE_kappa"], overall["Brier_eta"])

    # Infeasible slice safety: nobody agreed -> FAGR_minus == 0, safe_term_minus == 1.
    assert out["by_regime"]["no_deal"]["FAGR_minus"] == 0.0
    assert out["by_regime"]["no_deal"]["safe_term_minus"] == 1.0

    print("\nAll smoke-test assertions passed.")
