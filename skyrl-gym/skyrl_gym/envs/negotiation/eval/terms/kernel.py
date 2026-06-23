"""Counterpart simulator kernel pi_B for the TERMS-Bench reconstruction (SPEC.md Section 2).

Faithful reimplementation of the bilateral price-negotiation counterpart policy from
TERMS-Bench (arXiv:2605.13909), reconstructed from the paper appendices. All equation /
table references in comments point at SPEC.md (which mirrors the paper). Constants and the
data model are imported from `config.py`; nothing here redefines them.

Standard library only (`math`, `random`). Every stochastic draw goes through the injected
`random.Random` instance (`self.rng`) so a fixed seed reproduces the counterpart's behavior
given identical agent actions (SPEC Section 5).
"""

from __future__ import annotations

import math
import random

from config import (
    TermsConfig,
    Scenario,
    FAMILIES,
    STANCE_IDX,
    CounterpartMove,
    DEFAULT_CONFIG,
)

# Strategic-cue label order matches the logit vector (Concede, Hold, Pressure); SPEC 2e.
_STRAT = ("Concede", "Hold", "Pressure")


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigma(x) = 1/(1+exp(-x))."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _project(x: float, a: float, b: float) -> float:
    """Project x onto the closed interval whose endpoints are a and b (order-agnostic)."""
    return _clamp(x, a, b)


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class Counterpart:
    """Stateful counterpart policy pi_B for a single episode/scenario."""

    def __init__(self, scenario: Scenario, cfg: TermsConfig, rng: random.Random):
        self.scenario = scenario
        self.cfg = cfg
        self.rng = rng
        # Counterpart role + seller direction s_B (SPEC Section header).
        self.seller = scenario.counterpart_role == "seller"
        self.s_B = 1.0 if self.seller else -1.0
        # Agent concession direction s_A: +1 if AGENT is buyer (conceding = raising price).
        self.s_A = 1.0 if scenario.agent_role == "buyer" else -1.0

    # ------------------------------------------------------------------
    # History features over the agent's own offers (SPEC C.3).
    # ------------------------------------------------------------------
    def _history_features(self, agent_offers: list) -> tuple[float, float, int]:
        """Return (ConcedeMagnitude_k, ConcedeSpeed_k, Rigidity_k).

        Computed from the agent's own chronological offer sequence. The boundary rules in
        SPEC require that with >=2 agent offers the features are non-zero, so the most recent
        consecutive concession (the move into the current offer) is included; the window is
        the last up-to-3 consecutive concessions (J_k = {max(2,k-3)..k}). With <2 offers all
        three are 0.
        """
        a = agent_offers
        n = len(a)
        if n < 2:
            return 0.0, 0.0, 0
        R = self.cfg.R
        # Signed normalized concession for each consecutive pair.
        diffs = [self.s_A * (a[i] - a[i - 1]) / R for i in range(1, n)]
        window = diffs[-3:]
        concede_mag = _mean([max(0.0, d) for d in window])
        concede_speed = _mean(window)
        # Rigidity uses the most recent realized concession; 0 unless it exists.
        rigidity = 1 if max(0.0, diffs[-1]) < self.cfg.tau_rigid else 0
        return concede_mag, concede_speed, rigidity

    # ------------------------------------------------------------------
    # Cue model (SPEC 2e / C.5).
    # ------------------------------------------------------------------
    def _strategy_logits(self, C_B: float, Dtil: float) -> list:
        """Logits over (Concede, Hold, Pressure) for an Offer move (SPEC 2e)."""
        cfg = self.cfg
        eta = self.scenario.eta_B
        if eta == "conciliatory":
            b = (cfg.b_C, 0.0, -cfg.b_C)
        elif eta == "aggressive":
            b = (-cfg.b_P, 0.0, cfg.b_P)
        else:  # neutral
            b = (0.0, cfg.b_H, 0.0)
        l_concede = b[0] + cfg.alpha_C * (C_B - cfg.tau_conc)
        l_hold = b[1]
        l_pressure = b[2] + cfg.alpha_P * (Dtil - cfg.tau_dead) - cfg.beta_C * C_B
        return [l_concede, l_hold, l_pressure]

    def _softmax_sample(self, logits: list, temperature: float = 1.0) -> int:
        """Sample a Categorical(softmax(logits / T)) index using cumulative rng draw."""
        m = max(logits)
        exps = [math.exp((val - m) / temperature) for val in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        u = self.rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if u < cum:
                return i
        return len(probs) - 1

    def _sentiment_base(self, sigma_s: float) -> str:
        """Sample sentiment z = mu(eta) + N(0, sigma_s^2) thresholded by tau_s (SPEC 2e)."""
        cfg = self.cfg
        eta = self.scenario.eta_B
        if eta == "conciliatory":
            mu = cfg.mu_s
        elif eta == "aggressive":
            mu = -cfg.mu_s
        else:
            mu = 0.0
        z = mu + self.rng.gauss(0.0, sigma_s)
        if z > cfg.tau_s:
            return "positive"
        if z < -cfg.tau_s:
            return "negative"
        return "neutral"

    def _cues(self, decision: str, C_B: float, k: int) -> tuple[str, str]:
        """Produce (sentiment, strategy_cue) for a committed decision with the family override.

        Base strategic cue: Accept -> Concede, Reject -> Pressure, Offer -> sampled from logits.
        Family cue_mode overrides (SPEC C.5.3): 'base' (sample), 'uninformative' (neutral/Hold),
        'noisy' (sentiment with sigma_s_stoch, strategy softmax(l/T_stoch)),
        'pressuring' (negative/Pressure).
        """
        cfg = self.cfg
        mode = FAMILIES[self.scenario.family].cue_mode
        Dtil = math.sqrt(k / cfg.K)

        if mode == "uninformative":
            return "neutral", "Hold"
        if mode == "pressuring":
            return "negative", "Pressure"

        if mode == "noisy":
            sentiment = self._sentiment_base(cfg.sigma_s_stoch)
            if decision == "Accept":
                strat = "Concede"
            elif decision == "Reject":
                strat = "Pressure"
            else:
                idx = self._softmax_sample(self._strategy_logits(C_B, Dtil), cfg.T_stoch)
                strat = _STRAT[idx]
            return sentiment, strat

        # mode == "base"
        sentiment = self._sentiment_base(cfg.sigma_s)
        if decision == "Accept":
            strat = "Concede"
        elif decision == "Reject":
            strat = "Pressure"
        else:
            idx = self._softmax_sample(self._strategy_logits(C_B, Dtil), 1.0)
            strat = _STRAT[idx]
        return sentiment, strat

    def _concession_mag(self, new_price: float, prev_price: float) -> float:
        """Counterpart concession magnitude C^B_k for cue logits (SPEC 2e)."""
        r_B = self.scenario.r_counterpart
        denom = abs(prev_price - r_B) + self.cfg.eps_c
        return min(1.0, abs(new_price - prev_price) / denom)

    # ------------------------------------------------------------------
    # Opening offer (SPEC 2d, eqs. 15-16).
    # ------------------------------------------------------------------
    def _opening_move(self, k: int) -> CounterpartMove:
        cfg = self.cfg
        sc = self.scenario
        r_B = sc.r_counterpart
        R = cfg.R
        eta = sc.eta_B

        S_open = (cfg.p_max - r_B) if self.seller else (r_B - cfg.p_min)
        phi = 1.0 - cfg.omega_kappa * sc.kappa_B
        if eta == "aggressive":
            phi += cfg.omega_eta
        if eta == "conciliatory":
            phi -= cfg.omega_eta_prime
        phi = _clamp(phi, cfg.phi_min, cfg.phi_max)

        sigma0 = cfg.sigma0_bar * R
        eps0 = self.rng.gauss(0.0, sigma0)
        raw = r_B + self.s_B * sc.d0e * phi * S_open + eps0
        if self.seller:
            price = _project(raw, r_B, cfg.p_max)
        else:
            price = _project(raw, cfg.p_min, r_B)
        price = _clamp(price, cfg.p_min, cfg.p_max)

        sentiment, strat = self._cues("Offer", 0.0, k)
        return CounterpartMove("Offer", price, sentiment, strat)

    def opening_offer(self) -> CounterpartMove:
        """Counterpart's first offer when opener == 'CounterpartOpens' (round k=1)."""
        return self._opening_move(1)

    # ------------------------------------------------------------------
    # Response model (SPEC 2a -> 2b -> 2c).
    # ------------------------------------------------------------------
    def respond(
        self,
        agent_offer_price: float,
        k: int,
        agent_offers: list,
        counterpart_offers: list,
    ) -> CounterpartMove:
        cfg = self.cfg
        sc = self.scenario
        R = cfg.R
        r_B = sc.r_counterpart

        # Role-normalized favorability Delta_k (SPEC 2a).
        if self.seller:
            Delta = (agent_offer_price - r_B) / R
        else:
            Delta = (r_B - agent_offer_price) / R

        Dtil = math.sqrt(k / cfg.K)
        Dbar = 1.0 - Dtil
        concede_mag, concede_speed, rigidity = self._history_features(agent_offers)

        stance = STANCE_IDX[sc.eta_B]
        fam = FAMILIES[sc.family]

        # 2a. Acceptance.
        rho = fam.rho[stance]
        xi = fam.xi[stance]
        g = (
            cfg.alpha * Delta
            + cfg.beta * sc.kappa_B
            - cfg.gamma * Dbar
            + rho * concede_speed
            + xi * rigidity
        )
        a_k = _sigmoid(g) if Delta >= 0.0 else 0.0
        if self.rng.random() < a_k:
            sentiment, strat = self._cues("Accept", 0.0, k)
            return CounterpartMove("Accept", None, sentiment, strat)

        # 2b. Walk-away (terminal Reject), only if not accepted.
        denom = cfg.K - cfg.k_walk
        tauW = _clamp((k - cfg.k_walk) / denom, 0.0, 1.0) if denom > 0 else 1.0
        if k >= cfg.k_walk and Delta < 0.0:
            omega = _sigmoid(cfg.phi0 + cfg.phi_delta * max(0.0, -Delta) + cfg.phi_T * tauW)
        else:
            omega = 0.0
        if self.rng.random() < omega:
            sentiment, strat = self._cues("Reject", 0.0, k)
            return CounterpartMove("Reject", None, sentiment, strat)

        # 2c. Counter-offer. If the counterpart has not offered yet, use the opening model.
        if not counterpart_offers:
            return self._opening_move(k)

        prev = counterpart_offers[-1]
        lam2 = fam.lambda2[stance]
        lam = cfg.lambda0 + cfg.lambda1 * sc.kappa_B - lam2 * concede_mag
        if sc.eta_B == "aggressive":
            lam -= cfg.lambda3
        if sc.eta_B == "conciliatory":
            lam += cfg.lambda4
        lam = _clamp(lam, 0.0, 1.0)

        sigma_p = fam.price_noise * R
        eps = self.rng.gauss(0.0, sigma_p)
        p_target = prev + lam * (r_B - prev) + eps

        # Project onto the monotone feasible interval M_B(k); projecting onto [r_B, prev]
        # (seller) / [prev, r_B] (buyer) also holds at prev when noise reverses concession.
        if self.seller:
            price = _project(p_target, r_B, prev)
        else:
            price = _project(p_target, prev, r_B)
        price = _clamp(price, cfg.p_min, cfg.p_max)

        C_B = self._concession_mag(price, prev)
        sentiment, strat = self._cues("Offer", C_B, k)
        return CounterpartMove("Offer", price, sentiment, strat)


# ----------------------------------------------------------------------------------
# Smoke test: python3 kernel.py  (run from the terms/ dir so siblings import flat).
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = DEFAULT_CONFIG  # p_min=0, p_max=100, K=10
    scenario = Scenario(
        episode_id="smoke",
        regime="overlap",
        family="candid",
        agent_role="buyer",
        counterpart_role="seller",
        opener="CounterpartOpens",
        p_min=0.0,
        p_max=100.0,
        r_agent=60.0,
        r_counterpart=40.0,
        kappa_B=0.5,
        eta_B="neutral",
        d0e=0.5,
        seed=123,
    )
    rng = random.Random(42)
    cp = Counterpart(scenario, cfg, rng)

    VALID_SENT = {"positive", "neutral", "negative"}
    VALID_STRAT = {"Concede", "Hold", "Pressure"}
    r_B = scenario.r_counterpart

    def _check_cues(mv):
        assert mv.sentiment in VALID_SENT, f"bad sentiment: {mv.sentiment!r}"
        assert mv.strategy_cue in VALID_STRAT, f"bad strategy_cue: {mv.strategy_cue!r}"

    transcript = []
    counterpart_offers: list = []
    agent_offers: list = []

    # --- Opening offer (CounterpartOpens, k=1) ---
    opening = cp.opening_offer()
    assert opening.decision == "Offer" and opening.price is not None
    assert 0.0 <= opening.price <= 100.0, f"opening price out of bounds: {opening.price}"
    assert opening.price >= r_B - 1e-9, f"opening violates seller IR (>= r_B): {opening.price}"
    _check_cues(opening)
    counterpart_offers.append(opening.price)
    prev_cp = opening.price
    transcript.append(("CP opens", "Offer", round(opening.price, 2), opening.sentiment, opening.strategy_cue))

    # --- Scripted agent (buyer) conceding upward from 30 ---
    agent_script = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
    for i, agent_price in enumerate(agent_script):
        k = i + 2  # counterpart opened at k=1
        agent_offers.append(agent_price)
        mv = cp.respond(agent_price, k, list(agent_offers), list(counterpart_offers))
        _check_cues(mv)

        if mv.decision == "Accept":
            assert agent_price >= r_B - 1e-9, f"Accept below seller IR: {agent_price} < {r_B}"
            transcript.append((f"A offers {agent_price:.0f}", "Accept", None, mv.sentiment, mv.strategy_cue))
            break
        if mv.decision == "Reject":
            transcript.append((f"A offers {agent_price:.0f}", "Reject(walk)", None, mv.sentiment, mv.strategy_cue))
            break

        assert mv.decision == "Offer" and mv.price is not None
        assert 0.0 <= mv.price <= 100.0, f"counter price out of bounds: {mv.price}"
        assert mv.price >= r_B - 1e-9, f"counter violates seller IR (>= r_B): {mv.price}"
        assert mv.price <= prev_cp + 1e-9, f"seller offer not non-increasing: {mv.price} > {prev_cp}"
        counterpart_offers.append(mv.price)
        prev_cp = mv.price
        transcript.append((f"A offers {agent_price:.0f}", "Offer", round(mv.price, 2), mv.sentiment, mv.strategy_cue))

    print("--- TERMS-Bench kernel smoke test (seller counterpart, candid, neutral) ---")
    print(f"r_B (seller reservation) = {r_B}")
    for row in transcript:
        who, dec, price, sent, strat = row
        price_str = "  -  " if price is None else f"{price:>6}"
        print(f"  {who:<14} | {dec:<12} | price={price_str} | sentiment={sent:<8} | cue={strat}")
    print("SMOKE TEST PASSED")
