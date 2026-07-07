"""TERMS-Bench regime / scenario generator (SPEC Section 1; arXiv:2605.13909, S3.1).

Faithful reconstruction of the bilateral price-negotiation scenario sampler. Every
stochastic draw goes through a per-stream `random.Random` instance (Python's Mersenne
Twister is the documented RNG for this harness; SPEC S5), seeded so that a fixed
`base_seed` always reproduces the same ordered scenario set across evaluated models.

Stand-alone module: the `terms/` directory is on `sys.path`, so config symbols are
imported flat.

Seeding (SPEC S1, H.1.3). For each cell indexed by family `f_idx`, agent-role `r_idx`,
opener `o_idx`, and episode index `e`:

    cell = base_seed*10_000_000 + f_idx*100_000 + r_idx*10_000 + o_idx*1_000 + e*10

Note the cell base intentionally does NOT encode the regime: overlap / urgency_shift /
no_deal siblings that share (family, role, opener, episode) reuse the SAME geometry
percentile and so differ only by the sign of `delta` (and, for urgency_shift, the
urgency Beta). Disjoint latent streams `random.Random(cell + i)` are assigned as:

    i = 1  ->  kappa_B   (urgency; baseline or shifted Beta depending on regime)
    i = 2  ->  u_e       (shared geometry percentile in U(0,1))
    i = 3  ->  eta_B     (stance, via family stance_prior cumulative)
    i = 4  ->  d0e       (opening harshness ~ U(d0_min, d0_max))
    i = 5  ->  m         (midpoint, sampled so reservations stay in [p_min, p_max])

This assignment is fixed and part of the harness contract.
"""

from __future__ import annotations

import random

from config import (
    DEFAULT_CONFIG,
    FAMILIES,
    FAMILIES_ORDER,
    REGIMES,
    STANCES,
    Scenario,
    TermsConfig,
)

# Fixed cell axes. Index positions feed the seeding scheme, so the order is part of
# the harness contract and must not change.
ROLES: tuple[str, str] = ("buyer", "seller")
OPENERS: tuple[str, str] = ("AgentOpens", "CounterpartOpens")

# Disjoint stream offsets added to `cell` (documented above).
_STREAM_KAPPA = 1
_STREAM_GEOM = 2
_STREAM_STANCE = 3
_STREAM_D0 = 4
_STREAM_MIDPOINT = 5

N_PER_CELL_FULL_SUITE = 25  # SPEC H.1.2: reproduces 100 episodes per (regime, family).


def _cell_seed(base_seed: int, f_idx: int, r_idx: int, o_idx: int, e: int) -> int:
    return base_seed * 10_000_000 + f_idx * 100_000 + r_idx * 10_000 + o_idx * 1_000 + e * 10


def _draw_stance(rng: random.Random, prior: tuple[float, float, float]) -> str:
    """Sample a stance from `prior` over STANCES via cumulative inverse-CDF."""
    u = rng.random()
    acc = 0.0
    for idx, p in enumerate(prior):
        acc += p
        if u < acc:
            return STANCES[idx]
    return STANCES[-1]


def _build_scenario(
    regime: str,
    family: str,
    agent_role: str,
    opener: str,
    e: int,
    base_seed: int,
    cfg: TermsConfig,
) -> Scenario:
    f_idx = FAMILIES_ORDER.index(family)
    r_idx = ROLES.index(agent_role)
    o_idx = OPENERS.index(opener)
    cell = _cell_seed(base_seed, f_idx, r_idx, o_idx, e)

    # --- Urgency kappa_B (stream 1): baseline Beta, or shifted for urgency_shift. ---
    rng_kappa = random.Random(cell + _STREAM_KAPPA)
    if regime == "urgency_shift":
        kappa_B = rng_kappa.betavariate(cfg.urgency_shift_alpha, cfg.urgency_shift_beta)
    else:
        kappa_B = rng_kappa.betavariate(cfg.urgency_alpha, cfg.urgency_beta)

    # --- Shared geometry percentile u_e (stream 2). ---
    rng_geom = random.Random(cell + _STREAM_GEOM)
    u_e = rng_geom.random()
    z_e = cfg.zopa_min + u_e * (cfg.zopa_max - cfg.zopa_min)
    q_e = cfg.gap_min + u_e * (cfg.gap_max - cfg.gap_min)

    # --- Midpoint m (stream 5), sampled to keep reservations within [p_min, p_max]. ---
    rng_mid = random.Random(cell + _STREAM_MIDPOINT)
    if regime == "no_deal":
        half = cfg.gap_max / 2.0
        m = rng_mid.uniform(cfg.p_min + half, cfg.p_max - half)
        r_buyer = m - q_e / 2.0
        r_seller = m + q_e / 2.0
    else:  # overlap and urgency_shift share the positive-ZOPA geometry.
        max_half = cfg.zopa_max / 2.0
        m = rng_mid.uniform(cfg.p_min + max_half, cfg.p_max - max_half)
        r_buyer = m + z_e / 2.0
        r_seller = m - z_e / 2.0

    # --- Stance eta_B (stream 3) from the family prior. ---
    rng_stance = random.Random(cell + _STREAM_STANCE)
    eta_B = _draw_stance(rng_stance, FAMILIES[family].stance_prior)

    # --- Opening harshness d0e (stream 4). ---
    rng_d0 = random.Random(cell + _STREAM_D0)
    d0e = rng_d0.uniform(cfg.d0_min, cfg.d0_max)

    if agent_role == "buyer":
        r_agent, r_counterpart = r_buyer, r_seller
        counterpart_role = "seller"
    else:
        r_agent, r_counterpart = r_seller, r_buyer
        counterpart_role = "buyer"

    episode_id = f"{regime}-{family}-{agent_role}-{opener}-{e:04d}"

    return Scenario(
        episode_id=episode_id,
        regime=regime,
        family=family,
        agent_role=agent_role,
        counterpart_role=counterpart_role,
        opener=opener,
        p_min=cfg.p_min,
        p_max=cfg.p_max,
        r_agent=r_agent,
        r_counterpart=r_counterpart,
        kappa_B=kappa_B,
        eta_B=eta_B,
        d0e=d0e,
        seed=cell,
    )


def _cell_axes(regimes: tuple[str, ...], families: tuple[str, ...]) -> list[tuple[str, str, str, str]]:
    """Enumerate cells in deterministic generation order (used for remainder spread)."""
    cells: list[tuple[str, str, str, str]] = []
    for regime in regimes:
        for family in families:
            for agent_role in ROLES:
                for opener in OPENERS:
                    cells.append((regime, family, agent_role, opener))
    return cells


def _sort_key(s: Scenario) -> tuple[int, int, int, int, int]:
    """Canonical ordering: regime, family, agent_role, opener, episode_index."""
    regime_idx = REGIMES.index(s.regime) if s.regime in REGIMES else len(REGIMES)
    family_idx = FAMILIES_ORDER.index(s.family)
    role_idx = ROLES.index(s.agent_role)
    opener_idx = OPENERS.index(s.opener)
    e = int(s.episode_id.rsplit("-", 1)[1])
    return (regime_idx, family_idx, role_idx, opener_idx, e)


def generate_scenarios(
    n: int,
    base_seed: int = 0,
    cfg: TermsConfig = DEFAULT_CONFIG,
    families: tuple[str, ...] = FAMILIES_ORDER,
    regimes: tuple[str, ...] = REGIMES,
    full_suite: bool = False,
) -> list[Scenario]:
    """Generate a deterministic, balanced set of negotiation scenarios.

    Cells are the cross product
        regime x family x agent_role x opener
    over the requested `regimes` and `families` (roles and openers are fixed).

    If `full_suite` is True, every cell gets exactly `N_PER_CELL_FULL_SUITE` (25)
    episodes and `n` is ignored (1,800 total for the default 3 regimes x 6 families).
    Otherwise `n` total episodes are spread as evenly as possible: each cell gets
    `n // num_cells`, and the remaining `n % num_cells` episodes go one each to the
    first cells in deterministic generation order. When `n < num_cells`, the first
    `n` cells get a single episode and the rest get none.

    The returned list is sorted by (regime, family, agent_role, opener, episode_index)
    so a fixed `base_seed` always yields the same ordered set.
    """
    cells = _cell_axes(regimes, families)
    num_cells = len(cells)

    if full_suite:
        counts = [N_PER_CELL_FULL_SUITE] * num_cells
    elif num_cells == 0 or n <= 0:
        counts = [0] * num_cells
    else:
        base = n // num_cells
        remainder = n % num_cells
        counts = [base + (1 if i < remainder else 0) for i in range(num_cells)]

    scenarios: list[Scenario] = []
    for (regime, family, agent_role, opener), count in zip(cells, counts):
        for e in range(count):
            scenarios.append(_build_scenario(regime, family, agent_role, opener, e, base_seed, cfg))

    scenarios.sort(key=_sort_key)
    return scenarios


if __name__ == "__main__":
    cfg = DEFAULT_CONFIG

    # --- Full suite: 25 * 3 regimes * 6 families * 2 roles * 2 openers = 1800. ---
    suite = generate_scenarios(0, full_suite=True)
    expected = N_PER_CELL_FULL_SUITE * 3 * 6 * 2 * 2
    assert expected == 1800, expected
    assert len(suite) == 1800, len(suite)

    # --- Regime sign + bound + range invariants. ---
    for s in suite:
        if s.regime in ("overlap", "urgency_shift"):
            assert s.delta > 0, (s.episode_id, s.delta)
        elif s.regime == "no_deal":
            assert s.delta < 0, (s.episode_id, s.delta)
        else:
            raise AssertionError(f"unexpected regime {s.regime}")
        assert cfg.p_min <= s.r_buyer <= cfg.p_max, (s.episode_id, s.r_buyer)
        assert cfg.p_min <= s.r_seller <= cfg.p_max, (s.episode_id, s.r_seller)
        assert cfg.p_min <= s.r_agent <= cfg.p_max, (s.episode_id, s.r_agent)
        assert cfg.p_min <= s.r_counterpart <= cfg.p_max, (s.episode_id, s.r_counterpart)
        assert 0.0 <= s.kappa_B <= 1.0, (s.episode_id, s.kappa_B)
        assert s.eta_B in STANCES, (s.episode_id, s.eta_B)
        assert s.counterpart_role != s.agent_role, s.episode_id

    # --- Balanced spread: n == num_cells => exactly 1 per cell. ---
    num_cells = 3 * 6 * 2 * 2  # = 72
    assert num_cells == 72, num_cells
    balanced = generate_scenarios(num_cells, base_seed=7)
    assert len(balanced) == num_cells, len(balanced)
    seen: dict[tuple[str, str, str, str], int] = {}
    for s in balanced:
        key = (s.regime, s.family, s.agent_role, s.opener)
        seen[key] = seen.get(key, 0) + 1
    assert len(seen) == num_cells, len(seen)
    assert all(v == 1 for v in seen.values()), seen

    # --- Determinism: same base_seed => identical ordered scenario set. ---
    again = generate_scenarios(num_cells, base_seed=7)
    assert len(again) == len(balanced)
    for a, b in zip(balanced, again):
        assert (
            a.episode_id == b.episode_id
            and a.seed == b.seed
            and a.r_agent == b.r_agent
            and a.r_counterpart == b.r_counterpart
            and a.kappa_B == b.kappa_B
            and a.eta_B == b.eta_B
            and a.d0e == b.d0e
        ), a.episode_id

    # --- Summary table of counts by (regime, family) for the full suite. ---
    by_rf: dict[tuple[str, str], int] = {}
    for s in suite:
        by_rf[(s.regime, s.family)] = by_rf.get((s.regime, s.family), 0) + 1
    print(f"Full suite: {len(suite)} episodes  (expected {expected})")
    print(f"{'regime':<14}{'family':<13}{'count':>6}")
    print("-" * 33)
    for regime in REGIMES:
        for family in FAMILIES_ORDER:
            print(f"{regime:<14}{family:<13}{by_rf[(regime, family)]:>6}")
    print(f"\nBalanced n={num_cells}: {len(balanced)} episodes, exactly 1 per cell.")
    print("Determinism check: identical re-generation. All assertions passed.")
