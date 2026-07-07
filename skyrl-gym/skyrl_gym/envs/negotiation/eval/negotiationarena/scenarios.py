"""NegotiationArena scenario generator (SPEC §1; arXiv:2402.05863).

Faithful reconstruction of the game-instance sampler.  Every stochastic draw
goes through a per-stream ``random.Random`` instance (Python's Mersenne Twister
is the documented RNG for this harness; SPEC §5), seeded so that a fixed
``base_seed`` always reproduces the same ordered scenario set across evaluated
models.

Stand-alone module: the ``negotiationarena/`` directory is on ``sys.path``, so
config symbols are imported flat.

Seeding (SPEC §1).  For each cell indexed by game ``g_idx``, focal seat
``seat_idx``, and episode index ``e``:

    cell = base_seed*10_000_000 + g_idx*100_000 + seat_idx*10_000 + e*10

Disjoint latent streams ``random.Random(cell + i)`` are assigned as:

    i = 1  ->  seller_cost        (sell_buy only)
    i = 2  ->  buyer_willingness  (sell_buy only)
    i = 3  ->  amount_to_split    (ultimatum only; default unless cfg.vary_amount)

This assignment is fixed and part of the harness contract.
"""

from __future__ import annotations

import random

from config import (
    DEFAULT_CONFIG,
    GAME_PRESETS,
    GAMES_ORDER,
    N_PER_CELL_FULL_SUITE,
    NegArenaConfig,
    RESOURCE_EXCHANGE_ENDOWMENTS,
    ROLE_LABELS,
    SELL_BUY_COST_RANGE,
    SELL_BUY_WILLINGNESS_RANGE,
    Scenario,
)

# Disjoint stream offsets added to `cell` (documented above).
_STREAM_COST = 1
_STREAM_WILLINGNESS = 2
_STREAM_AMOUNT = 3

# Numerosity-probe amounts drawn when cfg.vary_amount is True (§5.2).
_AMOUNT_POOL: tuple[int, ...] = (100, 1000, 10_000, 100_000)


def _cell_seed(base_seed: int, g_idx: int, seat_idx: int, e: int) -> int:
    return base_seed * 10_000_000 + g_idx * 100_000 + seat_idx * 10_000 + e * 10


def _build_scenario(
    game: str,
    focal_seat: int,
    e: int,
    base_seed: int,
    cfg: NegArenaConfig,
    games: tuple,
) -> Scenario:
    g_idx = list(games).index(game)
    cell = _cell_seed(base_seed, g_idx, focal_seat, e)

    preset = GAME_PRESETS[game]
    role_label = ROLE_LABELS[game][focal_seat]
    episode_id = f"{game}-{role_label}-{e:04d}"

    if game == "resource_exchange":
        initial_resources: tuple[dict, dict] = (
            dict(RESOURCE_EXCHANGE_ENDOWMENTS[0]),
            dict(RESOURCE_EXCHANGE_ENDOWMENTS[1]),
        )
        money_token = None
        resource_tokens: tuple[str, ...] = ("X", "Y")
        first_mover = 0
        amount_to_split = None
        seller_cost = None
        buyer_willingness = None

    elif game == "ultimatum":
        rng_amount = random.Random(cell + _STREAM_AMOUNT)
        if cfg.vary_amount:
            amount = rng_amount.choice(_AMOUNT_POOL)
        else:
            amount = cfg.ultimatum_amount
        initial_resources = (
            {"Dollars": amount},
            {"Dollars": 0},
        )
        money_token = "Dollars"
        resource_tokens = ("Dollars",)
        first_mover = 0
        amount_to_split = amount
        seller_cost = None
        buyer_willingness = None

    elif game == "sell_buy":
        if cfg.vary_sell_buy:
            seller_cost = random.Random(cell + _STREAM_COST).randint(SELL_BUY_COST_RANGE[0], SELL_BUY_COST_RANGE[1])
            buyer_willingness = random.Random(cell + _STREAM_WILLINGNESS).randint(
                SELL_BUY_WILLINGNESS_RANGE[0], SELL_BUY_WILLINGNESS_RANGE[1]
            )
        else:
            seller_cost = cfg.sell_buy_cost
            buyer_willingness = cfg.sell_buy_willingness
        initial_resources = (
            {"X": 1},
            {"ZUP": cfg.buyer_budget},
        )
        money_token = "ZUP"
        resource_tokens = ("X", "ZUP")
        first_mover = 0
        amount_to_split = None

    else:
        raise ValueError(f"Unknown game: {game!r}")

    return Scenario(
        episode_id=episode_id,
        game=game,
        focal_seat=focal_seat,
        first_mover=first_mover,
        initial_resources=initial_resources,
        money_token=money_token,
        resource_tokens=resource_tokens,
        max_turns=preset.max_turns,
        number_of_proposals=preset.number_of_proposals,
        seed=cell,
        amount_to_split=amount_to_split,
        seller_cost=seller_cost,
        buyer_willingness=buyer_willingness,
        social_behaviour=("", ""),
    )


def _cell_axes(games: tuple) -> list[tuple[str, int]]:
    """Enumerate cells in deterministic generation order (used for remainder spread)."""
    cells: list[tuple[str, int]] = []
    for game in games:
        for seat in (0, 1):
            cells.append((game, seat))
    return cells


def _sort_key(s: Scenario, games: tuple) -> tuple[int, int, int]:
    """Canonical ordering: game_idx, focal_seat, episode_index."""
    g_idx = list(games).index(s.game)
    e = int(s.episode_id.rsplit("-", 1)[1])
    return (g_idx, s.focal_seat, e)


def generate_scenarios(
    n: int,
    base_seed: int = 0,
    cfg: NegArenaConfig = DEFAULT_CONFIG,
    games: tuple = GAMES_ORDER,
    full_suite: bool = False,
) -> list[Scenario]:
    """Generate a deterministic, balanced set of NegotiationArena scenarios.

    Cells are the cross product ``game × focal_seat`` (focal_seat ∈ {0, 1}) over
    the requested ``games``.

    If ``full_suite`` is True, every cell gets exactly ``N_PER_CELL_FULL_SUITE``
    episodes and ``n`` is ignored.  Otherwise ``n`` total episodes are spread as
    evenly as possible: each cell gets ``n // num_cells``, and the remaining
    ``n % num_cells`` episodes go one each to the first cells in deterministic
    generation order.  When ``n < num_cells``, the first ``n`` cells get a single
    episode and the rest get none.

    The returned list is sorted by ``(game_idx, focal_seat, episode_index)`` so a
    fixed ``base_seed`` always yields the same ordered set.
    """
    cells = _cell_axes(games)
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
    for (game, focal_seat), count in zip(cells, counts):
        for e in range(count):
            scenarios.append(_build_scenario(game, focal_seat, e, base_seed, cfg, games))

    scenarios.sort(key=lambda s: _sort_key(s, games))
    return scenarios


if __name__ == "__main__":
    # --- Full suite: N_PER_CELL_FULL_SUITE * len(GAMES_ORDER) * 2 total. ---
    suite = generate_scenarios(0, full_suite=True)
    expected_full = N_PER_CELL_FULL_SUITE * len(GAMES_ORDER) * 2
    assert len(suite) == expected_full, f"full_suite count mismatch: got {len(suite)}, expected {expected_full}"

    # --- Balanced spread: n == num_cells => exactly 1 per cell. ---
    num_cells = len(GAMES_ORDER) * 2
    balanced = generate_scenarios(num_cells, base_seed=42)
    assert len(balanced) == num_cells, f"balanced count mismatch: got {len(balanced)}, expected {num_cells}"
    seen: dict[tuple[str, int], int] = {}
    for s in balanced:
        key = (s.game, s.focal_seat)
        seen[key] = seen.get(key, 0) + 1
    assert len(seen) == num_cells, f"cell coverage: {len(seen)} != {num_cells}"
    assert all(v == 1 for v in seen.values()), f"unbalanced cells: {seen}"

    # --- Determinism: same base_seed => identical ordered scenario set. ---
    again = generate_scenarios(num_cells, base_seed=42)
    assert len(again) == len(balanced)
    for a, b in zip(balanced, again):
        assert a.episode_id == b.episode_id, f"episode_id mismatch: {a.episode_id!r} vs {b.episode_id!r}"
        assert a.seed == b.seed, f"seed mismatch: {a.episode_id}"
        assert a.seller_cost == b.seller_cost, f"seller_cost mismatch: {a.episode_id}"
        assert a.buyer_willingness == b.buyer_willingness, f"buyer_willingness mismatch: {a.episode_id}"
        assert a.amount_to_split == b.amount_to_split, f"amount_to_split mismatch: {a.episode_id}"

    # --- sell_buy: integer valuations within declared ranges (vary_sell_buy=True). ---
    sell_buy_scenarios = [s for s in suite if s.game == "sell_buy"]
    for s in sell_buy_scenarios:
        assert isinstance(s.seller_cost, int), f"seller_cost not int: {s.episode_id}"
        assert (
            SELL_BUY_COST_RANGE[0] <= s.seller_cost <= SELL_BUY_COST_RANGE[1]
        ), f"seller_cost={s.seller_cost} out of {SELL_BUY_COST_RANGE}: {s.episode_id}"
        assert isinstance(s.buyer_willingness, int), f"buyer_willingness not int: {s.episode_id}"
        assert SELL_BUY_WILLINGNESS_RANGE[0] <= s.buyer_willingness <= SELL_BUY_WILLINGNESS_RANGE[1], (
            f"buyer_willingness={s.buyer_willingness} out of " f"{SELL_BUY_WILLINGNESS_RANGE}: {s.episode_id}"
        )

    # --- ultimatum: proposer (seat 0) holds the full pot; responder (seat 1) holds 0. ---
    ult_scenarios = [s for s in suite if s.game == "ultimatum"]
    for s in ult_scenarios:
        assert s.amount_to_split is not None, f"amount_to_split is None: {s.episode_id}"
        assert s.initial_resources[0] == {"Dollars": s.amount_to_split}, (
            f"proposer resources wrong: {s.episode_id}, " f"got {s.initial_resources[0]}"
        )
        assert s.initial_resources[1] == {"Dollars": 0}, (
            f"responder resources wrong: {s.episode_id}, " f"got {s.initial_resources[1]}"
        )

    # --- Counts-by-(game, focal_role) table. ---
    by_gr: dict[tuple[str, str], int] = {}
    for s in suite:
        key = (s.game, s.focal_role)
        by_gr[key] = by_gr.get(key, 0) + 1

    print(f"Full suite: {len(suite)} episodes  (expected {expected_full})")
    print(f"\n{'game':<20}{'focal_role':<14}{'count':>6}")
    print("-" * 40)
    for game in GAMES_ORDER:
        for seat in (0, 1):
            role = ROLE_LABELS[game][seat]
            count = by_gr.get((game, role), 0)
            print(f"{game:<20}{role:<14}{count:>6}")

    print(f"\nBalanced n={num_cells}: {len(balanced)} episodes, exactly 1 per cell.")
    print("Determinism check: identical re-generation. All assertions passed.")
    print("scenarios.py smoke test passed.")
