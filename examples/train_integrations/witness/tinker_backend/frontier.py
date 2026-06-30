"""FrontierBuffer — per-game store of high-state_value snapshots for Go-Explore seeding.

Goal-part-3 (oversight → dense GRPO signal): on hard games a whole rollout group can be
~0 reward → zero GRPO advantage. We harvest promising mid-game states (partial progress
toward a win) from the policy's own rollouts, then restart a fraction of each group's
rollouts from one of those states so the policy (sampling ON-POLICY from a better start)
is more likely to complete a win → the group gets reward variance → real gradient.

See reports/2026-06-23_oversight_dense_signal_plan.md (mechanism (c)). Sparse by design:
state_value is called only to *score restart candidates*, not as a dense per-step reward.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class FrontierBuffer:
    def __init__(self, oracle, max_per_game: int = 8, sv_rollouts: int = 16,
                 min_success: float = 0.05, max_success: float = 0.95):
        """oracle: a WitnessOracle. A candidate snapshot is kept only if state_value is
        `informative` and success_rate in (min_success, max_success) — i.e. genuine *partial*
        progress: not already-won (no signal to add) and not hopeless (policy can't finish)."""
        self.oracle = oracle
        self.max_per_game = max_per_game
        self.sv_rollouts = sv_rollouts
        self.min_success = min_success
        self.max_success = max_success
        self._buf: Dict[str, List[Tuple[float, Any]]] = defaultdict(list)  # game_id -> [(score, snap)]
        self.n_harvested = 0
        self.n_scored = 0
        self.n_preseeded = 0

    def harvest(self, game_id: str, snap: Any) -> bool:
        """Score a candidate snapshot; keep if it's an informative partial-progress state.
        Returns True if kept. Catches oracle errors (non-informative games) and skips."""
        try:
            self.n_scored += 1
            ve = self.oracle.state_value(snap, rollouts=self.sv_rollouts)
        except Exception:
            return False
        if not getattr(ve, "informative", False):
            return False
        sr = float(getattr(ve, "success_rate", 0.0))
        if not (self.min_success < sr < self.max_success):
            return False
        b = self._buf[game_id]
        b.append((sr, snap))
        b.sort(key=lambda x: -x[0])                 # keep the most-winnable restart states
        del b[self.max_per_game:]
        self.n_harvested += 1
        return True

    def add(self, game_id: str, snap: Any, score: float) -> None:
        """Add a snapshot with an explicit score, BYPASSING the random-rollout state_value
        filter. Used by oracle pre-seeding, where the state is KNOWN to be on a winning path
        (the random filter would wrongly reject it — see the 2026-06-24 harvest probe). Same
        per-game cap + score-sort as `harvest`."""
        b = self._buf[game_id]
        b.append((float(score), snap))
        b.sort(key=lambda x: -x[0])
        del b[self.max_per_game:]

    def preseed_from_oracle(self, game_ids: List[str], levels: int = 3,
                            fractions: Tuple[float, ...] = (0.5, 0.7, 0.9),
                            seed: int = 0) -> int:
        """Option D: populate the frontier with guaranteed-on-winning-path mid-solution states.

        For each game, replay the oracle's optimal `solution_actions` to complete levels 0..L-1,
        and at level L snapshot the state after the first `frac` of that level's PATH moves (before
        the terminal CONFIRM). These states are on a known winning path, so we add them via `add`
        (no random-rollout filter — the filter is blind to witness puzzles, 2026-06-24 probe). The
        policy then samples its OWN on-policy continuation from the restored state (start closer to
        a win → more group members complete → real intra-group outcome variance). Returns # added."""
        from run_agent import load_game                       # CWD-safe loader (matches the bridge)
        from oversight import snapshot as _snap               # arc-witness-envs
        from arcengine import ActionInput, GameAction

        def _do(game, action_id):
            game.perform_action(ActionInput(id=GameAction.from_id(int(action_id))), raw=True)

        added = 0
        for g in game_ids:
            for L in range(levels):
                try:
                    sol, _status = self.oracle.solution(g, L)
                except IndexError:
                    break                                     # game has < L+1 levels
                if not sol:
                    break                                     # no solution at L → can't reach deeper
                path = sol[:-1]                               # drop terminal CONFIRM → path moves only
                if not path:
                    continue                                  # degenerate (CONFIRM-only) level → no mid-state
                try:
                    game = load_game(g, seed)
                    for prev in range(L):                     # advance through prior levels (full solutions)
                        psol, _ = self.oracle.solution(g, prev)
                        if not psol:
                            raise RuntimeError("solution gap below L")
                        for a in psol:
                            _do(game, a)
                    # snapshot at cumulative fractions of level L's path (mid-level, pre-CONFIRM)
                    ks = sorted({max(1, round(f * len(path))) for f in fractions})
                    done = 0
                    for k in ks:
                        for a in path[done:k]:
                            _do(game, a)
                        done = k
                        self.add(g, _snap(game), score=float(k) / len(path))
                        added += 1
                except Exception:
                    continue
        self.n_preseeded += added
        return added

    def sample(self, game_id: str, rng: Optional[random.Random] = None) -> Optional[Any]:
        """Sample a restart snapshot for a seeded rollout (None if the buffer is empty)."""
        b = self._buf.get(game_id)
        if not b:
            return None
        r = rng or random
        return r.choice([s for _, s in b])

    def has(self, game_id: str) -> bool:
        return bool(self._buf.get(game_id))

    def stats(self) -> dict:
        return {"games": len([g for g, b in self._buf.items() if b]),
                "snapshots": sum(len(b) for b in self._buf.values()),
                "scored": self.n_scored, "kept": self.n_harvested,
                "preseeded": self.n_preseeded}
