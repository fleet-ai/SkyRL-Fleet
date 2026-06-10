"""
PrimitivBench task PB-pilot-016: Polyomino tiling with rule-composition + exemption.

Game: 6x6 grid, 8 polyomino pieces, two simultaneous rules:
  R1 (rotation): a piece may not be rotated more than 90 degrees from its base
                 orientation. Allowed rotations are 0 and 90 only.
  R2 (color):    adjacent (edge-sharing) cells belonging to different pieces
                 must have COMPATIBLE colors per the declared relation.

Meta-rule: the player may declare EXACTLY ONE piece exempt from R1 OR R2
(not both). Without exemption, the puzzle is unsolvable.

Win condition: every cell of the 6x6 grid is filled; all placements legal
under R1 and R2 except for the one declared exemption (if any); exemption,
if used, is documented in state.
"""

import numpy as np
from copy import deepcopy

GRID_H = 6
GRID_W = 6

# Color compatibility: 4-cycle R-G-B-Y-R; diagonals (R,B) and (G,Y) incompatible.
COLORS = ["R", "G", "B", "Y"]
COMPAT = {
    ("R", "R"): True, ("G", "G"): True, ("B", "B"): True, ("Y", "Y"): True,
    ("R", "G"): True, ("G", "R"): True,
    ("G", "B"): True, ("B", "G"): True,
    ("B", "Y"): True, ("Y", "B"): True,
    ("Y", "R"): True, ("R", "Y"): True,
    ("R", "B"): False, ("B", "R"): False,
    ("G", "Y"): False, ("Y", "G"): False,
}

# Piece definitions: (id, base_cells (list of (dr,dc)), color).
# Base orientation is rotation=0. We allow rotation in {0, 90}.
# Cells listed relative to an anchor (top-left of bounding box at placement time
# uses the cells as-is offset by anchor).
#
# Designed solution (anchor = (row, col), rotation):
#   P1: I-pentomino vertical, color R, anchor (0,0), rot 0   -> cells (0..4,0)
#   P2: L-pentomino,         color G, anchor (0,1), rot 0
#   P3: P-pentomino,         color B, anchor (0,3), rot 0
#   P4: I-tetromino vert,    color Y, anchor (0,5), rot 0
#   P5: L-tetromino,         color R, anchor (4,1), rot 180 *** needs R1 exemption
#   P6: square tetromino,    color G, anchor (4,3), rot 0
#   P7: I-pentomino horiz,   color Y, anchor (5,0), rot 90 (but here piece 1 already at col 0...)
#
# To keep things tractable and verifiable, we hand-craft a tiling explicitly.
# Below we encode a concrete solved layout, then define each piece's base shape
# such that the documented anchor+rotation reproduces that layout.

# We design the solution grid first (piece id per cell), then derive piece shapes.
# Solution grid (1-indexed piece ids):
SOLUTION_GRID = [
    [1, 2, 2, 3, 3, 4],
    [1, 2, 3, 3, 3, 4],
    [1, 2, 2, 6, 6, 4],
    [1, 5, 5, 6, 6, 4],
    [7, 5, 8, 8, 8, 8],
    [7, 7, 7, 5, 5, 7],  # tweak below
]
# That layout is hard to balance. Instead use a verified-by-construction layout:

SOLUTION_GRID = [
    [1, 1, 2, 2, 3, 3],
    [1, 1, 2, 2, 3, 3],
    [4, 4, 5, 5, 3, 3],
    [4, 4, 5, 5, 6, 6],
    [7, 7, 8, 8, 6, 6],
    [7, 7, 8, 8, 6, 6],
]
# Piece sizes: 1->4, 2->4, 3->6, 4->4, 5->4, 6->6, 7->4, 8->4. Sum = 36. Good.
# All pieces are 2x2 squares except 3 and 6 which are 2x3 rectangles.

# Piece colors chosen so that the only conflict is at the (P5 <-> neighbors) interface.
PIECE_COLORS = {
    1: "R", 2: "G", 3: "B", 4: "Y",
    5: "B",  # P5 is B; its neighbors are P2(G), P4(Y), P6(Y), P8(G) -> G-B ok, B-Y ok. Fine.
    6: "Y", 7: "R", 8: "G",
}
# We need an actual color conflict to FORCE exemption. Re-color:
PIECE_COLORS = {
    1: "R",
    2: "G",
    3: "B",
    4: "G",
    5: "R",  # P5 borders P2(G), P4(G), P6(?), P8(?). R-G ok. R-B not ok.
    6: "B",  # P5(R) - P6(B) incompatible! This forces exemption on R2 for P5 (or P6).
    7: "Y",
    8: "Y",
}
# Check all adjacencies in SOLUTION_GRID:
#   P1(R)-P2(G): R-G ok
#   P1(R)-P4(G): R-G ok
#   P2(G)-P3(B): G-B ok
#   P2(G)-P5(R): G-R ok
#   P3(B)-P6(B): B-B ok
#   P4(G)-P5(R): G-R ok
#   P4(G)-P7(Y): G-Y NOT OK!
# Need another fix. Adjust P7 to R: P4(G)-P7(R) ok, P7(R)-P8(Y) ok (Y-R ok).
PIECE_COLORS = {
    1: "R", 2: "G", 3: "B", 4: "G", 5: "R", 6: "B", 7: "R", 8: "Y",
}
# Re-verify all piece-piece adjacencies:
#   P1(R)-P2(G): rows 0-1 cols 1-2; ok
#   P1(R)-P4(G): rows 1-2 cols 0-1 boundary (P1 at (1,0..1), P4 at (2,0..1)); R-G ok
#   P2(G)-P3(B): cols 3-4 rows 0-1; G-B ok
#   P2(G)-P5(R): row 1-2 cols 2-3 boundary (P2 ends row 1, P5 starts row 2); G-R ok
#   P3(B)-P5(R): row 2 col 3-4 boundary? P3 at (2,4..5), P5 at (2,2..3). Col 3/4 adj. row 2. B-R NOT OK!
# So P3-P5 boundary at (2,3)-(2,4) is R-B incompatible.
# Good: there is exactly one piece pair conflicting after coloring -> P3(B) and P5(R).
# Also check P4(G)-P7(R) row 3/4 col 0..1: G-R ok.
#   P4(G)-P5(R): row 2-3 col 2-3 vs col 0-1? P4 at (2..3, 0..1); P5 at (2..3, 2..3). Adj cols 1-2. G-R ok.
#   P5(R)-P6(B): P5 at (2..3, 2..3); P6 at (3..5, 4..5). Adj: (3,3)-(3,4): R-B NOT OK!
#   P5(R)-P8(Y): P5 at (2..3, 2..3); P8 at (4..5, 2..3). Adj: (3,2)-(4,2),(3,3)-(4,3): R-Y ok.
#   P6(B)-P8(Y): P6 at (3..5, 4..5); P8 at (4..5, 2..3). Adj cols 3-4 rows 4-5: B-Y ok.
#   P7(R)-P8(Y): P7 at (4..5, 0..1); P8 at (4..5, 2..3). Adj cols 1-2 rows 4-5: R-Y ok.
#
# So P5 conflicts with BOTH P3 and P6. Exemption on R2 for P5 covers both
# (exemption: P5's color edges are ignored). 

# Piece base shapes & valid rotations:
# All pieces here are rectangles, so rotation 90 changes 2x2 -> 2x2 (same),
# and 2x3 -> 3x2. We define base shapes.
PIECE_BASE = {
    1: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 2x2
    2: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 2x2
    3: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 3x2 (vertical rectangle)
    4: [(0, 0), (0, 1), (1, 0), (1, 1)],
    5: [(0, 0), (0, 1), (1, 0), (1, 1)],
    6: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
    7: [(0, 0), (0, 1), (1, 0), (1, 1)],
    8: [(0, 0), (0, 1), (1, 0), (1, 1)],
}
# Required-by-solution rotations (all 0 for this layout):
# Anchors:
PIECE_ANCHOR_SOL = {
    1: (0, 0),
    2: (0, 2),
    3: (0, 4),
    4: (2, 0),
    5: (2, 2),
    6: (3, 4),
    7: (4, 0),
    8: (4, 2),
}
PIECE_ROT_SOL = {pid: 0 for pid in range(1, 9)}


def rotate_cells(cells, rot):
    """Rotate a list of (dr, dc) cells by rot degrees in {0, 90, 180, 270}.
    Returns rotated cells normalized so min row = 0, min col = 0."""
    out = []
    for r, c in cells:
        if rot == 0:
            nr, nc = r, c
        elif rot == 90:
            nr, nc = c, -r
        elif rot == 180:
            nr, nc = -r, -c
        elif rot == 270:
            nr, nc = -c, r
        else:
            raise ValueError(f"bad rotation {rot}")
        out.append((nr, nc))
    min_r = min(x for x, _ in out)
    min_c = min(y for _, y in out)
    return [(r - min_r, c - min_c) for r, c in out]


class PrimitivBenchGame:
    """Polyomino tiling with composed rules and a single-rule exemption."""

    # R1 allows rotations {0, 90} unless the piece is exempt from R1.
    ALLOWED_ROTATIONS_R1 = {0, 90}
    ALL_ROTATIONS = {0, 90, 180, 270}

    RULE_NAMES = ("R1_rotation", "R2_color")

    def __init__(self):
        self.state = None

    def reset(self, seed: int = None) -> dict:
        # Deterministic puzzle; seed reserved but unused (puzzle is fixed by design).
        self.state = {
            "grid": [[0 for _ in range(GRID_W)] for _ in range(GRID_H)],  # 0 = empty
            "placements": {},  # pid -> {"anchor": (r,c), "rot": int}
            "exemption": None,  # None or {"piece": pid, "rule": "R1_rotation"|"R2_color"}
            "step_count": 0,
            "max_steps": 200,
            "pieces": {
                pid: {
                    "base_cells": list(PIECE_BASE[pid]),
                    "color": PIECE_COLORS[pid],
                    "size": len(PIECE_BASE[pid]),
                }
                for pid in range(1, 9)
            },
            "compat": {f"{a}-{b}": v for (a, b), v in COMPAT.items()},
        }
        return deepcopy(self.state)

    # ---------- helpers ----------

    def _piece_cells(self, pid, anchor, rot):
        base = PIECE_BASE[pid]
        rotated = rotate_cells(base, rot)
        ar, ac = anchor
        return [(ar + r, ac + c) for r, c in rotated]

    def _in_bounds(self, cells):
        return all(0 <= r < GRID_H and 0 <= c < GRID_W for r, c in cells)

    def _rule1_violated(self, pid, rot):
        # R1: rotation must be in {0,90}. If rot not in this set, R1 violated.
        return rot not in self.ALLOWED_ROTATIONS_R1

    def _adjacent_color_conflicts(self, pid, cells):
        """Return list of (other_pid, (r,c), (r2,c2)) conflicts under R2."""
        conflicts = []
        my_color = PIECE_COLORS[pid]
        cellset = set(cells)
        for (r, c) in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < GRID_H and 0 <= cc < GRID_W):
                    continue
                if (rr, cc) in cellset:
                    continue
                other = self.state["grid"][rr][cc]
                if other == 0 or other == pid:
                    continue
                other_color = PIECE_COLORS[other]
                if not COMPAT[(my_color, other_color)]:
                    conflicts.append((other, (r, c), (rr, cc)))
        return conflicts

    def _exemption_covers(self, pid, rule_name):
        ex = self.state["exemption"]
        return ex is not None and ex["piece"] == pid and ex["rule"] == rule_name

    def _placement_legal(self, pid, anchor, rot, check_color=True):
        """Returns (ok, reason). Considers current exemption."""
        if pid in self.state["placements"]:
            return False, "piece_already_placed"
        cells = self._piece_cells(pid, anchor, rot)
        if not self._in_bounds(cells):
            return False, "out_of_bounds"
        for r, c in cells:
            if self.state["grid"][r][c] != 0:
                return False, "overlap"
        # R1 check
        if self._rule1_violated(pid, rot):
            if not self._exemption_covers(pid, "R1_rotation"):
                return False, "R1_violation"
        # R2 check
        if check_color:
            conflicts = self._adjacent_color_conflicts(pid, cells)
            if conflicts and not self._exemption_covers(pid, "R2_color"):
                # Also: the *other* piece's exemption could cover the adjacency,
                # but in this puzzle exemption is granted to one piece's outgoing edges.
                # We treat exemption as "this piece's color edges ignored", which
                # is symmetric: if either endpoint piece is R2-exempt, edge ignored.
                filtered = [
                    cf for cf in conflicts
                    if not self._exemption_covers(cf[0], "R2_color")
                ]
                if filtered:
                    return False, f"R2_violation:{filtered[0]}"
        return True, "ok"

    # ---------- action API ----------

    def valid_actions(self) -> list:
        """Enumerate a representative set of legal actions.
        For brevity we list: place actions for each unplaced piece at every
        legal (anchor, rot) given current state; the exemption-declaration
        actions (if exemption slot empty); and a remove action per placed piece.
        """
        actions = []
        # Exemption declarations
        if self.state["exemption"] is None:
            for pid in range(1, 9):
                for rule in self.RULE_NAMES:
                    actions.append(("declare_exemption", pid, rule))
        # Remove actions
        for pid in list(self.state["placements"].keys()):
            actions.append(("remove", pid))
        # Place actions
        for pid in range(1, 9):
            if pid in self.state["placements"]:
                continue
            for rot in (0, 90, 180, 270):
                for r in range(GRID_H):
                    for c in range(GRID_W):
                        ok, _ = self._placement_legal(pid, (r, c), rot)
                        if ok:
                            actions.append(("place", pid, rot, r, c))
        return actions

    def step(self, action):
        self.state["step_count"] += 1
        info = {"action": action}
        reward = 0.0

        if not isinstance(action, tuple) or len(action) < 1:
            return deepcopy(self.state), -0.1, False, {"error": "malformed_action"}

        op = action[0]

        if op == "declare_exemption":
            _, pid, rule = action
            if self.state["exemption"] is not None:
                info["error"] = "exemption_already_declared"
                reward = -0.1
            elif pid not in range(1, 9) or rule not in self.RULE_NAMES:
                info["error"] = "bad_exemption_args"
                reward = -0.1
            else:
                self.state["exemption"] = {"piece": pid, "rule": rule}
                reward = 0.05
                info["ok"] = True

        elif op == "remove":
            _, pid = action
            if pid not in self.state["placements"]:
                info["error"] = "not_placed"
                reward = -0.1
            else:
                pl = self.state["placements"].pop(pid)
                cells = self._piece_cells(pid, pl["anchor"], pl["rot"])
                for r, c in cells:
                    self.state["grid"][r][c] = 0
                reward = -0.02
                info["ok"] = True

        elif op == "place":
            _, pid, rot, r, c = action
            ok, reason = self._placement_legal(pid, (r, c), rot)
            if not ok:
                info["error"] = reason
                reward = -0.1
            else:
                cells = self._piece_cells(pid, (r, c), rot)
                for cr, cc in cells:
                    self.state["grid"][cr][cc] = pid
                self.state["placements"][pid] = {"anchor": (r, c), "rot": rot}
                reward = 0.1
                info["ok"] = True

        else:
            info["error"] = "unknown_op"
            reward = -0.1

        done = self.is_won() or self.state["step_count"] >= self.state["max_steps"]
        if self.is_won():
            reward += 1.0
            info["won"] = True
        return deepcopy(self.state), reward, done, info

    def is_won(self) -> bool:
        # All 36 cells filled
        for row in self.state["grid"]:
            for v in row:
                if v == 0:
                    return False
        # All 8 pieces placed
        if len(self.state["placements"]) != 8:
            return False
        # All placements legal under current rules + exemption
        for pid, pl in self.state["placements"].items():
            anchor = pl["anchor"]
            rot = pl["rot"]
            # Temporarily clear this piece to test legality
            cells = self._piece_cells(pid, anchor, rot)
            for cr, cc in cells:
                self.state["grid"][cr][cc] = 0
            del self.state["placements"][pid]
            ok, _ = self._placement_legal(pid, anchor, rot)
            for cr, cc in cells:
                self.state["grid"][cr][cc] = pid
            self.state["placements"][pid] = pl
            if not ok:
                return False
        return True

    def render(self, mode: str = "text") -> str:
        lines = []
        lines.append("  " + " ".join(str(c) for c in range(GRID_W)))
        for r in range(GRID_H):
            row = [str(self.state["grid"][r][c]) if self.state["grid"][r][c] != 0 else "."
                   for c in range(GRID_W)]
            lines.append(f"{r} " + " ".join(row))
        lines.append("")
        lines.append("Pieces (id:color:size): " + ", ".join(
            f"{pid}:{PIECE_COLORS[pid]}:{len(PIECE_BASE[pid])}" for pid in range(1, 9)
        ))
        lines.append("Compat: 4-cycle R-G-B-Y-R; (R,B) and (G,Y) NOT compatible.")
        lines.append(f"Exemption: {self.state['exemption']}")
        lines.append(f"Placed: {sorted(self.state['placements'].keys())}")
        return "\n".join(lines)


# ----------------- self-test when run directly -----------------
if __name__ == "__main__":
    g = PrimitivBenchGame()
    g.reset(seed=0)
    print(g.render())
    # Optimal play: declare exemption for P5 on R2_color, then place per solution.
    print("\n-- declare exemption P5 / R2 --")
    s, r, d, info = g.step(("declare_exemption", 5, "R2_color"))
    print("reward", r, "info", info)
    for pid in [1, 2, 3, 4, 5, 6, 7, 8]:
        anchor = PIECE_ANCHOR_SOL[pid]
        rot = PIECE_ROT_SOL[pid]
        a = ("place", pid, rot, anchor[0], anchor[1])
        s, r, d, info = g.step(a)
        print(f"place P{pid} -> reward={r} done={d} info={info}")
    print(g.render())
    print("WON?", g.is_won())
