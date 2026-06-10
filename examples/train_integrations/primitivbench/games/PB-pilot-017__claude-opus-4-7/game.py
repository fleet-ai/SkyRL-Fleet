"""
PrimitivBench task PB-pilot-017
3-phase route planning on a cylinder grid.

Topology: H x W grid, columns wrap (col W-1 <-> col 0). Rows do not wrap.
The player draws 3 paths sequentially from S to T. Each path's interior cells
become forbidden for later paths (S and T are shared).

Phase 1: path P1 must avoid red zones (R1).
Phase 2: path P2 must avoid red zones AND use at least one wrap-edge (R1+R2).
Phase 3: path P3 must avoid red, use a wrap-edge, and visit the landmark (R1+R2+R3).
All three paths must be non-self-intersecting and pairwise non-crossing
(no shared interior cells).

Win: phase 3 path committed successfully.
"""

import numpy as np

# Fixed map (deterministic). Seed only affects rendering/info, not topology.
H = 5
W = 6
START = (2, 0)
TARGET = (2, 3)
RED_ZONES = {(2, 1), (2, 2)}
LANDMARK = (3, 3)


class PrimitivBenchGame:
    def __init__(self):
        self.reset()

    def reset(self, seed: int = None) -> dict:
        self.seed = seed if seed is not None else 0
        self._rng = np.random.default_rng(self.seed)
        self.phase = 1
        self.completed_paths = []         # list[list[(r,c)]]
        self.blocked_interior = set()     # cells used by completed paths (excluding S,T)
        self.current_path = [START]
        self.pos = START
        self.used_wrap_edge = False
        self.visited_landmark = (START == LANDMARK)
        self.done = False
        self.failed = False
        self.last_info = ""
        return self._state()

    # -------- core helpers --------

    def _neighbors(self, pos):
        r, c = pos
        out = []
        # north / south (no wrap)
        if r - 1 >= 0:
            out.append(("N", (r - 1, c)))
        if r + 1 < H:
            out.append(("S", (r + 1, c)))
        # east / west (wrap on columns)
        out.append(("E", (r, (c + 1) % W), (c + 1) >= W))   # wrap flag
        out.append(("W", (r, (c - 1) % W), (c - 1) < 0))
        return out

    def _phase_rules(self):
        # returns dict of active rules for current phase
        return {
            "R1_avoid_red": True,
            "R2_wrap_edge": self.phase >= 2,
            "R3_landmark":  self.phase >= 3,
        }

    def _cell_allowed(self, cell):
        if cell in RED_ZONES:
            return False
        # cannot revisit cells in current path
        if cell in self.current_path and cell != TARGET:
            return False
        # cannot enter cells used by previously committed paths (interior only)
        if cell in self.blocked_interior:
            return False
        return True

    # -------- gym API --------

    def valid_actions(self) -> list:
        if self.done:
            return []
        acts = []
        # movement actions
        for entry in self._neighbors(self.pos):
            direction = entry[0]
            nxt = entry[1]
            if self._cell_allowed(nxt):
                acts.append(direction)
        # commit action: only available when at TARGET and rules satisfied
        if self.pos == TARGET and len(self.current_path) > 1:
            acts.append("COMMIT")
        # abort current path (start over within phase)
        acts.append("RESET_PATH")
        return acts

    def step(self, action):
        if self.done:
            return self._state(), 0.0, True, {"msg": "already done"}

        reward = 0.0
        info = {}

        if action == "RESET_PATH":
            self.current_path = [START]
            self.pos = START
            self.used_wrap_edge = False
            self.visited_landmark = (START == LANDMARK)
            info["msg"] = f"phase {self.phase}: path reset"
            self.last_info = info["msg"]
            return self._state(), reward, False, info

        if action == "COMMIT":
            ok, msg = self._try_commit()
            info["msg"] = msg
            self.last_info = msg
            if ok:
                reward = 1.0 if self.phase == 3 else 0.25
                if self.phase == 3:
                    self.done = True
                else:
                    # advance to next phase
                    self.phase += 1
                    self.current_path = [START]
                    self.pos = START
                    self.used_wrap_edge = False
                    self.visited_landmark = (START == LANDMARK)
            return self._state(), reward, self.done, info

        # movement
        move_map = {}
        for entry in self._neighbors(self.pos):
            direction = entry[0]
            nxt = entry[1]
            wrap = entry[2] if len(entry) >= 3 else False
            move_map[direction] = (nxt, wrap)

        if action not in move_map:
            info["msg"] = f"invalid action {action}"
            self.last_info = info["msg"]
            return self._state(), -0.01, False, info

        nxt, wrap = move_map[action]
        if not self._cell_allowed(nxt):
            info["msg"] = f"cell {nxt} blocked"
            self.last_info = info["msg"]
            return self._state(), -0.01, False, info

        self.pos = nxt
        self.current_path.append(nxt)
        if wrap:
            self.used_wrap_edge = True
        if nxt == LANDMARK:
            self.visited_landmark = True

        info["msg"] = f"moved {action} -> {nxt}"
        self.last_info = info["msg"]
        return self._state(), reward, False, info

    def _try_commit(self):
        if self.pos != TARGET:
            return False, "commit failed: not at target"
        rules = self._phase_rules()
        # R1 always: path must not contain red cells (already enforced) — double check
        for cell in self.current_path:
            if cell in RED_ZONES:
                return False, "commit failed: R1 violated (red cell in path)"
        if rules["R2_wrap_edge"] and not self.used_wrap_edge:
            return False, "commit failed: R2 violated (no wrap-edge used)"
        if rules["R3_landmark"] and not self.visited_landmark:
            return False, "commit failed: R3 violated (landmark not visited)"
        # accept path
        self.completed_paths.append(list(self.current_path))
        for cell in self.current_path:
            if cell != START and cell != TARGET:
                self.blocked_interior.add(cell)
        return True, f"phase {self.phase} path committed (len={len(self.current_path)})"

    def is_won(self) -> bool:
        return self.done and len(self.completed_paths) == 3

    def render(self, mode: str = "text") -> str:
        # render grid with overlays
        grid = [["." for _ in range(W)] for _ in range(H)]
        for (r, c) in RED_ZONES:
            grid[r][c] = "R"
        lr, lc = LANDMARK
        if grid[lr][lc] == ".":
            grid[lr][lc] = "L"
        sr, sc = START
        tr, tc = TARGET
        grid[sr][sc] = "S"
        grid[tr][tc] = "T"
        # overlay committed paths with digits
        for i, p in enumerate(self.completed_paths, start=1):
            for (r, c) in p:
                if (r, c) in (START, TARGET):
                    continue
                grid[r][c] = str(i)
        # overlay current path with '*'
        for (r, c) in self.current_path:
            if (r, c) in (START, TARGET):
                continue
            if grid[r][c] in (".", "L"):
                grid[r][c] = "*"
        lines = []
        lines.append(f"phase={self.phase}  pos={self.pos}  wrap_used={self.used_wrap_edge}  L_visited={self.visited_landmark}")
        lines.append("cylinder (cols wrap): col 0 <-> col {}".format(W - 1))
        for r in range(H):
            lines.append("  " + " ".join(grid[r]))
        lines.append(f"valid_actions={self.valid_actions()}")
        lines.append(f"last: {self.last_info}")
        return "\n".join(lines)

    def _state(self) -> dict:
        return {
            "phase": self.phase,
            "pos": self.pos,
            "current_path": list(self.current_path),
            "completed_paths": [list(p) for p in self.completed_paths],
            "blocked_interior": sorted(self.blocked_interior),
            "used_wrap_edge": self.used_wrap_edge,
            "visited_landmark": self.visited_landmark,
            "rules_active": self._phase_rules(),
            "done": self.done,
            "won": self.is_won(),
            "H": H, "W": W,
            "start": START, "target": TARGET,
            "red_zones": sorted(RED_ZONES),
            "landmark": LANDMARK,
        }


# ---------- self-test ----------
if __name__ == "__main__":
    g = PrimitivBenchGame()
    g.reset(seed=0)
    print(g.render())
    print("valid:", g.valid_actions())
