"""
PrimitivBench task PB-pilot-001 — Courier's Choice.
Layer 3A: A1_objectness + A3_agency + A4_geometry_topology.
"""
from __future__ import annotations
import random
from typing import Any

GRID_W = 5
GRID_H = 5
MAX_STEPS = 60

# Zones: N = top row (y=0), S = bottom row (y=GRID_H-1)
ZONES = {"N": [(x, 0) for x in range(GRID_W)],
         "S": [(x, GRID_H - 1) for x in range(GRID_W)]}

OBJECT_IDS = ["G", "K", "O"]  # gem, key, orb — persistent distinguishable identity

DIRS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(src, dst, blocked):
    """Greedy 1-step move from src toward dst, avoiding `blocked` cells.
    Returns new position (may equal src if no improvement possible)."""
    best = src
    best_d = _manh(src, dst)
    # Try strictly improving moves first
    candidates = []
    for dx, dy in DIRS.values():
        nx, ny = src[0] + dx, src[1] + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in blocked:
            candidates.append(((nx, ny), _manh((nx, ny), dst)))
    # Prefer strict improvement; among ties, deterministic order
    candidates.sort(key=lambda c: (c[1], c[0][0], c[0][1]))
    if candidates and candidates[0][1] < best_d:
        return candidates[0][0]
    # Otherwise stay
    return src


class PrimitivBenchGame:
    def __init__(self):
        self.state = None
        self._rng = random.Random(0)

    # ---------- setup ----------
    def reset(self, seed: int = None) -> dict:
        if seed is None:
            seed = 0
        rng = random.Random(seed)
        self._rng = rng

        # Place objects at three fixed-but-seed-shuffled cells in middle band
        mid_cells = [(0, 2), (2, 2), (4, 2), (1, 2), (3, 2)]
        rng.shuffle(mid_cells)
        obj_positions = {oid: mid_cells[i] for i, oid in enumerate(OBJECT_IDS)}

        # Player starts center
        player_pos = (2, 2)
        # Resolve collision: if center is occupied by object, push player to (2,3)
        if player_pos in obj_positions.values():
            player_pos = (2, 3)

        # NPCs start at opposite corners (deterministic)
        npc_positions = [(0, 4), (4, 0)]

        # Assign private goals: each NPC has (object_id, zone)
        # NPCs must want DIFFERENT (obj, zone) pairs — but may share object or zone, not both.
        all_pairs = [(o, z) for o in OBJECT_IDS for z in ("N", "S")]
        rng.shuffle(all_pairs)
        npc0_goal = all_pairs[0]
        # pick a different pair for npc1
        for p in all_pairs[1:]:
            if p != npc0_goal:
                npc1_goal = p
                break

        # Choose which NPC is the ally (the one the player must help win)
        ally_idx = rng.randint(0, 1)

        self.state = {
            "step": 0,
            "player": player_pos,
            "player_carrying": None,  # object id or None
            "npcs": [
                {"pos": npc_positions[0], "carrying": None, "goal": npc0_goal, "delivered": False},
                {"pos": npc_positions[1], "carrying": None, "goal": npc1_goal, "delivered": False},
            ],
            "objects": obj_positions,  # oid -> (x,y)  (only when not carried)
            "ally_idx": ally_idx,
            "winner": None,  # 0, 1, or None
            "done": False,
            "history": [],  # observable history of NPC positions (for the player to infer)
        }
        return self._observation()

    # ---------- helpers ----------
    def _occupied_cells(self, exclude_player=False, exclude_npc=None):
        occ = set()
        if not exclude_player:
            occ.add(self.state["player"])
        for i, n in enumerate(self.state["npcs"]):
            if i == exclude_npc:
                continue
            if not n["delivered"]:
                occ.add(n["pos"])
        return occ

    def _object_pos(self, oid):
        """Where is object oid? Could be on grid, carried by player, or carried by an NPC."""
        if self.state["player_carrying"] == oid:
            return ("player", self.state["player"])
        for i, n in enumerate(self.state["npcs"]):
            if n["carrying"] == oid:
                return (f"npc{i}", n["pos"])
        if oid in self.state["objects"]:
            return ("grid", self.state["objects"][oid])
        return (None, None)

    def _observation(self):
        s = self.state
        obs = {
            "step": s["step"],
            "player": s["player"],
            "player_carrying": s["player_carrying"],
            "npc0_pos": s["npcs"][0]["pos"],
            "npc0_carrying": s["npcs"][0]["carrying"],
            "npc1_pos": s["npcs"][1]["pos"],
            "npc1_carrying": s["npcs"][1]["carrying"],
            "objects_on_grid": dict(s["objects"]),
            "ally_idx": s["ally_idx"],   # told to player: which NPC they back
            "winner": s["winner"],
            "done": s["done"],
            # private goals are NOT in observation; player must infer them.
        }
        return obs

    # ---------- actions ----------
    def valid_actions(self) -> list:
        if self.state is None or self.state["done"]:
            return []
        s = self.state
        acts = ["WAIT"]
        # Movement
        px, py = s["player"]
        for d, (dx, dy) in DIRS.items():
            nx, ny = px + dx, py + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                occ = self._occupied_cells(exclude_player=True)
                if (nx, ny) not in occ:
                    acts.append(f"MOVE_{d}")
        # Pickup: if standing on a free object and not carrying
        if s["player_carrying"] is None:
            for oid, pos in s["objects"].items():
                if pos == s["player"]:
                    acts.append(f"PICKUP_{oid}")
        # Drop: if carrying and current cell has no object on grid
        if s["player_carrying"] is not None:
            if s["player"] not in s["objects"].values():
                acts.append("DROP")
        return acts

    def step(self, action) -> tuple[dict, float, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        if self.state["done"]:
            return self._observation(), 0.0, True, {"reason": "already_done"}

        s = self.state
        info = {}
        reward = 0.0

        # --- player action ---
        if action not in self.valid_actions():
            # Treat illegal as WAIT, small penalty
            info["illegal"] = action
            action = "WAIT"
            reward -= 0.01

        if action.startswith("MOVE_"):
            d = action.split("_")[1]
            dx, dy = DIRS[d]
            s["player"] = (s["player"][0] + dx, s["player"][1] + dy)
        elif action.startswith("PICKUP_"):
            oid = action.split("_")[1]
            del s["objects"][oid]
            s["player_carrying"] = oid
        elif action == "DROP":
            oid = s["player_carrying"]
            s["objects"][oid] = s["player"]
            s["player_carrying"] = None
        # WAIT: no-op

        # --- NPC turns (sequential, NPC0 then NPC1) ---
        for i, npc in enumerate(s["npcs"]):
            if npc["delivered"]:
                continue
            goal_obj, goal_zone = npc["goal"]
            if npc["carrying"] == goal_obj:
                # head to nearest cell in goal zone
                zone_cells = ZONES[goal_zone]
                target = min(zone_cells, key=lambda c: _manh(npc["pos"], c))
            else:
                # Where is the object?
                loc_kind, loc_pos = self._object_pos(goal_obj)
                if loc_kind == "grid":
                    target = loc_pos
                else:
                    # object is held by player or other NPC: NPC waits adjacent / shadows
                    target = loc_pos if loc_pos is not None else npc["pos"]
            # Move one step toward target, avoiding occupied cells
            blocked = self._occupied_cells(exclude_npc=i)
            new_pos = _step_toward(npc["pos"], target, blocked)
            npc["pos"] = new_pos

            # Auto-pickup if on the goal object and free
            if npc["carrying"] is None and goal_obj in s["objects"]:
                if s["objects"][goal_obj] == npc["pos"]:
                    del s["objects"][goal_obj]
                    npc["carrying"] = goal_obj

            # Check delivery
            if npc["carrying"] == goal_obj and npc["pos"] in ZONES[goal_zone]:
                npc["delivered"] = True
                if s["winner"] is None:
                    s["winner"] = i

        # --- history log (for inference) ---
        s["history"].append({
            "step": s["step"],
            "npc0_pos": s["npcs"][0]["pos"],
            "npc0_carrying": s["npcs"][0]["carrying"],
            "npc1_pos": s["npcs"][1]["pos"],
            "npc1_carrying": s["npcs"][1]["carrying"],
        })

        s["step"] += 1

        # --- termination ---
        done = False
        if s["winner"] is not None:
            done = True
            if s["winner"] == s["ally_idx"]:
                reward += 1.0
            else:
                reward -= 1.0
        elif s["step"] >= MAX_STEPS:
            done = True
            reward -= 0.5  # timeout, no winner

        s["done"] = done
        return self._observation(), reward, done, info

    def is_won(self) -> bool:
        if self.state is None:
            return False
        return self.state["winner"] == self.state["ally_idx"]

    # ---------- rendering ----------
    def render(self, mode: str = "text") -> str:
        if self.state is None:
            return "<not reset>"
        s = self.state
        grid = [["." for _ in range(GRID_W)] for _ in range(GRID_H)]
        # Zones marker
        for x in range(GRID_W):
            if grid[0][x] == ".":
                grid[0][x] = "n"
            if grid[GRID_H - 1][x] == ".":
                grid[GRID_H - 1][x] = "s"
        # Objects on grid
        for oid, (x, y) in s["objects"].items():
            grid[y][x] = oid
        # NPCs
        for i, n in enumerate(s["npcs"]):
            if n["delivered"]:
                continue
            x, y = n["pos"]
            mark = f"{i}"
            if n["carrying"]:
                mark = f"{i}{n['carrying'].lower()}"
            grid[y][x] = mark
        # Player
        px, py = s["player"]
        pmark = "P"
        if s["player_carrying"]:
            pmark = f"P{s['player_carrying'].lower()}"
        grid[py][px] = pmark

        lines = []
        lines.append(f"Step {s['step']}  ally=NPC{s['ally_idx']}  winner={s['winner']}")
        for row in grid:
            lines.append(" ".join(f"{c:>3}" for c in row))
        lines.append(f"NPC0 pos={s['npcs'][0]['pos']} carry={s['npcs'][0]['carrying']} delivered={s['npcs'][0]['delivered']}")
        lines.append(f"NPC1 pos={s['npcs'][1]['pos']} carry={s['npcs'][1]['carrying']} delivered={s['npcs'][1]['delivered']}")
        lines.append(f"Player {s['player']} carrying={s['player_carrying']}")
        return "\n".join(lines)


# Convenience: self-test when run directly.
if __name__ == "__main__":
    g = PrimitivBenchGame()
    obs = g.reset(seed=1001)
    print(g.render())
    while not g.state["done"]:
        acts = g.valid_actions()
        a = random.choice(acts)
        obs, r, done, info = g.step(a)
    print("---")
    print(g.render())
    print("won?", g.is_won())
