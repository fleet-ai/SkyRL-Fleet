"""
PrimitivBench task PB-pilot-002
Game: "Lens Race" — player vs 2 NPCs racing for 4 colored items.
Primitives: A1 objectness, A3 agency, B1 perception transform.
"""

import numpy as np
from typing import Optional


GRID_H = 5
GRID_W = 5
NUM_ITEMS = 4
NUM_NPCS = 2
MAX_STEPS = 40

# Colors are integers 0..7 (3-bit). XOR mask is also 0..7.
NUM_COLORS = 8


class PrimitivBenchGame:
    """
    State:
      - player_pos: (r,c)
      - npc_pos: list of (r,c)
      - items: list of dicts {id, pos, observed_color, alive}
      - mask: current XOR mask (changes per turn)
      - npc_true_pref: list of true-color preferences for each NPC (hidden goal that
            player must reason about; NPCs ACT on raw observed color match, but
            their declared "want" — and thus win target — is the true color.)
      - turn: int
      - collected_by_player: list of item ids
      - collected_by_npc: list of (npc_idx, item_id)

    Win condition for player:
      Player has collected, for EACH NPC, the item whose true_color (== observed XOR mask_at_collection_time)
      matches that NPC's true preference. Equivalently: 2 specific target items
      (one per NPC) must be picked up by the player before any NPC picks ANY item.

    Lose:
      - any NPC picks up any item, OR
      - player picks up a non-target item (wrong identification), OR
      - MAX_STEPS reached without winning.

    NPC policy (A3):
      Each NPC at each turn picks the nearest item (Manhattan) whose
      OBSERVED color equals (npc_true_pref XOR current_mask). I.e. NPCs
      perceive raw observed color, and they "happen" to want the item
      whose raw color matches preference XOR mask — which is exactly the
      item with true_color == preference. The player must compute this.
      Move one step (greedy, deterministic tie-break by (dr, dc) order).
      If adjacent (distance 1) to its target item it picks it up.

    Mask changes deterministically each turn (mask = (base_mask + turn) % 8).
    """

    ACTIONS = ["N", "S", "E", "W", "PICK", "WAIT"]

    def __init__(self):
        self._rng = None
        self._initialized = False

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is None:
            seed = 0
        self._rng = np.random.RandomState(seed)
        rng = self._rng

        # Place player, NPCs, items at distinct cells.
        all_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W)]
        rng.shuffle(all_cells)
        idx = 0
        self.player_pos = tuple(all_cells[idx]); idx += 1
        self.npc_pos = [tuple(all_cells[idx + i]) for i in range(NUM_NPCS)]
        idx += NUM_NPCS

        # Choose NPC true preferences (distinct).
        prefs = list(rng.choice(NUM_COLORS, size=NUM_NPCS, replace=False))
        self.npc_true_pref = [int(p) for p in prefs]

        # Choose 4 items with TRUE colors. Exactly one item per NPC has true_color
        # matching that NPC's preference. The other 2 items have arbitrary
        # different true colors (distractors).
        true_colors = list(self.npc_true_pref)  # length 2
        remaining = [c for c in range(NUM_COLORS) if c not in self.npc_true_pref]
        rng.shuffle(remaining)
        true_colors.extend(remaining[:2])
        rng.shuffle(true_colors)

        self.base_mask = int(rng.randint(0, NUM_COLORS))
        self.turn = 0
        mask = self._current_mask()

        self.items = []
        for i in range(NUM_ITEMS):
            pos = tuple(all_cells[idx + i])
            tc = int(true_colors[i])
            oc = tc ^ mask  # observed at turn 0; we'll re-derive each turn
            self.items.append({
                "id": i,
                "pos": pos,
                "true_color": tc,
                "alive": True,
            })
        idx += NUM_ITEMS

        self.collected_by_player = []
        self.collected_by_npc = []  # list of (npc_idx, item_id)
        self.done = False
        self.win = False
        self.lose_reason = None

        # Player's targets: item ids whose true_color matches some NPC's preference.
        # Each NPC has exactly one matching item (since we placed them).
        self.targets_per_npc = {}
        for ni, p in enumerate(self.npc_true_pref):
            for it in self.items:
                if it["true_color"] == p:
                    self.targets_per_npc[ni] = it["id"]
                    break

        self._initialized = True
        return self._obs()

    # ------------------------------------------------------------------
    def _current_mask(self) -> int:
        return (self.base_mask + self.turn) % NUM_COLORS

    def _observed_color(self, item) -> int:
        return item["true_color"] ^ self._current_mask()

    def _obs(self) -> dict:
        mask = self._current_mask()
        items_obs = []
        for it in self.items:
            if it["alive"]:
                items_obs.append({
                    "id": it["id"],
                    "pos": it["pos"],
                    "observed_color": it["true_color"] ^ mask,
                })
        return {
            "turn": self.turn,
            "mask": mask,
            "player_pos": self.player_pos,
            "npcs": [
                {"idx": i, "pos": self.npc_pos[i], "true_pref": self.npc_true_pref[i]}
                for i in range(NUM_NPCS)
            ],
            "items": items_obs,
            "collected_by_player": list(self.collected_by_player),
            "done": self.done,
            "win": self.win,
        }

    # ------------------------------------------------------------------
    def valid_actions(self) -> list:
        if self.done:
            return []
        acts = ["WAIT"]
        r, c = self.player_pos
        if r > 0: acts.append("N")
        if r < GRID_H - 1: acts.append("S")
        if c < GRID_W - 1: acts.append("E")
        if c > 0: acts.append("W")
        # PICK valid if an alive item is at player's position or adjacent (distance <= 1)
        for it in self.items:
            if it["alive"] and self._manhattan(self.player_pos, it["pos"]) <= 1:
                acts.append("PICK")
                break
        return acts

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------
    def step(self, action):
        if self.done:
            return self._obs(), 0.0, True, {"error": "episode done"}
        if action not in self.valid_actions():
            # illegal -> small penalty, no state change except turn advance for NPCs
            reward = -0.1
            info = {"illegal": True, "action": action}
            # advance turn and let NPCs move
            self._npc_phase()
            self.turn += 1
            if self.turn >= MAX_STEPS and not self.done:
                self.done = True
                self.lose_reason = "timeout"
            return self._obs(), reward, self.done, info

        reward = 0.0
        info = {}

        # --- Player phase ---
        if action == "N":
            self.player_pos = (self.player_pos[0] - 1, self.player_pos[1])
        elif action == "S":
            self.player_pos = (self.player_pos[0] + 1, self.player_pos[1])
        elif action == "E":
            self.player_pos = (self.player_pos[0], self.player_pos[1] + 1)
        elif action == "W":
            self.player_pos = (self.player_pos[0], self.player_pos[1] - 1)
        elif action == "WAIT":
            pass
        elif action == "PICK":
            # pick nearest item within distance 1
            picked = None
            for it in self.items:
                if it["alive"] and self._manhattan(self.player_pos, it["pos"]) <= 1:
                    picked = it
                    break
            if picked is not None:
                picked["alive"] = False
                self.collected_by_player.append(picked["id"])
                info["picked_item"] = picked["id"]
                info["picked_true_color"] = picked["true_color"]
                # Check: is this item one of the target items?
                target_ids = set(self.targets_per_npc.values())
                if picked["id"] not in target_ids:
                    # wrong pick -> lose
                    self.done = True
                    self.lose_reason = "picked_wrong_item"
                    reward = -1.0
                    return self._obs(), reward, True, info
                else:
                    reward += 0.5
                    # win if both targets collected
                    if set(target_ids).issubset(set(self.collected_by_player)):
                        self.done = True
                        self.win = True
                        reward += 1.0
                        return self._obs(), reward, True, info

        # --- NPC phase ---
        self._npc_phase()
        if self.done:
            reward = -1.0
            return self._obs(), reward, True, info

        # --- Turn advance ---
        self.turn += 1
        if self.turn >= MAX_STEPS and not self.done:
            self.done = True
            self.lose_reason = "timeout"
            reward = -0.5

        return self._obs(), reward, self.done, info

    # ------------------------------------------------------------------
    def _npc_phase(self):
        """Each NPC moves one step toward the nearest item whose OBSERVED color
        matches (npc_true_pref XOR current_mask). If already adjacent (or
        on-cell), it picks the item, ending the game with player loss."""
        mask = self._current_mask()
        for ni in range(NUM_NPCS):
            target_obs_color = self.npc_true_pref[ni] ^ mask
            # candidate items: alive, observed_color == target_obs_color
            cands = [it for it in self.items
                     if it["alive"] and (it["true_color"] ^ mask) == target_obs_color]
            if not cands:
                continue
            # nearest by manhattan; deterministic tie-break by item id
            cands.sort(key=lambda it: (self._manhattan(self.npc_pos[ni], it["pos"]), it["id"]))
            target = cands[0]
            dist = self._manhattan(self.npc_pos[ni], target["pos"])
            if dist == 0:
                # pick
                target["alive"] = False
                self.collected_by_npc.append((ni, target["id"]))
                self.done = True
                self.lose_reason = f"npc_{ni}_picked_item_{target['id']}"
                return
            else:
                # step greedily; deterministic order: try row then col
                nr, nc = self.npc_pos[ni]
                tr, tc = target["pos"]
                if nr != tr:
                    nr += 1 if tr > nr else -1
                elif nc != tc:
                    nc += 1 if tc > nc else -1
                self.npc_pos[ni] = (nr, nc)
                # after moving, check if NPC stepped onto item position (distance 0)
                if self.npc_pos[ni] == target["pos"]:
                    target["alive"] = False
                    self.collected_by_npc.append((ni, target["id"]))
                    self.done = True
                    self.lose_reason = f"npc_{ni}_picked_item_{target['id']}"
                    return

    # ------------------------------------------------------------------
    def render(self, mode: str = "text") -> str:
        mask = self._current_mask()
        grid = [["." for _ in range(GRID_W)] for _ in range(GRID_H)]
        for it in self.items:
            if it["alive"]:
                r, c = it["pos"]
                grid[r][c] = f"i{it['id']}({it['true_color']^mask})"
        for ni, (r, c) in enumerate(self.npc_pos):
            grid[r][c] = f"N{ni}"
        pr, pc = self.player_pos
        grid[pr][pc] = "P"
        lines = []
        lines.append(f"Turn {self.turn} | mask={mask} | done={self.done} win={self.win}")
        lines.append(f"NPC true prefs: {self.npc_true_pref}")
        lines.append(f"Targets (per NPC, hidden goal): {self.targets_per_npc}")
        lines.append(f"Collected by player: {self.collected_by_player}")
        # build a width-padded grid
        col_w = 7
        for row in grid:
            lines.append(" ".join(cell.ljust(col_w) for cell in row))
        if self.lose_reason:
            lines.append(f"Lose reason: {self.lose_reason}")
        return "\n".join(lines)

    def is_won(self) -> bool:
        return bool(self.win)


# ---------------- self test ----------------
if __name__ == "__main__":
    g = PrimitivBenchGame()
    obs = g.reset(seed=1)
    print(g.render())
    print("valid:", g.valid_actions())
    # quick random play
    rng = np.random.RandomState(0)
    while not obs["done"]:
        a = rng.choice(g.valid_actions())
        obs, r, d, info = g.step(a)
    print("FINAL:")
    print(g.render())
