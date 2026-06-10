import random
from typing import List, Tuple, Dict, Optional

# Axial coordinates for hex grid: (q, r), with s = -q - r
# Distance: (|dq| + |dr| + |ds|) / 2

class PrimitivBenchGame:
    def __init__(self, radius: int = 4, tokens_per_side: int = 3, max_turns: int = 60):
        self.radius = radius
        self.tokens_per_side = tokens_per_side
        self.max_turns = max_turns
        self._rng = random.Random()
        self._seed = None

        # State
        self.all_cells = []           # list of all in-bounds coords
        self.occup: Dict[Tuple[int,int], int] = {}  # 0 empty, 1 player, 2 npc
        self.player_tokens: List[Tuple[int,int]] = []
        self.npc_tokens: List[Tuple[int,int]] = []
        self.turn_count = 0  # number of player turns completed
        self.done = False
        self._won = False
        self.info = {}
        self.invalid_count = 0
        self._hidden_target: Tuple[int,int] = (0,0)  # NPC target (private policy)
        self._npc_axis = 0  # unused; kept for extensibility

    # Public API
    def reset(self, seed: int = None) -> dict:
        if seed is None:
            seed = 0
        self._seed = seed
        self._rng.seed(seed)

        # Build board
        self.all_cells = self._generate_all_cells(self.radius)
        self.occup = {c: 0 for c in self.all_cells}
        self.turn_count = 0
        self.done = False
        self._won = False
        self.info = {}
        self.invalid_count = 0

        # Place tokens in a deterministic mirrored chain configuration, rotated by seed
        player_base = [(-4, 2), (-3, 2), (-3, 1)]
        npc_base = [(4, -2), (3, -2), (3, -1)]
        # If radius not 4, scale or adapt
        if self.radius != 4:
            # Project pattern to border rings near +/-radius
            R = self.radius
            player_base = [(-R, 2 if R >= 2 else 1), (-(R-1), 2 if R >= 2 else 1), (-(R-1), 1)]
            npc_base = [(R, -2 if R >= 2 else -1), ((R-1), -2 if R >= 2 else -1), ((R-1), -1)]

        rot = self._rng.randrange(6)
        self.player_tokens = [self._rotate_axial(p, rot) for p in player_base]
        self.npc_tokens = [self._rotate_axial(n, rot) for n in npc_base]

        # Validate in-bounds and adjust if needed (fallback to center-placed chains)
        if not self._positions_valid_initial(self.player_tokens, self.npc_tokens):
            # fallback deterministic chain near -q side and +q side
            self.player_tokens = [(-self.radius, 0), (-self.radius+1, 0), (-self.radius+1, 1)]
            self.npc_tokens = [(self.radius, 0), (self.radius-1, 0), (self.radius-1, -1)]
            # final rotation by rot
            self.player_tokens = [self._rotate_axial(p, rot) for p in self.player_tokens]
            self.npc_tokens = [self._rotate_axial(n, rot) for n in self.npc_tokens]

        # Fill occupancy
        self.occup = {c: 0 for c in self.all_cells}
        for p in self.player_tokens:
            self.occup[p] = 1
        for n in self.npc_tokens:
            self.occup[n] = 2

        # Choose hidden target on ring distance 2 from center, deterministically by seed
        ring2 = [c for c in self.all_cells if self._hex_distance((0,0), c) == 2]
        if ring2:
            idx = self._rng.randrange(len(ring2))
            self._hidden_target = ring2[idx]
        else:
            self._hidden_target = (0,0)

        # NPC axis not used but deterministic
        self._npc_axis = self._rng.randrange(3)

        return self._obs()

    def step(self, action) -> tuple[dict, float, bool, dict]:
        if self.done:
            return self._obs(), 0.0, True, self.info

        # If player already stuck at start of turn, they lose immediately
        legal = self._legal_moves_player()
        if not legal:
            self.done = True
            self._won = False
            self.info = {"result": "loss", "reason": "player_stuck"}
            return self._obs(), -1.0, True, self.info

        # Parse action
        parsed = self._parse_action(action)
        if parsed is None or parsed not in legal:
            # Invalid action penalty; do not advance state
            self.invalid_count += 1
            if self.invalid_count >= 5:
                self.done = True
                self._won = False
                self.info = {"result": "loss", "reason": "too_many_invalid_actions"}
                return self._obs(), -1.0, True, self.info
            return self._obs(), -0.2, False, {"result": "invalid_action", "invalid_count": self.invalid_count}

        # Apply player's move
        self._apply_move(for_player=True, move=parsed)

        # Step penalty
        reward = -0.01

        # After player's move, if NPC has no legal moves, player wins
        npc_legal = self._legal_moves_npc()
        if not npc_legal:
            self.done = True
            self._won = True
            self.info = {"result": "win", "reason": "npc_stuck"}
            return self._obs(), 1.0, True, self.info

        # Otherwise, NPC moves according to hidden deterministic policy
        npc_move = self._npc_choose_move(npc_legal)
        self._apply_move(for_player=False, move=npc_move)

        # Increment turn counter (player-NPC pair constitutes one turn)
        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            self.done = True
            self._won = False
            self.info = {"result": "draw", "reason": "max_turns_exceeded"}
            return self._obs(), 0.0, True, self.info

        return self._obs(), reward, False, {"result": "continue"}

    def render(self, mode: str = "text") -> str:
        # Render axial grid as rows of r from -radius to radius
        lines = []
        lines.append(f"Turn {self.turn_count} | radius={self.radius} | done={self.done}")
        for r in range(-self.radius, self.radius + 1):
            indent = " " * (self.radius - (r + self.radius)//2)  # rough spacing
            row = []
            q_min = max(-self.radius, -r - self.radius)
            q_max = min(self.radius, -r + self.radius)
            for q in range(q_min, q_max + 1):
                cell = (q, r)
                occ = self.occup.get(cell, 0)
                ch = "."
                if occ == 1:
                    ch = "P"
                elif occ == 2:
                    ch = "N"
                row.append(ch)
            lines.append(indent + " ".join(row))
        lines.append(f"Player tokens: {sorted(self.player_tokens)}")
        lines.append(f"NPC tokens: {sorted(self.npc_tokens)}")
        # Hidden target is private; not shown
        return "\n".join(lines)

    def valid_actions(self) -> List[Tuple[int,int,int,int]]:
        if self.done:
            return []
        return self._legal_moves_player()

    def is_won(self) -> bool:
        return self.done and self._won

    # Helpers and internal mechanics

    def _obs(self) -> dict:
        return {
            "radius": self.radius,
            "player_tokens": list(self.player_tokens),
            "npc_tokens": list(self.npc_tokens),
            "turn_count": self.turn_count,
            "done": self.done
        }

    def _generate_all_cells(self, radius: int) -> List[Tuple[int,int]]:
        cells = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                cells.append((q, r))
        return cells

    def _hex_distance(self, a: Tuple[int,int], b: Tuple[int,int]) -> int:
        dq = a[0] - b[0]
        dr = a[1] - b[1]
        ds = -(a[0] + a[1]) + (b[0] + b[1])
        return (abs(dq) + abs(dr) + abs(ds)) // 2

    def _neighbors(self, c: Tuple[int,int]) -> List[Tuple[int,int]]:
        q, r = c
        dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        res = []
        for dq, dr in dirs:
            cc = (q + dq, r + dr)
            if cc in self.occup:
                res.append(cc)
        return res

    def _is_connected(self, tokens: List[Tuple[int,int]]) -> bool:
        if not tokens:
            return True
        token_set = set(tokens)
        from collections import deque
        q = deque([tokens[0]])
        seen = {tokens[0]}
        while q:
            cur = q.popleft()
            for nb in self._neighbors(cur):
                if nb in token_set and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        return len(seen) == len(tokens)

    def _rotate_axial(self, c: Tuple[int,int], times: int) -> Tuple[int,int]:
        # Convert axial (q,r) to cube (x, y, z) with x=q, z=r, y=-x-z
        x, z = c
        y = -x - z
        times = times % 6
        for _ in range(times):
            # 60-degree rotation: (x,y,z) -> (-z, -x, -y)
            x, y, z = -z, -x, -y
        # Back to axial: q=x, r=z
        q, r = x, z
        if (q, r) not in self.occup and self.all_cells:
            # We may be rotating before occup is built in reset; check via bounds
            if max(abs(q), abs(r), abs(-q - r)) > self.radius:
                # Clamp or wrap not needed; caller will validate and fallback
                pass
        return (q, r)

    def _positions_valid_initial(self, P: List[Tuple[int,int]], N: List[Tuple[int,int]]) -> bool:
        # in bounds
        for c in P + N:
            if max(abs(c[0]), abs(c[1]), abs(-c[0] - c[1])) > self.radius:
                return False
        # no overlap
        if set(P) & set(N):
            return False
        # connectedness
        if not self._is_connected(P):
            return False
        if not self._is_connected(N):
            return False
        # cross distance >= 4
        for p in P:
            for n in N:
                if self._hex_distance(p, n) <= 3:
                    return False
        return True

    def _parse_action(self, action) -> Optional[Tuple[int,int,int,int]]:
        if isinstance(action, (list, tuple)) and len(action) == 4:
            fq, fr, tq, tr = action
            if all(isinstance(v, int) for v in (fq, fr, tq, tr)):
                return (fq, fr, tq, tr)
        if isinstance(action, dict):
            try:
                fq = int(action["from_q"]); fr = int(action["from_r"])
                tq = int(action["to_q"]); tr = int(action["to_r"])
                return (fq, fr, tq, tr)
            except Exception:
                return None
        return None

    def _legal_moves_player(self) -> List[Tuple[int,int,int,int]]:
        return self._enumerate_legal_moves(for_player=True)

    def _legal_moves_npc(self) -> List[Tuple[int,int,int,int]]:
        return self._enumerate_legal_moves(for_player=False)

    def _enumerate_legal_moves(self, for_player: bool) -> List[Tuple[int,int,int,int]]:
        me_tokens = self.player_tokens if for_player else self.npc_tokens
        opp_tokens = self.npc_tokens if for_player else self.player_tokens
        legal: List[Tuple[int,int,int,int]] = []

        me_set = set(me_tokens)
        opp_set = set(opp_tokens)

        for f in me_tokens:
            for t in self._neighbors(f):
                if self.occup.get(t, 1) != 0:
                    continue  # must move to empty
                # Check that t is adjacent (it is) and in-bounds (neighbors are in-bounds)
                # Simulate move
                new_me = [x for x in me_tokens if x != f] + [t]
                # Own connectivity must remain connected
                if not self._is_connected(new_me):
                    continue
                # Cross three-neighborhood exclusion: distance >= 4 to all opponent tokens
                ok = True
                for o in opp_tokens:
                    if self._hex_distance(t, o) <= 3:
                        ok = False
                        break
                if not ok:
                    continue
                # Opponent connectivity not affected; no need to check
                legal.append((f[0], f[1], t[0], t[1]))
        # Deterministic order
        legal.sort()
        return legal

    def _apply_move(self, for_player: bool, move: Tuple[int,int,int,int]):
        fq, fr, tq, tr = move
        f = (fq, fr); t = (tq, tr)
        assert self.occup.get(f, 0) == (1 if for_player else 2), "from must contain mover"
        assert self.occup.get(t, 0) == 0, "to must be empty"
        # Update lists
        if for_player:
            self.player_tokens = [x for x in self.player_tokens if x != f] + [t]
        else:
            self.npc_tokens = [x for x in self.npc_tokens if x != f] + [t]
        # Update occupancy map fresh
        self.occup = {c: 0 for c in self.all_cells}
        for p in self.player_tokens:
            self.occup[p] = 1
        for n in self.npc_tokens:
            self.occup[n] = 2

    def _npc_choose_move(self, legal_moves: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
        # Hidden policy:
        # Score = - min distance of any NPC token to hidden target after move
        #         + 0.01 * min cross distance to any player token
        #         - 0.001 * span (max pairwise distance within NPC tokens)
        # Deterministic tie-breaking by (to, from) lexicographic.
        best = None
        best_tuple = None  # comparison tuple
        target = self._hidden_target

        for mv in legal_moves:
            fq, fr, tq, tr = mv
            f = (fq, fr); t = (tq, tr)
            new_npc = [x for x in self.npc_tokens if x != f] + [t]
            # min distance to target
            dmin_t = min(self._hex_distance(x, target) for x in new_npc)
            # min cross distance to any player
            dmin_cross = min(self._hex_distance(x, p) for x in new_npc for p in self.player_tokens)
            # span
            span = 0
            for i in range(len(new_npc)):
                for j in range(i+1, len(new_npc)):
                    d = self._hex_distance(new_npc[i], new_npc[j])
                    if d > span:
                        span = d
            score = -float(dmin_t) + 0.01 * float(dmin_cross) - 0.001 * float(span)
            # Build comparison tuple for deterministic max
            # primary: score, secondary: to coord lexicographically reversed to prefer "larger" to be consistent
            cmp_tuple = (round(score, 6), tq, tr, fq, fr)
            if best_tuple is None or cmp_tuple > best_tuple:
                best_tuple = cmp_tuple
                best = mv

        if best is None:
            # Should not happen if legal_moves non-empty
            best = legal_moves[0]
        return best


# Simple manual test when running this module directly
if __name__ == "__main__":
    env = PrimitivBenchGame()
    s = env.reset(seed=1007)
    print(env.render())
    print("Valid actions:", env.valid_actions())
    # Play random valid moves for a few steps as a smoke test
    import random as pyrand
    for _ in range(5):
        acts = env.valid_actions()
        if not acts:
            break
        a = pyrand.choice(acts)
        s, r, d, info = env.step(a)
        print("Action:", a, "Reward:", r, "Done:", d, info)
        print(env.render())
        if d:
            break
