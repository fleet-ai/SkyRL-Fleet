import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Optional

# PrimitivBenchGame: Turn-based grid with two NPC agents and three labeled objects.
# Player must infer NPC hidden object-target goals and commit support before a goal is achieved.
# Win if the supported NPC's goal completes first.

Coord = Tuple[int, int]


class PrimitivBenchGame:
    def __init__(self):
        self.h = 6
        self.w = 6
        # Fixed interior walls to create geometric constraints
        self.fixed_walls = {(2, 2), (2, 3), (3, 2), (1, 4), (4, 1)}
        # Fixed visible targets (three)
        self.target_ids = [0, 1, 2]
        self.targets_fixed = {
            0: (0, 5),  # top-right corner
            1: (5, 0),  # bottom-left corner
            2: (5, 5),  # bottom-right corner
        }
        self.max_rounds = 60  # bounded runtime per episode
        self.rng = np.random.RandomState(0)
        self.reset()

    def reset(self, seed: int = None) -> dict:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        # Initialize state containers
        self.walls = set(self.fixed_walls)
        self.targets = dict(self.targets_fixed)
        # Place three labeled objects at random empty cells not on walls or targets
        empties = [(r, c) for r in range(self.h) for c in range(self.w)
                   if (r, c) not in self.walls and (r, c) not in self.targets.values()]
        self.rng.shuffle(empties)
        # Objects A, B, C
        self.objects: Dict[str, Coord] = {}
        obj_labels = ['A', 'B', 'C']
        for lab in obj_labels:
            # Ensure not on target initially
            while True:
                pos = empties.pop()
                if pos not in self.targets.values():
                    self.objects[lab] = pos
                    break
        # Place NPCs and player in empty cells not occupied by objects or walls/targets
        def next_empty():
            while True:
                p = empties.pop()
                if p not in self.objects.values() and p not in self.targets.values():
                    return p

        self.player_pos: Coord = next_empty()
        self.red_pos: Coord = next_empty()
        self.blue_pos: Coord = next_empty()

        # Hidden goals: each NPC is assigned a distinct (object_label, target_id) pair
        all_pairs = [(lab, tid) for lab in obj_labels for tid in self.target_ids]
        self.rng.shuffle(all_pairs)
        self.red_goal = all_pairs[0]
        # ensure distinct
        idx = 1
        while all_pairs[idx] == self.red_goal:
            idx += 1
        self.blue_goal = all_pairs[idx]

        # Game progress
        self.support: Optional[str] = None  # 'red' or 'blue' once declared
        self.step_count = 0  # rounds taken (each round: P, R, B)
        self.done = False
        self.winner_npc: Optional[str] = None  # 'red' or 'blue'
        # Order per round: player acts via step(); then env simulates red, then blue.
        return self._observation()

    # Utility geometry
    def in_bounds(self, p: Coord) -> bool:
        r, c = p
        return 0 <= r < self.h and 0 <= c < self.w

    def is_wall(self, p: Coord) -> bool:
        return p in self.walls

    def is_target(self, p: Coord) -> bool:
        return p in self.targets.values()

    def occ_objects(self) -> set:
        return set(self.objects.values())

    def occ_actors(self) -> set:
        return {self.player_pos, self.red_pos, self.blue_pos}

    def is_empty_cell(self, p: Coord, ignore: set = None) -> bool:
        if not self.in_bounds(p) or self.is_wall(p):
            return False
        if ignore is None:
            ignore = set()
        if p in ignore:
            return True
        if p in self.occ_objects():
            return False
        if p in self.occ_actors():
            return False
        return True

    def manhattan(self, a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors4(self, p: Coord) -> List[Coord]:
        r, c = p
        cand = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [q for q in cand if self.in_bounds(q) and not self.is_wall(q)]

    def _bfs(self, start: Coord, goals: set, blocked: set) -> Optional[List[Coord]]:
        # Return path (including start and goal) to the nearest goal among goals; avoid blocked cells
        if start in goals:
            return [start]
        q = deque([start])
        prev = {start: None}
        while q:
            u = q.popleft()
            for v in self._neighbors4(u):
                if v in prev:
                    continue
                if v in blocked:
                    continue
                prev[v] = u
                if v in goals:
                    # reconstruct
                    path = [v]
                    while u is not None:
                        path.append(u)
                        u = prev[u]
                    path.reverse()
                    return path
                q.append(v)
        return None

    def _push(self, actor: str, direction: Coord) -> bool:
        # Try to push an object if actor moves into it and beyond is empty
        if actor == 'player':
            apos = self.player_pos
        elif actor == 'red':
            apos = self.red_pos
        else:
            apos = self.blue_pos
        dr, dc = direction
        into = (apos[0] + dr, apos[1] + dc)
        if not self.in_bounds(into) or self.is_wall(into):
            return False
        # Is there an object?
        for lab, opos in self.objects.items():
            if opos == into:
                beyond = (into[0] + dr, into[1] + dc)
                if not self.is_empty_cell(beyond):
                    return False
                # Execute push: move object into beyond, actor moves into object's former cell
                self.objects[lab] = beyond
                if actor == 'player':
                    self.player_pos = into
                elif actor == 'red':
                    self.red_pos = into
                else:
                    self.blue_pos = into
                return True
        return False

    def _move_actor(self, actor: str, direction: Coord) -> bool:
        # Try standard move or push if valid; return True if any change
        if actor == 'player':
            apos = self.player_pos
        elif actor == 'red':
            apos = self.red_pos
        else:
            apos = self.blue_pos
        dr, dc = direction
        dest = (apos[0] + dr, apos[1] + dc)
        # First try standard move
        if self.is_empty_cell(dest):
            if actor == 'player':
                self.player_pos = dest
            elif actor == 'red':
                self.red_pos = dest
            else:
                self.blue_pos = dest
            return True
        # Try push if moving into object and beyond empty
        return self._push(actor, direction)

    def _npc_policy_step(self, npc: str):
        # NPC attempts to push its chosen object towards its target via local decision and BFS to pusher cells
        if npc == 'red':
            pos = self.red_pos
            obj_label, target_id = self.red_goal
        else:
            pos = self.blue_pos
            obj_label, target_id = self.blue_goal
        obj_pos = self.objects[obj_label]
        tgt_pos = self.targets[target_id]

        # 1) If adjacent in correct position, attempt push that reduces Manhattan distance
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UDLR
        for dr, dc in dirs:
            adj = (pos[0] + dr, pos[1] + dc)
            if adj == obj_pos:
                beyond = (obj_pos[0] + dr, obj_pos[1] + dc)
                if self.is_empty_cell(beyond) and self.manhattan(beyond, tgt_pos) < self.manhattan(obj_pos, tgt_pos):
                    # perform push
                    self._push(npc, (dr, dc))
                    return

        # 2) Move toward a "pusher cell": empty cell adjacent to object from which a push would reduce distance and is feasible
        pusher_cells = set()
        for dr, dc in dirs:
            pusher = (obj_pos[0] - dr, obj_pos[1] - dc)
            push_to = (obj_pos[0] + dr, obj_pos[1] + dc)
            if not self.in_bounds(pusher) or not self.in_bounds(push_to):
                continue
            # Cell where NPC stands must be empty and not a wall/object/actor
            if not self.is_empty_cell(pusher):
                continue
            # Push result cell must be empty and strictly reduce distance
            if not self.is_empty_cell(push_to):
                continue
            if self.manhattan(push_to, tgt_pos) < self.manhattan(obj_pos, tgt_pos):
                pusher_cells.add(pusher)

        # Build blocked set for BFS: walls, objects, actors (including the other NPC and player)
        blocked = set(self.walls) | set(self.occ_objects()) | set(self.occ_actors())
        # The NPC's own current cell shouldn't be blocked for itself
        blocked.discard(pos)

        if pusher_cells:
            path = self._bfs(pos, pusher_cells, blocked)
            if path and len(path) >= 2:
                step_to = path[1]
                self._move_specific(npc, step_to)
                return

        # 3) Fallback: move toward any empty neighbor of the object (try to approach)
        neighbor_cells = set()
        for dr, dc in dirs:
            neigh = (obj_pos[0] - dr, obj_pos[1] - dc)
            if self.is_empty_cell(neigh):
                neighbor_cells.add(neigh)
        if neighbor_cells:
            path = self._bfs(pos, neighbor_cells, blocked)
            if path and len(path) >= 2:
                step_to = path[1]
                self._move_specific(npc, step_to)
                return

        # 4) Else greedy single-step toward the object if possible
        best = None
        bestd = 1e9
        for dr, dc in dirs:
            cand = (pos[0] + dr, pos[1] + dc)
            if self.is_empty_cell(cand):
                d = self.manhattan(cand, obj_pos)
                if d < bestd:
                    bestd = d
                    best = cand
        if best is not None:
            self._move_specific(npc, best)
            return
        # 5) Otherwise, wait (no-op)

    def _move_specific(self, actor: str, dest: Coord) -> bool:
        # Move actor directly to dest if legal empty neighbor cell (without pushing)
        if actor == 'player':
            pos = self.player_pos
        elif actor == 'red':
            pos = self.red_pos
        else:
            pos = self.blue_pos
        if dest not in self._neighbors4(pos):
            return False
        if not self.is_empty_cell(dest):
            return False
        if actor == 'player':
            self.player_pos = dest
        elif actor == 'red':
            self.red_pos = dest
        else:
            self.blue_pos = dest
        return True

    def _check_goal_and_maybe_end(self, stage: str):
        # Determine if any NPC's goal is satisfied; stage helps for debugging ordering ("after_player", "after_red", "after_blue")
        # Evaluate red first then blue at each subturn as per order
        red_obj, red_tid = self.red_goal
        blue_obj, blue_tid = self.blue_goal
        if self.objects[red_obj] == self.targets[red_tid]:
            self.done = True
            self.winner_npc = 'red'
            return
        if self.objects[blue_obj] == self.targets[blue_tid]:
            self.done = True
            self.winner_npc = 'blue'
            return

    def step(self, action) -> tuple[dict, float, bool, dict]:
        if self.done:
            return self._observation(), 0.0, True, {}
        # Only accept player's action strings
        actions = ['up', 'down', 'left', 'right', 'wait', 'support_red', 'support_blue']
        if action not in actions:
            action = 'wait'

        # Process support declaration (can be done once, before any goal completes)
        if action == 'support_red' and self.support is None:
            self.support = 'red'
        elif action == 'support_blue' and self.support is None:
            self.support = 'blue'
        else:
            # Movement/push
            dir_map = {
                'up': (-1, 0),
                'down': (1, 0),
                'left': (0, -1),
                'right': (0, 1),
                'wait': (0, 0)
            }
            if action in dir_map and action != 'wait':
                self._move_actor('player', dir_map[action])
            # wait is a no-op

        # After player's move
        self._check_goal_and_maybe_end(stage="after_player")
        reward = -0.01  # per-round time penalty

        if not self.done:
            # Red NPC turn
            self._npc_policy_step('red')
            self._check_goal_and_maybe_end(stage="after_red")

        if not self.done:
            # Blue NPC turn
            self._npc_policy_step('blue')
            self._check_goal_and_maybe_end(stage="after_blue")

        # Increment round count after full cycle
        self.step_count += 1

        # Step limit termination
        if not self.done and self.step_count >= self.max_rounds:
            self.done = True
            # No winner if time-out; treated as loss

        # Final reward on terminal
        if self.done:
            if self.winner_npc is None:
                # timeout or no completion
                reward += -1.0
            else:
                if self.support == self.winner_npc:
                    reward += 1.0
                else:
                    reward += -1.0

        return self._observation(), reward, self.done, {}

    def _observation(self) -> dict:
        # Public state (no hidden goals exposed)
        obs = {
            "grid_size": (self.h, self.w),
            "player": self.player_pos,
            "red": self.red_pos,
            "blue": self.blue_pos,
            "objects": {k: tuple(v) for k, v in self.objects.items()},
            "targets": {int(k): tuple(v) for k, v in self.targets.items()},
            "support": self.support,
            "round": self.step_count,
            "done": self.done
        }
        return obs

    def render(self, mode: str = "text") -> str:
        # ASCII grid: walls '#', targets '1','2','3', objects 'A','B','C', NPCs 'R','B', player 'P'
        grid = [['.' for _ in range(self.w)] for _ in range(self.h)]
        for (r, c) in self.walls:
            grid[r][c] = '#'
        for tid, (r, c) in self.targets.items():
            grid[r][c] = str(tid + 1)
        # Place objects (override targets visually)
        for lab, (r, c) in self.objects.items():
            grid[r][c] = lab
        # Place actors (override objects visually)
        pr, pc = self.player_pos
        grid[pr][pc] = 'P'
        rr, rc = self.red_pos
        grid[rr][rc] = 'R'
        br, bc = self.blue_pos
        grid[br][bc] = 'B'
        rows = [''.join(row) for row in grid]
        s = []
        s.append("Round: %d  Support: %s" % (self.step_count, self.support))
        s.extend(rows)
        s.append("Legend: #:wall  1-3:targets  A/B/C:objects  R/B:NPCs  P:player")
        return "\n".join(s)

    def valid_actions(self) -> list:
        # Legal action list for the player's current state
        if self.done:
            return []
        acts = ['wait']
        # Support actions
        if self.support is None:
            acts += ['support_red', 'support_blue']
        # Movement/push actions if possible
        dir_map = {
                'up': (-1, 0),
                'down': (1, 0),
                'left': (0, -1),
                'right': (0, 1)
        }
        for name, (dr, dc) in dir_map.items():
            dest = (self.player_pos[0] + dr, self.player_pos[1] + dc)
            # simple move
            if self.is_empty_cell(dest):
                acts.append(name)
                continue
            # push if moving into object and beyond empty
            if self.in_bounds(dest) and not self.is_wall(dest):
                for lab, opos in self.objects.items():
                    if opos == dest:
                        beyond = (dest[0] + dr, dest[1] + dc)
                        if self.is_empty_cell(beyond):
                            acts.append(name)
                        break
        return acts

    def is_won(self) -> bool:
        if not self.done:
            return False
        if self.winner_npc is None:
            return False
        return self.support == self.winner_npc

