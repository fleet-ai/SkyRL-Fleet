from typing import Tuple, List, Dict, Optional
import random

class PrimitivBenchGame:
    """
    Hide-and-Seek Rotated View

    - World: 5x5 grid. Static walls, a goal corner, one seeker at center, a portal pair.
    - Player objective: reach the goal (0,0 in world coords) without being seen by the seeker's
      scanning cone, within max_steps.
    - Seeker: stationary at (2,2), scan cone range=3, rotates facing each turn in cycle [N,E,S,W].
      Detection happens AFTER the player acts, using the next facing in that cycle.
    - Player's observation: a 90° clockwise-rotated rendering of the current world state;
      the seeker_facing is reported in WORLD terms, not rotated.
    - Portal: stepping onto one portal teleports the player to the paired portal.
    """

    def __init__(self):
        # World size
        self.w = 5
        self.h = 5

        # Static world features (world coordinates: x to the right, y down)
        # Base map uses:
        # '.' empty, '#' wall, 'G' goal, 'A' portal endpoint, 'B' portal endpoint
        self.base_map = [['.' for _ in range(self.w)] for _ in range(self.h)]
        # Place goal at top-left
        self.goal_pos = (0, 0)
        self._set_cell(self.goal_pos, 'G')
        # Place some walls (do not block the optimal path)
        for wall in [(1, 3), (1, 2), (3, 3)]:
            self._set_cell(wall, '#')
        # Portals
        self.portal_A = (0, 3)
        self.portal_B = (4, 1)
        self._set_cell(self.portal_A, 'A')
        self._set_cell(self.portal_B, 'B')
        self.portal_pairs = {self.portal_A: self.portal_B, self.portal_B: self.portal_A}

        # Entities
        self.player_start = (0, 4)
        self.seeker_pos = (2, 2)

        # Seeker scan properties
        self.scan_cycle = ['N', 'E', 'S', 'W']  # rotation order
        self.vision_range = 3

        # Episode control
        self.max_steps = 40

        # Step penalty and terminal rewards
        self.step_penalty = -0.01
        self.win_reward = 1.0
        self.lose_penalty = -1.0
        self.timeout_penalty = -0.5

        # Runtime state
        self.rng = random.Random()
        self._seed = None
        self.reset_called = False
        self.reset()

    def reset(self, seed: int = None) -> dict:
        if seed is not None:
            self._seed = seed
            self.rng.seed(seed)
        # Reset dynamic state
        self.player_pos = tuple(self.player_start)
        self.turn_count = 0  # determines current seeker facing as scan_cycle[turn_count % 4]
        self.steps = 0
        self._done = False
        self._won = False
        self.reset_called = True
        return self._obs()

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """
        Step order:
        1) Validate and apply player action (move/portal).
        2) Increment turn_count.
        3) Seeker rotates to next facing; detect with that facing.
        4) If not caught, check if player on goal -> win.
        5) Check step limit.
        Reward: step penalty each step, plus terminal bonuses/penalties.
        """
        assert self.reset_called, "Call reset() before step()."
        if self._done:
            return self._obs(), 0.0, True, {"reason": "episode_already_done"}

        # Apply action if valid
        legal = self.valid_actions()
        if action not in legal:
            # Determinism: if invalid action, treat as 'wait'
            action = 'wait'

        # Move
        if action != 'wait':
            dx, dy = self._action_to_delta(action)
            newx = self.player_pos[0] + dx
            newy = self.player_pos[1] + dy
            newpos = (newx, newy)
            # movement into seeker position is disallowed (treated as invalid -> wait)
            if self._in_bounds(newpos) and self._cell(newpos) != '#' and newpos != self.seeker_pos:
                self.player_pos = newpos
                # Portal teleport if landed on a portal tile
                if self.player_pos in self.portal_pairs:
                    self.player_pos = self.portal_pairs[self.player_pos]

        self.steps += 1
        # Advance seeker cycle
        self.turn_count += 1

        # Detection with new facing
        facing = self.scan_cycle[self.turn_count % len(self.scan_cycle)]
        seen = self._is_in_cone(self.player_pos, self.seeker_pos, facing, self.vision_range)

        reward = self.step_penalty
        info = {"seeker_facing_used_for_detection": facing}

        if seen:
            self._done = True
            self._won = False
            reward += self.lose_penalty
            info["reason"] = "caught_in_cone"
        else:
            # Check goal
            if self.player_pos == self.goal_pos:
                self._done = True
                self._won = True
                reward += self.win_reward
                info["reason"] = "reached_goal"

        # Step limit
        if not self._done and self.steps >= self.max_steps:
            self._done = True
            self._won = False
            reward += self.timeout_penalty
            info["reason"] = "timeout"

        return self._obs(), reward, self._done, info

    def render(self, mode: str = "text") -> str:
        """
        Returns a human-readable view.
        Shows:
        - World (true) grid
        - Player-view (90° CW rotated) grid
        """
        world = self._compose_world_grid()
        rotated = self._rotate_grid_for_player(world)
        s = []
        s.append(f"Steps: {self.steps}/{self.max_steps}  Turn: {self.turn_count}")
        s.append(f"Seeker facing (current, pre-move): {self.scan_cycle[self.turn_count % 4]}")
        s.append("World:")
        s.extend([''.join(row) for row in world])
        s.append("Player-view (90° CW):")
        s.extend([''.join(row) for row in rotated])
        return "\n".join(s)

    def valid_actions(self) -> List[str]:
        if self._done:
            return []
        actions = ['wait']
        x, y = self.player_pos
        for name, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)), ('left', (-1, 0)), ('right', (1, 0))]:
            nx, ny = x + dx, y + dy
            npos = (nx, ny)
            if self._in_bounds(npos) and self._cell(npos) != '#' and npos != self.seeker_pos:
                actions.append(name)
        return actions

    def is_won(self) -> bool:
        return bool(self._won)

    # ----------------- Internal helpers -----------------

    def _set_cell(self, pos: Tuple[int, int], ch: str):
        x, y = pos
        self.base_map[y][x] = ch

    def _cell(self, pos: Tuple[int, int]) -> str:
        x, y = pos
        return self.base_map[y][x]

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.w and 0 <= y < self.h

    def _action_to_delta(self, action: str) -> Tuple[int, int]:
        mapping = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
        }
        return mapping.get(action, (0, 0))

    def _is_in_cone(self, target: Tuple[int, int], origin: Tuple[int, int], facing: str, rng: int) -> bool:
        """
        Triangle-like wedge extending from origin in 'facing' up to distance rng.
        No occlusion by walls.
        Facing:
          - 'N': rows above, centered on (ox, oy - d), width 2*(d-1)+1
          - 'S': rows below, centered on (ox, oy + d)
          - 'E': cols right, centered on (ox + d, oy)
          - 'W': cols left, centered on (ox - d, oy)
        """
        ox, oy = origin
        tx, ty = target
        cells = set()
        for d in range(1, rng + 1):
            if facing == 'N':
                y = oy - d
                for dx in range(-(d - 1), d):
                    x = ox + dx
                    if 0 <= x < self.w and 0 <= y < self.h:
                        cells.add((x, y))
            elif facing == 'S':
                y = oy + d
                for dx in range(-(d - 1), d):
                    x = ox + dx
                    if 0 <= x < self.w and 0 <= y < self.h:
                        cells.add((x, y))
            elif facing == 'E':
                x = ox + d
                for dy in range(-(d - 1), d):
                    y = oy + dy
                    if 0 <= x < self.w and 0 <= y < self.h:
                        cells.add((x, y))
            elif facing == 'W':
                x = ox - d
                for dy in range(-(d - 1), d):
                    y = oy + dy
                    if 0 <= x < self.w and 0 <= y < self.h:
                        cells.add((x, y))
        return (tx, ty) in cells

    def _compose_world_grid(self) -> List[List[str]]:
        # Start from base map
        world = [row[:] for row in self.base_map]
        # Overlay seeker
        sx, sy = self.seeker_pos
        world[sy][sx] = 'S'
        # Overlay player (dominates underlying tile for display)
        px, py = self.player_pos
        world[py][px] = 'P'
        # Ensure goal shows if player not on it
        gx, gy = self.goal_pos
        if (px, py) != (gx, gy):
            # Preserve goal mark if not overwritten by seeker or portal marker
            if world[gy][gx] == '.':
                world[gy][gx] = 'G'
        return world

    def _rotate_grid_for_player(self, world: List[List[str]]) -> List[List[str]]:
        """
        Rotate 90 degrees clockwise: (x, y) -> (y, H-1-x)
        """
        H = len(world)
        W = len(world[0]) if H > 0 else 0
        rotated = [[None for _ in range(H)] for _ in range(W)]
        for y in range(H):
            for x in range(W):
                rx, ry = y, H - 1 - x
                rotated[ry][rx] = world[y][x]
        # Return with same orientation as world printing (rows)
        return rotated

    def _obs(self) -> dict:
        world = self._compose_world_grid()
        rotated = self._rotate_grid_for_player(world)
        return {
            "player_view": [''.join(row) for row in rotated],
            "view_rotation_deg": 90,
            "seeker_facing_world_current": self.scan_cycle[self.turn_count % len(self.scan_cycle)],
            "step": self.steps,
            "max_steps": self.max_steps,
            "grid_size": (self.w, self.h)
        }


if __name__ == "__main__":
    # Simple manual sanity check
    env = PrimitivBenchGame()
    print(env.render())
    for a in ["wait", "up", "left", "wait", "left", "up", "left", "left"]:
        s, r, d, info = env.step(a)
        print("\nAction:", a, "Reward:", r, "Done:", d, "Info:", info)
        print(env.render())
        if d:
            break
