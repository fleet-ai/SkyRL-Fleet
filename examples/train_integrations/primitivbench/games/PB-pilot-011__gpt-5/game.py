import numpy as np
from typing import List, Tuple, Dict, Any

class PrimitivBenchGame:
    """
    Three-phase crafting with six persistent ingredients.

    Ingredients (fixed, deterministic):
      id: 0..5
      color: ['R','G','B','R','G','R']   # exactly one Blue at id=2
      weight: [1,2,3,4,5,6]

    Rules over the aggregate chosen set S (ids) at each phase:
      R1(S): exactly one Blue appears in S
      R2(S): sum of weights in S is even
      R3(S): sum of ids in S is divisible by 3

    Phases:
      - Phase 1: choose two distinct unused ingredients (pair) -> A must satisfy R1(S2)
      - Phase 2: add one unused ingredient -> B must satisfy R1(S3) AND R2(S3)
      - Phase 3: add one unused ingredient -> Product must satisfy R1(S4) AND R2(S4) AND R3(S4)

    Win on successful Phase 3 output. Episode terminates on win or step cap.
    """

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self._rng = np.random.RandomState(0)
        self.reset()

    # Public API
    def reset(self, seed: int = None) -> dict:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        # Fixed ingredient mapping (deterministic)
        self.ingredients = [
            {"id": 0, "color": "R", "weight": 1},
            {"id": 1, "color": "G", "weight": 2},
            {"id": 2, "color": "B", "weight": 3},  # unique blue
            {"id": 3, "color": "R", "weight": 4},
            {"id": 4, "color": "G", "weight": 5},
            {"id": 5, "color": "R", "weight": 6},
        ]
        self.color_map = {ing["id"]: ing["color"] for ing in self.ingredients}
        self.weight_map = {ing["id"]: ing["weight"] for ing in self.ingredients}
        self.blue_ids = {ing["id"] for ing in self.ingredients if ing["color"] == "B"}

        self.phase = 1  # 1 -> expect pair; 2 -> add third; 3 -> add fourth (final)
        self.current_set: List[int] = []
        self.used_ids: set = set()
        self.steps = 0
        self.done = False
        self.won = False
        self.last_info: Dict[str, Any] = {}

        return self._obs()

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """
        action formats:
          ('pair', i, j)  when phase == 1, i != j, both unused
          ('add', k)      when phase in {2,3}, k unused
        Any structurally invalid action: small penalty, no state change.
        Any rule-invalid attempt: small penalty, no state change.
        """
        if self.done:
            return self._obs(), 0.0, True, {"reason": "Episode already done."}

        self.steps += 1
        reward = 0.0
        info: Dict[str, Any] = {}

        # Structural validation
        if not isinstance(action, tuple) or len(action) < 2:
            return self._post_step_invalid("Malformed action.", -0.05)

        act_type = action[0]
        if act_type == 'pair':
            if self.phase != 1 or len(action) != 3:
                return self._post_step_invalid("Pair action not allowed in current phase or wrong arity.", -0.05)
            i, j = action[1], action[2]
            if not self._ids_exist([i, j]) or i == j or (i in self.used_ids) or (j in self.used_ids):
                return self._post_step_invalid("Invalid ids for pair.", -0.05)
            tentative = [i, j]
            if self._rule_R1(tentative):
                # commit
                self.current_set = tentative[:]
                self.used_ids = set(tentative)
                self.phase = 2
                reward += 0.2
                info["progress"] = "Phase 1 success"
            else:
                return self._post_step_invalid("R1 not satisfied for pair.", -0.05)

        elif act_type == 'add':
            if self.phase not in (2, 3) or len(action) != 2:
                return self._post_step_invalid("Add action not allowed in current phase or wrong arity.", -0.05)
            k = action[1]
            if not self._ids_exist([k]) or (k in self.used_ids):
                return self._post_step_invalid("Invalid id for add.", -0.05)
            tentative = self.current_set + [k]
            if self.phase == 2:
                if self._rule_R1(tentative) and self._rule_R2(tentative):
                    # commit
                    self.current_set = tentative
                    self.used_ids.add(k)
                    self.phase = 3
                    reward += 0.3
                    info["progress"] = "Phase 2 success"
                else:
                    return self._post_step_invalid("R1 and/or R2 not satisfied at Phase 2.", -0.05)
            else:  # phase == 3
                r1 = self._rule_R1(tentative)
                r2 = self._rule_R2(tentative)
                r3 = self._rule_R3(tentative)
                if r1 and r2 and r3:
                    # commit and win
                    self.current_set = tentative
                    self.used_ids.add(k)
                    self.phase = 4
                    reward += 1.0
                    self.done = True
                    self.won = True
                    info["progress"] = "Win: all rules satisfied"
                else:
                    return self._post_step_invalid("R1 and/or R2 and/or R3 not satisfied at Phase 3.", -0.05)
        else:
            return self._post_step_invalid("Unknown action type.", -0.05)

        # Step cap termination
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info["terminated"] = "step_cap"

        self.last_info = info
        return self._obs(), reward, self.done, info

    def render(self, mode: str = "text") -> str:
        lines = []
        lines.append("Three-Phase Crafting")
        lines.append(f"Phase: {self.phase} (1=pair, 2=add third, 3=add fourth)")
        lines.append(f"Steps: {self.steps} | Done: {self.done} | Won: {self.won}")
        lines.append("Ingredients:")
        for ing in self.ingredients:
            mark = "USED" if ing["id"] in self.used_ids else "free"
            lines.append(f"  id={ing['id']} color={ing['color']} weight={ing['weight']} [{mark}]")
        lines.append(f"Current selection S={self.current_set}")
        lines.append("Rules:")
        lines.append("  R1: exactly one Blue in S")
        lines.append("  R2: sum of weights in S is even")
        lines.append("  R3: sum of ids in S divisible by 3")
        if self.last_info:
            lines.append(f"Last info: {self.last_info}")
        return "\n".join(lines)

    def valid_actions(self) -> List[Tuple]:
        if self.done:
            return []
        acts = []
        remaining = [ing["id"] for ing in self.ingredients if ing["id"] not in self.used_ids]
        if self.phase == 1:
            for i_idx in range(len(remaining)):
                for j_idx in range(i_idx + 1, len(remaining)):
                    i = remaining[i_idx]
                    j = remaining[j_idx]
                    acts.append(('pair', i, j))
        elif self.phase in (2, 3):
            for k in remaining:
                acts.append(('add', k))
        return acts

    def is_won(self) -> bool:
        return self.won

    # Helpers
    def _obs(self) -> dict:
        return {
            "phase": self.phase,
            "steps": self.steps,
            "done": self.done,
            "won": self.won,
            "current_set": list(self.current_set),
            "used_ids": sorted(list(self.used_ids)),
            "remaining_ids": [ing["id"] for ing in self.ingredients if ing["id"] not in self.used_ids],
            "ingredients": [dict(ing) for ing in self.ingredients],
            "rules_summary": {
                "R1": "exactly one Blue in S",
                "R2": "sum of weights in S is even",
                "R3": "sum of ids in S divisible by 3"
            }
        }

    def _ids_exist(self, ids: List[int]) -> bool:
        for x in ids:
            if x not in self.color_map:
                return False
        return True

    def _rule_R1(self, S: List[int]) -> bool:
        # Exactly one blue id in S
        cnt_blue = sum(1 for x in S if x in self.blue_ids)
        return cnt_blue == 1

    def _rule_R2(self, S: List[int]) -> bool:
        total_w = sum(self.weight_map[x] for x in S)
        return (total_w % 2) == 0

    def _rule_R3(self, S: List[int]) -> bool:
        return (sum(S) % 3) == 0

    def _post_step_invalid(self, reason: str, penalty: float):
        info = {"invalid": reason}
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info["terminated"] = "step_cap"
        self.last_info = info
        return self._obs(), penalty, self.done, info


if __name__ == "__main__":
    # Simple manual sanity check
    env = PrimitivBenchGame()
    print(env.render())
    s, r, d, i = env.step(('pair', 2, 1))
    print("\nAfter action ('pair',2,1): reward", r, "done", d, i)
    print(env.render())
    s, r, d, i = env.step(('add', 4))
    print("\nAfter action ('add',4): reward", r, "done", d, i)
    print(env.render())
    s, r, d, i = env.step(('add', 5))
    print("\nAfter action ('add',5): reward", r, "done", d, i)
    print(env.render())
