import copy
from typing import List, Tuple, Dict, Any, Optional


class PrimitivBenchGame:
    """
    3-Phase Sequence Builder with Rule Suspension

    Rules:
    - R1 (from Phase 1): Strictly increasing. Each appended number must be > previous number.
    - R2 (from Phase 2): Parity alternation. Each appended number must have opposite parity to previous number.
    - R3 (from Phase 3): Running sum divisibility. After each append in phase 3, the total sum must be divisible by 3.

    Phase structure:
    - Phase 1: Build 2 elements under R1. No suspension available.
    - Phase 2: Extend by 2 elements under R2 + (R1 prior). Exactly one prior rule must be suspended this phase.
              (The only prior rule is R1, so R1 must be suspended in phase 2.)
    - Phase 3: Complete by 2 elements under R3 + (R1,R2 prior). Exactly one of {R1, R2} must be suspended.

    Suspension applies to validation at append time in that phase only.
    Earlier elements remain as previously accepted.

    Actions:
    - "append_<n>": append integer n in [1..9] if it satisfies currently active rules excluding the suspended one.
    - "choose_suspend_R1" / "choose_suspend_R2": choose the suspended prior rule for current phase (when required).
    - "undo": revert the last action (including suspension choices).

    Episode termination:
    - Win when all 6 elements appended validly across phases following the suspension protocol.
    - Lose if max_steps exceeded (done=True, is_won=False).

    Determinism: No RNG is used after reset; all legality is deterministic.

    Observation:
    Dict with keys:
      - phase (1,2,3 or 4 if completed)
      - sequence (list of ints)
      - phase_goal_remaining
      - current_suspension (str or None)
      - suspensions (dict {phase_index: "R1"/"R2"/None})
      - active_rules_now (list of rule names currently enforced)
      - can_undo (bool)
      - steps_taken (int)
      - done (bool)
    """

    def __init__(self):
        # Fixed configuration
        self.numbers_domain = list(range(1, 10))  # 1..9
        self.phase_lengths = [2, 2, 2]  # elements per phase
        self.max_steps = 100

        # Runtime state
        self.sequence: List[int] = []
        self.phase_index: int = 0  # 0,1,2; 3 means complete
        self.intra_phase_count: int = 0
        self.suspensions: Dict[int, Optional[str]] = {}  # phase_index -> "R1"/"R2"/None
        self.suspension_chosen_for_phase: bool = False
        self.done: bool = False
        self._won: bool = False
        self.steps_taken: int = 0

        # History for undo
        self._history: List[Dict[str, Any]] = []
        self._seed: Optional[int] = None

    # ============== Gym-like API ==============

    def reset(self, seed: int = None) -> dict:
        self._seed = seed
        self.sequence = []
        self.phase_index = 0
        self.intra_phase_count = 0
        self.suspensions = {0: None, 1: None, 2: None}
        self.suspension_chosen_for_phase = False  # phase 0 requires none
        self.done = False
        self._won = False
        self.steps_taken = 0
        self._history = []
        # No RNG usage; seed only stored
        return self._observation()

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        if self.done:
            return self._observation(), 0.0, True, {"reason": "episode_already_done"}

        self.steps_taken += 1

        # Save snapshot for undo
        self._push_history()

        reward = -0.01  # small step cost to encourage efficiency
        info = {}

        try:
            if isinstance(action, str) and action.startswith("append_"):
                n = int(action.split("_", 1)[1])
                self._append_number(n)
            elif action == "choose_suspend_R1":
                self._choose_suspension("R1")
            elif action == "choose_suspend_R2":
                self._choose_suspension("R2")
            elif action == "undo":
                self._undo_internal()
                # Undo does not change reward; keep step penalty
            else:
                # Invalid action: revert snapshot and penalize by not progressing (still pay step cost)
                self._pop_history()  # discard saved snapshot since we didn't mutate
                info["invalid_action"] = True
        except Exception as e:
            # On rule violation or illegal choice, revert to previous state
            self._restore_last_snapshot()
            info["exception"] = str(e)

        # Auto-advance when phase count reached
        self._advance_phase_if_needed()

        # Win condition
        if self.phase_index == 3 and not self.done:
            self.done = True
            self._won = True
            reward += 1.0  # winning bonus

        # Episode cap
        if self.steps_taken >= self.max_steps and not self.done:
            self.done = True
            self._won = False
            info["reason"] = "max_steps_reached"

        return self._observation(), reward, self.done, info

    def render(self, mode: str = "text") -> str:
        obs = self._observation()
        lines = []
        lines.append(f"Phase: {obs['phase']} (1..3; 4=complete)")
        lines.append(f"Sequence: {obs['sequence']}")
        lines.append(f"Remaining in this phase: {obs['phase_goal_remaining']}")
        lines.append(f"Active rules now: {obs['active_rules_now']}")
        lines.append(f"Current phase suspension: {obs['current_suspension']}")
        lines.append(f"Suspensions record: {obs['suspensions']}")
        lines.append(f"Can undo: {obs['can_undo']}")
        lines.append(f"Steps taken: {obs['steps_taken']}")
        lines.append(f"Done: {obs['done']} | Won: {self.is_won()}")
        return "\n".join(lines)

    def valid_actions(self) -> List[str]:
        if self.done:
            return []
        actions: List[str] = []

        # Suspension choices at start of phase (when required)
        if self._suspension_required_now() and not self.suspension_chosen_for_phase:
            for r in self._prior_rules_set():
                actions.append(f"choose_suspend_{r}")

            # No other actions until suspension chosen
            if self._can_undo():
                actions.append("undo")
            return actions

        # Append actions: all numbers that pass current active (non-suspended) rules
        for n in self.numbers_domain:
            if self._number_is_valid(n):
                actions.append(f"append_{n}")

        if self._can_undo():
            actions.append("undo")

        return actions

    def is_won(self) -> bool:
        return bool(self._won and self.done)

    # ============== Internal mechanics ==============

    def _append_number(self, n: int):
        if self.done:
            raise ValueError("Cannot append: episode done")
        if not (1 <= n <= 9):
            raise ValueError("n must be in 1..9")
        # If suspension is required now, must choose first
        if self._suspension_required_now() and not self.suspension_chosen_for_phase:
            raise ValueError("Must choose suspension before appending in this phase")

        if not self._number_is_valid(n):
            raise ValueError("Append violates active rules")

        self.sequence.append(n)
        self.intra_phase_count += 1

    def _number_is_valid(self, n: int) -> bool:
        # Gather active rules for this phase, excluding the suspended prior rule if applicable
        active_rules = self._active_rules_excluding_suspension()
        last = self.sequence[-1] if len(self.sequence) > 0 else None
        prior_sum = sum(self.sequence)

        # Evaluate rules
        for r in active_rules:
            if r == "R1":
                if last is not None and not (n > last):
                    return False
            elif r == "R2":
                if last is not None and not ((n % 2) != (last % 2)):
                    return False
            elif r == "R3":
                if ((prior_sum + n) % 3) != 0:
                    return False
            else:
                # Unknown rule (shouldn't happen)
                return False
        return True

    def _active_rules_excluding_suspension(self) -> List[str]:
        # Baseline active-by-phase:
        # Phase 0: R1
        # Phase 1: R1, R2
        # Phase 2: R1, R2, R3
        phase_active_map = {
            0: ["R1"],
            1: ["R1", "R2"],
            2: ["R1", "R2", "R3"],
        }
        rules = list(phase_active_map.get(self.phase_index, []))

        # Remove suspended prior rule for current phase
        suspended = self.suspensions.get(self.phase_index, None)
        if suspended in rules:
            # R3 is never suspendable per the rules we expose to the agent (we never set it as suspended)
            rules = [r for r in rules if r != suspended]
        return rules

    def _prior_rules_set(self) -> List[str]:
        # Prior rules exclude the newly introduced rule in this phase
        # Phase 1 (index 0): no prior rules
        # Phase 2 (index 1): prior={R1}
        # Phase 3 (index 2): prior={R1,R2}
        if self.phase_index == 0:
            return []
        elif self.phase_index == 1:
            return ["R1"]
        elif self.phase_index == 2:
            return ["R1", "R2"]
        else:
            return []

    def _suspension_required_now(self) -> bool:
        # Suspension is required at phases with at least one prior rule
        # i.e., phase_index 1 and 2 (phases 2 and 3)
        return self.phase_index in (1, 2)

    def _choose_suspension(self, rule: str):
        if not self._suspension_required_now():
            raise ValueError("Suspension not required in this phase")
        if self.suspension_chosen_for_phase:
            raise ValueError("Suspension already chosen for this phase")
        if rule not in self._prior_rules_set():
            raise ValueError("Can only suspend a prior rule in this phase")
        self.suspensions[self.phase_index] = rule
        self.suspension_chosen_for_phase = True

    def _advance_phase_if_needed(self):
        # If finished elements for the phase, move to next
        if self.phase_index >= 3:
            return
        if self.intra_phase_count >= self.phase_lengths[self.phase_index]:
            self.phase_index += 1
            self.intra_phase_count = 0
            self.suspension_chosen_for_phase = False
            # Initialize suspension placeholder for next phase if not present
            if self.phase_index < 3 and self.phase_index not in self.suspensions:
                self.suspensions[self.phase_index] = None

    def _can_undo(self) -> bool:
        return len(self._history) > 0

    def _push_history(self):
        snap = dict(
            sequence=list(self.sequence),
            phase_index=self.phase_index,
            intra_phase_count=self.intra_phase_count,
            suspensions=copy.deepcopy(self.suspensions),
            suspension_chosen_for_phase=self.suspension_chosen_for_phase,
            done=self.done,
            _won=self._won,
            steps_taken=self.steps_taken,
        )
        self._history.append(snap)

    def _restore_last_snapshot(self):
        if not self._history:
            return
        snap = self._history.pop()
        self.sequence = list(snap["sequence"])
        self.phase_index = snap["phase_index"]
        self.intra_phase_count = snap["intra_phase_count"]
        self.suspensions = copy.deepcopy(snap["suspensions"])
        self.suspension_chosen_for_phase = snap["suspension_chosen_for_phase"]
        self.done = snap["done"]
        self._won = snap["_won"]
        self.steps_taken = snap["steps_taken"]

    def _undo_internal(self):
        if not self._history:
            raise ValueError("No history to undo")
        self._restore_last_snapshot()

    def _pop_history(self):
        if self._history:
            self._history.pop()

    def _observation(self) -> dict:
        return {
            "phase": self.phase_index + 1 if self.phase_index < 3 else 4,
            "sequence": list(self.sequence),
            "phase_goal_remaining": (self.phase_lengths[self.phase_index] - self.intra_phase_count) if self.phase_index < 3 else 0,
            "current_suspension": self.suspensions.get(self.phase_index, None) if self.phase_index < 3 else None,
            "suspensions": copy.deepcopy(self.suspensions),
            "active_rules_now": self._active_rules_excluding_suspension() if self.phase_index < 3 else [],
            "can_undo": self._can_undo(),
            "steps_taken": self.steps_taken,
            "done": self.done,
        }


if __name__ == "__main__":
    # Simple manual run
    env = PrimitivBenchGame()
    obs = env.reset(seed=0)
    print(env.render())
    print("Valid actions:", env.valid_actions())
