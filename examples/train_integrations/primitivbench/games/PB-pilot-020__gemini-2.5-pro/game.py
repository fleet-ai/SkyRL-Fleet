import numpy as np
import math

class PrimitivBenchGame:
    """
    Phase Shift Sequencer is a 3-phase logical puzzle about building a sequence of numbers.

    - State: Consists of the current sequence, the game phase (1-3), and a log of
      suspended rules.
    - Actions: Are tuples of ('action_type', value).
        - ('suspend', rule_name): In phases 2 & 3, the player must choose to suspend
          a prior rule ('R1', 'R2') or 'None'. This is a meta-reasoning step.
        - ('add', number): Adds a number to the sequence, subject to the active rules.
    - Rules:
        - R1: The next number must be strictly greater than the previous one.
        - R2: The sum of the next number and the previous one must be a prime number.
        - R3: The next number must not share any digits with the previous one.
    - Phases:
        - Phase 1 (Sequence length 1-3): Only R1 is active.
        - Phase 2 (Sequence length 3-5): R1 and R2 are active. Player must suspend one or none.
        - Phase 3 (Sequence length 5-7): R1, R2, and R3 are active. Player must suspend one from {R1, R2} or none.
    - Win Condition: Successfully build a sequence of length 7.
    - Loss Condition: Make an invalid move or get into a state with no valid moves.
    """

    def __init__(self, max_number=25):
        self.max_number = max_number
        self.phase_lengths = {1: 3, 2: 5, 3: 7}
        self.rules = {
            'R1': self._is_r1_satisfied,
            'R2': self._is_r2_satisfied,
            'R3': self._is_r3_satisfied,
        }
        self.phase_rules = {
            1: {'R1'},
            2: {'R1', 'R2'},
            3: {'R1', 'R2', 'R3'},
        }

    def reset(self, seed: int = None) -> dict:
        """Resets the game to a new initial state."""
        self.rng = np.random.default_rng(seed)
        initial_number = self.rng.integers(1, 6)
        
        self.sequence = [initial_number]
        self.phase = 1
        self.suspension_choice_made_for_phase = True # No choice in phase 1
        self.suspended_rule_for_current_phase = None
        self.suspension_log = {2: None, 3: None}
        
        self.done = False
        self.reward = 0.0
        self.info = {'status': 'in_progress'}
        
        self._update_phase()
        
        return self._get_state()

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """Executes one time step within the environment."""
        if self.done:
            return self._get_state(), 0.0, True, {'status': 'game_over'}

        valid_actions_list = self.valid_actions()
        if action not in valid_actions_list:
            self.done = True
            self.reward = -1.0
            self.info['status'] = f'loss: invalid action {action}'
            return self._get_state(), self.reward, self.done, self.info
        
        action_type, value = action
        
        if action_type == 'suspend':
            self.suspended_rule_for_current_phase = value if value != 'None' else None
            self.suspension_log[self.phase] = self.suspended_rule_for_current_phase
            self.suspension_choice_made_for_phase = True
            
            # Check for dead ends immediately after suspension choice
            if not self.valid_actions():
                self.done = True
                self.reward = -1.0
                self.info['status'] = 'loss: dead end after suspension'
                return self._get_state(), self.reward, self.done, self.info

            return self._get_state(), 0.0, self.done, self.info

        elif action_type == 'add':
            self.sequence.append(value)
            self._update_phase()

            if self.is_won():
                self.done = True
                self.reward = 1.0
                self.info['status'] = 'win'
            else:
                 # Small reward for making progress
                self.reward = 0.1

            return self._get_state(), self.reward, self.done, self.info

        # Should not be reached
        return self._get_state(), -1.0, True, {'status': 'error: unknown action type'}

    def valid_actions(self) -> list:
        """Returns a list of valid actions for the current state."""
        if self.done:
            return []
        
        if not self.suspension_choice_made_for_phase:
            if self.phase == 2:
                return [('suspend', 'R1'), ('suspend', 'None')]
            if self.phase == 3:
                return [('suspend', 'R1'), ('suspend', 'R2'), ('suspend', 'None')]
        
        # If suspension choice is made (or not needed), list valid 'add' actions
        active_rules = self.phase_rules[self.phase].copy()
        if self.suspended_rule_for_current_phase:
            active_rules.remove(self.suspended_rule_for_current_phase)
            
        n_prev = self.sequence[-1]
        
        actions = []
        for n_new in range(1, self.max_number + 1):
            if n_new == n_prev: continue
            
            is_valid = True
            for rule_id in active_rules:
                if not self.rules[rule_id](n_prev, n_new):
                    is_valid = False
                    break
            if is_valid:
                actions.append(('add', n_new))
                
        return actions

    def render(self, mode: str = "text") -> str:
        """Returns a human-readable representation of the state."""
        active_rules = self.phase_rules[self.phase].copy()
        if self.suspended_rule_for_current_phase:
            active_rules.remove(self.suspended_rule_for_current_phase)

        return (
            f"Phase: {self.phase}\n"
            f"Sequence: {self.sequence}\n"
            f"Suspension Log: {self.suspension_log}\n"
            f"Active Rules for next step: {sorted(list(active_rules))}\n"
            f"Status: {self.info.get('status', 'in_progress')}"
        )

    def is_won(self) -> bool:
        """Checks if the game has been won."""
        return len(self.sequence) == self.phase_lengths[3]

    def _get_state(self) -> dict:
        """Returns the current state as a dictionary."""
        return {
            "sequence": self.sequence,
            "phase": self.phase,
            "suspension_log": self.suspension_log,
            "suspension_choice_made": self.suspension_choice_made_for_phase,
            "active_rules": sorted(list(self._get_active_ruleset()))
        }
        
    def _get_active_ruleset(self):
        """Helper to get the current set of active rules."""
        ruleset = self.phase_rules[self.phase].copy()
        if self.suspended_rule_for_current_phase:
            ruleset.remove(self.suspended_rule_for_current_phase)
        return ruleset

    def _update_phase(self):
        """Updates the game phase based on sequence length."""
        current_len = len(self.sequence)
        if self.phase == 1 and current_len >= self.phase_lengths[1]:
            self.phase = 2
            self.suspension_choice_made_for_phase = False
            self.suspended_rule_for_current_phase = None
        elif self.phase == 2 and current_len >= self.phase_lengths[2]:
            self.phase = 3
            self.suspension_choice_made_for_phase = False
            self.suspended_rule_for_current_phase = None

    def _is_prime(self, n):
        if n < 2: return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0: return False
        return True

    def _is_r1_satisfied(self, n_prev, n_new):
        return n_new > n_prev

    def _is_r2_satisfied(self, n_prev, n_new):
        return self._is_prime(n_prev + n_new)

    def _is_r3_satisfied(self, n_prev, n_new):
        return set(str(n_prev)).isdisjoint(set(str(n_new)))