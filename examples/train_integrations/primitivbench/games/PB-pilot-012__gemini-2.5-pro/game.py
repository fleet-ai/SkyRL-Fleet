import numpy as np
from typing import Optional, List, Tuple, Dict, Any

class PrimitivBenchGame:
    """
    Cipher Cell is a 3-round deduction game where the player must identify a traitor
    among four NPCs. The traitor's identity is unmasked by progressively decoding
    encrypted messages.

    Primitives Targeted:
    - A3_agency: The player must reason about the intentions of NPCs. Honest NPCs
      truthfully report encoded information, while the single traitor consistently
      reports encoded misinformation to mislead the player. The player must model
      this deceptive agency to isolate the traitor.
    - B1_perception_transform: All NPC reports are presented as ciphertext. The raw
      observations are meaningless. The player must deduce the correct cipher keys
      and apply a decoding transformation to convert the ciphertext into
      meaningful evidence about NPC behavior.
    - B4_multi_step_composition: Winning requires a sequence of dependent reasoning
      steps. (1) Deduce key1 from Round 1 data. (2) Calculate key2, which is a
      function of key1. (3) Use both key1 and key2 to decode a final, compositely
      encrypted clue in Round 3. Success in step 3 is impossible without the
      correct outputs from steps 1 and 2.
    """
    _ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    _ALPHABET_SIZE = len(_ALPHABET)
    _CHAR_TO_INT = {c: i for i, c in enumerate(_ALPHABET)}
    _INT_TO_CHAR = {i: c for i, c in enumerate(_ALPHABET)}
    NUM_NPCS = 4

    def __init__(self):
        self.rng = np.random.default_rng()
        self.traitor_idx = 0
        self.key1 = 0
        self.key2 = 0
        self.truth1 = ''
        self.truth2 = ''
        self.lie1 = ''
        self.lie2 = ''
        self.final_clue_char = ''
        self.round = 0
        self.done = True
        self.won = False

    def _encode_char(self, char_val: str, key: int) -> str:
        """Encodes a single character using a Caesar cipher."""
        if char_val not in self._CHAR_TO_INT:
            return char_val
        val = self._CHAR_TO_INT[char_val]
        encoded_val = (val + key) % self._ALPHABET_SIZE
        return self._INT_TO_CHAR[encoded_val]

    def _decode_char(self, char_val: str, key: int) -> str:
        """Decodes a single character from a Caesar cipher."""
        if char_val not in self._CHAR_TO_INT:
            return char_val
        val = self._CHAR_TO_INT[char_val]
        decoded_val = (val - key + self._ALPHABET_SIZE) % self._ALPHABET_SIZE
        return self._INT_TO_CHAR[decoded_val]
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resets the game to the initial state for Round 1."""
        self.rng = np.random.default_rng(seed)
        self.traitor_idx = self.rng.choice(range(self.NUM_NPCS))
        
        # Key generation with dependency
        # key2 is dependent on key1, creating a multi-step composition requirement.
        # We must avoid key1=7, as (7*3+5)%26 = 0, a null cipher.
        valid_key1s = [k for k in range(1, self._ALPHABET_SIZE) if k != 7]
        self.key1 = self.rng.choice(valid_key1s)
        self.key2 = (self.key1 * 3 + 5) % self._ALPHABET_SIZE

        # Set ground truths and lies for the rounds
        self.truth1 = self._INT_TO_CHAR[self.rng.choice(self._ALPHABET_SIZE)]
        self.truth2 = self._INT_TO_CHAR[self.rng.choice(self._ALPHABET_SIZE)]
        
        possible_lies1 = [c for c in self._ALPHABET if c != self.truth1]
        self.lie1 = self.rng.choice(possible_lies1)
        
        possible_lies2 = [c for c in self._ALPHABET if c != self.truth2]
        self.lie2 = self.rng.choice(possible_lies2)
        
        # Final clue is the traitor's index, 0-3, mapped to A-D
        self.final_clue_char = self._INT_TO_CHAR[self.traitor_idx]

        self.round = 1
        self.done = False
        self.won = False
        
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        """Constructs the observation dictionary for the current state."""
        obs = {
            "round": self.round,
            "player_clues": {},
            "npc_reports": {},
            "round_3_locked_file": None
        }

        # Round 1 data
        if self.round >= 1:
            obs["player_clues"]["round_1_truth"] = self.truth1
            reports_r1 = {}
            for i in range(self.NUM_NPCS):
                msg_content = self.lie1 if i == self.traitor_idx else self.truth1
                reports_r1[f"agent_{i}"] = self._encode_char(msg_content, self.key1)
            obs["npc_reports"]["round_1"] = reports_r1

        # Round 2 data
        if self.round >= 2:
            obs["player_clues"]["round_2_truth"] = self.truth2
            reports_r2 = {}
            for i in range(self.NUM_NPCS):
                msg_content = self.lie2 if i == self.traitor_idx else self.truth2
                reports_r2[f"agent_{i}"] = self._encode_char(msg_content, self.key2)
            obs["npc_reports"]["round_2"] = reports_r2
            
        # Round 3 data
        if self.round >= 3:
            # The final clue is compositely encrypted
            encoded_1 = self._encode_char(self.final_clue_char, self.key1)
            encoded_2 = self._encode_char(encoded_1, self.key2)
            obs["round_3_locked_file"] = encoded_2
            
        return obs

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Processes an action and returns the new state, reward, and done flag."""
        if self.done:
            raise ValueError("Game is over. Call reset() to start a new game.")
        
        if not action in self.valid_actions():
            raise ValueError(f"Invalid action '{action}' for the current state.")

        reward = 0.0

        if action.startswith("ACCUSE_"):
            parts = action.split('_')
            accused_idx = int(parts[1])
            self.done = True
            if accused_idx == self.traitor_idx:
                self.won = True
                reward = 1.0
        elif action == "PROCEED":
            if self.round < 3:
                self.round += 1
            else:
                # Should not happen if action validation is correct
                self.done = True 
        
        obs = self._get_observation()
        info = {'is_won': self.won}
        
        return obs, reward, self.done, info

    def render(self, mode: str = "text") -> str:
        """Returns a human-readable string representation of the state."""
        import json
        return json.dumps(self._get_observation(), indent=2)

    def valid_actions(self) -> List[str]:
        """Returns the list of legal actions in the current state."""
        if self.done:
            return []
        
        actions = [f"ACCUSE_{i}" for i in range(self.NUM_NPCS)]
        if self.round < 3:
            actions.append("PROCEED")
            
        return actions

    def is_won(self) -> bool:
        """Returns true if the player has won the game."""
        return self.won
