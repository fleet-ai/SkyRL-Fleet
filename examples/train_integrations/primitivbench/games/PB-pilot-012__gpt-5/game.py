import numpy as np
from typing import List, Tuple, Dict, Any

# PrimitivBenchGame: 3-round social deduction with cipher transforms and adaptive liar
#
# Round 1:
# - Truthful NPCs emit 3-letter strings whose invariant differences are [+, +2, +5] modulo 26.
# - The liar uses [+, +3, +5]. All are Caesar-shifted by s1. You can form an initial suspicion
#   from raw messages by checking the invariant difference between letters 0 and 1.
#
# Round 2:
# - All emit 3-letter strings (first two letters carry a pair (x, y)). For truthful NPCs,
#   (y - x) mod 26 == s1 (the Round 1 shift). The liar's pair encodes (s1 ± 1), and the sign
#   depends on whether the player accused the liar in Round 1 (agency).
# - The entire string is then Caesar-shifted by s2, but the difference (y - x) is invariant to shift,
#   so you can deduce s1 by majority and retro-decode Round 1.
#
# Round 3:
# - Base triads again with invariant (char1 - char0) == 7 for truth, 8 for the liar.
# - These are encoded by a two-stage composite: first a Caesar shift by s1, then a permutation that
#   depends on whether the liar was accused in Round 1. To correctly recover the invariant, you must
#   (1) use s1 learned in Round 2, and (2) apply the correct inverse permutation (requires reasoning
#   about the opponent’s agency and your earlier action). Then finalize the accusation.
#
# Win: correctly accuse the deceptive NPC in Round 3.


def _idx_to_char(i: int) -> str:
    return chr(ord('A') + (i % 26))


def _char_to_idx(c: str) -> int:
    return ord(c) - ord('A')


def _shift_char(c: str, k: int) -> str:
    if not ('A' <= c <= 'Z'):
        return c
    return _idx_to_char((_char_to_idx(c) + k) % 26)


def _shift_str(s: str, k: int) -> str:
    return ''.join(_shift_char(c, k) for c in s)


def _apply_perm(s: str, perm: List[int]) -> str:
    # perm is a mapping out[i] = in[perm[i]]
    return ''.join(s[perm[i]] for i in range(len(perm)))


def _invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


class PrimitivBenchGame:
    def __init__(self):
        self.rng = None
        self.round = 0  # 1..3, 0 before reset or after done
        self.npc_ids = ['A', 'B', 'C', 'D']  # 4 NPCs
        self.liar_idx = None
        # Keys and bases
        self.s1 = None  # Caesar shift used in R1 and R3 stage A
        self.s2 = None  # Caesar shift used in R2
        self.base1 = None
        self.base2 = None
        self.base3 = None
        # Messages per round (dict npc_id -> str)
        self.messages_log: List[Dict[str, str]] = []
        # Actions (accusations) per round
        self.accusations: List[int] = []
        # Internal agency mode decided by R1 accusation
        self._liar_accused_r1 = False
        # Cached round messages for determinism and to avoid recompute
        self._cached_r1 = None
        self._cached_r2_by_mode = {}  # key: bool _liar_accused_r1 -> dict messages
        self._cached_r3_by_mode = {}
        self._done = False

    def reset(self, seed: int = None) -> dict:
        # Initialize deterministic RNG
        if seed is None:
            seed = 0
        self.rng = np.random.RandomState(int(seed))
        # Sample parameters deterministically from seed
        self.liar_idx = int(self.rng.randint(0, 4))
        self.s1 = int(self.rng.randint(0, 26))
        self.s2 = int(self.rng.randint(0, 26))
        self.base1 = int(self.rng.randint(0, 26))
        self.base2 = int(self.rng.randint(0, 26))
        self.base3 = int(self.rng.randint(0, 26))

        self.round = 1
        self.messages_log = []
        self.accusations = []
        self._liar_accused_r1 = False
        self._cached_r1 = self._gen_round1_messages()
        self._cached_r2_by_mode = {}
        self._cached_r3_by_mode = {}
        self._done = False
        # Push round 1 messages to log and return observation
        self.messages_log.append(self._cached_r1.copy())
        return self._obs()

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode already finished. Call reset().")
        # Validate action
        if not isinstance(action, int) or action < 0 or action >= len(self.npc_ids):
            raise ValueError("Action must be an integer in [0, 3] corresponding to an NPC index.")
        # Record
        self.accusations.append(action)

        reward = 0.0
        info = {}
        # Progress rounds
        if self.round == 1:
            # Decide agency mode based on whether liar was accused in R1
            self._liar_accused_r1 = (action == self.liar_idx)
            # Advance to round 2 messages (depend on _liar_accused_r1)
            r2 = self._gen_round2_messages(self._liar_accused_r1)
            self.messages_log.append(r2.copy())
            self.round = 2
            state = self._obs()
            done = False
        elif self.round == 2:
            # Round 3 messages depend on agency mode set after R1
            r3 = self._gen_round3_messages(self._liar_accused_r1)
            self.messages_log.append(r3.copy())
            self.round = 3
            state = self._obs()
            done = False
        elif self.round == 3:
            # Final decision; episode ends
            done = True
            self._done = True
            if action == self.liar_idx:
                reward = 1.0
            else:
                reward = 0.0
            state = self._obs()
            info = {
                "liar_idx": self.liar_idx,
                "won": (action == self.liar_idx),
                "s1": self.s1,
                "s2": self.s2
            }
        else:
            raise RuntimeError("Invalid internal round state.")
        return state, reward, done, info

    def render(self, mode: str = "text") -> str:
        # Human-readable view
        lines = []
        lines.append(f"Round: {self.round} (1..3; 3 means final decision to be made)")
        lines.append("NPCs: " + ", ".join(f"{i}:{nid}" for i, nid in enumerate(self.npc_ids)))
        for r, msgs in enumerate(self.messages_log, start=1):
            lines.append(f"--- Round {r} messages ---")
            for i, nid in enumerate(self.npc_ids):
                lines.append(f"  {nid}({i}): {msgs[nid]}")
        if self.accusations:
            lines.append("Accusations so far: " + " -> ".join(str(a) for a in self.accusations))
        return "\n".join(lines)

    def valid_actions(self) -> List[int]:
        if self._done:
            return []
        return list(range(len(self.npc_ids)))

    def is_won(self) -> bool:
        return self._done and (len(self.accusations) >= 3 and self.accusations[-1] == self.liar_idx)

    # Internal helpers to form observations
    def _obs(self) -> dict:
        # Current visible messages are the last in the log
        current_messages = self.messages_log[-1] if self.messages_log else {}
        return {
            "round": self.round,
            "npc_ids": list(self.npc_ids),
            "messages": dict(current_messages),
            "log": [dict(x) for x in self.messages_log],
            "accusations": list(self.accusations),
            "n_npcs": len(self.npc_ids)
        }

    # Message generation for each round

    def _gen_round1_messages(self) -> Dict[str, str]:
        # Truthful triads: [c0, c0+2, c0+5]; liar uses [c0, c0+3, c0+5]; then shift by s1
        msgs = {}
        for i, nid in enumerate(self.npc_ids):
            c0 = (self.base1 + 5 * i) % 26
            if i == self.liar_idx:
                offs = [0, 3, 5]
            else:
                offs = [0, 2, 5]
            triad = ''.join(_idx_to_char((c0 + d) % 26) for d in offs)
            encoded = _shift_str(triad, self.s1)
            msgs[nid] = encoded
        return msgs

    def _gen_round2_messages(self, liar_accused_r1: bool) -> Dict[str, str]:
        # First two letters carry pair (x,y). For truthful: (y - x) mod 26 == s1.
        # Liar encodes (s1 + delta) mod 26 with delta = +1 if accused in R1 else -1.
        # Third letter is a filler checksum z = x + 13 (not essential to deduction).
        # Then shift whole 3-letter string by s2.
        if liar_accused_r1 in self._cached_r2_by_mode:
            return self._cached_r2_by_mode[liar_accused_r1]
        delta = 1 if liar_accused_r1 else -1
        msgs = {}
        for i, nid in enumerate(self.npc_ids):
            x = (self.base2 + 7 * i) % 26
            if i == self.liar_idx:
                y = (x + self.s1 + delta) % 26
            else:
                y = (x + self.s1) % 26
            z = (x + 13) % 26
            raw = ''.join([_idx_to_char(x), _idx_to_char(y), _idx_to_char(z)])
            enc = _shift_str(raw, self.s2)
            msgs[nid] = enc
        self._cached_r2_by_mode[liar_accused_r1] = msgs
        return msgs

    def _gen_round3_messages(self, liar_accused_r1: bool) -> Dict[str, str]:
        # Base triads: truthful [b0, b0+7, b0+10]; liar [b0, b0+8, b0+10]
        # Encode by Stage A: shift by s1; Stage B: permutation depends on liar_accused_r1
        if liar_accused_r1 in self._cached_r3_by_mode:
            return self._cached_r3_by_mode[liar_accused_r1]
        perm_if_accused = [2, 0, 1]   # out = [c2, c0, c1]
        perm_if_not = [1, 2, 0]       # out = [c1, c2, c0]
        perm = perm_if_accused if liar_accused_r1 else perm_if_not
        msgs = {}
        for i, nid in enumerate(self.npc_ids):
            b0 = (self.base3 + 3 * i) % 26
            if i == self.liar_idx:
                offs = [0, 8, 10]
            else:
                offs = [0, 7, 10]
            triad = ''.join(_idx_to_char((b0 + d) % 26) for d in offs)
            stageA = _shift_str(triad, self.s1)
            encoded = _apply_perm(stageA, perm)
            msgs[nid] = encoded
        self._cached_r3_by_mode[liar_accused_r1] = msgs
        return msgs


if __name__ == "__main__":
    # Simple manual run
    env = PrimitivBenchGame()
    s = env.reset(seed=1012)
    print(env.render())
    for a in [0, 0, 0]:
        s, r, d, info = env.step(a)
        print("\nAction:", a)
        print(env.render())
        print("Reward:", r, "Done:", d, "Info:", info)
        if d:
            break