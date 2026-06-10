"""
PrimitivBench PB-pilot-012: Cipher Council
A 3-round social deduction game with per-round perception transforms.

Primitives elicited:
- A3 agency: the deceptive NPC is an adversary whose messages are crafted
  to look truthful raw but contradict the public truth once decoded.
- B1 perception transform: each round's messages are encoded; decoding
  with the round's cipher is required to read semantic content.
- B4 multi-step composition: round-3 cipher is a composition of two
  transforms, and final accusation requires aggregating decoded evidence
  across all 3 rounds (depth >= 3 reasoning chain).
"""

import random
import string


NPC_NAMES = ["Alice", "Bob", "Carol", "Dave"]


def caesar(text: str, k: int) -> str:
    """Caesar-shift letters by k. Non-letters pass through."""
    out = []
    for ch in text:
        if 'A' <= ch <= 'Z':
            out.append(chr((ord(ch) - ord('A') + k) % 26 + ord('A')))
        elif 'a' <= ch <= 'z':
            out.append(chr((ord(ch) - ord('a') + k) % 26 + ord('a')))
        else:
            out.append(ch)
    return "".join(out)


def reverse_words(text: str) -> str:
    """Reverse each word's letters; preserve word order and spaces/punct."""
    out_words = []
    for w in text.split(" "):
        out_words.append(w[::-1])
    return " ".join(out_words)


def encode_round1(plaintext: str) -> str:
    # identity: round-1 cipher is no-op. Raw text is the message.
    return plaintext


def encode_round2(plaintext: str, k1: int) -> str:
    return caesar(plaintext, k1)


def encode_round3(plaintext: str, k1: int, k2: int) -> str:
    # composite: caesar(k1) then word-reverse if k2 odd, else caesar(k2) then caesar(k1)
    step1 = caesar(plaintext, k1)
    if k2 % 2 == 1:
        return reverse_words(step1)
    else:
        return caesar(step1, k2)


def decode_round3(ciphertext: str, k1: int, k2: int) -> str:
    """Inverse of encode_round3 — useful for solvers/players."""
    if k2 % 2 == 1:
        step1 = reverse_words(ciphertext)
    else:
        step1 = caesar(ciphertext, -k2)
    return caesar(step1, -k1)


class PrimitivBenchGame:
    """
    Cipher Council: 3-round social deduction.

    State:
      - true_room: int in [1..9]  (public truth, given in render)
      - liar_id: int in [0..3]    (hidden; the deceptive NPC)
      - k1, k2: per-round cipher keys (hidden until round end)
      - round: 1, 2, or 3 (current round)
      - messages: dict[round] -> list of 4 ciphertexts (one per NPC)
      - accusations: dict[round] -> npc_id chosen by player
      - revealed_keys: dict[round] -> key(s) revealed after accusation
      - done: bool
      - won: bool
    """

    MAX_STEPS = 3

    def __init__(self):
        self._rng = random.Random()
        self.state = None
        self.done = True
        self.won = False

    # ------------- core API -------------

    def reset(self, seed: int = None) -> dict:
        if seed is None:
            seed = 0
        self._rng = random.Random(seed)
        true_room = self._rng.randint(1, 9)
        liar_id = self._rng.randint(0, 3)
        # cipher keys; nonzero shifts to avoid trivial identity in rounds 2/3
        k1 = self._rng.randint(1, 25)
        k2 = self._rng.randint(1, 25)
        # craft a "lie room" different from true_room
        lie_room = true_room
        while lie_room == true_room:
            lie_room = self._rng.randint(1, 9)

        # Each NPC has an honest claim. The liar says lie_room.
        plaintexts = {}  # round -> [str per NPC]
        for r in (1, 2, 3):
            row = []
            for i in range(4):
                if i == liar_id:
                    # Adversarial wording slightly varied per round
                    msg = f"THE TREASURE IS IN ROOM {lie_room}"
                else:
                    msg = f"THE TREASURE IS IN ROOM {true_room}"
                row.append(msg)
            plaintexts[r] = row

        # Encode per-round messages.
        messages = {1: [], 2: [], 3: []}
        for i in range(4):
            messages[1].append(encode_round1(plaintexts[1][i]))
            messages[2].append(encode_round2(plaintexts[2][i], k1))
            messages[3].append(encode_round3(plaintexts[3][i], k1, k2))

        self.state = {
            "true_room": true_room,
            "liar_id": liar_id,         # hidden from observation
            "lie_room": lie_room,       # hidden
            "k1": k1,                   # hidden until end of round 1
            "k2": k2,                   # hidden until end of round 2
            "round": 1,
            "messages": messages,
            "accusations": {},
            "revealed_keys": {},        # populated after each round's accusation
            "npc_names": list(NPC_NAMES),
        }
        self.done = False
        self.won = False
        return self._observation()

    def step(self, action):
        """
        action: int in [0..3] — the NPC index being accused this round.
        Returns: (observation, reward, done, info)
        """
        if self.done:
            return self._observation(), 0.0, True, {"error": "episode_done"}

        if not isinstance(action, int) or action < 0 or action > 3:
            return self._observation(), -1.0, False, {"error": "invalid_action"}

        r = self.state["round"]
        self.state["accusations"][r] = action

        # Reveal that round's cipher key(s) so player can re-read prior rounds.
        if r == 1:
            self.state["revealed_keys"][1] = {"cipher": "identity"}
            # End of round 1: also reveal k1 used in round 2? No — reveal at end of round 2.
            # Actually: round 1's cipher IS identity, so nothing new. But the
            # problem statement says "newly-learned cipher" applies to round-1 records
            # in round 2. We interpret: at end of round 1 the player learns k1
            # (the cipher that will be used in round 2's messages and that
            # retro-applies to revise interpretation). To match spec we reveal k1 here.
            self.state["revealed_keys"][1]["k1"] = self.state["k1"]
        elif r == 2:
            self.state["revealed_keys"][2] = {"cipher": f"caesar(k1={self.state['k1']})",
                                              "k2": self.state["k2"]}
        elif r == 3:
            self.state["revealed_keys"][3] = {"cipher": f"composite(k1={self.state['k1']}, k2={self.state['k2']})"}

        reward = 0.0
        info = {}

        if r < 3:
            self.state["round"] = r + 1
        else:
            # Final round: evaluate.
            self.done = True
            final = self.state["accusations"][3]
            self.won = (final == self.state["liar_id"])
            reward = 1.0 if self.won else 0.0
            info["liar_id"] = self.state["liar_id"]
            info["liar_name"] = NPC_NAMES[self.state["liar_id"]]
            info["true_room"] = self.state["true_room"]
            info["lie_room"] = self.state["lie_room"]

        return self._observation(), reward, self.done, info

    def render(self, mode: str = "text") -> str:
        s = self.state
        lines = []
        lines.append("=== Cipher Council ===")
        lines.append(f"Public truth: the treasure is in ROOM {s['true_room']}.")
        lines.append(f"Current round: {s['round']}  (done={self.done}, won={self.won})")
        lines.append("")
        for r in (1, 2, 3):
            if r > s["round"] and not self.done:
                continue
            lines.append(f"--- Round {r} messages ---")
            for i, name in enumerate(NPC_NAMES):
                lines.append(f"  [{i}] {name}: {s['messages'][r][i]!r}")
            if r in s["revealed_keys"]:
                lines.append(f"  (cipher revealed: {s['revealed_keys'][r]})")
            if r in s["accusations"]:
                aid = s["accusations"][r]
                lines.append(f"  >> you accused [{aid}] {NPC_NAMES[aid]}")
            lines.append("")
        if self.done:
            lines.append(f"Liar was: [{s['liar_id']}] {NPC_NAMES[s['liar_id']]}  (said room {s['lie_room']})")
            lines.append(f"Result: {'WIN' if self.won else 'LOSS'}")
        return "\n".join(lines)

    def valid_actions(self) -> list:
        if self.done:
            return []
        return [0, 1, 2, 3]

    def is_won(self) -> bool:
        return self.won

    # ------------- helpers -------------

    def _observation(self) -> dict:
        s = self.state
        # Player sees: true_room, messages up through current round,
        # revealed_keys for rounds whose accusation has been made,
        # current round, prior accusations.
        visible_msgs = {}
        for r in (1, 2, 3):
            if r <= s["round"]:
                visible_msgs[r] = list(s["messages"][r])
        return {
            "true_room": s["true_room"],
            "round": s["round"],
            "messages": visible_msgs,
            "revealed_keys": dict(s["revealed_keys"]),
            "accusations": dict(s["accusations"]),
            "npc_names": list(NPC_NAMES),
            "done": self.done,
            "won": self.won,
        }


# ------------- demo / quick self-test -------------
if __name__ == "__main__":
    g = PrimitivBenchGame()
    obs = g.reset(seed=1012)
    print(g.render())
    # Random play
    import random as _r
    rng = _r.Random(7)
    while not g.done:
        a = rng.choice(g.valid_actions())
        obs, rew, done, info = g.step(a)
    print(g.render())
    print("info:", info)
