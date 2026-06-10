"""
PrimitivBench task PB-pilot-003: 3-player token negotiation.

Player + 2 NPCs trade among 5 token types {A,B,C,D,E} with fixed weights.
Win for the player iff:
  (R1) player holds exactly one of each token type, AND
  (R2) player's total token weight <= THRESHOLD, AND
  (R3) at least one NPC fails (R1 AND R2).

Primitives elicited:
  A1 objectness: token type identity is load-bearing (types differ in weight,
                 and the goal references the set of distinct types).
  A3 agency: NPCs accept/reject trades and initiate inter-NPC trades based on
             hidden valuation functions; a random-walker NPC trivializes play.
  B3 multi-rule composition: both R1 and R2 must hold simultaneously; R3 is a
             third conjunct that requires reasoning about NPC states too.
"""

import random
from typing import Optional

TOKEN_TYPES = ["A", "B", "C", "D", "E"]
TOKEN_WEIGHTS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
WEIGHT_THRESHOLD = 15
MAX_ROUNDS = 30


def _empty_inv():
    return {t: 0 for t in TOKEN_TYPES}


def _satisfies_rules(inv):
    if any(inv[t] != 1 for t in TOKEN_TYPES):
        return False
    total = sum(TOKEN_WEIGHTS[t] * inv[t] for t in TOKEN_TYPES)
    return total <= WEIGHT_THRESHOLD


def _total_weight(inv):
    return sum(TOKEN_WEIGHTS[t] * inv[t] for t in TOKEN_TYPES)


class PrimitivBenchGame:
    def __init__(self):
        self.state = None
        self._rng = None
        self._last_action_was_pass = False
        self._done = False

    # ----- core API -----
    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is None:
            seed = 1003
        self._rng = random.Random(seed)
        # Total token pool: 4A, 3B, 3C, 2D, 2E  (14 tokens)
        pool = ["A"] * 4 + ["B"] * 3 + ["C"] * 3 + ["D"] * 2 + ["E"] * 2

        # Deterministic but seed-dependent allocation across 3 players.
        # Each player gets ~5 tokens, but allocation is uneven to force trading.
        self._rng.shuffle(pool)
        inventories = [_empty_inv() for _ in range(3)]
        # Distribute: first 5 to player(0), next 5 to npc1(1), last 4 to npc2(2)
        sizes = [5, 5, 4]
        idx = 0
        for pid, n in enumerate(sizes):
            for _ in range(n):
                inventories[pid][pool[idx]] += 1
                idx += 1

        # Hidden NPC valuations: each NPC has a value per token type plus a
        # threshold. Generated from seed; player cannot directly observe.
        npc_vals = []
        for npc_id in range(2):
            vals = {t: self._rng.randint(1, 9) for t in TOKEN_TYPES}
            npc_vals.append(vals)
        npc_threshold = 0  # accept iff receive_value - give_value >= 0
        # tie-break: prefer tokens that complete the NPC's set.

        self.state = {
            "round": 0,
            "inventories": inventories,    # [player, npc1, npc2]
            "npc_valuations": npc_vals,    # hidden
            "npc_threshold": npc_threshold,
            "history": [],                 # list of dicts describing each step
            "last_was_pass": False,
            "done": False,
            "winner_info": None,
        }
        self._done = False
        return self._public_state()

    def _public_state(self):
        # Player sees: their own inventory, both NPCs' inventories (public),
        # round number, history of accepted/rejected trades (which leaks
        # information about hidden valuations -> agency modeling).
        s = self.state
        return {
            "round": s["round"],
            "you": dict(s["inventories"][0]),
            "npc1": dict(s["inventories"][1]),
            "npc2": dict(s["inventories"][2]),
            "you_weight": _total_weight(s["inventories"][0]),
            "npc1_weight": _total_weight(s["inventories"][1]),
            "npc2_weight": _total_weight(s["inventories"][2]),
            "weight_threshold": WEIGHT_THRESHOLD,
            "token_weights": dict(TOKEN_WEIGHTS),
            "history": list(s["history"]),
            "done": s["done"],
        }

    def valid_actions(self) -> list:
        if self._done:
            return []
        s = self.state
        actions = [("pass",)]
        player_inv = s["inventories"][0]
        for npc_id in (1, 2):
            npc_inv = s["inventories"][npc_id]
            for give in TOKEN_TYPES:
                if player_inv[give] <= 0:
                    continue
                for recv in TOKEN_TYPES:
                    if recv == give:
                        continue
                    if npc_inv[recv] <= 0:
                        continue
                    actions.append(("trade", npc_id, give, recv))
        return actions

    def step(self, action):
        if self._done:
            return self._public_state(), 0.0, True, {"error": "game over"}

        s = self.state
        s["round"] += 1
        reward = 0.0
        info = {}

        if action == ("pass",) or action == "pass":
            event = {"round": s["round"], "type": "player_pass"}
            s["history"].append(event)
            if s["last_was_pass"]:
                # Two passes in a row -> end.
                self._finalize()
                return self._public_state(), self._final_reward(), True, {"end": "double_pass"}
            s["last_was_pass"] = True
            # NPC sub-turn after player's pass
            self._npc_turn()
            if s["round"] >= MAX_ROUNDS:
                self._finalize()
                return self._public_state(), self._final_reward(), True, {"end": "max_rounds"}
            return self._public_state(), reward, False, info

        s["last_was_pass"] = False

        if not (isinstance(action, tuple) and len(action) == 4 and action[0] == "trade"):
            info["error"] = "invalid action format"
            return self._public_state(), -0.01, False, info

        _, npc_id, give, recv = action
        if npc_id not in (1, 2):
            info["error"] = "invalid npc id"
            return self._public_state(), -0.01, False, info
        if give not in TOKEN_TYPES or recv not in TOKEN_TYPES or give == recv:
            info["error"] = "invalid token types"
            return self._public_state(), -0.01, False, info

        player_inv = s["inventories"][0]
        npc_inv = s["inventories"][npc_id]
        if player_inv[give] <= 0 or npc_inv[recv] <= 0:
            info["error"] = "tokens not available"
            return self._public_state(), -0.01, False, info

        accept = self._npc_decides(npc_id, give_to_npc=give, recv_from_npc=recv)
        if accept:
            player_inv[give] -= 1
            player_inv[recv] += 1
            npc_inv[recv] -= 1
            npc_inv[give] += 1
            event = {
                "round": s["round"],
                "type": "trade_accepted",
                "with_npc": npc_id,
                "player_gave": give,
                "player_received": recv,
            }
        else:
            event = {
                "round": s["round"],
                "type": "trade_rejected",
                "with_npc": npc_id,
                "player_offered_give": give,
                "player_offered_recv": recv,
            }
        s["history"].append(event)

        # NPC sub-turn (they may also trade among themselves)
        self._npc_turn()

        if s["round"] >= MAX_ROUNDS:
            self._finalize()
            return self._public_state(), self._final_reward(), True, {"end": "max_rounds"}

        return self._public_state(), reward, False, info

    # ----- NPC logic (hidden valuations -> A3 agency) -----
    def _npc_value_of(self, npc_id, token):
        vals = self.state["npc_valuations"][npc_id - 1]
        inv = self.state["inventories"][npc_id]
        base = vals[token]
        # Need-bonus: if NPC is missing this type, value rises sharply.
        if inv[token] == 0:
            base += 5
        # Anti-duplicate: if NPC already has >=1 of this type, extra copies
        # have diminishing value (they don't help win and add weight).
        if inv[token] >= 1:
            base -= 3 * inv[token]
        # Weight penalty: avoid pushing over threshold.
        if _total_weight(inv) + TOKEN_WEIGHTS[token] > WEIGHT_THRESHOLD:
            base -= 4
        return base

    def _npc_decides(self, npc_id, give_to_npc, recv_from_npc):
        # NPC receives `give_to_npc` from player, loses `recv_from_npc`.
        gain = self._npc_value_of(npc_id, give_to_npc)
        # value of losing the outgoing token: use current value before loss
        loss = self._npc_value_of(npc_id, recv_from_npc)
        # Decision: accept if net >= threshold
        return (gain - loss) >= self.state["npc_threshold"]

    def _npc_turn(self):
        # The two NPCs each consider one trade with the other NPC.
        # Deterministic given state.
        s = self.state
        for proposer in (1, 2):
            counter = 2 if proposer == 1 else 1
            best = None  # (net_gain_for_proposer, give, recv)
            prop_inv = s["inventories"][proposer]
            counter_inv = s["inventories"][counter]
            for give in TOKEN_TYPES:
                if prop_inv[give] <= 0:
                    continue
                for recv in TOKEN_TYPES:
                    if recv == give or counter_inv[recv] <= 0:
                        continue
                    # proposer's net from doing this trade
                    proposer_net = (self._npc_value_of(proposer, recv)
                                    - self._npc_value_of(proposer, give))
                    if proposer_net <= 0:
                        continue
                    # counter would accept iff their net >= 0
                    counter_net = (self._npc_value_of(counter, give)
                                   - self._npc_value_of(counter, recv))
                    if counter_net < 0:
                        continue
                    score = proposer_net + counter_net
                    key = (score, -TOKEN_WEIGHTS[give], give, recv)
                    if best is None or key > best[0]:
                        best = (key, give, recv)
            if best is not None:
                _, give, recv = best
                prop_inv[give] -= 1
                prop_inv[recv] += 1
                counter_inv[recv] -= 1
                counter_inv[give] += 1
                s["history"].append({
                    "round": s["round"],
                    "type": "npc_trade",
                    "proposer": proposer,
                    "counter": counter,
                    "proposer_gave": give,
                    "proposer_received": recv,
                })

    # ----- termination -----
    def _finalize(self):
        s = self.state
        s["done"] = True
        self._done = True
        results = []
        for pid in range(3):
            results.append(_satisfies_rules(s["inventories"][pid]))
        player_ok = results[0]
        # Player wins iff player_ok AND at least one NPC fails.
        npc_fail = (not results[1]) or (not results[2])
        win = player_ok and npc_fail
        s["winner_info"] = {
            "player_satisfies_rules": results[0],
            "npc1_satisfies_rules": results[1],
            "npc2_satisfies_rules": results[2],
            "player_won": win,
        }

    def _final_reward(self):
        if self.state["winner_info"] is None:
            return 0.0
        return 1.0 if self.state["winner_info"]["player_won"] else -1.0

    def is_won(self) -> bool:
        if not self._done:
            return False
        wi = self.state["winner_info"]
        return bool(wi and wi["player_won"])

    def render(self, mode: str = "text") -> str:
        s = self.state
        if s is None:
            return "<game not initialized>"
        lines = []
        lines.append(f"Round {s['round']}/{MAX_ROUNDS}  (threshold={WEIGHT_THRESHOLD})")
        lines.append(f"Weights: {TOKEN_WEIGHTS}")
        for pid, name in enumerate(["YOU ", "NPC1", "NPC2"]):
            inv = s["inventories"][pid]
            w = _total_weight(inv)
            sat = "OK" if _satisfies_rules(inv) else "no"
            lines.append(f"  {name}: {dict(inv)}  weight={w}  rules={sat}")
        if s["history"]:
            lines.append("Last events:")
            for ev in s["history"][-4:]:
                lines.append(f"  {ev}")
        if s["done"]:
            lines.append(f"DONE: {s['winner_info']}")
        return "\n".join(lines)


# Minimal self-test runnable as script.
if __name__ == "__main__":
    g = PrimitivBenchGame()
    obs = g.reset(seed=1003)
    print(g.render())
    print("valid actions sample:", g.valid_actions()[:6], "...")
