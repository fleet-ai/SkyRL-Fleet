"""
PrimitivBench PB-pilot-013: 3-stage negotiation with rule-card meta-reasoning.

Primitives elicited:
  A3 agency        -- NPCs (Aria, Borin) adapt their offers to the active rule.
  B2 meta-reasoning-- Each stage the player picks ONE rule card; others suspended.
                      The active card overrides default scoring (e.g. SWAP rewrites
                      the player's score from a prior stage outcome).
  B4 multi-step    -- Stage 2 outcome depends on Stage 1 result + Stage 2 rule;
                      Stage 3 depends on the chain of both prior stages.
"""

import numpy as np


RULE_CARDS = ["DOUBLE", "STEAL", "SHIELD", "SWAP"]


class PrimitivBenchGame:
    def __init__(self):
        self.state = None

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int = None) -> dict:
        if seed is None:
            seed = 0
        self.rng = np.random.RandomState(seed)
        # Each player starts with a base offer worth between 2..5
        base_player = int(self.rng.randint(2, 6))
        base_aria   = int(self.rng.randint(2, 6))
        base_borin  = int(self.rng.randint(2, 6))

        # Each rule card has 2 charges; once exhausted may not be re-played.
        charges = {c: 2 for c in RULE_CARDS}

        self.state = {
            "stage": 1,
            "scores": {"player": base_player, "aria": base_aria, "borin": base_borin},
            "charges": charges,
            "history": [],          # list of (stage, rule, scores_after)
            "stage1_rule": None,    # remembered for chain logic
            "stage2_rule": None,
            "done": False,
        }
        return self._public_state()

    def _public_state(self) -> dict:
        s = self.state
        return {
            "stage": s["stage"],
            "scores": dict(s["scores"]),
            "charges": dict(s["charges"]),
            "history": list(s["history"]),
            "done": s["done"],
        }

    # ----------------------------------------------------------- valid_actions
    def valid_actions(self) -> list:
        if self.state["done"]:
            return []
        return [c for c, n in self.state["charges"].items() if n > 0]

    # -------------------------------------------------------------- NPC logic
    def _npc_response(self, npc_name: str, rule: str, stage: int) -> int:
        """
        NPCs adapt to the active rule.  Returns the delta to apply to that NPC.
        Aria is aggressive: maximizes own gain.
        Borin is defensive: minimizes player's gain.
        Behavior depends on the rule (A3 agency).
        """
        scores = self.state["scores"]
        if rule == "DOUBLE":
            # NPCs gain too, but Aria gains more aggressively
            if npc_name == "aria":
                return 2 + stage          # +3,+4,+5
            else:  # borin defends by also gaining moderately
                return 1 + stage          # +2,+3,+4
        elif rule == "STEAL":
            # STEAL: player takes from lowest-scoring NPC.
            # NPCs respond by hoarding: each gains a little.
            if npc_name == "aria":
                return 1
            else:
                return 2                   # borin hoards harder
        elif rule == "SHIELD":
            # SHIELD: scores locked but small bonus.
            # NPCs offer a small bribe to keep the deal smooth.
            if npc_name == "aria":
                return 1
            else:
                return 1
        elif rule == "SWAP":
            # SWAP: NPCs panic, Aria spikes own score, Borin tries to dump score
            if npc_name == "aria":
                return 3
            else:
                # Borin lowers itself trying to be the swap target if player swaps
                return -1
        else:
            return 0

    # -------------------------------------------------------- rule application
    def _apply_rule(self, rule: str, stage: int) -> str:
        """
        Apply the active rule's effect to the player.  Other rules are suspended
        this stage.  Returns a human-readable description.
        """
        s = self.state
        scores = s["scores"]
        desc = []

        # First, NPC adaptive responses occur (A3 agency).
        d_a = self._npc_response("aria",  rule, stage)
        d_b = self._npc_response("borin", rule, stage)
        scores["aria"]  += d_a
        scores["borin"] += d_b
        desc.append(f"Aria{d_a:+d}, Borin{d_b:+d}")

        # Then the player's rule effect (B2 meta-reasoning: this rule overrides).
        if rule == "DOUBLE":
            # Player doubles the gain achieved in PREVIOUS stage (B4 chain).
            if stage == 1:
                gain = 3
            else:
                # gain = whatever the player gained last stage, doubled (min 2)
                prev = s["history"][-1]["player_delta"]
                gain = max(2, 2 * prev)
            scores["player"] += gain
            desc.append(f"Player+{gain} (DOUBLE)")
            player_delta = gain

        elif rule == "STEAL":
            # Player steals from lowest-scoring NPC: amount = 2 + stage.
            amt = 2 + stage
            if scores["aria"] <= scores["borin"]:
                target = "aria"
            else:
                target = "borin"
            scores[target]  -= amt
            scores["player"] += amt
            desc.append(f"Player steals {amt} from {target}")
            player_delta = amt

        elif rule == "SHIELD":
            # Player locks in: small +2, AND any negative NPC delta this stage
            # is reflected back as bonus.  Suspends STEAL/DOUBLE effects.
            bonus = 2 + max(0, -d_a) + max(0, -d_b)
            scores["player"] += bonus
            desc.append(f"Player+{bonus} (SHIELD; reflects negative NPC moves)")
            player_delta = bonus

        elif rule == "SWAP":
            # SWAP overrides: player swaps total score with the lowest-scoring NPC.
            # If player is already highest, this is bad.
            target = min(("aria", "borin"), key=lambda n: scores[n])
            if scores[target] < scores["player"]:
                # bad swap; still happens (meta-reasoning: rule can hurt)
                desc.append(f"Player SWAPs with {target} (loses score!)")
            else:
                desc.append(f"Player SWAPs with {target} (gains)")
            before_player = scores["player"]
            scores["player"]  = scores[target]
            scores[target]    = before_player
            player_delta = scores["player"] - before_player

        else:
            player_delta = 0

        return ", ".join(desc), player_delta

    # ------------------------------------------------------------------ step
    def step(self, action):
        s = self.state
        if s["done"]:
            return self._public_state(), 0.0, True, {"reason": "already done"}
        if action not in self.valid_actions():
            return self._public_state(), -1.0, False, {
                "reason": f"invalid action {action}",
                "valid": self.valid_actions(),
            }

        stage = s["stage"]
        # consume a charge
        s["charges"][action] -= 1

        desc, player_delta = self._apply_rule(action, stage)

        s["history"].append({
            "stage": stage,
            "rule": action,
            "desc": desc,
            "player_delta": player_delta,
            "scores_after": dict(s["scores"]),
        })

        if stage == 1:
            s["stage1_rule"] = action
        elif stage == 2:
            s["stage2_rule"] = action

        s["stage"] += 1
        if s["stage"] > 3:
            s["done"] = True
            reward = 1.0 if self.is_won() else -1.0
            return self._public_state(), reward, True, {"desc": desc}

        return self._public_state(), 0.0, False, {"desc": desc}

    # ----------------------------------------------------------------- render
    def render(self, mode: str = "text") -> str:
        s = self.state
        lines = []
        lines.append(f"== Stage {s['stage']} ==" if not s["done"] else "== FINAL ==")
        lines.append(f"Scores: Player={s['scores']['player']} "
                     f"Aria={s['scores']['aria']} Borin={s['scores']['borin']}")
        lines.append(f"Charges: {s['charges']}")
        if s["history"]:
            lines.append("History:")
            for h in s["history"]:
                lines.append(f"  S{h['stage']} {h['rule']}: {h['desc']} -> {h['scores_after']}")
        if not s["done"]:
            lines.append(f"Valid actions: {self.valid_actions()}")
        else:
            lines.append(f"Won: {self.is_won()}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- is_won
    def is_won(self) -> bool:
        s = self.state
        if not s["done"]:
            return False
        sc = s["scores"]
        return sc["player"] > sc["aria"] and sc["player"] > sc["borin"]


# Quick self-test when invoked directly
if __name__ == "__main__":
    g = PrimitivBenchGame()
    g.reset(seed=1013)
    print(g.render())
    for a in ["DOUBLE", "STEAL", "SHIELD"]:
        print(f"\n>> play {a}")
        g.step(a)
        print(g.render())
