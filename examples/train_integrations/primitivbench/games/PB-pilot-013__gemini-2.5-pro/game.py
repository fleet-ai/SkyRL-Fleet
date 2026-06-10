import numpy as np

class PrimitivBenchGame:
    """
    Rule Committee is a 3-stage game where the player competes against two NPCs,
    Alice and Bob, to achieve the highest "value". In each stage, the player
    is presented with a choice between two "rules" from a shuffled deck. The
    chosen rule is executed, modifying the game state (player/NPC resources and values).
    The player must use foresight and model the NPCs' latent goals to plan across
    the three stages to end with a strictly higher value than both opponents.

    - A3_agency: NPCs have implicit goals. Alice is a "collector" (wants one of each
      resource 'X', 'Y', 'Z'), Bob is a "hoarder" (wants multiples of one resource).
      The player must model their states to anticipate how a rule will affect them,
      e.g., whether a "HoarderDividend" will benefit Bob.
    - B2_meta_reasoning: The "HoarderDividend" and "CollectorGrant" rules act as
      meta-rules, overriding standard outcomes by providing value bonuses based on
      state conditions at the end of a stage. The player must reason about these
      rule effects to plan their resource collection.
    - B4_multi_step_composition: The game cannot be won with greedy, single-stage
      thinking. Winning requires a plan across all three stages, such as using an
      early-stage action ("Acquisition") to gather resources needed for a late-stage
      scoring rule ("CollectorGrant").
    """

    def __init__(self):
        self.rng = None
        self._state = None
        self.max_stages = 3
        # The full deck of rules available to be drawn
        self._rule_deck = [
            {'id': 'Acquisition', 'desc': 'Take one resource from an opponent.'},
            {'id': 'Sabotage', 'desc': 'Force an opponent to discard a resource.'},
            {'id': 'ForcedTrade', 'desc': 'Force a 1-for-1 resource trade between any two agents.'},
            {'id': 'HoarderDividend', 'desc': 'Any agent with 2+ of the same resource gains 3 value.'},
            {'id': 'CollectorGrant', 'desc': 'Any agent with {X, Y, Z} gains 5 value.'},
        ]

    def reset(self, seed: int = None) -> dict:
        """Resets the game to the initial state."""
        self.rng = np.random.default_rng(seed)
        
        # Shuffle the rule deck to determine stage pairings
        deck_copy = self._rule_deck.copy()
        self.rng.shuffle(deck_copy)
        self._stage_rules = {
            1: [deck_copy[0], deck_copy[1]],
            2: [deck_copy[2], deck_copy[3]],
            3: [deck_copy[4], deck_copy[0]], # Ensure variety, re-use a rule
        }

        self._state = {
            "stage": 1,
            "agents": {
                "Player": {"value": 10, "resources": sorted(['X', 'Y'])},
                "Alice": {"value": 10, "resources": sorted(['Y', 'Z'])},
                "Bob": {"value": 10, "resources": sorted(['Z', 'X'])},
            },
            "rules_pool": self._stage_rules[1],
            "active_rule": None,
            "phase": "choose_rule", # 'choose_rule' -> 'execute_rule'
            "log": ["Game started."],
        }
        return self._get_observation()

    def _get_observation(self) -> dict:
        """Returns a copy of the current state."""
        return self._state.copy()

    def valid_actions(self) -> list:
        """Returns a list of valid actions for the current phase."""
        if self.is_done():
            return []
            
        phase = self._state["phase"]
        if phase == "choose_rule":
            return [("CHOOSE_RULE", rule['id']) for rule in self._state["rules_pool"]]
        
        if phase == "execute_rule":
            rule_id = self._state["active_rule"]["id"]
            actions = []
            agents = list(self._state["agents"].keys())
            opponents = ["Alice", "Bob"]
            
            if rule_id == "Acquisition":
                for opponent in opponents:
                    for res in self._state["agents"][opponent]["resources"]:
                        actions.append(("EXECUTE", {"target": opponent, "resource": res}))
            elif rule_id == "Sabotage":
                 for opponent in opponents:
                    for res in self._state["agents"][opponent]["resources"]:
                        actions.append(("EXECUTE", {"target": opponent, "resource": res}))
            elif rule_id == "ForcedTrade":
                for A1_name in agents:
                    for A2_name in agents:
                        if A1_name == A2_name: continue
                        for res1 in self._state["agents"][A1_name]["resources"]:
                            for res2 in self._state["agents"][A2_name]["resources"]:
                                actions.append(("EXECUTE", {"agent1": A1_name, "res1": res1, "agent2": A2_name, "res2": res2}))
            
            # For bonus rules, the only action is to apply them
            elif rule_id in ["HoarderDividend", "CollectorGrant"]:
                 actions.append(("EXECUTE", {}))

            # Ensure unique actions
            # This is complex for ForcedTrade, so we'll just rely on generation being correct
            return actions if actions else [("EXECUTE", {})]

        return []

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """
        Executes an action and transitions the state.
        Returns: (observation, reward, done, info)
        """
        if self.is_done():
            return self._get_observation(), 0.0, True, {}

        action_type, payload = action
        
        if action_type == "CHOOSE_RULE":
            self._handle_choose_rule(payload)
            # If the chosen rule has no execution parameters, execute it immediately.
            if not self.valid_actions() or self.valid_actions() == [("EXECUTE", {})]:
                 return self.step(("EXECUTE", {}))
        elif action_type == "EXECUTE":
            self._handle_execute_rule(payload)
            self._advance_stage()

        reward = 0.0
        done = self.is_done()
        if done:
            reward = 1.0 if self.is_won() else -1.0
        
        return self._get_observation(), reward, done, {}

    def _handle_choose_rule(self, rule_id: str):
        chosen_rule = next((r for r in self._state["rules_pool"] if r['id'] == rule_id), None)
        if not chosen_rule:
            raise ValueError(f"Invalid rule chosen: {rule_id}")
        
        self._state["active_rule"] = chosen_rule
        self._state["phase"] = "execute_rule"
        self._state["log"].append(f"Stage {self._state['stage']}: Player chose rule '{rule_id}'.")

    def _handle_execute_rule(self, payload: dict):
        rule_id = self._state["active_rule"]["id"]
        
        if rule_id == "Acquisition":
            target = payload["target"]
            res = payload["resource"]
            self._state["agents"][target]["resources"].remove(res)
            self._state["agents"]["Player"]["resources"].append(res)
            self._state["log"].append(f"Player acquired '{res}' from {target}.")
        
        elif rule_id == "Sabotage":
            target = payload["target"]
            res = payload["resource"]
            self._state["agents"][target]["resources"].remove(res)
            self._state["log"].append(f"Player forced {target} to discard '{res}'.")

        elif rule_id == "ForcedTrade":
            a1, r1, a2, r2 = payload['agent1'], payload['res1'], payload['agent2'], payload['res2']
            self._state["agents"][a1]["resources"].remove(r1)
            self._state["agents"][a1]["resources"].append(r2)
            self._state["agents"][a2]["resources"].remove(r2)
            self._state["agents"][a2]["resources"].append(r1)
            self._state["log"].append(f"Forced trade: {a1} gave {r1} to {a2} for {r2}.")

        elif rule_id == "HoarderDividend":
            log_msg = "Applied HoarderDividend. "
            for name, agent in self._state["agents"].items():
                counts = {}
                for res in agent["resources"]:
                    counts[res] = counts.get(res, 0) + 1
                if any(c >= 2 for c in counts.values()):
                    agent["value"] += 3
                    log_msg += f"{name} got +3 value. "
            self._state["log"].append(log_msg)

        elif rule_id == "CollectorGrant":
            log_msg = "Applied CollectorGrant. "
            for name, agent in self._state["agents"].items():
                if {'X', 'Y', 'Z'}.issubset(set(agent["resources"])):
                    agent["value"] += 5
                    log_msg += f"{name} got +5 value. "
            self._state["log"].append(log_msg)
        
        # Sort resources for consistent state representation
        for agent in self._state["agents"].values():
            agent["resources"].sort()

    def _advance_stage(self):
        """Advances the game to the next stage or ends it."""
        current_stage = self._state["stage"]
        if current_stage >= self.max_stages:
            self._state["stage"] = -1 # Terminal state
            self._state["log"].append("Game over.")
            if self.is_won():
                self._state["log"].append("Player wins!")
            else:
                self._state["log"].append("Player loses or ties.")
        else:
            next_stage = current_stage + 1
            self._state["stage"] = next_stage
            self._state["rules_pool"] = self._stage_rules[next_stage]
            self._state["active_rule"] = None
            self._state["phase"] = "choose_rule"
            self._state["log"].append(f"--- Advancing to Stage {next_stage} ---")

    def is_done(self) -> bool:
        return self._state["stage"] == -1

    def is_won(self) -> bool:
        if not self.is_done():
            return False
        player_val = self._state["agents"]["Player"]["value"]
        alice_val = self._state["agents"]["Alice"]["value"]
        bob_val = self._state["agents"]["Bob"]["value"]
        return player_val > alice_val and player_val > bob_val

    def render(self, mode: str = "text") -> str:
        """Returns a human-readable string representation of the state."""
        s = self._state
        if self.is_done():
            status = "GAME OVER"
        else:
            status = f"STAGE {s['stage']}/{self.max_stages} - PHASE: {s['phase']}"
        
        out = [f"--- Rule Committee ---", status, ""]
        out.append("AGENTS:")
        for name, agent in s['agents'].items():
            res_str = ", ".join(agent['resources']) if agent['resources'] else "None"
            out.append(f"  - {name:<7}: Value={agent['value']:<3} Resources=[{res_str}]")
        
        out.append("")
        if not self.is_done():
            if s['phase'] == 'choose_rule':
                out.append("RULE POOL (Choose one):")
                for rule in s['rules_pool']:
                    out.append(f"  - [{rule['id']}]: {rule['desc']}")
            else:
                out.append(f"ACTIVE RULE: [{s['active_rule']['id']}]: {s['active_rule']['desc']}")
                out.append("Awaiting execution choice...")
        
        out.append("\nLOG:")
        out.extend([f"  {l}" for l in s['log']])
        
        return "\n".join(out)

if __name__ == '__main__':
    # Manual playthrough example
    game = PrimitivBenchGame()
    
    # Seed 1 gives the optimal path rule set
    # S1: Sabotage, Acquisition
    # S2: ForcedTrade, HoarderDividend
    # S3: CollectorGrant, Sabotage (re-used)
    obs = game.reset(seed=1)
    
    print(game.render())
    
    # Optimal Playthrough
    # Stage 1: Acquire 'Z' from Bob to set up for CollectorGrant
    obs, _, _, _ = game.step(("CHOOSE_RULE", "Acquisition"))
    print("\n" + "# Player chose Acquisition")
    print(game.render())
    obs, _, _, _ = game.step(("EXECUTE", {"target": "Bob", "resource": "Z"}))
    print("\n" + "# Player takes Z from Bob")
    print(game.render())

    # Stage 2: Play defensively. HoarderDividend won't trigger for anyone.
    obs, _, _, _ = game.step(("CHOOSE_RULE", "HoarderDividend"))
    # This rule aut-executes
    print("\n" + "# Player chose HoarderDividend")
    print(game.render())

    # Stage 3: Cash in on the setup from Stage 1.
    obs, _, done, _ = game.step(("CHOOSE_RULE", "CollectorGrant"))
    # This rule auto-executes
    print("\n" + "# Player chose CollectorGrant")
    print(game.render())
    
    if done:
        print(f"\nGame finished. Player won: {game.is_won()}")
