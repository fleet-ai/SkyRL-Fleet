import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class PrimitivBenchGame:
    """
    3-stage code-breaking puzzle on a 6-hex-digit cyclic sequence.

    Transforms:
    - T1(k1): alternating add: even indices +k1, odd indices -k1 (mod 16)
    - T2(k2): rotate-left by k2 (on a 6-cycle)
    - T3(k3): uniform add: all indices +k3 (mod 16)

    Rules (must all hold cumulatively):
    - R1: Existence of at least one equal adjacent pair on the cycle
    - R2: Existence of at least one adjacent difference equal to target_D (nonzero) on the cycle
    - R3: Minimum value occurs at index 0 (ties allowed; value at index 0 equals global min)
    - R4: Weighted sum S = sum(i * v[i]) % 16 equals anchor_A

    Stage progression:
    - Stage 1: apply T1 to ciphertext; must satisfy R1 + R2
    - Stage 2: apply T1∘T2; must satisfy R1 + R2 + R3
    - Stage 3: apply T1∘T2∘T3; must satisfy R1 + R2 + R3 + R4 (win)

    Deterministic puzzle generation guarantees solvability for a given seed.
    """

    def __init__(self, max_steps: int = 60):
        self.max_steps = max_steps
        self.rng: Optional[np.random.RandomState] = None
        self.seed_val: Optional[int] = None
        self.state: Dict[str, Any] = {}
        self.steps = 0
        self.done = False
        self.won = False

    # ---------- Helper transforms and checks ----------

    @staticmethod
    def _to_hex(vals: List[int]) -> str:
        return "".join("{:X}".format(x % 16) for x in vals)

    @staticmethod
    def _alt_add(arr: List[int], k: int) -> List[int]:
        # even indices +k, odd indices -k modulo 16
        out = []
        for i, a in enumerate(arr):
            if i % 2 == 0:
                out.append((a + k) % 16)
            else:
                out.append((a - k) % 16)
        return out

    @staticmethod
    def _rot_left(arr: List[int], k: int) -> List[int]:
        k = k % len(arr)
        return list(arr[k:] + arr[:k])

    @staticmethod
    def _add_all(arr: List[int], k: int) -> List[int]:
        return [((a + k) % 16) for a in arr]

    @staticmethod
    def _cyclic_diffs(arr: List[int]) -> List[int]:
        n = len(arr)
        return [((arr[(i + 1) % n] - arr[i]) % 16) for i in range(n)]

    @staticmethod
    def _check_R1(arr: List[int]) -> bool:
        n = len(arr)
        for i in range(n):
            if arr[i] == arr[(i + 1) % n]:
                return True
        return False

    @staticmethod
    def _check_R2(arr: List[int], target_D: int) -> bool:
        diffs = PrimitivBenchGame._cyclic_diffs(arr)
        return any(d == (target_D % 16) for d in diffs)

    @staticmethod
    def _check_R3(arr: List[int]) -> bool:
        return arr[0] == min(arr)

    @staticmethod
    def _weighted_sum_mod16(arr: List[int]) -> int:
        return sum(i * v for i, v in enumerate(arr)) % 16

    @staticmethod
    def _check_R4(arr: List[int], anchor_A: int) -> bool:
        return PrimitivBenchGame._weighted_sum_mod16(arr) == (anchor_A % 16)

    # ---------- Internal: compute working buffer given stage/key ----------

    def _compute_work(self) -> List[int]:
        st = self.state["stage"]
        if st == 1:
            return self._alt_add(self.state["ciphertext"], self.state["k1"])
        elif st == 2:
            # base is committed1
            return self._rot_left(self.state["committed1"], self.state["k2"])
        elif st == 3:
            # base is committed2
            return self._add_all(self.state["committed2"], self.state["k3"])
        else:
            # If already done, just return last committed or ciphertext
            if self.state.get("committed2") is not None:
                return list(self.state["committed2"])
            if self.state.get("committed1") is not None:
                return list(self.state["committed1"])
            return list(self.state["ciphertext"])

    def _rules_ok(self, arr: List[int], up_to_stage: int) -> bool:
        # R1+R2 always checked; add R3 at stage>=2; add R4 at stage>=3
        ok = self._check_R1(arr) and self._check_R2(arr, self.state["target_D"])
        if up_to_stage >= 2:
            ok = ok and self._check_R3(arr)
        if up_to_stage >= 3:
            ok = ok and self._check_R4(arr, self.state["anchor_A"])
        return ok

    # ---------- API ----------

    def reset(self, seed: int = None) -> dict:
        # Seed and deterministic RNG
        self.seed_val = 0 if seed is None else int(seed)
        self.rng = np.random.RandomState(self.seed_val)
        self.steps = 0
        self.done = False
        self.won = False

        # Construct stage-1 target array a1 that satisfies R1 and R2 by design
        # 1) Choose k1_star
        k1_star = int(self.rng.randint(0, 16))

        # 2) Build a1 with properties:
        #    - a1[0] == a1[1] (ensures R1 with diff 0 on edge 0->1)
        #    - pick D in 1..15, set a1[2] = a1[1] + D mod 16 so edge 1->2 has diff D (ensures R2)
        a1 = [0] * 6
        a1[0] = int(self.rng.randint(0, 16))
        a1[1] = a1[0]
        target_D = int(self.rng.randint(1, 16))
        a1[2] = (a1[1] + target_D) % 16
        # Fill remaining with random values (no special constraints needed)
        a1[3] = int(self.rng.randint(0, 16))
        a1[4] = int(self.rng.randint(0, 16))
        a1[5] = int(self.rng.randint(0, 16))

        # 3) Compute ciphertext by inverting T1 with k1_star
        #    Even idx: b[i] = a1[i] - k1_star; Odd idx: b[i] = a1[i] + k1_star (mod16)
        ciphertext = []
        for i, v in enumerate(a1):
            if i % 2 == 0:
                ciphertext.append((v - k1_star) % 16)
            else:
                ciphertext.append((v + k1_star) % 16)

        # 4) Stage-2 target: rotate so that min at index 0
        #    This will always be attainable by some k2
        min_val = min(a1)
        candidate_min_indices = [i for i, v in enumerate(a1) if v == min_val]
        k2_star = candidate_min_indices[0] if candidate_min_indices else 0
        committed1 = list(a1)  # result of correct T1

        # 5) Stage-3 target: pick a solvable anchor_A via a random k3_star
        stage2_base = self._rot_left(committed1, k2_star)
        k3_star = int(self.rng.randint(0, 16))
        # weighted sum after T3(k3_star) equals anchor_A
        # S_after = S_base + 15*k3_star (mod16)
        anchor_A = (self._weighted_sum_mod16(stage2_base) + (15 * k3_star) % 16) % 16

        # Initialize keys and state
        self.state = {
            "stage": 1,
            "ciphertext": ciphertext,
            "k1": 0,
            "k2": 0,
            "k3": 0,
            "work": None,  # lazily computed on apply or queried via render
            "committed1": None,
            "committed2": None,
            "target_D": target_D,
            "anchor_A": anchor_A,
            # hidden solution hints (not exposed): k1_star, k2_star, k3_star (for internal consistency)
            "_hidden_k1_star": k1_star,
            "_hidden_k2_star": k2_star,
            "_hidden_k3_star": k3_star,
        }

        # Initialize a displayed work buffer for player convenience
        self.state["work"] = self._compute_work()
        return self._observation()

    def _observation(self) -> dict:
        # Return an observation sufficient for programmatic play (no hidden fields)
        obs = {
            "stage": self.state["stage"],
            "ciphertext": list(self.state["ciphertext"]),
            "k1": self.state["k1"],
            "k2": self.state["k2"],
            "k3": self.state["k3"],
            "work": list(self.state["work"]) if self.state["work"] is not None else None,
            "committed1": list(self.state["committed1"]) if self.state["committed1"] is not None else None,
            "committed2": list(self.state["committed2"]) if self.state["committed2"] is not None else None,
            "target_D": self.state["target_D"],
            "anchor_A": self.state["anchor_A"],
        }
        # Add a helper: current_rules_satisfied
        stg = self.state["stage"]
        obs["rules_satisfied"] = self._rules_ok(self.state["work"], up_to_stage=stg)
        return obs

    def step(self, action) -> tuple[dict, float, bool, dict]:
        if self.done:
            return self._observation(), 0.0, True, {"msg": "episode_done"}

        if action not in self.valid_actions():
            raise ValueError(f"Invalid action for stage {self.state['stage']}: {action}")

        self.steps += 1

        # Apply action
        stg = self.state["stage"]
        if action == "inc_k1":
            self.state["k1"] = (self.state["k1"] + 1) % 16
        elif action == "dec_k1":
            self.state["k1"] = (self.state["k1"] - 1) % 16
        elif action == "apply_T1":
            self.state["work"] = self._alt_add(self.state["ciphertext"], self.state["k1"])

        elif action == "inc_k2":
            self.state["k2"] = (self.state["k2"] + 1) % 6
        elif action == "dec_k2":
            self.state["k2"] = (self.state["k2"] - 1) % 6
        elif action == "apply_T2":
            base = self.state["committed1"]
            if base is None:
                # Safety: if not committed, base can't exist
                raise RuntimeError("Stage 2 base not set; lock stage 1 first.")
            self.state["work"] = self._rot_left(base, self.state["k2"])

        elif action == "inc_k3":
            self.state["k3"] = (self.state["k3"] + 1) % 16
        elif action == "dec_k3":
            self.state["k3"] = (self.state["k3"] - 1) % 16
        elif action == "apply_T3":
            base = self.state["committed2"]
            if base is None:
                raise RuntimeError("Stage 3 base not set; lock stage 2 first.")
            self.state["work"] = self._add_all(base, self.state["k3"])

        elif action == "lock":
            # Attempt to advance or win
            if stg == 1:
                cand = self._alt_add(self.state["ciphertext"], self.state["k1"])
                if self._rules_ok(cand, up_to_stage=1):
                    self.state["committed1"] = cand
                    self.state["stage"] = 2
                    # Initialize work for stage 2
                    self.state["work"] = self._rot_left(self.state["committed1"], self.state["k2"])
                else:
                    # lock fails; keep stage, work resets to current T1 for visibility
                    self.state["work"] = cand
            elif stg == 2:
                base = self.state["committed1"]
                if base is None:
                    raise RuntimeError("Stage 2 lock attempted without stage 1 commit.")
                cand = self._rot_left(base, self.state["k2"])
                if self._rules_ok(cand, up_to_stage=2):
                    self.state["committed2"] = cand
                    self.state["stage"] = 3
                    # Initialize work for stage 3
                    self.state["work"] = self._add_all(self.state["committed2"], self.state["k3"])
                else:
                    self.state["work"] = cand
            elif stg == 3:
                base = self.state["committed2"]
                if base is None:
                    raise RuntimeError("Stage 3 lock attempted without stage 2 commit.")
                cand = self._add_all(base, self.state["k3"])
                if self._rules_ok(cand, up_to_stage=3):
                    # Win!
                    self.state["work"] = cand
                    self.done = True
                    self.won = True
                else:
                    self.state["work"] = cand

        # Update done for step cap
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            self.won = False

        # Reward only on successful final lock
        reward = 1.0 if (self.done and self.won) else 0.0
        info = {"steps": self.steps, "won": self.won, "done": self.done}
        return self._observation(), reward, self.done, info

    def render(self, mode: str = "text") -> str:
        st = self.state
        lines = []
        lines.append(f"Stage: {st['stage']}{' (WIN)' if self.won else ''}")
        lines.append(f"Ciphertext (hex): {self._to_hex(st['ciphertext'])}")
        lines.append(f"Keys: k1={st['k1']}, k2={st['k2']}, k3={st['k3']}")
        if st.get("committed1") is not None:
            lines.append(f"Committed after Stage 1 (hex): {self._to_hex(st['committed1'])}")
        if st.get("committed2") is not None:
            lines.append(f"Committed after Stage 2 (hex): {self._to_hex(st['committed2'])}")
        if st.get("work") is not None:
            lines.append(f"Work buffer (hex): {self._to_hex(st['work'])}")
        lines.append(f"Targets: R2 target_D={st['target_D']}, R4 anchor_A={st['anchor_A']}")
        # Rules status
        stg = st["stage"]
        arr = st["work"] if st["work"] is not None else self._compute_work()
        r1 = self._check_R1(arr)
        r2 = self._check_R2(arr, st["target_D"])
        r3 = self._check_R3(arr)
        r4 = self._check_R4(arr, st["anchor_A"])
        lines.append(f"Rules now: R1={'OK' if r1 else 'no'}, R2={'OK' if r2 else 'no'}, R3={'OK' if r3 else 'no'}, R4={'OK' if r4 else 'no'}")
        return "\n".join(lines)

    def valid_actions(self) -> list:
        stg = self.state.get("stage", 1)
        if stg == 1:
            return ["inc_k1", "dec_k1", "apply_T1", "lock"]
        elif stg == 2:
            return ["inc_k2", "dec_k2", "apply_T2", "lock"]
        elif stg == 3:
            return ["inc_k3", "dec_k3", "apply_T3", "lock"]
        else:
            return []

    def is_won(self) -> bool:
        return bool(self.won)


if __name__ == "__main__":
    # Simple sanity run
    env = PrimitivBenchGame()
    obs = env.reset(seed=1019)
    print(env.render())
    # Random agent rollout
    rs = np.random.RandomState(0)
    for t in range(10):
        acts = env.valid_actions()
        a = acts[rs.randint(0, len(acts))]
        o, r, d, i = env.step(a)
        print(f"\nAction: {a}")
        print(env.render())
        if d:
            break
