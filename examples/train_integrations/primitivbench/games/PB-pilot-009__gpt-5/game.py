import numpy as np
from typing import Tuple, List, Dict, Any


class PrimitivBenchGame:
    """
    Sorting-with-transform puzzle:
    - 8 tokens arrive in a fixed queue. Each token has a raw color label in {0..K-1}, K in [3..5].
    - The agent may:
        * Assign the next token to one of 3 bins (append-only).
        * Set a global swap rule (a,b) that swaps all occurrences of colors a and b (identity allowed if a==b).
    - Win condition (checked by is_won()):
        * All 8 tokens have been assigned (no unassigned tokens remain).
        * The count of tokens in each bin exactly matches the provided target distribution.
        * Under the CURRENT swap rule, each bin's sequence of transformed colors is nondecreasing (monotonic gradient).
        * The swap rule must have been explicitly documented at least once via set_swap (documented=True).
    - Deterministic given seed; bounded by a per-episode step limit.
    - Dense reward shaping based on progress toward both rules; +1 on win.

    Required primitives engagement:
    - A1 Objectness: next-token identity and persistent per-token placement are essential.
    - B1 Perception transform: monotonic checks use transformed colors under the current swap, which the agent selects.
    - B3 Multi-rule composition: both monotonic-per-bin and per-bin counts must be satisfied simultaneously; satisfying
      either alone fails the win condition.
    """

    def __init__(self, max_steps: int = 200):
        self.N = 8
        self.max_steps = max_steps
        self._rng = None  # numpy Generator
        self._seed = None

        # Puzzle parameters (set in reset)
        self.palette_size = None  # K
        self.target_counts: List[int] = None  # length 3, sum to N
        self.queue_raw: List[int] = None  # length N, raw colors
        self._secret_swap: Tuple[int, int] = None  # used only for instance construction (not exposed)

        # Dynamic state
        self.next_index = 0
        self.bins_raw: List[List[int]] = None  # 3 lists of raw colors in append order
        self.current_swap: Tuple[int, int] = None
        self.documented: bool = False
        self.steps = 0

        # Progress shaping
        self.prev_progress_score = 0.0

    # ---------------- Core API ----------------

    def reset(self, seed: int = None) -> dict:
        """
        Resets to a new puzzle instance, reproducible by seed.
        Returns the initial observation dict.
        """
        self._seed = int(seed) if seed is not None else None
        self._rng = np.random.default_rng(self._seed)

        self.steps = 0
        self.next_index = 0
        self.documented = False

        # Palette size K in [3..5]
        self.palette_size = int(self._rng.integers(3, 6))
        colors = list(range(self.palette_size))

        # Choose a secret swap pair (a,b) with a!=b
        a, b = self._rng.choice(colors, size=2, replace=False)
        self._secret_swap = (int(a), int(b))

        # Choose target distribution counts (positive) summing to N
        # Sample two cut points in {1..N-1}, unique and sorted
        c1, c2 = sorted(self._rng.choice(np.arange(1, self.N), size=2, replace=False))
        c0 = c1
        c1b = c2 - c1
        c2b = self.N - c2
        self.target_counts = [int(c0), int(c1b), int(c2b)]

        # Construct per-bin transformed nondecreasing sequences
        per_bin_transformed = []
        for j in range(3):
            length = self.target_counts[j]
            # Sample with replacement from 0..K-1 then sort
            seq = list(np.sort(self._rng.integers(0, self.palette_size, size=length)))
            per_bin_transformed.append([int(x) for x in seq])

        # Convert transformed sequences back to raw via the (secret) swap (self-inverse)
        per_bin_raw = []
        for j in range(3):
            raw_seq = [self._apply_swap_to_color(x, self._secret_swap) for x in per_bin_transformed[j]]
            per_bin_raw.append(raw_seq)

        # Riffle merge preserving per-bin internal order to form the queue
        order_labels = []
        for j in range(3):
            order_labels += [j] * len(per_bin_raw[j])
        order_labels = list(order_labels)
        self._rng.shuffle(order_labels)
        # Pointers for per-bin popping
        ptrs = [0, 0, 0]
        queue_raw = []
        for lbl in order_labels:
            idx = ptrs[lbl]
            queue_raw.append(per_bin_raw[lbl][idx])
            ptrs[lbl] += 1

        self.queue_raw = queue_raw

        # Initialize empty bins
        self.bins_raw = [[], [], []]

        # Start with identity swap (0,0) as default (identity regardless of palette),
        # but must be explicitly set (documented) to satisfy win condition.
        self.current_swap = (0, 0)
        self.documented = False

        self.prev_progress_score = self._progress_score()

        return self._observation()

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """
        Executes an action:
        - ('assign', bin_index) to append the next token to a bin.
        - ('set_swap', a, b) to set the global swap rule (identity allowed via a==b).
        Returns: (state, reward, done, info)
        """
        if not isinstance(action, (tuple, list)) or len(action) < 2:
            raise ValueError("Action must be a tuple: ('assign', bin) or ('set_swap', a, b)")

        kind = action[0]

        if kind == 'assign':
            if len(action) != 2:
                raise ValueError("Assign action format: ('assign', bin_index)")
            bin_index = int(action[1])
            if bin_index not in (0, 1, 2):
                raise ValueError("Invalid bin index")
            if self.next_index >= self.N:
                raise ValueError("No tokens left to assign")
            # Append next token
            raw_color = self.queue_raw[self.next_index]
            self.bins_raw[bin_index].append(raw_color)
            self.next_index += 1

        elif kind == 'set_swap':
            if len(action) != 3:
                raise ValueError("set_swap action format: ('set_swap', a, b)")
            a = int(action[1])
            b = int(action[2])
            if not (0 <= a < self.palette_size and 0 <= b < self.palette_size):
                raise ValueError("Swap colors must be within current palette size")
            self.current_swap = (a, b)
            self.documented = True
        else:
            raise ValueError("Unknown action type")

        self.steps += 1

        # Compute reward
        progress = self._progress_score()
        reward = progress - self.prev_progress_score
        won = self.is_won()
        if won:
            reward += 1.0  # terminal bonus
        self.prev_progress_score = progress

        done = won or (self.steps >= self.max_steps)

        info = {
            "won": won,
            "reason": ("max_steps" if (not won and self.steps >= self.max_steps) else ("win" if won else "in_progress"))
        }
        return self._observation(), float(reward), bool(done), info

    def render(self, mode: str = "text") -> str:
        """
        Human-readable text rendering of the current state.
        """
        if mode != "text":
            raise ValueError("Only text mode is supported")

        lines = []
        lines.append(f"Tokens remaining: {self.N - self.next_index} / {self.N}")
        counts = [len(b) for b in self.bins_raw]
        lines.append(f"Target counts: {self.target_counts} | Current counts: {counts}")
        lines.append(f"Current swap: {self.current_swap} (documented={self.documented})")
        next_tok = (self.queue_raw[self.next_index] if self.next_index < self.N else None)
        lines.append(f"Next token (raw): {next_tok}")

        # Show bins with raw and transformed views and monotonicity
        for i, br in enumerate(self.bins_raw):
            transformed = [self._apply_swap_to_color(c, self.current_swap) for c in br]
            mono = self._is_nondecreasing(transformed)
            lines.append(f"Bin {i}: raw={br} | transformed={transformed} | monotonic={mono}")

        won = self.is_won()
        lines.append(f"Win status: {won}")

        return "\n".join(lines)

    def valid_actions(self) -> list:
        """
        Lists legal actions in current state.
        Always includes all possible ('set_swap', a, b).
        Includes ('assign', b) for b in {0,1,2} only if a token remains.
        """
        acts = []
        if self.next_index < self.N:
            acts += [('assign', 0), ('assign', 1), ('assign', 2)]
        for a in range(self.palette_size):
            for b in range(self.palette_size):
                acts.append(('set_swap', a, b))
        return acts

    def is_won(self) -> bool:
        """
        Checks the win condition.
        """
        if self.next_index != self.N:
            return False
        if not self.documented:
            return False
        # Counts must match exactly
        cur_counts = [len(b) for b in self.bins_raw]
        if cur_counts != self.target_counts:
            return False
        # Each bin must be nondecreasing under current swap
        for br in self.bins_raw:
            transformed = [self._apply_swap_to_color(c, self.current_swap) for c in br]
            if not self._is_nondecreasing(transformed):
                return False
        return True

    # ---------------- Internals ----------------

    def _apply_swap_to_color(self, color: int, swap: Tuple[int, int]) -> int:
        a, b = swap
        if color == a:
            return b
        if color == b:
            return a
        return color

    def _is_nondecreasing(self, seq: List[int]) -> bool:
        return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))

    def _progress_score(self) -> float:
        """
        Dense progress in [0,1] combining:
        - counts proximity (L1 distance normalized)
        - number of monotonic bins under current swap
        """
        # Counts proximity
        cur_counts = np.array([len(b) for b in self.bins_raw], dtype=int)
        tgt = np.array(self.target_counts, dtype=int)
        l1 = int(np.sum(np.abs(cur_counts - tgt)))
        # Maximum possible L1 between two size-N 3-bin histograms is 2N
        counts_score = 1.0 - (l1 / (2.0 * self.N))

        # Monotonic bins
        mono_bins = 0
        for br in self.bins_raw:
            transformed = [self._apply_swap_to_color(c, self.current_swap) for c in br]
            if self._is_nondecreasing(transformed):
                mono_bins += 1
        monotonic_score = mono_bins / 3.0

        # Combine equally
        return max(0.0, min(1.0, 0.5 * counts_score + 0.5 * monotonic_score))

    def _observation(self) -> Dict[str, Any]:
        return {
            "next_token_raw": (self.queue_raw[self.next_index] if self.next_index < self.N else None),
            "bins_raw": [list(b) for b in self.bins_raw],
            "current_swap": tuple(self.current_swap),
            "documented": bool(self.documented),
            "target_counts": list(self.target_counts),
            "counts_so_far": [len(b) for b in self.bins_raw],
            "remaining_to_assign": self.N - self.next_index,
            "palette_size": int(self.palette_size),
            "steps": int(self.steps),
        }


if __name__ == "__main__":
    # Simple manual smoke test
    env = PrimitivBenchGame()
    s = env.reset(seed=1009)
    print(env.render())
    for _ in range(5):
        va = env.valid_actions()
        # pick a random valid action for demo
        a = va[np.random.randint(0, len(va))]
        s, r, d, info = env.step(a)
        print(f"\nAction: {a}, reward={r:.3f}, done={d}, info={info}")
        print(env.render())
        if d:
            break
