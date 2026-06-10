"""
PrimitivBench task PB-pilot-018
Symbolic rewrite puzzle eliciting:
  B1 perception_transform (view filter on observation)
  B2 meta_reasoning       (suspend-rule meta action)
  B3 multi_rule_composition (R1 swap + R2 reverse fire simultaneously)
"""
from typing import Any
import random


ALPHABET = ['A', 'B', 'C', 'D']
LEN = 6
MAX_STEPS = 40


class PrimitivBenchGame:
    def __init__(self):
        self.state = None  # underlying truth (list of chars)
        self.target = None
        self.visible = None  # dict symbol -> bool (True = visible in render)
        self.suspend_next = None  # None | 'R1' | 'R2'
        self.steps = 0
        self.done = False
        self._won = False

    # ------------- core lifecycle -------------
    def reset(self, seed: int = None) -> dict:
        rng = random.Random(seed if seed is not None else 1018)
        # Choose a fixed-difficulty puzzle deterministic in seed.
        # Pick start, then construct a 2-step optimal path:
        #   step1: APPLY (R1+R2)
        #   step2: SUSPEND R2; APPLY (only R1)  -- requires meta-reasoning
        # This guarantees a solvable puzzle.
        start = [rng.choice(ALPHABET) for _ in range(LEN)]
        # ensure non-trivial: at least 2 distinct symbol types
        while len(set(start)) < 2:
            start = [rng.choice(ALPHABET) for _ in range(LEN)]

        # build target by applying a known op chain
        s1 = list(start)
        x, y = rng.sample(ALPHABET, 2)
        i = rng.randrange(0, LEN - 1)
        j = rng.randrange(i + 1, LEN)
        s1 = self._apply(s1, x, y, i, j, do_r1=True, do_r2=True)

        # second step: only R1 (suspend R2)
        x2, y2 = rng.sample(ALPHABET, 2)
        # i2/j2 still need to be provided but R2 is suspended
        i2 = rng.randrange(0, LEN - 1)
        j2 = rng.randrange(i2 + 1, LEN)
        s2 = self._apply(s1, x2, y2, i2, j2, do_r1=True, do_r2=False)

        self.state = list(start)
        self.target = list(s2)
        # store canonical solution for traces/debug only
        self._solution = [
            ('APPLY', x, y, i, j),
            ('SUSPEND', 'R2'),
            ('APPLY', x2, y2, i2, j2),
        ]

        # view filter starts: all symbols visible EXCEPT one chosen symbol
        # forces toggling to see truth fully -> perception transform
        self.visible = {s: True for s in ALPHABET}
        hidden = rng.choice(ALPHABET)
        self.visible[hidden] = False

        self.suspend_next = None
        self.steps = 0
        self.done = False
        self._won = (self.state == self.target)
        return self._observe()

    # ------------- rewrite primitives -------------
    @staticmethod
    def _apply(s, x, y, i, j, do_r1=True, do_r2=True):
        s = list(s)
        # R1 and R2 fire simultaneously: compute both from same input.
        # Implementation: build R1 result, build R2 result, then COMPOSE
        # by treating both as acting on the original; we define
        # "simultaneous" as: apply R1 globally first to produce s', then
        # R2 reverses the substring of s'. Either may be suspended.
        # The "simultaneous" semantics: both transformations contribute
        # to the single step's output; suspending one removes its
        # contribution.
        if do_r1:
            s = [y if c == x else (x if c == y else c) for c in s]
        if do_r2:
            sub = s[i:j+1]
            sub.reverse()
            s = s[:i] + sub + s[j+1:]
        return s

    # ------------- gym API -------------
    def step(self, action):
        if self.done:
            return self._observe(), 0.0, True, {'msg': 'episode over'}
        self.steps += 1
        info = {}
        reward = 0.0

        if not isinstance(action, tuple) or len(action) == 0:
            info['error'] = 'malformed action'
            return self._finish(reward, info)

        kind = action[0]

        if kind == 'NOOP':
            pass

        elif kind == 'TOGGLE':
            # ('TOGGLE', symbol)
            if len(action) != 2 or action[1] not in ALPHABET:
                info['error'] = 'bad toggle'
            else:
                sym = action[1]
                self.visible[sym] = not self.visible[sym]

        elif kind == 'SUSPEND':
            # ('SUSPEND', 'R1' | 'R2')
            if len(action) != 2 or action[1] not in ('R1', 'R2'):
                info['error'] = 'bad suspend'
            else:
                self.suspend_next = action[1]

        elif kind == 'APPLY':
            # ('APPLY', x, y, i, j)
            if len(action) != 5:
                info['error'] = 'bad apply arity'
            else:
                _, x, y, i, j = action
                if (x not in ALPHABET or y not in ALPHABET or x == y
                        or not (0 <= i < j < LEN)):
                    info['error'] = 'bad apply args'
                else:
                    do_r1 = self.suspend_next != 'R1'
                    do_r2 = self.suspend_next != 'R2'
                    self.state = self._apply(
                        self.state, x, y, i, j, do_r1=do_r1, do_r2=do_r2
                    )
                    self.suspend_next = None  # suspension consumed
        else:
            info['error'] = f'unknown action {kind}'

        # check win
        if self.state == self.target:
            self._won = True
            self.done = True
            reward = 1.0
        elif self.steps >= MAX_STEPS:
            self.done = True
            reward = 0.0

        return self._finish(reward, info)

    def _finish(self, reward, info):
        return self._observe(), reward, self.done, info

    # ------------- observation / perception transform -------------
    def _filter(self, s):
        return ''.join(c if self.visible.get(c, True) else '?' for c in s)

    def _observe(self):
        return {
            'view': self._filter(self.state),
            'target_view': self._filter(self.target),
            'visible': dict(self.visible),
            'suspend_next': self.suspend_next,
            'steps': self.steps,
            'max_steps': MAX_STEPS,
        }

    def render(self, mode: str = 'text') -> str:
        lines = [
            f"step {self.steps}/{MAX_STEPS}",
            f"view   : {self._filter(self.state)}    (raw: {''.join(self.state)})",
            f"target : {self._filter(self.target)}    (raw: {''.join(self.target)})",
            f"visible: {self.visible}",
            f"suspend_next: {self.suspend_next}",
            f"done={self.done} won={self._won}",
        ]
        return '\n'.join(lines)

    def valid_actions(self) -> list:
        acts = [('NOOP',)]
        for s in ALPHABET:
            acts.append(('TOGGLE', s))
        for r in ('R1', 'R2'):
            acts.append(('SUSPEND', r))
        for x in ALPHABET:
            for y in ALPHABET:
                if x == y:
                    continue
                for i in range(LEN - 1):
                    for j in range(i + 1, LEN):
                        acts.append(('APPLY', x, y, i, j))
        return acts

    def is_won(self) -> bool:
        return self._won


# ------------- self-test / random rollouts when run directly -------------
if __name__ == '__main__':
    import sys
    g = PrimitivBenchGame()
    obs = g.reset(seed=1018)
    print(g.render())
    print('solution oracle:', g._solution)
    for act in g._solution:
        obs, r, d, info = g.step(act)
        print('->', act)
        print(g.render())
        print()
    print('won?', g.is_won())
