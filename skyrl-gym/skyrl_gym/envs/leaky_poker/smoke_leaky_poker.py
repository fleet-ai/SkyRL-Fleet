#!/usr/bin/env python3
"""No-GPU / no-network smoke test for LeakyPokerEnv: scripted villain + stub reader.
Exercises init -> step loop, dense reward, validity gate, metrics, and both reward modes.

  cd /workspace/allie
  /workspace/allie/performative/.venv/bin/python \
    skyrl-neg-wt/skyrl-gym/skyrl_gym/envs/leaky_poker/smoke_leaky_poker.py
"""
import random
import re
import sys

sys.path.insert(0, "/workspace/allie/skyrl-neg-wt/skyrl-gym")
sys.path.insert(0, "/workspace/allie/TextArena")

from skyrl_gym.envs.leaky_poker.env import LeakyPokerEnv, POKER_SYSTEM  # noqa: E402


def hero_policy(obs, rng):
    m = re.search(r"Current bet: (\d+)", obs)
    cur = int(m.group(1)) if m else 0
    if cur == 0:
        return rng.choice(["[Bet 40]", "[Check]", "[Bet 80]"])
    return rng.choice(["[Call]", "[Fold]", "[Raise 40]"])


def run(reward_mode, seed=3):
    env = LeakyPokerEnv(
        env_config={"opponent_mode": "scripted", "reader_mode": "stub",
                    "reward_mode": reward_mode, "leak_lambda": 100.0, "num_rounds": 4,
                    "hold_lie_rate": True},
        extras={"reward_spec": {"ground_truth": {"seed": seed, "num_rounds": 4, "hero": 0}},
                "max_turns": 80},
    )
    prompt, _ = env.init([{"role": "system", "content": POKER_SYSTEM}])
    rng = random.Random(seed)
    obs = prompt[-1]["content"]
    total, steps = 0.0, 0
    while steps < 200:
        action = hero_policy(obs, rng)
        out = env.step(action)
        total += out["reward"]
        steps += 1
        if out["done"]:
            break
        obs = out["observations"][0]["content"]
    env.close()
    m = env.get_metrics()
    print(f"[{reward_mode:6}] steps={steps} total_reward={total:8.2f} "
          f"hero_net={m['hero_net_chips']:+.0f} bluffs={m['n_bluffs']:.0f} "
          f"lie_rate={m['lie_rate']:.2f} mean_leak={m['mean_bluff_leakage']:.2f} "
          f"foldrate={m['bluff_fold_rate']:.2f} invalid={m['invalid_rate']:.2f}")
    return m


if __name__ == "__main__":
    print("LeakyPokerEnv smoke (scripted villain, stub reader):")
    run("dense")
    run("sparse")
    print("OK")
