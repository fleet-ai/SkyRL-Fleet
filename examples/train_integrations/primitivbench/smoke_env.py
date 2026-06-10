"""Keyless local smoke test for PrimitivBenchEnv.

Stubs skyrl_gym (so no SkyRL install needed), loads the env, plays one
episode per portfolio game with a scripted (cycling-index) player, and
prints turn counts / rewards / terminal states.

Run: python3 smoke_env.py [--games-dir <curated_v2/games>] [--limit 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- stub skyrl_gym before importing env.py ----
if "skyrl_gym" not in sys.modules:
    skyrl_gym = types.ModuleType("skyrl_gym")
    envs = types.ModuleType("skyrl_gym.envs")
    base = types.ModuleType("skyrl_gym.envs.base_text_env")

    class BaseTextEnv:  # minimal protocol stand-in
        def __init__(self):
            self.turns = 0
            self.max_turns = 1

    def BaseTextEnvStepOutput(**kw):  # the real one is a TypedDict
        return dict(kw)

    base.BaseTextEnv = BaseTextEnv
    base.BaseTextEnvStepOutput = BaseTextEnvStepOutput
    base.ConversationType = list
    envs.base_text_env = base
    skyrl_gym.envs = envs
    sys.modules["skyrl_gym"] = skyrl_gym
    sys.modules["skyrl_gym.envs"] = envs
    sys.modules["skyrl_gym.envs.base_text_env"] = base

sys.path.insert(0, _HERE)
import env as pb_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.games_dir:
        os.environ["PRIMITIVBENCH_GAMES_DIR"] = args.games_dir

    gdir = pb_env._games_dir()
    games = sorted(d for d in os.listdir(gdir) if os.path.isdir(os.path.join(gdir, d)))
    if args.limit:
        games = games[: args.limit]
    print(f"smoke: {len(games)} games from {gdir}")

    for name in games:
        e = pb_env.PrimitivBenchEnv(None, {"game_name": name, "seed": 1000, "max_turns": 20})
        chat, meta = e.init([])
        assert chat[0]["role"] == "system" and "Valid actions" in chat[1]["content"], "init malformed"
        total_r, turns, done, won = 0.0, 0, False, False
        i = 0
        while not done and turns < 25:
            out = e.step(f"I'll try option {i % 7}. <action>{i % 7}</action>")
            total_r += out["reward"]
            done = out["done"]
            won = out["metadata"].get("won", False)
            turns += 1
            i += 1
        print(f"  {name:42s} turns={turns:2d} reward={total_r:+.2f} done={done} won={won}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
