"""GRPO entrypoint for the PrimitivBench mini-flywheel.

Registers BOTH env classes so a single mixed dataset (arm B: witness rows +
primitivbench rows, each row carrying its own `env_class`) trains in one run.

Mirrors examples/train_integrations/witness/entrypoints/main_witness.py.
"""

import os
import sys

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
import ray
from skyrl_gym.envs import register


def _strip_hydra_prefixes(args: list[str]) -> list[str]:
    cleaned = []
    for arg in args:
        if arg.startswith("++"):
            cleaned.append(arg[2:])
        elif arg.startswith("+"):
            cleaned.append(arg[1:])
        else:
            cleaned.append(arg)
    return cleaned


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # witness deps (arm A/B/C rows)
    witness_dir = os.environ.get("WITNESS_ENVS_DIR", os.path.expanduser("~/arc-witness-envs"))
    if witness_dir and witness_dir not in sys.path:
        sys.path.insert(0, witness_dir)
    agent_dir = os.environ.get("ARC_WITNESS_AGENT_DIR", os.path.expanduser("~/arc-witness-agent"))
    if agent_dir and agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    register(
        id="witness",
        entry_point="examples.train_integrations.witness.env:WitnessEnv",
    )
    # v5b7 rows: merged witness parquets may carry env_class="witness_agent"
    register(
        id="witness_agent",
        entry_point="examples.train_integrations.witness.env_agent:WitnessAgentEnv",
    )
    register(
        id="primitivbench",
        entry_point="examples.train_integrations.primitivbench.env:PrimitivBenchEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(_strip_hydra_prefixes(sys.argv[1:]))
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
