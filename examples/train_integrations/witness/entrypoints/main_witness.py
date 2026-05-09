import os
import sys

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
import ray
from skyrl_gym.envs import register


def _strip_hydra_prefixes(args: list[str]) -> list[str]:
    """Strip Hydra ++ and + prefixes from CLI args."""
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
    # Ensure arc-witness-envs is importable in this Ray worker
    witness_dir = os.environ.get("WITNESS_ENVS_DIR", os.path.expanduser("~/arc-witness-envs"))
    if witness_dir and witness_dir not in sys.path:
        sys.path.insert(0, witness_dir)
    # Also ensure arc-witness-agent (new B7 RL agent) is importable.
    agent_dir = os.environ.get("ARC_WITNESS_AGENT_DIR", os.path.expanduser("~/arc-witness-agent"))
    if agent_dir and agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    # v3 GRPO env (light-harness, retained for backward compat).
    register(
        id="witness",
        entry_point="examples.train_integrations.witness.env:WitnessEnv",
    )
    # v5b7 RL env (full ORAI agent; bridged mode). Dataset rows must set
    # `env_class: "witness_agent"` to use this.
    register(
        id="witness_agent",
        entry_point="examples.train_integrations.witness.env_agent:WitnessAgentEnv",
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
