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
    register(
        id="witness",
        entry_point="examples.train_integrations.witness.env:WitnessEnv",
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
