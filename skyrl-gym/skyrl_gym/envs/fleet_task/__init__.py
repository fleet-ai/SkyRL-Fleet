"""Fleet Task Environment for SkyRL-Gym.

Provides a multi-turn tool-use environment backed by Fleet-hosted environments,
using OpenEnv's FleetTaskEnv as the abstraction layer.

FleetTaskEnv is lazy-loaded via PEP 562 `__getattr__` so leaf modules
(families, tool_call_parser) can be imported without pulling env.py's
heavyweight deps (mcp, loguru, openenv). The tinker_shim consumes
families + parser directly without needing the training stack installed.
"""

from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call

__all__ = ["FleetTaskEnv", "parse_tool_call"]


def __getattr__(name):
    if name == "FleetTaskEnv":
        from skyrl_gym.envs.fleet_task.env import FleetTaskEnv
        return FleetTaskEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
