from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any, Dict
from skyrl_gym.envs.ifeval.ifeval_utils import compute_score


class IFEvalEnv(BaseTextEnv):
    """Single-turn environment for IFEval (Instruction-Following Evaluation).

    Reward = fraction of verifiable constraints satisfied (0.0-1.0).
    ground_truth is a JSON string with instruction_id_list and kwargs.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _get_reward(self, action: str) -> float:
        return compute_score(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward = self._get_reward(action)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
