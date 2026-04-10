"""Gym-Anything Environment for SkyRL-Gym.

Wraps gym-anything's GymAnythingEnv (Docker-based desktop software environments)
as a SkyRL-compatible BaseTextEnv for computer-use RL training.

Action flow:
  Model output (Qwen [0,1000] coords) → parse tool_call → scale to pixels →
  convert to gym-anything action dict → env.step() → capture screenshot →
  return as base64 multimodal observation

Reward: from gym-anything's programmatic verifier (0-100 score, normalized to 0-1).
"""

import base64
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)

# Global cache for task index files
_TASK_INDEX_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# Cache sysbox-runc availability check
_SYSBOX_AVAILABLE: Optional[bool] = None


def _sysbox_available() -> bool:
    """Check whether sysbox-runc is registered with Docker (cached)."""
    global _SYSBOX_AVAILABLE
    if _SYSBOX_AVAILABLE is not None:
        return _SYSBOX_AVAILABLE
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True, timeout=5,
        )
        _SYSBOX_AVAILABLE = b"sysbox-runc" in result.stdout
    except Exception:
        _SYSBOX_AVAILABLE = False
    return _SYSBOX_AVAILABLE


def load_task_index(tasks_file: str) -> List[Dict[str, Any]]:
    """Load task index JSON. Format: list of {env_dir, task_id, env_id, description, ...}."""
    if tasks_file not in _TASK_INDEX_CACHE:
        path = os.path.expanduser(tasks_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task index not found: {path}")
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "tasks" in data:
            data = data["tasks"]
        _TASK_INDEX_CACHE[tasks_file] = {t["task_key"]: t for t in data}
    return _TASK_INDEX_CACHE[tasks_file]


class GymAnythingTaskEnv(BaseTextEnv):
    """SkyRL environment wrapping gym-anything desktop software environments.

    Constructor signature follows upstream convention:
        __init__(self, env_config=None, extras={})

    Where:
        env_config: Dict from skyrl_gym_config YAML (gym_anything section)
        extras: Per-sample data from training dataset (task_key, env_dir, task_id, etc.)
    """

    def __init__(self, env_config=None, extras: Dict[str, Any] = {}):
        super().__init__()

        if env_config is None:
            env_config = {}

        self.extras = extras
        self.max_turns = extras.get("max_turns", 50)

        # Task identification
        self.task_key = extras.get("task_key")
        self.tasks_file = (
            env_config.get("tasks_file") if hasattr(env_config, "get") else None
        ) or extras.get("tasks_file")

        if not self.task_key:
            raise ValueError("task_key must be provided in extras")

        # Load task config from index
        if self.tasks_file:
            tasks = load_task_index(self.tasks_file)
            self.task_config = tasks.get(self.task_key)
            if not self.task_config:
                raise ValueError(f"Task '{self.task_key}' not found in {self.tasks_file}")
        else:
            # Direct config in extras
            self.task_config = extras

        # Gym-anything env/task paths
        self.env_dir = self.task_config.get("env_dir")
        self.task_id = self.task_config.get("task_id")

        if not self.env_dir:
            raise ValueError(f"env_dir must be provided for task {self.task_key}")

        # Gym-anything settings
        self.use_cache = env_config.get("use_cache", True) if hasattr(env_config, "get") else True
        self.cache_level = env_config.get("cache_level", "post_start") if hasattr(env_config, "get") else "post_start"

        # Screen dimensions (from env.json, typically 1920x1080)
        self.screen_width = 1920
        self.screen_height = 1080

        # State
        self.ga_env = None  # GymAnythingEnv instance
        self.chat_history: ConversationType = []
        self.turns = 0
        self.last_reward: Optional[float] = None

    def _build_system_prompt(self, task_description: str) -> str:
        """Build system prompt with computer_use tool definition for Qwen VL."""
        current_date = datetime.now().strftime("%Y-%m-%d")

        tools_def = {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": (
                    f"Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
                    f"* This is an interface to a desktop GUI. You do not have access to a terminal.\n"
                    f"* Some applications may take time to start or process actions, so you may need to wait.\n"
                    f"* The screen's resolution is 1000x1000.\n"
                    f"* Coordinates use a [0, 1000] grid. (0,0) is top-left, (999,999) is bottom-right.\n"
                    f"* Click the center of elements, not their edges."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "key", "type", "mouse_move", "click", "left_click",
                                "drag", "right_click", "middle_click", "double_click",
                                "triple_click", "scroll", "wait", "screenshot", "terminate",
                            ],
                            "description": "The action to perform.",
                        },
                        "keys": {
                            "type": "array",
                            "description": "Key names for action=key (e.g. [\"ctrl\", \"s\"]).",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type for action=type.",
                        },
                        "coordinate": {
                            "type": "array",
                            "description": "The [x, y] coordinates (0-999) for mouse actions.",
                        },
                        "coordinate2": {
                            "type": "array",
                            "description": "End [x, y] coordinates for action=drag.",
                        },
                        "pixels": {
                            "type": "number",
                            "description": "Scroll amount in pixels (positive=down).",
                        },
                        "time": {
                            "type": "number",
                            "description": "Seconds to wait for action=wait.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["success", "failure"],
                            "description": "Task completion status for action=terminate.",
                        },
                    },
                },
            },
        }

        tools_json = json.dumps([tools_def], indent=2)

        return (
            f"You are a helpful agent controlling a desktop application. Complete the task by interacting with the GUI.\n\n"
            f"## Current Date\n{current_date}\n\n"
            f"## Interaction Strategy\n"
            f"1. **Act**: Perform ONE action (click, type, scroll, etc.)\n"
            f"2. **Observe**: Take a screenshot to see the result\n"
            f"3. **Adapt**: If the screen hasn't changed, try a DIFFERENT action\n\n"
            f"Key rules:\n"
            f"- After clicking or typing, ALWAYS take a screenshot next to see what happened\n"
            f"- NEVER repeat the same action more than twice\n"
            f"- Use wait() only ONCE after actions that take time, then screenshot\n"
            f"- When done, use action=terminate with status=success or status=failure\n\n"
            f"## Available Tools\n{tools_json}\n\n"
            f"## Tool Call Format\n"
            f"Use: <tool_call>{{\"name\": \"computer_use\", \"arguments\": {{...}}}}</tool_call>\n\n"
            f"## Response Format\n"
            f"EVERY response MUST end with exactly ONE of:\n"
            f"1. A tool call: <tool_call>...</tool_call>\n"
            f"2. Done signal: <done> - ONLY when the task is fully complete\n"
        )

    def _scale_coordinate(self, x: int, y: int) -> Tuple[int, int]:
        """Convert Qwen [0,1000] coordinates to pixel coordinates."""
        px = int(x / 1000 * self.screen_width)
        py = int(y / 1000 * self.screen_height)
        return px, py

    def _tool_call_to_ga_actions(self, tool_call: Dict[str, Any]) -> Tuple[List[Dict], bool, bool]:
        """Convert parsed tool_call to gym-anything action dicts.

        Returns:
            (actions, is_terminal, is_screenshot_only)
        """
        args = tool_call.get("arguments", {})
        action_type = args.get("action", "")

        if action_type == "terminate":
            return [], True, False

        if action_type == "screenshot":
            return [], False, True

        if action_type == "wait":
            wait_time = args.get("time", 1.0)
            return [{"action": "wait", "time": wait_time}], False, False

        if action_type == "key":
            keys = args.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            return [{"keyboard": {"keys": keys}}], False, False

        if action_type == "type":
            actions = []
            if args.get("clear"):
                actions.append({"keyboard": {"keys": ["ctrl", "a"]}})
            actions.append({"keyboard": {"text": args.get("text", "")}})
            if args.get("enter"):
                actions.append({"keyboard": {"keys": ["Return"]}})
            return actions, False, False

        if action_type == "mouse_move":
            coord = args.get("coordinate", [500, 500])
            x, y = self._scale_coordinate(coord[0], coord[1])
            return [{"mouse": {"move": [x, y]}}], False, False

        if action_type in ("click", "left_click"):
            coord = args.get("coordinate", [500, 500])
            x, y = self._scale_coordinate(coord[0], coord[1])
            return [{"mouse": {"left_click": [x, y]}}], False, False

        if action_type == "right_click":
            coord = args.get("coordinate", [500, 500])
            x, y = self._scale_coordinate(coord[0], coord[1])
            return [{"mouse": {"right_click": [x, y]}}], False, False

        if action_type == "double_click":
            coord = args.get("coordinate", [500, 500])
            x, y = self._scale_coordinate(coord[0], coord[1])
            return [{"mouse": {"double_click": [x, y]}}], False, False

        if action_type == "triple_click":
            coord = args.get("coordinate", [500, 500])
            x, y = self._scale_coordinate(coord[0], coord[1])
            return [{"mouse": {"triple_click": [x, y]}}], False, False

        if action_type in ("drag", "left_click_drag"):
            coord = args.get("coordinate", [500, 500])
            coord2 = args.get("coordinate2", coord)
            x1, y1 = self._scale_coordinate(coord[0], coord[1])
            x2, y2 = self._scale_coordinate(coord2[0], coord2[1])
            return [{"mouse": {"left_click_drag": [[x1, y1], [x2, y2]]}}], False, False

        if action_type == "middle_click":
            coord = args.get("coordinate", [500, 500])
            x, y = self._scale_coordinate(coord[0], coord[1])
            return [{"mouse": {"left_click": [x, y]}}], False, False  # fallback

        if action_type == "scroll":
            pixels = args.get("pixels", 0)
            actions = []
            if "coordinate" in args:
                coord = args["coordinate"]
                x, y = self._scale_coordinate(coord[0], coord[1])
                actions.append({"mouse": {"move": [x, y]}})
            actions.append({"mouse": {"scroll": int(pixels)}})
            return actions, False, False

        logger.warning(f"Unknown action type: {action_type}, taking screenshot")
        return [], False, True

    def _screenshot_to_base64(self, obs: Dict[str, Any]) -> Optional[str]:
        """Extract screenshot from gym-anything observation and encode as base64."""
        screen = obs.get("screen", {})
        # If inline base64 is already available
        if "png_b64" in screen:
            return screen["png_b64"]
        # Otherwise read from file path
        path = screen.get("path")
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        return None

    def _make_screenshot_observation(self, screenshot_b64: str) -> Dict[str, Any]:
        """Build multimodal user message with screenshot."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                }
            ],
        }

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize gym-anything environment and return initial observation."""
        import os
        import gym_anything
        from gym_anything import from_config

        # Close previous env if any
        self.close()

        # Force Docker runner (default auto-detect prefers QEMU on Linux)
        os.environ.setdefault("GYM_ANYTHING_RUNNER", "docker")

        # gym-anything resolves Dockerfile paths relative to CWD. Presets use
        # paths like "gym_anything/presets/.../Dockerfile" which resolves from
        # the `src/` directory (the parent of the gym_anything package).
        ga_src_dir = Path(gym_anything.__file__).parent.parent  # .../src/
        prev_cwd = os.getcwd()
        try:
            os.chdir(ga_src_dir)
        except Exception:
            pass

        # Create gym-anything environment
        env_dir = Path(self.env_dir)
        self.ga_env = from_config(env_dir, task_id=self.task_id)

        # Resolve Dockerfile to absolute path. The preset stores it as a
        # relative path (gym_anything/presets/.../Dockerfile) which only
        # works if CWD == src/. Make it absolute so it works from any CWD.
        try:
            df = self.ga_env.env_spec.dockerfile
            if df and not os.path.isabs(df):
                abs_df = ga_src_dir / df
                if abs_df.exists():
                    self.ga_env.env_spec.dockerfile = str(abs_df)
        except Exception:
            pass

        # If sysbox-runc isn't available, strip the runtime override so Docker
        # falls back to default runc. Most envs don't strictly need systemd.
        if not _sysbox_available():
            try:
                if hasattr(self.ga_env.env_spec.security, "runtime"):
                    self.ga_env.env_spec.security.runtime = None
                if hasattr(self.ga_env.env_spec.security, "use_systemd"):
                    self.ga_env.env_spec.security.use_systemd = False
            except Exception:
                pass

        # Restore CWD
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass

        # Read resolution from env spec
        screen_spec = next(
            (o for o in self.ga_env.env_spec.observation if o.type == "rgb_screen"),
            None,
        )
        if screen_spec and screen_spec.resolution:
            self.screen_width, self.screen_height = screen_spec.resolution

        # Reset environment (starts Docker container, runs setup scripts)
        obs = self.ga_env.reset(
            use_cache=self.use_cache,
            cache_level=self.cache_level,
        )

        # Reset state
        self.turns = 0
        self.last_reward = None

        # Get task description
        task_description = ""
        if self.ga_env.task_spec:
            task_description = self.ga_env.task_spec.description or ""

        # Build system prompt
        system_content = self._build_system_prompt(task_description)
        system_message = {"role": "system", "content": system_content}

        # Build initial user message with screenshot
        screenshot_b64 = self._screenshot_to_base64(obs)
        if screenshot_b64:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_description},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                ],
            }
        else:
            user_message = {"role": "user", "content": task_description}

        self.chat_history = [system_message, user_message]

        metadata = {
            "task_key": self.task_key,
            "env_dir": str(self.env_dir),
            "task_id": self.task_id,
            "screen_resolution": (self.screen_width, self.screen_height),
        }

        return self.chat_history.copy(), metadata

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step: parse tool_call, inject action, capture screenshot."""
        step_start = time.time()
        self.turns += 1

        assistant_msg = {"role": "assistant", "content": action}
        self.chat_history.append(assistant_msg)

        max_turns_reached = self.turns >= self.max_turns

        # Check done signals
        agent_done = "<done>" in action.lower() or "[done]" in action.lower()

        # Parse tool call
        tool_call = parse_tool_call(action)

        reward = 0.0
        is_terminal = False

        # Check terminate action inside tool call
        if (
            not agent_done
            and tool_call
            and tool_call.get("arguments", {}).get("action") == "terminate"
        ):
            agent_done = True
            is_terminal = True

        if tool_call and self.ga_env and not is_terminal:
            ga_actions, is_term, is_screenshot = self._tool_call_to_ga_actions(tool_call)

            if is_term:
                agent_done = True
            elif is_screenshot:
                # Just capture a fresh screenshot without injecting actions
                obs = self.ga_env.capture_observation()
                screenshot_b64 = self._screenshot_to_base64(obs)
                if screenshot_b64:
                    new_obs = self._make_screenshot_observation(screenshot_b64)
                    self.chat_history.append(new_obs)
                    return BaseTextEnvStepOutput(
                        observations=[new_obs],
                        reward=0.0,
                        done=False,
                        metadata={"turn": self.turns, "action": "screenshot"},
                    )
            elif ga_actions:
                # Inject actions and get observation
                obs, step_reward, done, info = self.ga_env.step(
                    ga_actions, mark_done=False,
                )
                screenshot_b64 = self._screenshot_to_base64(obs)
                if screenshot_b64:
                    new_obs = self._make_screenshot_observation(screenshot_b64)
                    self.chat_history.append(new_obs)

                    episode_done = agent_done or max_turns_reached or done
                    if episode_done:
                        reward = self._get_final_reward()

                    return BaseTextEnvStepOutput(
                        observations=[new_obs],
                        reward=reward,
                        done=episode_done,
                        metadata={
                            "turn": self.turns,
                            "tool_call": tool_call,
                            "step_time": time.time() - step_start,
                        },
                    )

        # Handle done (agent_done or max_turns)
        episode_done = agent_done or max_turns_reached
        if episode_done:
            # Mark done in gym-anything to trigger verifier
            if self.ga_env:
                try:
                    obs, step_reward, done, info = self.ga_env.step(
                        [{"action": "screenshot"}], mark_done=True,
                    )
                    reward = self._get_reward_from_info(info)
                except Exception as e:
                    logger.warning(f"Error during final step: {e}")
                    reward = 0.0

        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"done_reason": "max_turns", "task_key": self.task_key},
            )

        # No valid action — nudge the model
        if not tool_call:
            obs_content = (
                "No tool call found. Use "
                '<tool_call>{"name": "computer_use", "arguments": {"action": "...", ...}}</tool_call> '
                "format."
            )
        elif agent_done:
            obs_content = "Task marked as complete."
        else:
            obs_content = "Action executed."

        new_obs = {"role": "user", "content": obs_content}
        self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=episode_done,
            metadata={
                "turn": self.turns,
                "tool_call": tool_call,
                "step_time": time.time() - step_start,
            },
        )

    def _get_final_reward(self) -> float:
        """Trigger verifier by marking done and extract reward."""
        if not self.ga_env:
            return 0.0
        try:
            obs, step_reward, done, info = self.ga_env.step(
                [{"action": "screenshot"}], mark_done=True,
            )
            return self._get_reward_from_info(info)
        except Exception as e:
            logger.warning(f"Verifier error: {e}")
            return 0.0

    def _get_reward_from_info(self, info: Dict[str, Any]) -> float:
        """Extract normalized reward (0-1) from gym-anything verifier info."""
        verifier = info.get("verifier", {})
        if not verifier:
            return 0.0
        # Verifier returns score 0-100, normalize to 0-1
        score = verifier.get("score", 0)
        self.last_reward = score / 100.0
        return self.last_reward

    def close(self):
        """Close the gym-anything environment (stops Docker container)."""
        if self.ga_env:
            try:
                self.ga_env.close()
            except Exception as e:
                logger.warning(f"Error closing gym-anything env: {e}")
            self.ga_env = None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "reward": self.last_reward or 0.0,
            "task_key": self.task_key,
        }
