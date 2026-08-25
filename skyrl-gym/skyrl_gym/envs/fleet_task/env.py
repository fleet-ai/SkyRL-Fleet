"""Fleet Task Environment for SkyRL-Gym.

Provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks.
Uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments,
keeping a clean separation between SkyRL's training interface and Fleet's
environment management.

Multi-modal support: When the task modality is "computer_use" or "browser_use", step() returns
multimodal observations in OpenAI format (image_url content blocks). Upstream
SkyRL's generator already handles these via extract_images_from_conversation()
and passes them as multi_modal_data to vLLM — no upstream changes needed.
"""

import ast
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.fleet_task.families import get_family
from skyrl_gym.envs.fleet_task.screenshot_compress import compress_content_blocks
from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call

# Reduce MCP client log noise
try:
    from loguru import logger as loguru_logger

    loguru_logger.disable("mcp")
except ImportError:
    pass
logging.getLogger("mcp").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global task cache to avoid reloading JSON for each env instance
_TASK_CACHE: Dict[str, Dict[str, Any]] = {}

# Truncate any single tool output past this many chars (caught a 5.8M-char query_data_lake return).
MAX_TOOL_OUTPUT_CHARS = 16_000

# Per-turn observation scaffold (turn_indicator + per_turn_reminder) is built
# from the per-family YAML templates in fleet_task.yaml. Source of truth is
# the YAML; env.py is the consumer. Canonical-v3 had a threshold-based nudge
# ("Emit <done> NOW or reward 0") that fired every turn for the last 5; 98 of
# 271 such events were immediately followed by bare <done> with no answer
# (learned surrender). Continuous low-key framing (turn indicator) avoids the
# threshold cliff. The per_turn_reminder reinforces the canonical tool-call
# format on every turn — addressing the NAKED-format failure where the model
# drops <|tool_call_begin|> / <|tool_call_argument_begin|> literally.


_STRICT_VERDICT_RE = re.compile(r">>> STRICT_VERDICT >>>\s*\n\s*(-?[\d.]+)\s*\n\s*<<< STRICT_VERDICT <<<")


def parse_strict_verdict(verifier_stdout: Optional[str]) -> Optional[float]:
    """Extract the oracle verdict a dual-scoring verifier prints to stdout.

    Dual-scoring verifiers (rl-experiments/make_arms.py) run the unmodified strict
    verifier observationally alongside the weakened one that produces the reward,
    so both scores describe the same rollout end-state. Returns None when the block
    is absent (ordinary single-verifier task) and -1.0 when the oracle itself
    errored — callers must exclude -1 from the strict curve rather than score it 0.
    """
    if not verifier_stdout:
        return None
    m = _STRICT_VERDICT_RE.search(verifier_stdout)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def load_tasks_from_json(tasks_file: str) -> Dict[str, Any]:
    """Load tasks from JSON file with caching.

    Returns a dict mapping task_key -> task_config dict.
    """
    if tasks_file not in _TASK_CACHE:
        expanded_path = os.path.expanduser(tasks_file)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Tasks file not found: {expanded_path}")

        with open(expanded_path, "r") as f:
            data = json.load(f)

        # Handle both formats: array or {"tasks": [...]}
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict) and "tasks" in data:
            tasks = data["tasks"]
        else:
            raise ValueError(f"Invalid JSON format in {tasks_file}: expected array or object with 'tasks' key")

        if not tasks:
            raise ValueError(f"No tasks found in {tasks_file}")

        # Index by task_key (support both 'key' and 'task_key' fields)
        _TASK_CACHE[tasks_file] = {t.get("key") or t.get("task_key"): t for t in tasks}

    return _TASK_CACHE[tasks_file]


def clear_caches():
    """Clear global caches. Useful for testing."""
    global _TASK_CACHE
    _TASK_CACHE = {}


def truncate_tool_result(tool_result: Any, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> Any:
    """Cap the on-wire size of a single tool result before it lands in chat_history.

    Long query results, screenshots-as-strings, or large dicts can fill the
    next prompt and trigger `stop=length` long before max_turns.

    Contract:
      - None passes through unchanged.
      - Multimodal list-of-content-blocks ([{"type": "text"|"image_url", ...}]):
        preserved as a list. Only the text blocks are length-checked; the
        image_url blocks pass through untouched. `[image]` placeholders the
        VL pipeline reads do NOT get counted against the text budget.
      - Anything else: serialized once; if within `max_chars` passes through
        unchanged; otherwise coerced to a truncated string with a marker.

    Previous behavior counted the entire JSON repr of multimodal content —
    including base64 image bytes — against the text budget, which forced
    every screenshot tool result into the truncated-string fallback. That
    leaked base64 into the tokenizer as `Tool result:\\n[{"type": ...`
    instead of letting the VL pipeline see actual images.
    """
    if tool_result is None:
        return None

    # Multimodal content blocks: never serialize whole structure.
    if isinstance(tool_result, list) and tool_result and all(isinstance(b, dict) and "type" in b for b in tool_result):
        out: list[dict] = []
        for b in tool_result:
            if b.get("type") == "text":
                t = b.get("text", "")
                if isinstance(t, str) and len(t) > max_chars:
                    elided = len(t) - max_chars
                    out.append(
                        {
                            "type": "text",
                            "text": t[:max_chars] + f"\n\n[TRUNCATED — {elided} chars elided.]",
                        }
                    )
                else:
                    out.append(b)
            else:
                # image_url or anything else: pass through unchanged.
                out.append(b)
        return out

    if isinstance(tool_result, str):
        text = tool_result
    else:
        try:
            text = json.dumps(tool_result, default=str)
        except Exception:
            text = str(tool_result)
    if len(text) <= max_chars:
        return tool_result
    elided = len(text) - max_chars
    return text[:max_chars] + f"\n\n[TRUNCATED — {elided} chars elided.]"


def tool_result_to_message_content(tool_result: Any) -> Any:
    """Normalize a Fleet tool_result into a user-message `content` value.

    Returns one of:
      - the input list unchanged, if it is a non-empty list of OpenAI
        content blocks (each element a dict carrying a "type" key —
        text / image_url / etc.). These ride through `apply_chat_template`
        as native multimodal content.
      - a string body, for every other shape (str / int / dict / plain
        list[dict] / list[str] / empty list / mixed list). Sending those
        raw as `content` crashes `apply_chat_template` with "Input is
        not valid. Should be a string, a list/tuple of strings or a
        list/tuple of integers."
    """
    if isinstance(tool_result, list) and tool_result and all(isinstance(b, dict) and "type" in b for b in tool_result):
        return tool_result
    body = json.dumps(tool_result, indent=2) if isinstance(tool_result, (list, dict)) else tool_result
    return f"Tool result:\n{body}"


_ACTION_VOCAB = (
    "screenshot, left_click, right_click, double_click, triple_click, "
    "middle_click, mouse_move, left_click_drag, type, key, scroll, wait, "
    "cursor_position, left_mouse_down, left_mouse_up, hold_key"
)


# Trailing chars stripped from the end of a response before checking for
# a done signal. Catches "<done>", "<done>.", "<done>\n", "<done> ", and
# common combinations the sampler likes to emit.
_DONE_RSTRIP_CHARS = " \t\n\r.!?'\"`*"


def is_done_signal(action: str, signals: List[str]) -> bool:
    """Return True iff the response ends with one of the done signals.

    Endswith (after stripping trailing whitespace + punctuation) is the
    literal reading of the system prompt's contract: "EVERY response MUST
    end with exactly ONE of: a tool call, OR <done>". The previous
    substring match (``"<done>" in action.lower()``) fired on any
    occurrence anywhere in the response, including when the model quoted
    the system prompt back to itself while debugging format issues —
    every BU rollout in job c4b429ae terminated this way with score=0.

    Implementation:
      - Strip trailing whitespace + common terminal punctuation so
        "<done>." / "<done>\\n" / "<done> " all match.
      - Lowercase both sides.
      - endswith() against each configured signal in order.
    """
    s = action.rstrip(_DONE_RSTRIP_CHARS).lower()
    return any(s.endswith(sig.lower()) for sig in signals)


def _bu_interaction_hints(portal_url: Optional[str]) -> str:
    """browser_use system-prompt addendum.

    Includes the live portal URL (when available) and a hard ban on the two
    failure modes the workflow analysis caught in 47/48 BU rollouts: opening
    with `navigate("/lifeline/")` (relative URL — gets resolved to localhost
    which Django ALLOWED_HOSTS-rejects), and typing made-up localhost domains.
    """
    if portal_url:
        url_line = f"You are on a Fleet env at {portal_url} . Stay on this domain.\n"
    else:
        url_line = (
            "You are on a Fleet env page that was already loaded for you. "
            "Stay on the current domain — do NOT type a new hostname.\n"
        )
    return (
        "\n## Browser Interaction Strategy\n"
        + url_line
        + "\n- To open an app (Lifeline, Latch, Medora, etc.), CLICK the visible card.\n"
        "  DO NOT use `navigate` with relative URLs like `/lifeline/` — that escapes the env.\n"
        "- DO NOT type `http://localhost...` or any made-up domain. The page renders\n"
        "  Django ALLOWED_HOSTS 403 if you guess wrong and you lose the working session URL.\n"
        "- After any action: take a screenshot to verify the screen changed.\n"
        "- If `navigate` returns a 403 page, you guessed a wrong host. Recover with\n"
        "  the browser back-button keystroke — call `computer` with arguments\n"
        '  `{"action": "key", "text": "alt+Left"}` (xdotool combo syntax in the\n'
        "  `text` field; the schema has no `keys` field).\n"
        "- When the task is fully complete, output your final answer and emit <done>.\n"
        f"\n## Action Vocabulary\n"
        f"Valid `action` values: {_ACTION_VOCAB}, navigate.\n"
        f"Use `left_click`, NOT `click`. Coordinates are integer pixels.\n"
    )


def _cu_interaction_hints() -> str:
    """computer_use system-prompt addendum.

    fos-* envs are Linux desktops with SaaS apps in Chrome tabs, NOT a single
    web browser. The default 'browser strategy' framing causes 13/44 CU rollouts
    to type invented localhost URLs and 21/44 to escape to a sqlite terminal.
    """
    return (
        "\n## Desktop Interaction Strategy\n"
        "You are on a Linux desktop. Apps run as Chrome tabs (Sentry, Jira,\n"
        "QuickBooks, Outlook, HR, expenses). Navigate WITHIN each app by\n"
        "clicking sidebar / menu items.\n\n"
        "- DO NOT type URLs in the address bar.\n"
        "- DO NOT open a terminal to inspect SQLite — the verifier checks UI state,\n"
        "  so data you read from a terminal does not count toward task completion.\n"
        "- DO NOT use `navigate` — that's for the browser_use modality, not here.\n"
        "- After any action: take a screenshot to verify the screen changed.\n"
        "- When the task is fully complete, output your final answer and emit <done>.\n\n"
        "## App Alias Map (in-env name → SaaS app)\n"
        "  Signal  = Sentry\n"
        "  Kernel  = Jira\n"
        "  Ledger  = QuickBooks\n"
        "  Latch   = Outlook\n"
        "  Cadence = HR (BambooHR-style)\n"
        "  Float   = expenses\n"
        "  Ramp    = expenses\n"
        f"\n## Action Vocabulary\n"
        f"Valid `action` values: {_ACTION_VOCAB}.\n"
        f"Use `left_click`, NOT `click`. Coordinates are integer pixels in the viewport.\n"
    )


def build_system_content(
    tools: List[Dict[str, Any]],
    *,
    modality: str = "tool_use",
    env_variables: Optional[Dict[str, Any]] = None,
    env_key: Optional[str] = None,
    use_tools_channel: bool = False,
    now: Optional[datetime] = None,
    portal_url: Optional[str] = None,
    model_family: Optional[str] = None,
) -> str:
    """Build the system-message text for a Fleet rollout.

    When `use_tools_channel=True`, the in-prompt `## Available Tools` JSON
    dump and `## Tool Call Format` example are omitted: the caller is
    expected to pass tools via `apply_chat_template(tools=...)` so the
    model's native tool_declare block handles them. This is the canonical
    path for Tinker rollouts with Kimi-K2 / Qwen3+.

    When `use_tools_channel=False` (default), tools are embedded as text in
    the system message for compatibility with vLLM/SkyRL paths where the
    rendered prompt is the only channel the model sees.

    Pure function (no env state); broken out for direct unit testing.
    """
    env_variables = env_variables or {}
    now = now or datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    # Environment context section from env_variables
    env_context = ""
    if env_variables:
        env_lines = []
        if "LOGGED_IN_USER" in env_variables:
            env_lines.append(f"- Logged in user ID: {env_variables['LOGGED_IN_USER']}")
        if "LOGGED_IN_NAME" in env_variables:
            env_lines.append(f"- Logged in as: {env_variables['LOGGED_IN_NAME']}")
        for key, value in env_variables.items():
            if key not in ("LOGGED_IN_USER", "LOGGED_IN_NAME", "CURRENT_DATE"):
                env_lines.append(f"- {key}: {value}")
        if env_lines:
            env_context = "\n## Environment Context\n" + "\n".join(env_lines) + "\n"

    env_hints = ""
    if env_key == "fostgres":
        env_hints = (
            "\n## Database Exploration\n"
            "Before writing SQL queries, first explore the database schema:\n"
            "- List tables: SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'\n"
            "- List columns: SELECT column_name, data_type FROM "
            "information_schema.columns WHERE table_name = 'your_table'\n"
        )

    # Modality-specific interaction hints. The previous single-string version
    # for both modalities mis-framed CU as a web browser (CU is a Linux desktop
    # with SaaS apps in Chrome tabs) and omitted the live portal URL on BU
    # (47/48 BU rollouts wasted turn 1 guessing the hostname). See workflow
    # synthesis wf_1c2d5247 for evidence.
    computer_use_hints = ""
    if modality == "browser_use":
        computer_use_hints = _bu_interaction_hints(portal_url)
    elif modality == "computer_use":
        computer_use_hints = _cu_interaction_hints()

    # Two independent decisions are bundled into this block:
    #   (a) ALWAYS dump the `## Available Tools` schema. Even when the model
    #       has a native tool channel (Kimi <|tool_declare_begin|>, Qwen3+
    #       <tools> block), the in-prompt JSON dump is the canonical reference
    #       the model can fall back to when prose hints in the system prompt
    #       are ambiguous. Cost is ~400-600 tokens on a 131K context (<0.5%).
    #       Tinker job 4746408e proved the cost of omitting it: every BU/CU
    #       rollout where the model hallucinated `keys: [...]` instead of
    #       `text: "alt+Left"` for the `key` action (matching the prose hint
    #       rather than the schema) burned the entire trajectory.
    #   (b) The `## Tool Call Format` example (`<tool_call>{...}</tool_call>`)
    #       is Qwen's call syntax. Models with native channels (Kimi, Qwen3+
    #       via tool_declare) must NOT be pushed off their native channel,
    #       so this block is conditional on `use_tools_channel=False`.
    # Split, not bundled — see job 4746408e_4 root cause.
    tools_json = json.dumps(tools, indent=2)
    tools_block = f"## Available Tools\n{tools_json}\n\n"
    if not use_tools_channel:
        tool_names = [t["function"]["name"] for t in tools if "function" in t]
        tool_names_str = ", ".join(tool_names)
        tools_block += (
            f"## Tool Call Format\n"
            f"Use the tools listed above by name ({tool_names_str}). "
            f"Format each call as:\n"
            f'<tool_call>{{"name": "<tool_name_from_above>", "arguments": '
            f"{{...}}}}</tool_call>\n\n"
        )
    else:
        # use_tools_channel=True: the trainer is passing tools via
        # apply_chat_template(tools=...), so the model's native tool
        # declaration block already renders the schema. We add a
        # `## Tool Call Format` block here ONLY with the canonical shape
        # the model emits — the literal special tokens of the chosen
        # family. The tokenizer re-encodes the <|...|> markers below
        # back into single special-token IDs when this prompt is
        # tokenized for the model, anchoring the model's marginal
        # probability on those IDs at every turn so format drift can't
        # compound over long rollouts. Without this, BU job c4b429ae
        # showed 43% parse rate / 57% reject (format drift starting
        # ~turn 5-13 in 14/14 sessions).
        #
        # When model_family is None or has no canonical_tool_call (e.g.
        # Qwen, whose chat template renders the format spec via the
        # `tools` argument), omit the block — safer than guessing.
        family = get_family(model_family)
        canonical = family.canonical_tool_call if family else None
        if canonical:
            tools_block += (
                f"## Tool Call Format\n"
                f"End each response with exactly this shape (the <|...|> "
                f"markers below are single special tokens, not literal "
                f"text — they encode as the corresponding vocab IDs):\n"
                f"{canonical}\n\n"
            )

    return (
        f"You are a helpful agent. Complete the task by calling tools.\n\n"
        f"## Current Date\n"
        f"Today's date is {current_date}. When dates are mentioned without "
        f"a year, assume the current year ({now.year}) or a "
        f"future date.\n"
        f"{env_context}{env_hints}{computer_use_hints}\n"
        f"{tools_block}"
        f"## Error Handling\n"
        f"If a tool call returns an error:\n"
        f"- Read the error message carefully\n"
        f"- Do NOT repeat the same call with identical arguments\n"
        f"- Change your approach: use different parameters, try a different "
        f"tool, or break the task into smaller steps\n\n"
        f"## Response Format\n"
        f"EVERY response MUST end with exactly ONE of:\n"
        f"1. A tool call to perform an action\n"
        f"2. Done signal: <done> - ONLY when the task is fully complete\n\n"
        f"IMPORTANT: When the task is complete, first output your final "
        f"answer with the requested information, THEN say <done>. Do not "
        f"just say <done> without providing the answer.\n\n"
        f'NEVER respond with just a message. NEVER say "feel free to ask" '
        f"or offer further help.\n"
        f"If the task is complete, provide your answer then say <done>. "
        f"Otherwise, make a tool call."
    )


class FleetTaskEnv(BaseTextEnv):
    """SkyRL environment for Fleet-hosted tasks.

    Uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments.
    This provides a clean separation between SkyRL's training interface and
    Fleet's environment management.

    Constructor signature follows upstream convention:
        __init__(self, env_config=None, extras={})

    Where:
        env_config: Dict or DictConfig from skyrl_gym_config YAML
        extras: Per-sample data from the training dataset (task_key, max_turns, etc.)
    """

    _trace_config: Optional[Dict[str, str]] = None

    @classmethod
    def set_trace_config(cls, job_id: str, model: str):
        """Set trace config for uploading rollout traces to Fleet."""
        cls._trace_config = {"job_id": job_id, "model": model}

    @classmethod
    def clear_trace_config(cls):
        """Clear trace config after traced rollouts are done."""
        cls._trace_config = None

    def __init__(
        self,
        env_config=None,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        if env_config is None:
            env_config = {}

        self.extras = extras
        self.max_turns = extras.get("max_turns", 50)
        # Screenshot compression — applied to image_url blocks in tool
        # results before they enter chat_history. 0 disables (default),
        # byte-identical to historical behavior. See screenshot_compress.py.
        self.screenshot_max_dim = int(extras.get("screenshot_max_dim", 0) or 0)
        # Resolved from task_config at init (in init() below). Default here
        # so step_async's post-action wait gating doesn't AttributeError on
        # rollouts that fail before init() runs.
        self.modality: str = "tool_use"

        # Task configuration from extras (set by dataset)
        self.task_key = extras.get("task_key")
        self.tasks_file = (env_config.get("tasks_file") if hasattr(env_config, "get") else None) or extras.get(
            "tasks_file"
        )

        if not self.task_key:
            raise ValueError("task_key must be provided in extras (from dataset)")
        if not self.tasks_file:
            raise ValueError("tasks_file must be provided in env_config or extras")

        # Expand path
        self.tasks_file = os.path.expanduser(self.tasks_file)

        # Load task config from JSON
        tasks = load_tasks_from_json(self.tasks_file)
        self.task_config = tasks.get(self.task_key)
        if not self.task_config:
            available_keys = list(tasks.keys())[:5]
            raise ValueError(
                f"Task '{self.task_key}' not found in {self.tasks_file}. " f"Available keys (first 5): {available_keys}"
            )

        # API key
        self.api_key = (env_config.get("api_key") if hasattr(env_config, "get") else None) or os.environ.get(
            "FLEET_API_KEY"
        )
        if not self.api_key:
            raise ValueError("FLEET_API_KEY must be set in env_config or environment")

        # Logfire telemetry (no-op if LOGFIRE_TOKEN is not set)
        logfire_token = os.environ.get("LOGFIRE_TOKEN")
        if logfire_token:
            try:
                from envs.fleet_env import configure_fleet_telemetry

                configure_fleet_telemetry(token=logfire_token)
            except ImportError:
                pass

        # TTL for Fleet environment instances
        self.ttl_seconds = env_config.get("ttl_seconds") if hasattr(env_config, "get") else None

        # Partial reward: use verifier accumulator counts instead of binary 0/1
        self.partial_reward = env_config.get("partial_reward", False) if hasattr(env_config, "get") else False

        # Hint config
        self.enable_hints = env_config.get("enable_hints", False) if hasattr(env_config, "get") else False

        # Environment state (initialized on init())
        self.openenv_task_env = None
        self.chat_history: ConversationType = []
        # Parallel to chat_history: the exact per-turn scaffold suffix
        # appended to that message's text destination, or "" if none. Used
        # by chat_history_for_trace() to strip the scaffold before upload
        # so the trace viewer shows only env content + image payloads.
        self._scaffold_per_msg: List[str] = []
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.last_reward: Optional[float] = None
        self.tools: List[Dict[str, Any]] = []

        # Verifier feedback (captured at close time for hint generation)
        self._verifier_stdout: Optional[str] = None
        self._verifier_error: Optional[str] = None
        self._tool_error_messages: List[str] = []
        # Oracle verdict from a dual-scoring verifier (see rl-experiments/make_arms.py).
        # None for ordinary single-verifier tasks.
        self._strict_reward: Optional[float] = None

        # Context management (uses OpenEnv's ContextManager)
        self.enable_context_tools = (
            env_config.get("enable_context_tools", False) if hasattr(env_config, "get") else False
        )
        self.context_manager = None
        if self.enable_context_tools:
            try:
                from envs.fleet_env import ContextManager

                logger.info(
                    "Enabling context management tools with "
                    f"max_output_chars={extras.get('max_output_chars', 10000)}"
                )
                self.context_manager = ContextManager(max_output_chars=extras.get("max_output_chars", 10000))
            except ImportError:
                logger.warning("ContextManager not available, disabling context tools")

    def _adapt_computer_tool_for_qwen(self):
        """Adapt computer tool description for Qwen VL's [0, 1000] coordinate space.

        Qwen3-VL/3.5 output coordinates in a normalized [0, 1000] grid regardless
        of screen resolution. This rewrites tool descriptions to match, and
        _convert_qwen_coordinates() converts back to pixels before MCP execution.
        """
        for tool in self.tools:
            func = tool.get("function", {})
            if func.get("name") != "computer":
                continue

            desc = func.get("description", "")

            # Parse actual screen dimensions
            res_match = re.search(r"Screen resolution:\s*(\d+)x(\d+)", desc)
            if res_match:
                self.screen_width = int(res_match.group(1))
                self.screen_height = int(res_match.group(2))
            else:
                self.screen_width = 1366
                self.screen_height = 768

            w, h = self.screen_width, self.screen_height

            # Rewrite description for Qwen's [0, 1000] coordinate space
            desc = re.sub(
                r"Screen resolution:\s*\d+x\d+\s*pixels\s*(\([^)]*\))?",
                "Screen resolution: 1000x1000",
                desc,
            )
            desc = re.sub(
                r"\(0, 0\) is top-left,\s*\(\d+, \d+\) is bottom-right",
                "(0, 0) is top-left, (999, 999) is bottom-right",
                desc,
            )
            desc = re.sub(
                r"valid range: x=0-\d+, y=0-\d+",
                "valid range: x=0-999, y=0-999",
                desc,
            )
            desc = re.sub(
                r"JPEG format at \d+x\d+",
                "JPEG format at 1000x1000",
                desc,
            )
            func["description"] = desc

            logger.info(
                f"Adapted computer tool for Qwen VL: actual_screen={w}x{h}, " f"model coordinate space=[0, 1000]"
            )
            break

    def _convert_normalized_coordinates(self, tool_call: Dict[str, Any]):
        """Normalize-to-pixel conversion that handles BOTH conventions:
          - Kimi [0, 1.0] float regression       → x * screen_w
          - Qwen-VL [0, 1000] integer convention → x / 1000 * screen_w
          - Pixel                                → leave alone

        Detection by range of `max(x, y)`:
          0 < max ≤ 1.0    → Kimi [0,1]   (e.g. [0.273, 0.298] → pixel(372,228))
          1.0 < max ≤ 1000 → Qwen [0,1000] (e.g. [200, 400] → pixel(273,307))
          max > 1000       → pixel, no-op (e.g. [1216, 90])

        Why Kimi case matters: per the K2.6 model card the system prompt
        instructs pixel coords, but Kimi's VL training prior occasionally
        regresses to [0,1] floats. Observed in session 827f4376 turns 6 & 8
        (`[0.273, 0.298]` when the model meant pixel `(372, 228)` on a
        1366x768 viewport). The MCP server then treats the float as pixel,
        rounding to (0, 0) — dead space, no state change, model gets the
        same screenshot back and assumes its click failed.

        The previous version of this method only handled the Qwen branch;
        the Kimi branch was a no-op (and worse: `0.273 / 1000 * 1366 = 0` —
        the Qwen scaler ACTIVELY broke the Kimi regression case by
        scaling it down by another 1000x). Unified here.

        Excludes (0, 0) clicks (max == 0) and mixed-format inputs (one
        pixel + one float) as a no-op — surfaces as a bug rather than
        getting silently papered over.

        Modifies tool_call arguments in-place.
        """
        if not getattr(self, "screen_width", None) or not getattr(self, "screen_height", None):
            return
        args = tool_call.get("arguments", {})
        if not args or tool_call.get("name") != "computer":
            return
        for field in ("coordinate", "start_coordinate"):
            coords = args.get(field)
            if not (coords and isinstance(coords, (list, tuple)) and len(coords) == 2):
                continue
            try:
                x, y = float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                continue
            mx = max(x, y)
            if 0 < x <= 1.0 and 0 < y <= 1.0:
                # Kimi [0,1] regression
                args[field] = [
                    int(x * self.screen_width),
                    int(y * self.screen_height),
                ]
            elif 1.0 < mx <= 1000 and x >= 0 and y >= 0:
                # Qwen-VL [0,1000] convention
                args[field] = [
                    int(x / 1000 * self.screen_width),
                    int(y / 1000 * self.screen_height),
                ]
            # else: already pixels, leave alone

    # Back-compat alias for the old name — anything else in the codebase
    # that called `_convert_qwen_coordinates` keeps working.
    _convert_qwen_coordinates = _convert_normalized_coordinates

    def _normalize_task_config(self) -> Dict[str, Any]:
        """Normalize task config to OpenEnv's expected format."""
        config = self.task_config.copy()

        # Map field names if needed
        if "key" in config and "task_key" not in config:
            config["task_key"] = config["key"]
        if "env_id" in config and "env_key" not in config:
            config["env_key"] = config["env_id"]
        if "version" in config and "env_version" not in config:
            config["env_version"] = config["version"]

        return config

    async def init_async(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the Fleet environment and return initial observation.

        Creates Fleet environment via OpenEnv's FleetTaskEnv and returns
        the task prompt with tool definitions.
        """
        from envs.fleet_env import FleetTaskEnv as OpenEnvFleetTaskEnv

        # Close any existing environment
        self.close()

        # Create OpenEnv's FleetTaskEnv with normalized config
        task_config = self._normalize_task_config()

        try:
            self.openenv_task_env = OpenEnvFleetTaskEnv(
                task_config=task_config,
                api_key=self.api_key,
                ttl_seconds=self.ttl_seconds,
                max_steps=self.max_turns,
                partial_reward=self.partial_reward,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenEnv FleetTaskEnv: {e}") from e

        # Reset episode state (tools are already cached from __init__)
        obs = await self.openenv_task_env.reset_async()

        # Reset state
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.last_reward = None

        # Reset context manager if enabled
        if self.context_manager:
            self.context_manager.reset()

        # Get tools from observation
        self.tools = obs.get("tools", [])

        # Add context management tools if enabled
        if self.context_manager:
            self.tools = self.tools + self.context_manager.get_tools()
        if not self.tools:
            raise RuntimeError(f"Task {self.task_key}: no tools found. Fleet env requires tools.")

        # VL: adapt computer tool for Qwen's normalized coordinate space.
        # Cache on self for step_async (post-action wait gating).
        modality = self.task_config.get("task_modality", "tool_use")
        self.modality = modality
        if modality in ("computer_use", "browser_use"):
            self._adapt_computer_tool_for_qwen()

        # Build initial prompt with task instruction
        task_prompt = self.task_config.get("prompt", "")

        # Inject hint from previous failed attempt if provided
        hint = self.extras.get("hint")
        if hint:
            task_prompt = f"{task_prompt}\n\nHere is feedback from a previous attempt " f"to help you:\n{hint}"

        # Build system prompt. Tools are either embedded as text below (legacy
        # path for vLLM/SkyRL where the rendered prompt is the only channel) or
        # passed out-of-band via `apply_chat_template(tools=...)` (Tinker /
        # HF-standard channel — Kimi-K2, Qwen3+ render them in the model's
        # native tool_declare block at the top of the prompt). When the caller
        # sets extras["use_tools_channel"]=True, we skip the in-system-message
        # injection so they don't double up.
        use_tools_channel = bool(self.extras.get("use_tools_channel", False))
        env_key = self.task_config.get("env_key") or self.task_config.get("env_id")
        # Pull the live portal URL so BU rollouts don't waste turn 1 guessing
        # hostnames (47/48 of them did in the prior canonical run).
        portal_url: Optional[str] = None
        if modality == "browser_use":
            orch = getattr(self.openenv_task_env, "_orch", None)
            fleet_env = getattr(orch, "_fleet_env", None) if orch else None
            urls = getattr(fleet_env, "urls", None) if fleet_env else None
            root = getattr(urls, "root", None) if urls else None
            if root:
                portal_url = str(root).rstrip("/")
        system_content = build_system_content(
            tools=self.tools,
            modality=modality,
            env_variables=self.task_config.get("env_variables", {}),
            env_key=env_key,
            use_tools_channel=use_tools_channel,
            portal_url=portal_url,
            # Trainer sets this from model_name prefix; env reads it to
            # pick the canonical tool-call shape inserted in the system
            # prompt (and later in reject messages). Unknown family → no
            # format example block; safe fallback.
            model_family=self.extras.get("model_family"),
        )

        system_message = {"role": "system", "content": system_content}

        # VL: include initial screenshot in multimodal user message
        initial_screenshot = obs.get("initial_screenshot")
        if initial_screenshot and isinstance(initial_screenshot, list):
            user_content = [{"type": "text", "text": task_prompt}]
            for item in initial_screenshot:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    user_content.append(item)
            user_message = {"role": "user", "content": user_content}
        else:
            user_message = {"role": "user", "content": task_prompt}

        self.chat_history = [system_message, user_message]
        # System + initial user carry no per-turn scaffold; they pass
        # through chat_history_for_trace() unchanged.
        self._scaffold_per_msg = ["", ""]

        metadata = {
            "task_key": self.task_key,
            "env_key": env_key,
            "tools": self.tools,
            "modality": self.task_config.get("task_modality", "tool_use"),
        }

        return self.chat_history.copy(), metadata

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the Fleet environment (sync wrapper).

        Uses asyncio.run() for sync contexts. For async contexts, the upstream
        generator's _run_in_executor_if_available will call this in a thread pool,
        where asyncio.run() is safe.
        """
        return asyncio.run(self.init_async(prompt))

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step in the Fleet environment.

        Parses the action for tool calls, executes via OpenEnv's FleetTaskEnv,
        and returns observation. Reward is computed by the verifier on completion.

        For computer_use/browser_use modality, observations may include multimodal content
        (image_url blocks with base64 screenshots). Upstream SkyRL's generator
        handles these via extract_images_from_conversation().
        """
        step_start = time.time()
        # Episode already finished on a prior step (OpenEnv ran verifier).
        # Re-entry would raise "Episode is done", get silently caught, and
        # then re-upload with reward=0.0 and clobber a real 1.0 score.
        if self.openenv_task_env is not None and getattr(self.openenv_task_env, "_done", False):
            return BaseTextEnvStepOutput(
                observations=[],
                reward=self.last_reward or 0.0,
                done=True,
                metadata={"done_reason": "already_finalized", "task_key": self.task_key},
            )
        self.turns += 1

        # Parse + coord-scale the model's emission BEFORE building the
        # assistant message so the per-family adapter can structure the
        # message correctly with reasoning_content / tool_calls fields.
        from .config import get_config

        _cfg = get_config()
        agent_done = is_done_signal(action, _cfg.done_signals)
        tool_call = parse_tool_call(action)

        # VL: catch done signal wrapped in a computer tool call.
        if not agent_done and tool_call and tool_call.get("arguments", {}).get("action") == "done":
            agent_done = True
            tool_call = None

        # VL: convert normalized coordinates to pixels. Single function
        # handles both Qwen [0,1000] and Kimi [0,1.0] by range detection.
        if tool_call and getattr(self, "screen_width", None):
            self._convert_normalized_coordinates(tool_call)

        # Per-family adapter owns the assistant-message shape (Kimi splits
        # <think> into reasoning_content to avoid the Kimi chat template's
        # double-think bug; Qwen passes content through because its
        # template extracts <think> inline). When no family is registered,
        # fall back to the raw-content shape (byte-identical to pre-family
        # behavior — preserves any caller that doesn't set model_family).
        family = get_family(self.extras.get("model_family"))
        if family is not None:
            assistant_msg = family.build_assistant_message(action, tool_call, self.turns)
        else:
            assistant_msg = {"role": "assistant", "content": action}
            if tool_call:
                assistant_msg["tool_calls"] = [
                    {
                        "id": f"call_{self.turns}",
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call.get("arguments", {})),
                        },
                    }
                ]
        self.chat_history.append(assistant_msg)
        self._scaffold_per_msg.append("")  # assistant turns carry no scaffold
        if self.context_manager:
            self.context_manager.track_message(assistant_msg)

        max_turns_reached = self.turns >= self.max_turns

        tool_result = None
        error = None
        reward = 0.0
        mcp_time = 0.0

        # Send done=True at max_turns even without <done> — otherwise OpenEnv
        # never trips _done and the verifier never runs.
        force_done = agent_done or max_turns_reached

        # Hand the verifier the full transcript so judge-style verifiers can
        # grade produced artifacts + observable actions (not only a submitted
        # final answer). OpenEnv forwards it only to verifiers whose signature
        # accepts `conversation`, so submit-answer tasks are unaffected.
        if self.openenv_task_env is not None:
            self.openenv_task_env.conversation_messages = self.chat_history

        # Handle context management tools locally (no MCP call)
        if tool_call and self.context_manager and self.context_manager.is_context_tool(tool_call["name"]):
            tool_result, self.chat_history = self.context_manager.execute_tool(
                tool_call["name"],
                tool_call.get("arguments", {}),
                self.chat_history,
            )
        # Execute tool call if present via OpenEnv
        elif tool_call and self.openenv_task_env:
            self.tool_calls += 1
            openenv_action = {
                "tool": tool_call["name"],
                "params": tool_call.get("arguments", {}),
                "done": force_done,
            }

            try:
                mcp_start = time.time()
                obs, reward, done, info = await self.openenv_task_env.step_async(openenv_action)
                mcp_time = time.time() - mcp_start
                tool_result = obs.get("observation")
                if "tool_error" in info:
                    error = info["tool_error"]

                # Cap tool_result size before it lands in chat_history.
                if tool_result is not None:
                    if self.context_manager:
                        text = tool_result if isinstance(tool_result, str) else json.dumps(tool_result, default=str)
                        tool_result = self.context_manager.truncate_output(text)
                    else:
                        tool_result = truncate_tool_result(tool_result)

                # Modality-gated post-action wait. Browser-based modalities
                # (browser_use, computer_use) take the MCP tool result
                # synchronously but the browser DOM may still be repainting,
                # so the screenshot returned (or the next screenshot the
                # model takes) can capture a blank/mid-load state. The
                # post_action_wait_for(modality) returns 0 for tool_use
                # where SQL/REST results are atomic.
                # getattr with default keeps existing tests passing —
                # some fixtures construct FleetTaskEnv without running
                # init() to skip MCP setup, leaving self.modality unset.
                _mod = getattr(self, "modality", "tool_use")
                wait_s = _cfg.post_action_wait_for(_mod)
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
            except Exception as e:
                mcp_time = time.time() - mcp_start
                error = str(e)
        elif force_done and self.openenv_task_env:
            # <done> without a tool call, or max_turns hit with no tool call — still need the verifier.
            openenv_action = {"done": True}
            try:
                mcp_start = time.time()
                obs, reward, done, info = await self.openenv_task_env.step_async(openenv_action)
                mcp_time = time.time() - mcp_start
            except Exception as e:
                mcp_time = time.time() - mcp_start
                error = str(e)

        # Detect error patterns in tool_result
        if not error and tool_result:
            result_str = str(tool_result) if not isinstance(tool_result, str) else tool_result
            if result_str.strip().startswith("Error:") or result_str.strip().startswith("error:"):
                error = result_str
                tool_result = None
            elif isinstance(tool_result, dict) and tool_result.get("error"):
                error = tool_result["error"]
                tool_result = None

        episode_done = agent_done or max_turns_reached

        # Cache before the upload await so a CancelledError during upload
        # (the wait_for race in main_fleet_tinker) can't lose the real reward.
        if episode_done:
            self.last_reward = reward

        # Upload trace at episode end if trace config is set
        if episode_done and FleetTaskEnv._trace_config:
            try:
                from envs.fleet_env.trace import upload_trace

                inst_id = None
                orch = getattr(self.openenv_task_env, "_orch", None)
                if orch:
                    fleet_env = getattr(orch, "_fleet_env", None)
                    if fleet_env:
                        inst_id = getattr(fleet_env, "instance_id", None)
                exec_id = getattr(self.openenv_task_env, "_last_verifier_execution_id", None)
                logger.info(
                    f"[{self.task_key}] upload_trace exec_id={exec_id} reward={reward} agent_done={agent_done} max_turns_reached={max_turns_reached}"
                )
                await upload_trace(
                    api_key=self.api_key,
                    job_id=FleetTaskEnv._trace_config["job_id"],
                    task_key=self.task_key,
                    model=FleetTaskEnv._trace_config["model"],
                    # Stripped of per-turn scaffold so the trace viewer
                    # shows only env content + image payloads; the model
                    # still sees the scaffold via self.chat_history during
                    # rollout.
                    chat_history=self.chat_history_for_trace(),
                    reward=reward,
                    instance_id=inst_id,
                    metadata={
                        "env_key": self.task_config.get("env_key"),
                        "turns": self.turns,
                    },
                    verifier_execution_id=exec_id,
                )
            except Exception as e:
                logger.warning(f"Failed to upload trace for {self.task_key}: {e}")

        # Build observation message
        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={
                    "done_reason": "max_turns",
                    "task_key": self.task_key,
                },
            )

        # Per-turn observation scaffold (turn indicator + canonical-format
        # reminder). Owned by the per-family adapter — Kimi emits both,
        # Qwen indicator-only, unknown family emits nothing. "" path means
        # downstream concatenations are no-ops, byte-identical to a caller
        # that doesn't set model_family.
        scaffold = family.per_turn_reminder(self.turns, self.max_turns) if family else ""

        # Build response observation
        if error:
            self.tool_errors += 1
            self._tool_error_messages.append(str(error)[:500])
            obs_content = f"Error: {error}"
        elif tool_result:
            content = tool_result_to_message_content(tool_result)
            if isinstance(content, list):
                # Compress screenshot image_url blocks if configured. No-op
                # when screenshot_max_dim=0 (default) — byte-identical to
                # pre-flag behavior.
                if self.screenshot_max_dim > 0:
                    content = compress_content_blocks(
                        content,
                        max_dim=self.screenshot_max_dim,
                    )
                # Multimodal obs — pass blocks through; append scaffold as
                # trailing text block (leading newlines stripped because the
                # screenshot+text composition handles spacing).
                content = list(content) + [{"type": "text", "text": scaffold.lstrip("\n")}]
                new_obs = {"role": "user", "content": content}
                self.chat_history.append(new_obs)
                self._scaffold_per_msg.append(scaffold)
                if self.context_manager:
                    self.context_manager.track_message(new_obs)
                return BaseTextEnvStepOutput(
                    observations=[new_obs],
                    reward=reward,
                    done=episode_done,
                    metadata={
                        "task_key": self.task_key,
                        "turn": self.turns,
                        "tool_call": tool_call,
                        "error": None,
                        "done_reason": "agent_done" if agent_done else None,
                        "step_time": time.time() - step_start,
                        "mcp_time": mcp_time,
                    },
                )
            obs_content = content
        elif agent_done:
            obs_content = "Task marked as complete."
        elif not tool_call:
            # No tool call landed. Owned by the per-family adapter so the
            # canonical-format anchor (Kimi <|...|> specials, Qwen text
            # grammar) matches what that family's template renders.
            #
            # History: the previous reject text blamed truncation ("be
            # more concise") which was the wrong diagnosis in 55/231 cases
            # observed in job c4b429ae — responses were well under
            # MAX_GENERATE_LENGTH but the model dropped markers under
            # format-stress and spiralled trying to be "concise." Family
            # adapter echoes a concrete canonical example instead.
            if family is not None:
                obs_content = family.reject_message()
            else:
                obs_content = (
                    "No tool call landed. End your response with a tool " "call in the canonical format for your model."
                )
        else:
            obs_content = "Action executed."

        if not isinstance(obs_content, str):
            obs_content = str(obs_content)
        obs_content += scaffold

        new_obs = {"role": "user", "content": obs_content}
        self.chat_history.append(new_obs)
        self._scaffold_per_msg.append(scaffold)
        if self.context_manager:
            self.context_manager.track_message(new_obs)

        step_time = time.time() - step_start
        metadata = {
            "task_key": self.task_key,
            "turn": self.turns,
            "tool_call": tool_call,
            "tool_result": (
                tool_result[:200] if isinstance(tool_result, str) and len(tool_result) > 200 else tool_result
            ),
            "error": error,
            "done_reason": "agent_done" if agent_done else None,
            "step_time": step_time,
            "mcp_time": mcp_time,
        }

        # If context was modified, return full chat_history so the generator
        # can replace its copy (required for stepwise training).
        if tool_call and self.context_manager and self.context_manager.is_context_tool(tool_call["name"]):
            if tool_call["name"] == "manage_context":
                metadata["modified_chat_history"] = self.chat_history.copy()

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=episode_done,
            metadata=metadata,
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step in the Fleet environment (sync wrapper)."""
        return asyncio.run(self.step_async(action))

    def chat_history_for_trace(self) -> ConversationType:
        """chat_history with the per-turn scaffold (turn indicator +
        canonical-format reminder) stripped from observation messages, so
        the trace viewer shows only the env content + image payloads. The
        model still sees the scaffold during rollout via self.chat_history;
        this projection is for upload only.

        Strip is exact-match via str.removesuffix using the scaffold string
        recorded in self._scaffold_per_msg at append time — no regex, no
        guessing.

        Multimodal: only the trailing text block (the one step_async
        appended) is touched; earlier text blocks pass through unchanged
        even if they happen to end with the same suffix. If the trailing
        text block strips to empty it is dropped; image_url blocks pass
        through byte-identical.

        Raises ValueError on a chat_history / _scaffold_per_msg length
        mismatch rather than silently truncating via zip().
        """
        if len(self.chat_history) != len(self._scaffold_per_msg):
            raise ValueError(
                f"chat_history ({len(self.chat_history)}) and _scaffold_per_msg "
                f"({len(self._scaffold_per_msg)}) length mismatch — programmer error"
            )
        out: ConversationType = []
        for msg, scaffold in zip(self.chat_history, self._scaffold_per_msg):
            if not scaffold:
                out.append(msg)
                continue
            c = msg.get("content")
            if isinstance(c, list):
                # Multimodal: step_async appended the scaffold as the last
                # text block via `scaffold.lstrip("\n")`. Locate the
                # trailing text block by index and strip it only; earlier
                # blocks (image_url + any text the tool result carried)
                # pass through unchanged.
                lstripped = scaffold.lstrip("\n")
                new_blocks = list(c)
                last_text_idx = next(
                    (
                        i
                        for i in range(len(new_blocks) - 1, -1, -1)
                        if isinstance(new_blocks[i], dict) and new_blocks[i].get("type") == "text"
                    ),
                    None,
                )
                if last_text_idx is not None:
                    block = new_blocks[last_text_idx]
                    stripped = (block.get("text") or "").removesuffix(lstripped)
                    if stripped:
                        new_blocks[last_text_idx] = {**block, "text": stripped}
                    else:
                        del new_blocks[last_text_idx]
                out.append({**msg, "content": new_blocks})
            elif isinstance(c, str):
                out.append({**msg, "content": c.removesuffix(scaffold)})
            else:
                out.append(msg)
        return out

    def _capture_verifier_feedback(self):
        """Capture verifier feedback from OpenEnv before nulling the env."""
        if self.openenv_task_env:
            self._verifier_stdout = getattr(self.openenv_task_env, "verifier_stdout", None)
            self._verifier_error = getattr(self.openenv_task_env, "verifier_error", None)
            self._tool_error_messages = getattr(self.openenv_task_env, "tool_errors_list", [])
            self._strict_reward = parse_strict_verdict(self._verifier_stdout)
            # Surface verifier output so failure modes (LLM-judge details,
            # tracebacks, GRADING_DETAILS blocks) are grep'able from logs.
            if self._verifier_stdout:
                logger.info(f"[{self.task_key}] verifier stdout: {self._verifier_stdout[:1500]}")
            if self._verifier_error:
                logger.warning(f"[{self.task_key}] verifier error: {self._verifier_error[:1500]}")

    def close(self):
        """Close the Fleet environment and cleanup resources."""
        if self.openenv_task_env:
            try:
                self.openenv_task_env.close()
                if self.openenv_task_env.final_reward is not None:
                    self.last_reward = self.openenv_task_env.final_reward
                self._capture_verifier_feedback()
            except Exception as e:
                logger.warning(f"Failed to close Fleet environment: {e}")
            self.openenv_task_env = None

    async def close_async(self):
        """Close the Fleet environment (async version).

        Runs verifier via OpenEnv's close_async() to get actual reward for
        orphaned rollouts (context overflow, early termination by SkyRL).
        """
        if self.openenv_task_env:
            try:
                await self.openenv_task_env.close_async()
                if self.openenv_task_env.final_reward is not None:
                    self.last_reward = self.openenv_task_env.final_reward
                self._capture_verifier_feedback()
            except Exception as e:
                logger.warning(f"Failed to close Fleet environment: {e}")
            self.openenv_task_env = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return environment metrics for this episode."""
        metrics = {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key") or self.task_config.get("env_id"),
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "is_hinted": bool(self.extras.get("hint")),
        }
        if self.last_reward is not None:
            metrics["final_reward"] = self.last_reward
        # Oracle verdict on the SAME end-state that produced final_reward. The gap
        # between the two is the reward-hacking signal; -1 means the oracle errored
        # and the rollout must be excluded from the strict curve, not counted as 0.
        if self._strict_reward is not None:
            metrics["strict_reward"] = self._strict_reward
        # Include verifier feedback for hint generation
        if self._verifier_stdout is not None:
            metrics["verifier_stdout"] = self._verifier_stdout
        if self._verifier_error is not None:
            metrics["verifier_error"] = self._verifier_error
        if self._tool_error_messages:
            metrics["tool_error_messages"] = self._tool_error_messages
        # Include chat_history for LLM hint synthesis (consumed then deleted by generator)
        if self.chat_history:
            metrics["chat_history"] = self.chat_history
        return metrics

    @staticmethod
    def build_hint_text(
        verifier_stdout: Optional[str],
        verifier_error: Optional[str],
        tool_error_messages: Optional[List[str]],
    ) -> str:
        """Build hint text from verifier feedback. No LLM call.

        Parses ERROR_ACCUMULATOR / SUCCESS_ACCUMULATOR from verifier stdout
        and formats tool errors into structured feedback for the next attempt.
        """
        parts = []

        if verifier_stdout:
            err_match = re.search(
                r">>> ERROR_ACCUMULATOR >>>\n(.+?)\n<<< ERROR_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            suc_match = re.search(
                r">>> SUCCESS_ACCUMULATOR >>>\n(.+?)\n" r"<<< SUCCESS_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            if err_match or suc_match:
                try:
                    errors = ast.literal_eval(err_match.group(1)) if err_match else []
                    successes = ast.literal_eval(suc_match.group(1)) if suc_match else []
                except Exception:
                    errors, successes = [], []
                if successes:
                    parts.append(f"Checks passed ({len(successes)}): " + ", ".join(str(s)[:100] for s in successes[:5]))
                if errors:
                    parts.append(f"Checks failed ({len(errors)}): " + ", ".join(str(e)[:100] for e in errors[:5]))

        if verifier_error:
            parts.append(f"Verifier: {verifier_error}")

        if tool_error_messages:
            unique = list(dict.fromkeys(tool_error_messages))[:5]
            parts.append("Tool errors: " + "; ".join(e[:200] for e in unique))

        return "\n".join(parts) if parts else "The previous attempt failed. Try a different approach."

    @staticmethod
    def aggregate_metrics(
        metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate metrics across episodes with per-env breakdown."""
        if not metrics:
            return {}

        env_init_failures: Dict[str, int] = {}
        total_init_failures = 0

        env_data: Dict[str, Dict[str, List[int]]] = {}
        for m in metrics:
            # Check for init failure metrics
            for key, value in m.items():
                if key.startswith("env_init_failed/"):
                    env_key = key.split("/", 1)[1]
                    env_init_failures[env_key] = env_init_failures.get(env_key, 0) + int(value)
                    total_init_failures += int(value)

            env_key = m.get("env_key")
            if env_key:
                if env_key not in env_data:
                    env_data[env_key] = {
                        "turns": [],
                        "tool_calls": [],
                        "tool_errors": [],
                    }
                env_data[env_key]["turns"].append(m.get("turns", 0))
                env_data[env_key]["tool_calls"].append(m.get("tool_calls", 0))
                env_data[env_key]["tool_errors"].append(m.get("tool_errors", 0))

        result: Dict[str, Any] = {}
        total_turns = 0
        total_tool_calls = 0
        total_tool_errors = 0
        total_episodes = 0

        for env_key, data in env_data.items():
            turns_list = data["turns"]
            tool_calls_list = data["tool_calls"]
            tool_errors_list = data["tool_errors"]

            avg_turns = sum(turns_list) / len(turns_list)
            avg_tool_calls = sum(tool_calls_list) / len(tool_calls_list)
            avg_tool_errors = sum(tool_errors_list) / len(tool_errors_list)
            total_env_turns = sum(turns_list)
            total_env_tool_calls = sum(tool_calls_list)
            total_env_tool_errors = sum(tool_errors_list)
            tool_calls_per_turn = total_env_tool_calls / total_env_turns if total_env_turns > 0 else 0
            tool_error_rate = total_env_tool_errors / total_env_tool_calls if total_env_tool_calls > 0 else 0

            result[f"{env_key}/avg_turns"] = avg_turns
            result[f"{env_key}/min_turns"] = min(turns_list)
            result[f"{env_key}/max_turns"] = max(turns_list)
            result[f"{env_key}/avg_tool_calls"] = avg_tool_calls
            result[f"{env_key}/tool_calls_per_turn"] = tool_calls_per_turn
            result[f"{env_key}/avg_tool_errors"] = avg_tool_errors
            result[f"{env_key}/total_tool_errors"] = total_env_tool_errors
            result[f"{env_key}/tool_error_rate"] = tool_error_rate
            result[f"{env_key}/num_episodes"] = len(turns_list)

            total_turns += total_env_turns
            total_tool_calls += total_env_tool_calls
            total_tool_errors += total_env_tool_errors
            total_episodes += len(turns_list)

        result["avg_turns"] = total_turns / total_episodes if total_episodes > 0 else 0
        result["avg_tool_calls"] = total_tool_calls / total_episodes if total_episodes > 0 else 0
        result["tool_calls_per_turn"] = total_tool_calls / total_turns if total_turns > 0 else 0
        result["avg_tool_errors"] = total_tool_errors / total_episodes if total_episodes > 0 else 0
        result["total_tool_errors"] = total_tool_errors
        result["tool_error_rate"] = total_tool_errors / total_tool_calls if total_tool_calls > 0 else 0
        result["total_episodes"] = total_episodes

        for env_key, failures in env_init_failures.items():
            result[f"{env_key}/env_init_failed"] = failures
        if total_init_failures > 0:
            result["total_env_init_failed"] = total_init_failures

        return result
