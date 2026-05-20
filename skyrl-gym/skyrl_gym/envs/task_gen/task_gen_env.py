"""
Task Generation Environment for SkyRL.

Multi-turn BaseTextEnv where the LLM can explore the seed database via
``query_db`` meta-tool before generating a task.

When ``max_turns > 1`` (the default), the model explores the DB first
and then produces a ``<task>`` block.  When ``max_turns == 1`` it
behaves identically to the original single-turn variant.

Reward:

    R(task) = base_quality + llm_validity * (alpha * var(raw_scores) + (p_hint - p_raw))

    base_quality:     Small reward for passing sandbox+judge (default 0.1)
    llm_validity:     Binary 0/1 from LLM-as-a-judge (is the task well-formed?)
    var(raw_scores):  Variance of k raw evaluator rollouts (difficulty calibration)
    p_hint - p_raw:   Hint gap — solvable with hints but not without (learnability)
    alpha:            Weight balancing variance vs hint gap (default 0.5)
"""

import ast
import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_calls
from skyrl_gym.envs.task_gen.verifier_sandbox import (
    VerifierSandbox,
    parse_task_output,
)

logger = logging.getLogger(__name__)


class FleetHarnessJobError(RuntimeError):
    """Error raised after a Fleet harness job may have been created."""

    def __init__(self, message: str, *, job_id: str = "", task_key: str = "", status: str = ""):
        super().__init__(message)
        self.job_id = job_id
        self.task_key = task_key
        self.status = status


def _json_safe(value: Any, *, max_depth: int = 4) -> Any:
    if max_depth <= 0:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, max_depth=max_depth - 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, max_depth=max_depth - 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(), max_depth=max_depth - 1)
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict(), max_depth=max_depth - 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value), max_depth=max_depth - 1)
        except Exception:
            pass
    return repr(value)


def _first_present_mapping_value(mapping: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


_SESSION_TRAJECTORY_KEYS = [
    "trajectory",
    "trajectories",
    "rollout",
    "rollouts",
    "transcript",
    "messages",
    "steps",
    "actions",
    "events",
]
_SESSION_TRANSCRIPT_KEYS = ["session_transcript", "transcript_payload", *_SESSION_TRAJECTORY_KEYS]
_TRANSCRIPT_CONTAINER_KEYS = ["session_transcript", "transcript_payload", "transcript", "payload", "data", "result"]
_TRANSCRIPT_MESSAGE_KEYS = ["messages", "conversation", "chat_history"]


def _looks_like_messages(value: Any) -> bool:
    return isinstance(value, list) and any(
        isinstance(item, dict) and ("role" in item or "content" in item) for item in value
    )


def _extract_transcript_fields(payload: Any) -> Dict[str, Any]:
    """Pull common trajectory fields from Fleet transcript payload shapes."""
    if not payload:
        return {}
    if isinstance(payload, list):
        fields = {"transcript": payload, "trajectory": payload}
        if _looks_like_messages(payload):
            fields["messages"] = payload
        return fields
    if not isinstance(payload, dict):
        return {}

    fields: Dict[str, Any] = {}
    containers = [payload]
    for key in _TRANSCRIPT_CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, dict) and value not in containers:
            containers.append(value)
        elif isinstance(value, list) and value:
            fields.setdefault("transcript", value)
            fields.setdefault("trajectory", value)
            if _looks_like_messages(value):
                fields.setdefault("messages", value)

    for container in containers:
        messages = _first_present_mapping_value(container, _TRANSCRIPT_MESSAGE_KEYS)
        if messages:
            fields.setdefault("messages", messages)
        for key in ("steps", "actions", "events"):
            value = container.get(key)
            if value:
                fields.setdefault(key, value)
        trajectory = _first_present_mapping_value(container, _SESSION_TRAJECTORY_KEYS)
        if trajectory:
            fields.setdefault("trajectory", trajectory)
        transcript = _first_present_mapping_value(
            container,
            ["transcript", "session_transcript", "transcript_payload"],
        )
        if transcript:
            fields.setdefault("transcript", transcript)

    if "messages" in fields and "trajectory" not in fields:
        fields["trajectory"] = fields["messages"]
    if any(key in fields for key in ("trajectory", "messages", "steps", "actions", "events")):
        fields.setdefault("transcript", payload)
    return fields


def _format_compact_schema(describe_result: Any) -> str:
    """Convert a DescribeResponse dict to compact 'table: col (type), ...' format."""
    if not isinstance(describe_result, dict):
        return str(describe_result) if describe_result else ""
    tables = describe_result.get("tables")
    if not tables or not isinstance(tables, list):
        return ""
    lines = []
    for t in tables:
        name = t.get("name", "")
        cols = t.get("columns", [])
        col_parts = []
        for c in cols:
            col_name = c.get("name", "")
            col_type = c.get("type", "").lower()
            col_parts.append(f"{col_name} ({col_type})" if col_type else col_name)
        lines.append(f"{name}: {', '.join(col_parts)}")
    return "\n".join(lines)


def build_task_gen_db_schema_prompt(
    env_key: str,
    env_tools_schema: List[Dict[str, Any]],
    env_tools: List[str],
    env_variables: Dict[str, Any],
    env_variable_keys: List[str],
    env_schema: str,
) -> str:
    """Build environment, tool summary, variable, and database schema context."""
    parts = []

    parts.append(f'You are a task designer for the "{env_key}" environment.')

    current_date = env_variables.get("CURRENT_DATE", "")
    if current_date:
        parts.append(
            f"\n**IMPORTANT — Current Date: {current_date}**\n"
            f"The environment's current date is {current_date}. "
            "All dates in generated tasks MUST be on or after this date. "
            "Do NOT use past dates — the environment will reject them "
            "(e.g., check-in dates, event dates, appointment dates must be in the future)."
        )

    parts.append(f"\n## Environment: {env_key}")
    parts.append("\n### Available Tools")

    api_schemas = [t for t in env_tools_schema if t.get("function", {}).get("name") != "computer"]
    api_tool_names = [t for t in env_tools if t != "computer"]

    if api_schemas:
        for tool in api_schemas:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            parts.append(f"- **{name}**: {desc}")
    elif api_tool_names:
        parts.append("\n".join(f"- {t}" for t in api_tool_names))
    else:
        parts.append("No tools discovered for this environment.")

    if env_variables:
        parts.append("\n### Environment Variables (embed as constants)")
        parts.append(
            "These variables describe the user/session context. "
            "**Embed them directly as string constants** in your verifier code. "
            "Do NOT use `env.env_variables` — it is not available at verifier runtime."
        )
        for var_key, var_val in env_variables.items():
            parts.append(f'- `{var_key}` = `"{var_val}"`')
        parts.append(
            "\nExample usage in verifier:\n"
            "```python\n"
            f'LOGGED_IN_USER = "{env_variables.get("LOGGED_IN_USER", "user@example.com")}"\n'
            f'# Use as: rows = current.table("users").eq("email", LOGGED_IN_USER).all()\n'
            "```"
        )
    elif env_variable_keys:
        parts.append("\n### Environment Variables")
        parts.append(
            "These variables parameterize each environment instance. "
            "Look up values from the database instead of using env.env_variables."
        )
        for var_key in env_variable_keys:
            parts.append(f"- `{var_key}`")

    if env_schema:
        parts.append("\n### Database Schema")
        parts.append(
            "Use these exact table and column names in verifiers "
            '(e.g., `current.table("bookings").eq("guest_email", val).all()`):'
        )
        parts.append(f"```\n{env_schema}\n```")

    return "\n".join(parts)


def build_task_gen_task_verifier_instructions(current_date: str) -> str:
    """Build task-design and verifier-construction instructions."""
    if current_date:
        date_guidance = (
            f"### Date Awareness\n"
            f"The environment's current date is **{current_date}**. "
            f"ALL dates in your task MUST be on or after {current_date}. "
            "Tasks with past dates will always fail because the environment "
            "rejects them (e.g., 'checkIn date cannot be in the past'). "
            "Use `query_db` to check what date ranges exist in the data, "
            "and always generate future dates."
        )
    else:
        date_guidance = (
            "### Date Awareness\n"
            "If the environment works with dates, verify what date ranges "
            "are valid before generating tasks. Use `query_db` to check."
        )

    return f"""
## Verifier Guidelines

The verifier checks whether the agent completed the task by inspecting database state changes.

Signature: `def validate_task(env: Environment, final_answer: str | None = None) -> int`

**IMPORTANT**: The function MUST be named `validate_task` and return `TASK_FAILED_SCORE` (0) or `TASK_SUCCESSFUL_SCORE` (1).

### Verifier API
```python
env.instance.load()              # Load current state (call first)
seed = env.db("seed")            # Original DB before agent acted
current = env.db("current")      # Current DB after agent acted

# Query tables — ALL results are Python dicts, use row["column"] NOT row.column:
rows = current.table("table_name").eq("column", value).all()   # -> List[dict]
row = current.table("table_name").eq("column", value).first()  # -> dict or None
rows = current.table("table_name").neq("column", value).all()  # -> List[dict]
count = current.table("table_name").eq("column", value).count() # -> int
rows = current.table("table_name").select("col1", "col2").all() # -> List[dict]
# Access fields: row["id"], row["name"], row["email"] — NEVER row.id or row.name
# Only methods: .table(), .eq(), .neq(), .select(), .all(), .first(), .count()
# NO .like(), .gt(), .lt(), .contains(), .in_() — use Python filtering instead

# Compare seed vs current to detect NEW entries:
def find_new_entries(seed, current, table_name, id_field="id", filter_conditions=None):
    before_query = seed.table(table_name)
    after_query = current.table(table_name)
    if filter_conditions:
        for key, value in filter_conditions.items():
            before_query = before_query.eq(key, value)
            after_query = after_query.eq(key, value)
    before_ids = {{entry[id_field] for entry in before_query.select(id_field).all()}}
    return [e for e in after_query.all() if e[id_field] not in before_ids]
```

### Error Tracking (REQUIRED)
Every verifier MUST track errors and successes using accumulator lists, and print them
before returning. This enables automated feedback for hint-based evaluation.

```python
error_accumulator = []
success_accumulator = []

# ... check conditions ...
if condition_met:
    success_accumulator.append("[C] Booking was created")
else:
    error_accumulator.append("[X] Expected booking not found")

# ALWAYS print accumulators before returning:
if error_accumulator:
    print(">>> ERROR_ACCUMULATOR >>>")
    print(error_accumulator)
    print("<<< ERROR_ACCUMULATOR <<<")
if success_accumulator:
    print(">>> SUCCESS_ACCUMULATOR >>>")
    print(success_accumulator)
    print("<<< SUCCESS_ACCUMULATOR <<<")
```

### Verifier Template (follow this structure)
```python
def validate_task(env: Environment, final_answer: str | None = None) -> int:
    error_accumulator = []
    success_accumulator = []
    env.instance.load()
    seed = env.db("seed")
    current = env.db("current")

    def find_new_entries(table_name, id_field="id", filter_conditions=None):
        \"\"\"Compare seed vs current to find rows added by the agent.

        Args:
            table_name: Table to compare.
            id_field: Primary key column (default "id").
            filter_conditions: Optional dict of {{column: value}} filters
                applied to BOTH seed and current before comparison.

        Returns:
            List[dict] — rows present in current but not in seed.
        \"\"\"
        before_query = seed.table(table_name)
        after_query = current.table(table_name)
        if filter_conditions:
            for key, value in filter_conditions.items():
                before_query = before_query.eq(key, value)
                after_query = after_query.eq(key, value)
        before_ids = set(entry[id_field] for entry in before_query.select(id_field).all())
        return [e for e in after_query.all() if e[id_field] not in before_ids]

    # --- Validation: use SET-BASED comparison, never row-index ---
    # GOOD: compare by content/ID sets, order-independent
    #   expected_ids = {{"id_1", "id_2"}}
    #   actual_ids = {{row["id"] for row in new_entries}}
    #   if not expected_ids.issubset(actual_ids): ...
    #
    # BAD: comparing by row index (fragile, order-dependent)
    #   if new_entries[0]["id"] == "id_1": ...

    # Check conditions...
    # On early failure:
    if critical_failure:
        error_accumulator.append("[X] Critical check failed")
        print(">>> ERROR_ACCUMULATOR >>>")
        print(error_accumulator)
        print("<<< ERROR_ACCUMULATOR <<<")
        return TASK_FAILED_SCORE

    # Final result:
    if error_accumulator:
        print(">>> ERROR_ACCUMULATOR >>>")
        print(error_accumulator)
        print("<<< ERROR_ACCUMULATOR <<<")
        return TASK_FAILED_SCORE
    print(">>> SUCCESS_ACCUMULATOR >>>")
    print(success_accumulator)
    print("<<< SUCCESS_ACCUMULATOR <<<")
    return TASK_SUCCESSFUL_SCORE
```

### Rules
- **NEVER hardcode database IDs** (user_id, hotel_id, etc.) — always query the DB to find them
- **NEVER use `env.env_variables`** — it is not available at runtime. Embed env var values as string constants at the top of your verifier (e.g., `LOGGED_IN_USER = "riley3318"`)
- **DB rows are dicts** — use `row["id"]`, `row["name"]`, NOT `row.id`, `row.name`. Using dot notation will crash with `AttributeError: 'dict' object has no attribute 'id'`
- **Only use supported query methods**: `.eq()`, `.neq()`, `.select()`, `.all()`, `.first()`, `.count()`. NO `.like()`, `.gt()`, `.lt()`, `.order()`, `.limit()`, `.contains()`, `.in_()` — filter and sort in Python instead (e.g., `sorted([r for r in rows if r["score"] > 8.0], key=lambda r: r["score"], reverse=True)[:5]`)
- **`.eq()` takes exactly 2 args**: `.eq(column, value)`. NO operator arg like `.eq("rating", ">", 8)` — use Python: `[r for r in rows if r["rating"] > 8]`
- **Use timezone-tolerant comparisons** for datetimes — the DB may store `"2025-08-08T14:00:00Z"` while you expect `"2025-08-08T14:00:00"`. Use `.startswith()` or strip the trailing `"Z"` before comparing
- **If you use `.select()`, only access the selected columns** — accessing other columns raises `KeyError`. Prefer `.all()` without `.select()` unless you specifically need to limit columns
- **Define `find_new_entries` inside your verifier function** — it is NOT a built-in. Copy it from the template above into your `validate_task()` function body. Do NOT call `find_new_entries()` without defining it first
- **List comprehensions produce tuples if you use tuple syntax** — `[(a, b) for ...]` creates tuples, not dicts. If you need dict-like access later, keep the original dicts: `[row for row in rows if condition]`
- **NEVER hardcode expected values the agent must create** — e.g., don't check for a specific phone number or email the agent would need to invent. Instead, check that the field was changed from its original value: `current_val != seed_val`
- Look up the logged-in user by name/email from the users table, don't assume an ID
- Compare `seed` (before) vs `current` (after) to detect what the agent did
- Must return `TASK_FAILED_SCORE` on a fresh environment (before agent acts)
- **NEVER call `.table("X").all()` without a preceding `.eq()` or `.neq()` filter** — unfiltered `.all()` fetches every row, which is wasteful and causes warm-pool saturation with large tables. Always filter first: `current.table("orders").eq("user_id", uid).all()`. The only exception is inside `find_new_entries` where `.select(id_field).all()` fetches just IDs for comparison
- **Use order-independent (set-based) comparison** — never compare results by row index or list position. Rows may be returned in any order. Use sets: `actual_ids = {{r["id"] for r in rows}}; assert expected_ids.issubset(actual_ids)`. NEVER do `rows[0]["id"] == expected` — it breaks when row order changes
- **Verifier MUST return 0 on unmodified DB** — the verifier must fail when the agent has not acted. Always compare `seed` vs `current` state. A verifier that only checks `current` without comparing to `seed` is permissive — it may return 1 even when the agent did nothing. Pattern: `new_entries = find_new_entries("table"); if not new_entries: return TASK_FAILED_SCORE`
- Use `final_answer` for tasks that require the agent to report a value
- Reference actual tool names from this environment

## Task Design Guidelines

Design tasks that maximize learnability: an ideal task is one that a capable agent can solve with effort, but not trivially. Tasks that are too easy (always solved) or too hard (never solved) produce no learning signal.

{date_guidance}

### Realism
Write prompts as a real user would — natural language, concrete parameters, plausible intent. The task should sound like something a person would actually ask, not a test case.

BAD:  "Call get_user with id=5, then call update_user to set email to test@example.com"
GOOD: "Update the email address for Jamie Chen to jamie.chen@newdomain.com"

### Avoiding Underspecification
A prompt is underspecified when multiple valid solutions exist but the verifier only accepts one. This creates false negatives — the agent solves the task correctly but gets reward 0.

BAD prompt:  "Find a designer in Mexico" (3 designers exist, verifier checks for one specific one)
FIX option 1: Make the prompt specific: "Find the designer in Mexico City who joined after 2023"
FIX option 2: Make the verifier accept all valid answers: check that ANY designer in Mexico is returned

Use `query_db` to check the actual data before writing the prompt. If a query returns multiple rows, either narrow the prompt or widen the verifier. Always verify your assumptions by querying — don't guess.

### Avoiding Overspecification
A prompt is overspecified when it dictates HOW to accomplish the task rather than WHAT outcome is needed. This makes the task trivially easy (no learning signal) and doesn't test real problem-solving.

BAD:  "First call list_tables, then call get_bookings with check_in_date='2024-03-15', then count the results and call submit_answer with the count"
GOOD: "How many bookings have a check-in date of March 15, 2024?"

The prompt should specify the desired outcome. The agent should figure out which tools to use and in what order.

### Complexity
Aim for tasks solvable in 2-8 tool calls. Tasks requiring 1 tool call are too easy (no signal). Tasks requiring 15+ calls are too hard (agent gives up). The sweet spot is 3-6 calls with some reasoning required.

### Diversity
Vary tasks across multiple dimensions:
- Operations: reads (lookup, search, aggregate) AND writes (create, update, delete)
- Complexity: simple (2-3 tool calls) through moderate (4-8 tool calls with dependencies)
- Reasoning: some tasks need multi-step logic (find X, use X to look up Y, modify Y based on Z)
- Data entities: use different tables, columns, and relationships in the schema

### Verifier-Prompt Consistency
The verifier must check exactly what the prompt asks — no more, no less. Before writing, verify:
1. Is there exactly one correct outcome for this prompt? (If not, widen the verifier or narrow the prompt)
2. Does the verifier return 0.0 on a fresh environment? (It must — the agent hasn't acted yet)
3. Does the verifier avoid hardcoded values? (Query the DB instead)
4. Could a different valid approach fool the verifier? (If so, fix the verifier to accept it)"""


def build_task_gen_tool_call_instructions() -> str:
    """Build the RL/Qwen XML tool-call protocol instructions."""
    return """
## Exploration Tools

The database schema is provided above. Use BOTH `query_db` AND environment API tools during exploration.

### Database Tools
<tool_call>{"name": "query_db", "arguments": {"sql": "SELECT * FROM table_name LIMIT 5"}}</tool_call>
Runs a read-only SQL query against the seed database.

### Environment API Tools
<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>
Calls the environment API tool and returns its result. **You MUST call at least one API tool** (e.g., searchEvents, getAvailability) during exploration to understand what the solver agent will experience. The solver uses these API tools, not SQL — if you only explore via SQL, you won't know whether the API tools actually work for your task.

### Workflow
1. **Inspect data**: Call `query_db` to inspect real data (values, ranges, row counts).
2. **Try API tools**: Call at least one environment API tool to understand its behavior, input/output format, and what data it returns. This is critical — your task must be achievable using these tools.
3. **Draft a task idea**: Based on the data AND tool behavior you've observed.
4. **Validate**: Before outputting, verify:
   - Does the data your prompt references actually exist? (Query to confirm.)
   - Is the task achievable using the available API tools? (You tested them.)
   - Does your verifier check for a DB mutation (e.g., new order, new cart item)? If so, does the task actually cause that mutation?
   - Will the verifier return 0 on the unmodified DB? (If it uses `find_new_entries`, the task MUST involve a write action like buy/reserve/create — NOT just search/list.)
5. **Output**: Only when confident, output the final task in the format below."""


def build_task_gen_output_format_instructions() -> str:
    """Build final task submission instructions."""
    return """
## Output Format

Generate exactly ONE task. Output it in this format:

<task>
<prompt>
[Natural language task instruction for the agent. Be specific about what needs to be done.]
</prompt>
<verifier>
[Python function: def validate_task(env, final_answer=None) -> int]
</verifier>
</task>"""


# Meta-tools the model can call to explore the seed database.
_META_TOOLS = {"query_db"}

# All callable tools = meta-tools + any MCP env tools discovered at init time.
# Populated per-instance in init_async().


class TaskGenEnv(BaseTextEnv):
    """Environment for RL-based task generation.

    The LLM generates (prompt, verifier) pairs for Fleet environments.
    Supports multi-turn: the model can explore the seed DB via ``query_db``
    meta-tool before outputting a ``<task>`` block. Schema is in the prompt.

    Reward = llm_validity * (alpha * var(raw_scores) + (p_hint - p_raw))

    Evaluation uses Fleet harness jobs (POST /v1/jobs) to run an LLM agent
    against the generated task, rather than a stub evaluator.

    Constructor args (via extras, from dataset):
        env_key, env_version, data_key, data_version
        env_tools, env_tools_schema, env_variable_keys

    Constructor args (via env_config, from Hydra):
        max_turns: Max turns before forced termination (default 10)
        judge_model: Model ID for LLM-as-a-judge gate
        k_rollouts: Number of rollouts per condition (raw/hinted, default 4)
        max_eval_steps: Max agent steps per evaluator rollout (default 30)
        evaluator_model: Fleet harness model for task evaluation (default anthropic/claude-sonnet-4.5)
        base_quality_reward: Optional reward for passing sandbox+judge (default 0.0).
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        # Configurable multi-turn (default 10; set to 1 for single-turn)
        self.max_turns = int(env_config.get("max_turns", 10)) if env_config else 10

        # Fleet orchestrator for DB exploration (set in init_async)
        self.orch = None
        # MCP tools client for calling env tools (set in init_async)
        self.mcp_tools = None
        # Set of all callable tool names (meta-tools + MCP tools)
        self.callable_tools = set(_META_TOOLS)
        # Exploration sequence tracking (reset in init_async)
        self.called_query_db = False

        # Environment context from dataset (extras)
        self.env_key = extras.get("env_key") or extras.get("data_source", "unknown")
        self.env_version = extras.get("env_version", "")
        self.data_key = extras.get("data_key", "")
        self.data_version = extras.get("data_version", "")

        # Parse env_tools_schema (full tool schemas for prompt building)
        env_tools_schema_raw = extras.get("env_tools_schema", "[]")
        if isinstance(env_tools_schema_raw, str):
            try:
                self.env_tools_schema: List[Dict[str, Any]] = json.loads(env_tools_schema_raw)
            except json.JSONDecodeError:
                self.env_tools_schema: List[Dict[str, Any]] = []
        else:
            self.env_tools_schema: List[Dict[str, Any]] = env_tools_schema_raw or []

        # Parse env_tools (tool name list for sandbox validation)
        env_tools_raw = extras.get("env_tools", [])
        if isinstance(env_tools_raw, str):
            try:
                self.env_tools: List[str] = json.loads(env_tools_raw)
            except json.JSONDecodeError:
                self.env_tools: List[str] = []
        else:
            self.env_tools: List[str] = env_tools_raw or []

        # If env_tools is empty but we have schemas, extract names from schemas
        if not self.env_tools and self.env_tools_schema:
            self.env_tools = [
                t["function"]["name"] for t in self.env_tools_schema if "function" in t and "name" in t["function"]
            ]

        # Parse env_variable_keys (available context variables for this env)
        env_var_keys_raw = extras.get("env_variable_keys", "[]")
        if isinstance(env_var_keys_raw, str):
            try:
                self.env_variable_keys: List[str] = json.loads(env_var_keys_raw)
            except json.JSONDecodeError:
                self.env_variable_keys: List[str] = []
        else:
            self.env_variable_keys: List[str] = env_var_keys_raw or []

        # Parse env_variables (actual values for harness evaluation)
        env_vars_raw = extras.get("env_variables", "{}")
        if isinstance(env_vars_raw, str):
            try:
                self.env_variables: Dict[str, Any] = json.loads(env_vars_raw)
            except json.JSONDecodeError:
                self.env_variables: Dict[str, Any] = {}
        else:
            self.env_variables: Dict[str, Any] = env_vars_raw or {}

        # Parse env_schema (compact DB schema: table→columns)
        self.env_schema: str = extras.get("env_schema", "") or ""

        # Verifier sandbox — filters out CUA-only tool "computer" from available tools
        api_tools = set(self.env_tools) - {"computer"} if self.env_tools else None
        self.sandbox = VerifierSandbox(
            available_tools=api_tools if api_tools else None,
            min_ast_nodes=env_config.get("verifier_min_ast_nodes") if env_config else None,
            max_ast_nodes=env_config.get("verifier_max_ast_nodes") if env_config else None,
        )

        # Judge config (from Hydra env_config)
        self.judge_model = str(env_config.get("judge_model", "")) if env_config else ""

        # Evaluator config (from Hydra env_config)
        self.k_rollouts = int(env_config.get("k_rollouts", 4)) if env_config else 4
        self.max_eval_steps = int(env_config.get("max_eval_steps", 30)) if env_config else 30
        self.evaluator_model = (
            str(env_config.get("evaluator_model", "anthropic/claude-sonnet-4.5"))
            if env_config
            else "anthropic/claude-sonnet-4.5"
        )

        # API keys from environment variables (set by SkyPilot YAML)
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.fleet_api_key = os.environ.get("FLEET_API_KEY", "")

        # Eval mode: k=8 raw only (no hints)
        self.is_eval = extras.get("training_phase") == "eval"
        self.eval_k_rollouts = int(env_config.get("eval_k_rollouts", 8)) if env_config else 8
        # Whether to run hinted evaluation jobs (2nd harness job with verifier feedback).
        # Default off — hints were net negative in iter#11 (verifier code dump confused evaluator).
        self.enable_hints = bool(env_config.get("enable_hints", False)) if env_config else False

        # Lazy-init Fleet SDK client for harness evaluation
        self._fleet_client = None

        # Rollout dump directory (full prompt/verifier/scores per eval)
        default_rollout_dir = os.path.join(os.path.expanduser("~"), "reward_rollouts")
        self._rollout_dir = os.environ.get("REWARD_ROLLOUT_DIR", default_rollout_dir)
        os.makedirs(self._rollout_dir, exist_ok=True)

        # Base quality reward for tasks passing sandbox + judge gate.
        # Default 0.0 keeps the learning signal tied to solver rollout variance.
        self.base_quality_reward = float(env_config.get("base_quality_reward", 0.0)) if env_config else 0.0

        # Small per-tool-call reward to incentivize DB exploration (query_db).
        # Default 0.0 = off (no behavior change for existing runs).
        self.tool_call_reward_per_call = float(env_config.get("tool_call_reward_per_call", 0.0)) if env_config else 0.0

        logger.info(
            f"TaskGenEnv: env={self.env_key}, max_turns={self.max_turns}, "
            f"judge={self.judge_model or 'none'}, "
            f"tools={len(self.env_tools)}, k={self.k_rollouts}, eval_k={self.eval_k_rollouts}, "
            f"evaluator={self.evaluator_model}, is_eval={self.is_eval}, "
            f"base_quality={self.base_quality_reward}, tool_call_reward={self.tool_call_reward_per_call}"
        )

    def _format_tool_schema(self, tool: Dict[str, Any]) -> str:
        """Format a single tool schema for the system prompt."""
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        lines = [f"**{name}**: {desc}"]
        if properties:
            lines.append("  Parameters:")
            for pname, pschema in properties.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req_marker = " (required)" if pname in required else ""
                lines.append(f"  - {pname} ({ptype}{req_marker}): {pdesc}")

        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Build the system prompt with environment context and priors."""
        parts = [
            build_task_gen_db_schema_prompt(
                env_key=self.env_key,
                env_tools_schema=self.env_tools_schema,
                env_tools=self.env_tools,
                env_variables=self.env_variables,
                env_variable_keys=self.env_variable_keys,
                env_schema=self.env_schema,
            ),
            build_task_gen_task_verifier_instructions(self.env_variables.get("CURRENT_DATE", "")),
        ]
        if self.max_turns > 1:
            parts.append(build_task_gen_tool_call_instructions())

        # Few-shot examples were removed because they anchored the model to
        # generate near-copies of the examples (especially booking/wishlist tasks),
        # causing mode collapse and zero reward signal. The verifier template +
        # guidelines above provide enough structure for the model to generate
        # diverse tasks from the actual DB schema and tools.
        parts.append(build_task_gen_output_format_instructions())
        return "\n".join(parts)

    def _judge_task(self, prompt: str, verifier: str) -> float:
        """LLM classifier gate: returns 0.0 (reject) or 1.0 (accept).

        Predicts whether the (prompt, verifier) pair will produce meaningful
        evaluation signal. Optimized for very low false positive rate — only
        rejects tasks that are near-certain to waste harness compute.

        Checks:
            1. Phantom tables: verifier references tables not in env schema
            2. Undefined references: calls to functions/constants not defined
            3. Vacuous checks: verifier only checks user existence or len>0
        """
        if not self.judge_model or not self.openrouter_api_key:
            return 1.0  # No judge configured, pass through

        # Build context for the classifier
        tool_names = [t for t in self.env_tools if t != "computer"]
        tools_str = ", ".join(tool_names[:20]) if tool_names else "none discovered"

        schema_block = self.env_schema if self.env_schema else "Schema not available."

        judge_prompt = (
            "You are a verifier quality judge for an AI task-generation system. You evaluate "
            "whether a generated verifier function can reliably determine if an AI agent "
            "correctly completed a task.\n\n"
            "## Context\n\n"
            "The verifier has access to:\n"
            '- `env.db("seed")` — database state BEFORE the agent acted\n'
            '- `env.db("current")` — database state AFTER the agent acted\n'
            "- `final_answer` — the agent's text response\n"
            "- DB query methods: `.table(name)`, `.eq(col, val)`, `.first()`, `.all()`, "
            "`.select()`, `.neq()`, `.gt()`, `.lt()`\n\n"
            f"Database schema (valid tables and columns):\n```\n{schema_block}\n```\n\n"
            f'Environment: "{self.env_key}"\n'
            f"Available tools: {tools_str}\n\n"
            "## Classification Criteria\n\n"
            "### ACCEPT if the verifier does ANY of:\n\n"
            "1. **Mutation verification**: Compares seed vs current database state to detect "
            "that the agent created, modified, or deleted records.\n\n"
            "2. **DB-grounded answer validation**: Queries the database for specific records "
            "and validates that values FROM those records appear in `final_answer`. The "
            "expected values must come from the database, not from hardcoded strings or "
            "the task prompt.\n\n"
            "3. **Specific record validation**: Looks up a record by ID or unique field and "
            "checks its field values match expected values.\n\n"
            "### REJECT if the verifier does ANY of:\n\n"
            "1. **Generic keyword checking**: Checks if generic category words appear in "
            '`final_answer` (e.g., "event", "venue", "concert", "price", "bedroom", '
            '"listing"). These words appear in any topically-relevant response regardless '
            "of task completion.\n\n"
            "2. **Prompt echo checking**: Checks if values from the task prompt appear in "
            '`final_answer` (e.g., "Los Angeles" when the prompt asked about LA events). '
            "The agent could echo prompt values without doing any work.\n\n"
            "3. **Exists-check-only**: Only checks `final_answer is not None` or "
            "`len(answer) > 0`.\n\n"
            "4. **Dead code DB queries**: Has `seed.table()` or `current.table()` calls but "
            "never uses the query results in conditional logic that affects the return value.\n\n"
            "5. **Nonexistent API access**: References `env.instance.tool_calls`, "
            "`get_call_history()`, or `env.call_tool()` — these don't exist in the verifier "
            "runtime.\n\n"
            "6. **Cargo-cult DB**: Queries the DB only for user/account existence (which always "
            "passes for pre-existing entities), then gates on keyword checks for actual "
            "validation.\n\n"
            '7. **Phantom tables**: The verifier calls `.table("X")` where X does not exist '
            "in the schema above.\n\n"
            "8. **Undefined references**: The verifier calls functions or uses constants that "
            "are not defined in the code and are not Python builtins.\n\n"
            "### Edge Cases:\n\n"
            "- Read-only tasks with DB-grounded keywords: ACCEPT — if the verifier queries a "
            "DB table to get specific values then checks those values appear in `final_answer`.\n"
            "- JSON structure validation without DB cross-reference: REJECT.\n"
            '- Existence checks on initially-empty tables (e.g., orders after "place order"): '
            "weak ACCEPT.\n\n"
            f"## Generated Task\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Verifier:\n```python\n{verifier}\n```\n\n"
            "Answer with exactly one word: ACCEPT or REJECT"
        )

        try:
            import litellm

            response = litellm.completion(
                model=f"openrouter/{self.judge_model}",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=10,
                api_key=self.openrouter_api_key,
            )
            answer = response.choices[0].message.content.strip().upper()
            accepted = "ACCEPT" in answer and "REJECT" not in answer
            logger.info(f"LLM classifier [{self.env_key}]: {answer} -> " f"{'ACCEPT' if accepted else 'REJECT'}")
            return 1.0 if accepted else 0.0
        except Exception as e:
            logger.warning(f"LLM classifier failed, defaulting to accept: {e}")
            return 1.0

    @staticmethod
    def _build_hint_text(
        verifier_stdout: Optional[str],
        verifier_error: Optional[str],
        tool_error_messages: Optional[List[str]],
    ) -> str:
        """Build hint text from verifier feedback. No LLM call.

        Parses ERROR_ACCUMULATOR / SUCCESS_ACCUMULATOR from verifier stdout
        and formats tool errors into structured feedback for hinted rollouts.
        """
        parts: List[str] = []

        if verifier_stdout:
            err_match = re.search(
                r">>> ERROR_ACCUMULATOR >>>\n(.+?)\n<<< ERROR_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            suc_match = re.search(
                r">>> SUCCESS_ACCUMULATOR >>>\n(.+?)\n<<< SUCCESS_ACCUMULATOR <<<",
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

    def _get_fleet_client(self):
        """Lazy-init Fleet SDK client."""
        if self._fleet_client is None:
            from fleet import Fleet

            timeout = float(os.environ.get("FLEET_CLIENT_TIMEOUT", "60"))
            self._fleet_client = Fleet(api_key=self.fleet_api_key, timeout=timeout)
        return self._fleet_client

    def _record_fleet_job_event(
        self,
        *,
        job_id: str,
        task_key: str,
        rollout_label: str,
        pass_k: int,
        event: str,
        status: str = "",
        elapsed: Optional[float] = None,
        error: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist Fleet job status while the remote harness is still running."""
        try:
            run_name = os.environ.get("RUN_NAME", "unknown")
            record: Dict[str, Any] = {
                "timestamp": time.time(),
                "run_name": run_name,
                "event": event,
                "job_id": job_id,
                "task_key": task_key,
                "rollout_label": rollout_label,
                "status": status,
                "elapsed_seconds": elapsed,
                "pass_k": pass_k,
                "max_eval_steps": self.max_eval_steps,
                "env_key": self.env_key,
                "data_key": self.data_key,
                "data_version": self.data_version,
                "evaluator_model": self.evaluator_model,
                "error": error,
            }
            if extra:
                record.update(extra)

            os.makedirs(self._rollout_dir, exist_ok=True)
            status_path = os.path.join(self._rollout_dir, "fleet_job_status.jsonl")
            with open(status_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if job_id:
                jobs_dir = os.path.join(self._rollout_dir, "fleet_jobs")
                os.makedirs(jobs_dir, exist_ok=True)
                snapshot_path = os.path.join(jobs_dir, f"{job_id}.json")
                with open(snapshot_path, "w") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                    f.write("\n")
        except Exception as e:
            logger.warning(f"Failed to record Fleet job event for {job_id}: {e}")

    async def _poll_job(
        self,
        fleet,
        job_id: str,
        *,
        task_key: str,
        rollout_label: str,
        pass_k: int,
        poll_interval: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Poll Fleet job until completion or timeout.

        Returns:
            Final job status string.
        """
        poll_interval = poll_interval or int(os.environ.get("FLEET_JOB_POLL_INTERVAL", "10"))
        timeout = timeout or int(os.environ.get("FLEET_JOB_POLL_TIMEOUT", "600"))
        start = time.time()
        poll_count = 0
        while time.time() - start < timeout:
            elapsed = time.time() - start
            poll_count += 1
            try:
                request_start = time.time()
                job = fleet.get_job(job_id)
                request_seconds = time.time() - request_start
                status = job.status
                self._record_fleet_job_event(
                    job_id=job_id,
                    task_key=task_key,
                    rollout_label=rollout_label,
                    pass_k=pass_k,
                    event="poll",
                    status=status,
                    elapsed=elapsed,
                    extra={"poll_count": poll_count, "request_seconds": round(request_seconds, 3)},
                )
                print(
                    f"[task-gen eval] Job {job_id} status={status} "
                    f"elapsed={elapsed:.0f}s poll={poll_count}",
                    flush=True,
                )
                if status in ("completed", "cancelled", "errored"):
                    self._record_fleet_job_event(
                        job_id=job_id,
                        task_key=task_key,
                        rollout_label=rollout_label,
                        pass_k=pass_k,
                        event="terminal",
                        status=status,
                        elapsed=time.time() - start,
                        extra={"poll_count": poll_count},
                    )
                    return status
            except Exception as e:
                self._record_fleet_job_event(
                    job_id=job_id,
                    task_key=task_key,
                    rollout_label=rollout_label,
                    pass_k=pass_k,
                    event="poll_error",
                    elapsed=elapsed,
                    error=str(e),
                    extra={"poll_count": poll_count},
                )
                logger.warning(f"Error polling job {job_id}: {e}")
                print(f"[task-gen eval] Error polling job {job_id}: {e}", flush=True)
            await asyncio.sleep(poll_interval)

        logger.error(f"Job {job_id} timed out after {timeout}s")
        print(f"[task-gen eval] Job {job_id} timed out after {timeout}s", flush=True)
        self._record_fleet_job_event(
            job_id=job_id,
            task_key=task_key,
            rollout_label=rollout_label,
            pass_k=pass_k,
            event="timeout",
            status="timeout",
            elapsed=time.time() - start,
        )
        return "timeout"

    def _query_supabase_scores(self, job_id: str) -> Dict[str, float]:
        """Query Supabase for session verifier scores as fallback.

        When Fleet backend doesn't populate verifier_execution FK (regression
        since 2026-03-23), the score is still available in session metadata.

        Returns:
            Dict mapping session_id -> verifier_score.
        """
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        if not supabase_url or not supabase_key:
            return {}
        try:
            import httpx

            resp = httpx.get(
                f"{supabase_url}/rest/v1/sessions",
                params={"job_id": f"eq.{job_id}", "select": "id,metadata"},
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"Supabase query failed: {resp.status_code}")
                return {}
            scores = {}
            for row in resp.json():
                meta = row.get("metadata") or {}
                sid = row.get("id")
                v_score = meta.get("verifier_score")
                if sid and v_score is not None:
                    scores[sid] = float(v_score)
            return scores
        except Exception as e:
            logger.warning(f"Supabase fallback failed: {e}")
            return {}

    def _extract_job_results(
        self,
        fleet,
        job_id: str,
        *,
        task_key: str,
        rollout_label: str = "raw",
        pass_k: int = 0,
    ) -> List[Dict[str, Any]]:
        """Extract per-session solver results from completed job sessions.

        Primary path: read from session.verifier_execution (Fleet SDK).
        Fallback: query Supabase for metadata.verifier_score when VE is null
        (Fleet backend regression since 2026-03-23 stopped populating VE FK).

        Returns:
            List of JSON-serializable session records.
        """
        results: List[Dict[str, Any]] = []
        self._record_fleet_job_event(
            job_id=job_id,
            task_key=task_key,
            rollout_label=rollout_label,
            pass_k=pass_k,
            event="fetch_sessions_start",
            status="completed",
        )
        sessions_response = fleet.list_job_sessions(job_id)
        session_count = sum(len(tg.sessions) for tg in sessions_response.tasks)
        self._record_fleet_job_event(
            job_id=job_id,
            task_key=task_key,
            rollout_label=rollout_label,
            pass_k=pass_k,
            event="fetch_sessions_done",
            status="completed",
            extra={"session_count": session_count},
        )

        # Check if any session has verifier_execution populated
        all_ve_null = all(s.verifier_execution is None for tg in sessions_response.tasks for s in tg.sessions)

        # Fallback: query Supabase only when needed
        supabase_scores: Dict[str, float] = {}
        if all_ve_null:
            supabase_scores = self._query_supabase_scores(job_id)
            if supabase_scores:
                logger.info(f"[{job_id[:8]}] Using Supabase fallback for {len(supabase_scores)} session scores")

        for task_group in sessions_response.tasks:
            for session in task_group.sessions:
                raw_session = _json_safe(session, max_depth=8)
                session_payload = raw_session if isinstance(raw_session, dict) else {}
                session_id = (
                    getattr(session, "session_id", "")
                    or getattr(session, "id", "")
                    or str(session_payload.get("session_id") or session_payload.get("id") or "")
                )
                existing_transcript_payload = _first_present_mapping_value(session_payload, _SESSION_TRANSCRIPT_KEYS)
                session_transcript = None
                session_transcript_error = ""
                if session_id and not existing_transcript_payload:
                    try:
                        session_transcript = _json_safe(fleet.get_session_transcript(session_id), max_depth=8)
                    except Exception as e:
                        session_transcript_error = str(e)
                        logger.warning(
                            f"Failed to fetch Fleet transcript for session {session_id} in job {job_id}: {e}"
                        )
                elif existing_transcript_payload:
                    session_transcript = existing_transcript_payload

                transcript_fields = _extract_transcript_fields(session_transcript)
                trajectory = _first_present_mapping_value(
                    session_payload,
                    _SESSION_TRAJECTORY_KEYS,
                ) or transcript_fields.get("trajectory")
                messages = session_payload.get("messages") or transcript_fields.get("messages")
                steps = session_payload.get("steps") or transcript_fields.get("steps")
                actions = session_payload.get("actions") or transcript_fields.get("actions")
                events = session_payload.get("events") or transcript_fields.get("events")
                transcript = session_payload.get("transcript") or transcript_fields.get("transcript")
                score = 0.0
                stdout = None
                error = None
                verifier_success = None
                verifier_execution_id = ""
                if session.verifier_execution:
                    verifier_execution_id = (
                        getattr(session.verifier_execution, "id", "")
                        or getattr(session.verifier_execution, "verifier_execution_id", "")
                    )
                    if session.verifier_execution.score is not None:
                        score = float(session.verifier_execution.score)
                    elif session.verifier_execution.success:
                        score = 1.0
                    verifier_success = bool(session.verifier_execution.success)
                    stdout = session.verifier_execution.stdout
                    # Capture error from verifier crashes — error is nested in result.error
                    ve_result = session.verifier_execution.result
                    if ve_result:
                        ve_error = ve_result.get("error") if isinstance(ve_result, dict) else ve_result.error
                        if ve_error:
                            error = ve_error.get("message", "") if isinstance(ve_error, dict) else ve_error.message
                            traceback_str = (
                                ve_error.get("traceback", "") if isinstance(ve_error, dict) else ve_error.traceback
                            )
                            if traceback_str:
                                # Extract just the last line of traceback (the actual error)
                                error = traceback_str.strip().split("\n")[-1] if traceback_str else error
                elif session_id in supabase_scores:
                    # Fallback: use Supabase metadata.verifier_score
                    score = supabase_scores[session_id]
                session_result = {
                    "session_id": session_id,
                    "task_key": getattr(task_group, "task_key", "") or getattr(task_group, "key", "") or task_key,
                    "score": score,
                    "verifier_stdout": stdout,
                    "verifier_error": error,
                    "verifier_success": verifier_success,
                    "verifier_execution_id": verifier_execution_id,
                    "transcript": transcript,
                    "trajectory": trajectory,
                    "messages": messages,
                    "steps": steps,
                    "actions": actions,
                    "events": events,
                    "session_payload": session_payload,
                    "raw_session": raw_session,
                }
                if session_transcript:
                    session_result["session_transcript"] = session_transcript
                if session_transcript_error:
                    session_result["session_transcript_error"] = session_transcript_error
                results.append(session_result)
                print(
                    f"[task-gen eval] Finished {rollout_label} solver rollout {len(results)} "
                    f"for job {job_id}: score={score}",
                    flush=True,
                )
        self._record_fleet_job_event(
            job_id=job_id,
            task_key=task_key,
            rollout_label=rollout_label,
            pass_k=pass_k,
            event="results_extracted",
            status="completed",
            extra={
                "scores": [result["score"] for result in results],
                "session_ids": [result.get("session_id", "") for result in results],
                "session_count": len(results),
            },
        )
        return results

    async def _run_harness_job(
        self, prompt: str, verifier: str, k: int, rollout_label: str = "raw"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Run a single Fleet harness job and return per-session results + job ID.

        1. Import task to Fleet
        2. Create harness job with pass_k=k
        3. Poll until completion
        4. Extract results

        Returns:
            Tuple of (job_id, results) where results is a list of per-session records.
        """
        from fleet.tasks import Task

        fleet = self._get_fleet_client()
        task_key = f"taskgen_{uuid.uuid4().hex[:12]}"
        print(
            f"[task-gen eval] Preparing {rollout_label} harness job for {k} solver rollout(s) "
            f"(model={self.evaluator_model}, env={self.env_key})",
            flush=True,
        )
        for rollout_index in range(1, k + 1):
            print(
                f"[task-gen eval] Starting {rollout_label} solver rollout {rollout_index}/{k} " f"for task {task_key}",
                flush=True,
            )

        task = Task(
            key=task_key,
            prompt=prompt,
            env_id=self.env_key,
            version=self.env_version or None,
            verifier_func=verifier,
            data_id=self.data_key or None,
            data_version=self.data_version or None,
            env_variables=self.env_variables or {},
        )

        try:
            print(f"[task-gen eval] Importing generated task {task_key} into Fleet", flush=True)
            import_response = fleet.import_single_task(task)
        except Exception as e:
            logger.error(f"[{task_key}] Failed to import task to Fleet: {e}")
            print(f"[task-gen eval] Failed to import task {task_key}: {e}", flush=True)
            raise RuntimeError(f"Failed to import task {task_key} to Fleet: {e}") from e
        if import_response is None:
            logger.error(
                f"[{task_key}] Failed to import task to Fleet (returned None, api_key set: {bool(self.fleet_api_key)})"
            )
            print(f"[task-gen eval] Failed to import task {task_key}: Fleet returned None", flush=True)
            raise RuntimeError(f"Failed to import task {task_key} to Fleet: import returned None")

        print(
            f"[task-gen eval] Creating {rollout_label} Fleet harness job for task {task_key} "
            f"with pass_k={k}, max_steps={self.max_eval_steps}",
            flush=True,
        )
        try:
            job_response = fleet.create_job(
                models=[self.evaluator_model],
                task_keys=[task_key],
                pass_k=k,
                max_steps=self.max_eval_steps,
                mode="tool-use",
                name=f"taskgen-eval-{task_key}",
            )
        except Exception as e:
            logger.error(f"[{task_key}] Failed to create Fleet harness job: {e}")
            print(f"[task-gen eval] Failed to create Fleet harness job for task {task_key}: {e}", flush=True)
            raise RuntimeError(f"Failed to create Fleet harness job for task {task_key}: {e}") from e
        if job_response is None or not getattr(job_response, "job_id", None):
            raise RuntimeError(f"Failed to create Fleet harness job for task {task_key}: missing job_id")
        job_id = job_response.job_id
        logger.info(f"[{task_key}] Harness job created: {job_id} (model={self.evaluator_model}, k={k})")
        print(f"[task-gen eval] Harness job created: {job_id} for task {task_key}", flush=True)
        self._record_fleet_job_event(
            job_id=job_id,
            task_key=task_key,
            rollout_label=rollout_label,
            pass_k=k,
            event="created",
            status="created",
        )

        try:
            status = await self._poll_job(
                fleet,
                job_id,
                task_key=task_key,
                rollout_label=rollout_label,
                pass_k=k,
            )
            if status != "completed":
                logger.warning(f"[{task_key}] Job {job_id} ended with status: {status}")
                print(f"[task-gen eval] Job {job_id} ended with status={status}; failing evaluation", flush=True)
                raise FleetHarnessJobError(
                    f"Fleet harness job {job_id} ended with status={status}",
                    job_id=job_id,
                    task_key=task_key,
                    status=status,
                )

            results = self._extract_job_results(
                fleet,
                job_id,
                task_key=task_key,
                rollout_label=rollout_label,
                pass_k=k,
            )
            if len(results) != k:
                raise FleetHarnessJobError(
                    f"Fleet harness job {job_id} returned {len(results)} session(s), expected {k}",
                    job_id=job_id,
                    task_key=task_key,
                    status="incomplete_sessions",
                )
        except FleetHarnessJobError:
            raise
        except Exception as e:
            raise FleetHarnessJobError(
                f"Fleet harness job {job_id} failed while collecting results: {e}",
                job_id=job_id,
                task_key=task_key,
                status="result_collection_failed",
            ) from e
        print(
            f"[task-gen eval] Completed {rollout_label} job {job_id}: "
            f"scores={[result['score'] for result in results]}",
            flush=True,
        )
        return (job_id, results)

    async def _evaluate_task(self, prompt: str, verifier: str) -> Dict[str, float]:
        """Run hint-based evaluation via Fleet harness jobs.

        1. Raw job: k rollouts without hints
        2. Build hint from first failing session's verifier stdout
        3. Hinted job: k rollouts with hint appended to prompt
        4. Compute reward via compute_task_reward()

        Returns:
            Reward breakdown dict from compute_task_reward.
        """
        from integrations.fleet.task_gen_reward import compute_task_reward

        zero_result = compute_task_reward([], [], validity=1.0)

        if not self.fleet_api_key:
            raise RuntimeError("FLEET_API_KEY is required for task-generation solver evaluation.")

        task_id = f"taskgen_{uuid.uuid4().hex[:8]}"
        start = time.time()
        raw_job_id = None
        hinted_job_id = None
        raw_scores: List[float] = []
        hinted_scores: List[float] = []
        raw_results: List[Dict[str, Any]] = []
        hinted_results: List[Dict[str, Any]] = []
        hint_text = ""

        try:
            # Eval: k=eval_k_rollouts for pass rate; Train: k=k_rollouts
            eval_k = self.eval_k_rollouts if self.is_eval else self.k_rollouts
            print(
                f"[task-gen eval] Starting task evaluation: phase={'eval' if self.is_eval else 'train'}, "
                f"raw_rollouts={eval_k}, hinted_rollouts={self.k_rollouts if self.enable_hints and not self.is_eval else 0}, "
                f"model={self.evaluator_model}, max_steps={self.max_eval_steps}",
                flush=True,
            )

            # 1. Raw job: k rollouts without hints
            try:
                raw_job_id, raw_results = await self._run_harness_job(prompt, verifier, k=eval_k, rollout_label="raw")
            except FleetHarnessJobError as e:
                raw_job_id = e.job_id or raw_job_id
                raise
            raw_scores = [float(result.get("score", 0.0)) for result in raw_results]
            print(f"[task-gen eval] Raw rollout scores: {raw_scores}", flush=True)

            if self.enable_hints and not self.is_eval:
                # Hinted training: k raw + k hinted for hint_gap signal
                # 2. Build hint from first failing session's stdout/error
                hint_stdout = None
                hint_error = None
                for result in raw_results:
                    score = float(result.get("score", 0.0))
                    if score < 1.0:
                        stdout = result.get("verifier_stdout")
                        error = result.get("verifier_error")
                        if stdout:
                            hint_stdout = stdout
                        if error:
                            hint_error = error
                        if hint_stdout or hint_error:
                            break
                hint_text = self._build_hint_text(hint_stdout, hint_error, None)

                # Fallback: if hint is generic (no VE stdout due to backend regression),
                # use the verifier source code as the hint. This tells the hinted agent
                # exactly what checks to satisfy, creating hint_gap signal.
                if hint_text == "The previous attempt failed. Try a different approach.":
                    # Truncate verifier to avoid blowing up prompt length
                    verifier_hint = verifier[:2000]
                    hint_text = (
                        "Here is the verification function that will be used to check your work. "
                        "Make sure your actions satisfy all the checks:\n\n"
                        f"```python\n{verifier_hint}\n```"
                    )

                # 3. Hinted job: k rollouts with hint
                hinted_prompt = f"{prompt}\n\nHere is feedback from a previous attempt to help you:\n{hint_text}"
                try:
                    hinted_job_id, hinted_results = await self._run_harness_job(
                        hinted_prompt, verifier, k=self.k_rollouts, rollout_label="hinted"
                    )
                except FleetHarnessJobError as e:
                    hinted_job_id = e.job_id or hinted_job_id
                    raise
                hinted_scores = [float(result.get("score", 0.0)) for result in hinted_results]
                print(f"[task-gen eval] Hinted rollout scores: {hinted_scores}", flush=True)

                # 4. Compute reward
                result = compute_task_reward(raw_scores, hinted_scores, validity=1.0)
            else:
                # No hints — reward based on raw variance only
                hinted_job_id = None
                result = compute_task_reward(raw_scores, raw_scores, validity=1.0)

            duration = time.time() - start
            print(
                f"[task-gen eval] Reward computed: p_raw={result['p_raw']:.4f}, "
                f"var_raw={result['var_raw']:.4f}, hint_gap={result['hint_gap']:.4f}, "
                f"total={result['total']:.4f}, duration={duration:.0f}s",
                flush=True,
            )

            # --- Iron-clad eval logging ---
            # Truncate prompt/verifier for log readability
            prompt_log = prompt[:300].replace("\n", " ")
            verifier_log = verifier[:200].replace("\n", " ")
            hint_log = hint_text[:200].replace("\n", " ")
            logger.info(
                f"[{task_id}] EVAL | "
                f"raw_job={raw_job_id} hinted_job={hinted_job_id} | "
                f"raw={raw_scores} hinted={hinted_scores} | "
                f"var={result['var_raw']:.4f} gap={result['hint_gap']:.4f} total={result['total']:.4f} | "
                f"time={duration:.0f}s | "
                f"prompt={prompt_log} | "
                f"verifier={verifier_log} | "
                f"hint={hint_log}"
            )

            # Save full rollout to local JSONL
            self._save_rollout(
                task_id=task_id,
                env_key=self.env_key,
                data_key=self.data_key,
                prompt=prompt,
                verifier=verifier,
                hint=hint_text,
                raw_scores=raw_scores,
                hinted_scores=hinted_scores,
                raw_job_id=raw_job_id,
                hinted_job_id=hinted_job_id,
                raw_sessions=raw_results,
                hinted_sessions=hinted_results,
                result=result,
                duration=duration,
            )

            return result

        except Exception as e:
            duration = time.time() - start
            logger.exception(f"[{task_id}] Evaluation failed: {e}")
            print(f"[task-gen eval] Evaluation failed for {task_id}: {e}", flush=True)
            self._save_rollout(
                task_id=task_id,
                env_key=self.env_key,
                data_key=self.data_key,
                prompt=prompt,
                verifier=verifier,
                hint=hint_text,
                raw_scores=raw_scores,
                hinted_scores=hinted_scores,
                raw_job_id=raw_job_id,
                hinted_job_id=hinted_job_id,
                raw_sessions=raw_results,
                hinted_sessions=hinted_results,
                result=zero_result,
                duration=duration,
                error=str(e),
            )
            self._record_fleet_job_event(
                job_id=raw_job_id or "",
                task_key=task_id,
                rollout_label="eval",
                pass_k=self.eval_k_rollouts if self.is_eval else self.k_rollouts,
                event="evaluation_failed",
                status="failed",
                elapsed=duration,
                error=str(e),
            )
            raise

    def _save_rollout(
        self,
        task_id,
        env_key,
        data_key,
        prompt,
        verifier,
        hint,
        raw_scores,
        hinted_scores,
        raw_job_id,
        hinted_job_id,
        raw_sessions,
        hinted_sessions,
        result,
        duration,
        error="",
    ):
        """Append full rollout data to a local JSONL file."""
        try:
            run_name = os.environ.get("RUN_NAME", "unknown")
            path = os.path.join(self._rollout_dir, f"{run_name}.jsonl")
            record = {
                "status": "failed" if error else "completed",
                "task_id": task_id,
                "env_key": env_key,
                "data_key": data_key,
                "prompt": prompt,
                "verifier": verifier,
                "hint": hint,
                "raw_scores": raw_scores,
                "hinted_scores": hinted_scores,
                "raw_sessions": raw_sessions,
                "hinted_sessions": hinted_sessions,
                "raw_job_id": raw_job_id,
                "hinted_job_id": hinted_job_id,
                "var_raw": result["var_raw"],
                "hint_gap": result["hint_gap"],
                "total": result["total"],
                "duration": duration,
                "timestamp": time.time(),
                "error": error,
            }
            with open(path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[{task_id}] Failed to save rollout: {e}")

    async def dryrun_verifier(self, verifier: str) -> Tuple[bool, str]:
        """Run verifier against seed DB (no agent actions). Returns (ok, error_msg).

        A correct verifier should return 0 on unmodified DB (task not done yet).
        Returns 1 → broken (permissive). Crashes → broken.
        """
        if self.orch is None:
            return True, ""  # Can't dry-run without orchestrator, skip
        try:
            from fleet._async.tasks import Task as AsyncFleetTask

            task = AsyncFleetTask(
                key=f"dryrun_{self.env_key}",
                prompt="dry-run",
                env_id=self.env_key,
                verifier_func=verifier,
            )
            result = await task.verify_detailed_async(self.orch._fleet_env)
            if not result.success:
                return False, f"Verifier execution failed on seed DB: {result.error}"
            verifier_score = result.result
            try:
                verifier_passed = float(verifier_score) > 0.0
            except (TypeError, ValueError):
                verifier_passed = bool(verifier_score)
            if verifier_passed:
                return (
                    False,
                    "Verifier returned 1 on the unmodified database — it passes even when no agent has acted. Your verifier must return 0 on seed state. Check that your task involves a write/mutation action and your verifier checks for that mutation (e.g., find_new_entries).",
                )
            return True, ""
        except Exception as e:
            err_msg = str(e)
            # Truncate long tracebacks
            if len(err_msg) > 500:
                err_msg = err_msg[:500] + "..."
            return False, f"Verifier crashed on seed DB: {err_msg}"

    async def _handle_task_generation(self, action: str) -> BaseTextEnvStepOutput:
        """Evaluate a generated task through the full pipeline.

        Pipeline:
            1. Parse <task> output -> fail = reward 0
            2. Sandbox validation -> fail = reward 0
            3. Verifier dry-run on seed DB -> if broken, return feedback (retry)
            4. LLM-as-a-judge -> gate (0/1), fail = reward 0
            5. Hint-based evaluation via Fleet harness (k raw + k hinted rollouts)
            6. R = base_quality + binary_eval_signal

        base_quality rewards structural validity (sandbox+judge pass) when enabled.
        """
        metadata: Dict[str, Any] = {"env_key": self.env_key, "turn": self.turns}
        max_turns_reached = self.turns >= self.max_turns

        # 1. Parse
        parsed = parse_task_output(action)
        if parsed is None:
            metadata["error"] = "parse_failed"
            metadata["reward_breakdown"] = {"total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        prompt = parsed["prompt"]
        verifier = parsed["verifier"]
        metadata["generated_prompt"] = prompt
        metadata["generated_verifier"] = verifier

        # 2. Sandbox validation
        validation = self.sandbox.validate(verifier, prompt)
        metadata["validation"] = {
            "valid": validation.valid,
            "passed": validation.checks_passed,
            "failed": validation.checks_failed,
            "error": validation.error,
        }
        if not validation.valid:
            if not max_turns_reached:
                remaining = self.max_turns - self.turns
                obs = {
                    "role": "user",
                    "content": f"Sandbox rejected your verifier: {', '.join(validation.checks_failed)}. Fix and resubmit. {remaining} turn(s) left.",
                }
                return BaseTextEnvStepOutput(observations=[obs], reward=0.0, done=False, metadata=metadata)
            metadata["reward_breakdown"] = {"sandbox": 0.0, "total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        # 3. Verifier dry-run on seed DB
        dryrun_ok, dryrun_error = await self.dryrun_verifier(verifier)
        metadata["dryrun_ok"] = dryrun_ok
        if not dryrun_ok:
            metadata["dryrun_error"] = dryrun_error
            logger.info(f"TaskGenEnv [{self.env_key}]: Verifier dry-run failed: {dryrun_error[:200]}")
            if not max_turns_reached:
                remaining = self.max_turns - self.turns
                obs = {
                    "role": "user",
                    "content": f"⚠️ Verifier dry-run FAILED: {dryrun_error}\n\nFix your verifier and resubmit. {remaining} turn(s) left.",
                }
                return BaseTextEnvStepOutput(observations=[obs], reward=0.0, done=False, metadata=metadata)
            metadata["reward_breakdown"] = {"dryrun": 0.0, "total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        # 4. LLM-as-a-judge gate
        judge_gate = self._judge_task(prompt, verifier)
        metadata["judge_gate"] = judge_gate

        if judge_gate == 0.0:
            metadata["reward_breakdown"] = {"sandbox": 1.0, "judge": 0.0, "total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        # 5. Hint-based evaluation via Fleet harness
        eval_result = await self._evaluate_task(prompt, verifier)

        # 6. R = base_quality + binary_eval_signal
        base_quality = self.base_quality_reward
        reward = base_quality + eval_result["total"]

        metadata["reward_breakdown"] = {
            "sandbox": 1.0,
            "dryrun": 1.0,
            "judge": judge_gate,
            "base_quality": base_quality,
            **eval_result,
            "total": reward,
        }

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Sync wrapper for step_async."""
        return asyncio.run(self.step_async(action))

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step — tool call, task generation, or nudge.

        Multi-turn flow:
            1. <task> block detected  → evaluation pipeline (done=True)
            2. <tool_call> detected   → execute query_db/MCP tools (done=False)
            3. Neither                → nudge observation (done=False)
            4. max_turns reached      → done=True, reward=0
        """
        self.turns += 1
        max_turns_reached = self.turns >= self.max_turns

        # 1. Check for <task> block → evaluation pipeline
        if "<task>" in action:
            # Exploration gate: in multi-turn mode, bounce back if model hasn't
            # called query_db yet and still has turns remaining. Prevents
            # single-turn collapse where model skips DB exploration entirely.
            if self.max_turns > 1 and not self.called_query_db and not max_turns_reached:
                remaining = self.max_turns - self.turns
                nudge = (
                    "You must explore the database with `query_db` before submitting a task. "
                    "Use SELECT queries to inspect actual data — table contents, value ranges, "
                    f"row counts — so your task and verifier are grounded in real data. "
                    f"You have {remaining} turn(s) remaining."
                )
                observation = {"role": "user", "content": nudge}
                return BaseTextEnvStepOutput(
                    observations=[observation],
                    reward=0.0,
                    done=False,
                    metadata={"env_key": self.env_key, "turn": self.turns, "exploration_gate": True},
                )
            return await self._handle_task_generation(action)

        # 2. Check for tool calls → execute all via Fleet orchestrator or MCP
        tool_calls = parse_tool_calls(action)
        tool_calls = [tc for tc in tool_calls if tc["name"] in self.callable_tools]
        if tool_calls:
            results = []
            for tc in tool_calls:
                if tc["name"] in _META_TOOLS:
                    self.meta_tool_calls += 1
                    if tc["name"] == "query_db":
                        self.called_query_db = True
                    result = await self._execute_meta_tool(tc)
                else:
                    self.mcp_tool_calls += 1
                    result = await self._execute_mcp_tool(tc)
                results.append(f"[{tc['name']}] {result}")

            if max_turns_reached:
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata={"env_key": self.env_key, "turn": self.turns, "done_reason": "max_turns"},
                )

            obs_content = "\n\n".join(results)
            remaining = self.max_turns - self.turns
            if remaining <= 3 and self.called_query_db:
                obs_content += (
                    f"\n\n⚠️ You have {remaining} turn(s) left. "
                    "You MUST output your <task> block NOW. "
                    "Stop exploring and generate the task."
                )
            observation = {"role": "user", "content": obs_content}
            return BaseTextEnvStepOutput(
                observations=[observation],
                reward=0.0,
                done=False,
                metadata={"env_key": self.env_key, "turn": self.turns, "tool_calls": [tc["name"] for tc in tool_calls]},
            )

        # 3. Neither task nor tool call → nudge
        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={
                    "env_key": self.env_key,
                    "turn": self.turns,
                    "done_reason": "max_turns",
                },
            )

        remaining = self.max_turns - self.turns
        if self.max_turns == 1:
            nudge = "No <task> block found. Output your task in <task>...</task> format."
        elif remaining <= 2:
            nudge = (
                f"You have {remaining} turn(s) left. Output your <task> block NOW or you will "
                "get reward 0. Stop exploring and generate the task."
            )
        else:
            nudge = "Use <tool_call> to explore the database or call environment tools, then generate a <task> block."
        observation = {"role": "user", "content": nudge}
        return BaseTextEnvStepOutput(
            observations=[observation],
            reward=0.0,
            done=False,
            metadata={"env_key": self.env_key, "turn": self.turns},
        )

    async def _execute_meta_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute a query_db meta-tool call via the Fleet orchestrator."""
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if self.orch is None:
            return "Error: Fleet environment not provisioned. Generate a <task> directly."

        if name != "query_db":
            return f"Error: Unknown meta-tool '{name}'."

        sql = args.get("sql", "")
        if not sql:
            return "Error: query_db requires a 'sql' argument."

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await self.orch.query_db_async(sql=sql, db_name=args.get("db_name", "seed"))
                if isinstance(result, dict):
                    # Truncate rows to save context — model only needs a sample
                    if "rows" in result and isinstance(result["rows"], list) and len(result["rows"]) > 5:
                        result["rows"] = result["rows"][:5]
                        result["message"] = "Query returned more rows; showing first 5."
                    formatted = json.dumps(result, indent=2, default=str)
                    if len(formatted) > 3000:
                        formatted = formatted[:3000] + "\n... (truncated)"
                    return f"Tool result:\n{formatted}"
                return f"Tool result:\n{str(result)[:3000]}"
            except Exception as e:
                if attempt < max_retries - 1 and (
                    "closed" in str(e).lower() or "transport" in str(e).lower() or "connection" in str(e).lower()
                ):
                    await asyncio.sleep(1)
                    continue
                return f"Error: {e}"

    async def _execute_mcp_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute an MCP tool call via FleetMCPTools."""
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if self.mcp_tools is None:
            return "Error: MCP tools not available. Use query_db or generate a <task>."

        try:
            result = await self.mcp_tools.call_tool(name, args)
            if isinstance(result, dict):
                return f"Tool result:\n{json.dumps(result, indent=2, default=str)}"
            return f"Tool result:\n{result}"
        except Exception as e:
            return f"Error calling {name}: {e}"

    async def init_async(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the environment, optionally provisioning a Fleet env for DB exploration.

        When ``max_turns > 1``, provisions a Fleet environment via
        ``FleetEnvClient.from_fleet_async`` so the model can call
        ``query_db`` during exploration turns.
        Falls back to single-turn if provisioning fails.
        """
        self.turns = 0
        self.meta_tool_calls = 0
        self.mcp_tool_calls = 0
        self.called_query_db = False
        self.orch = None
        self.mcp_tools = None
        self.callable_tools = set(_META_TOOLS)

        # Provision Fleet env for multi-turn exploration (DB + MCP tools)
        if self.max_turns > 1 and self.fleet_api_key and self.data_key:
            try:
                from envs.fleet_env import FleetEnvClient

                self.orch, self.mcp_tools = await FleetEnvClient.from_fleet_async(
                    api_key=self.fleet_api_key,
                    env_key=self.env_key,
                    data_key=self.data_key,
                    data_version=self.data_version,
                    image_type="standard",
                    ttl_seconds=900,
                )
                # Load instance resources so db("seed") works
                # instance.load() is async — must await directly, not via to_thread
                await self.orch._fleet_env.instance.load()
                logger.info(f"TaskGenEnv [{self.env_key}]: Fleet env provisioned for DB + tool exploration")

                # Auto-populate env_schema from describe_db if not provided in dataset.
                # Compact format: "table: col1 (type), col2 (type), ..." — one line per table.
                if not self.env_schema:
                    try:
                        schema_result = await self.orch.describe_db_async(db_name="seed")
                        self.env_schema = _format_compact_schema(schema_result)
                        if self.env_schema:
                            logger.info(
                                f"TaskGenEnv [{self.env_key}]: Auto-populated env_schema ({len(self.env_schema)} chars)"
                            )
                    except Exception as e:
                        logger.warning(f"TaskGenEnv [{self.env_key}]: Failed to auto-populate env_schema: {e}")

                # Discover MCP tools so the model can call them
                if self.mcp_tools:
                    try:
                        tools_action = await self.mcp_tools.list_tools()
                        mcp_tools = [
                            t for t in tools_action.tools if "function" in t and t["function"].get("name") != "computer"
                        ]
                        mcp_tool_names = {t["function"]["name"] for t in mcp_tools}
                        self.callable_tools = set(_META_TOOLS) | mcp_tool_names
                        # Update tool schemas for system prompt if dataset didn't have them
                        if not self.env_tools_schema:
                            self.env_tools_schema = mcp_tools
                            self.env_tools = [t["function"]["name"] for t in mcp_tools]
                        logger.info(f"TaskGenEnv [{self.env_key}]: {len(mcp_tool_names)} MCP tools available")
                    except Exception as e:
                        logger.warning(f"TaskGenEnv [{self.env_key}]: Failed to list MCP tools: {e}")
            except Exception as e:
                logger.warning(
                    f"TaskGenEnv [{self.env_key}]: Fleet provisioning failed, " f"falling back to single-turn: {e}"
                )
                self.max_turns = 1

        system_prompt = self._build_system_prompt()

        user_content = (
            f"Generate a task for the {self.env_key} environment. "
            "First explore the database to understand the data, then draft a prompt and verifier. "
            "Before outputting, query the DB to verify your assumptions are correct — "
            "iterate on your draft until you're confident the data supports it."
            if self.max_turns > 1
            else f"Generate a task for the {self.env_key} environment."
        )

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        metadata = {
            "env_key": self.env_key,
            "env_version": self.env_version,
            "num_tools": len(self.env_tools),
            "multi_turn": self.max_turns > 1,
        }

        return conversation, metadata

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Sync wrapper for init_async."""
        return asyncio.run(self.init_async(prompt))

    def close(self):
        """Close the Fleet orchestrator if provisioned."""
        if self.orch is not None:
            try:
                self.orch.close()
            except Exception:
                pass
            self.orch = None

    async def close_async(self):
        """Async close — release Fleet orchestrator resources."""
        if self.orch is not None:
            try:
                await self.orch.close_async()
            except Exception:
                pass
            self.orch = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return per-episode metrics."""
        return {
            "env_key": self.env_key,
            "turns": self.turns,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes."""
        if not metrics:
            return {}

        # Group by env_key
        env_counts: Dict[str, int] = {}
        for m in metrics:
            env_key = m.get("env_key", "unknown")
            env_counts[env_key] = env_counts.get(env_key, 0) + 1

        result = {"total_episodes": len(metrics)}
        for env_key, count in env_counts.items():
            result[f"{env_key}/episodes"] = count

        return result
