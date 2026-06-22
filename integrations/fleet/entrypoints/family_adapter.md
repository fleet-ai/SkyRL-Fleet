# Per-family adapter

The Tinker trainer entrypoint ([`main_fleet_tinker.py`](main_fleet_tinker.py))
runs an agentic GRPO loop against Fleet envs. Each model family (Kimi-K2.6,
Qwen3, …) has its own chat-template expectations for how assistant
messages should be structured. A per-family adapter
([`families.py`](../../../skyrl-gym/skyrl_gym/envs/fleet_task/families.py))
owns that delta; the env is family-agnostic.

## Why

The Kimi-K2.6 chat template injects an empty `<think></think>` when the
assistant message has no `reasoning_content` field. If we stuff the
model's raw emission (which contains its own `<think>...</think>`) into
`content`, the next-turn prompt renders TWO `<think>` blocks per
assistant turn. The model imitates → naked tool-call emissions + format-
debugging cascades. A/B-verified via `tokenizer.apply_chat_template`:

| Shape | `<think>` count per asst turn |
|---|---|
| raw string in `content` (pre-fix) | 3 |
| `reasoning_content` + structured `tool_calls` (post-fix) | 2 |

Qwen's chat template parses inline `<think>` from `content` directly, so
its adapter is passthrough. New families register one class.

## What's family-specific

| Concern | Kimi-K2.6 | Qwen3 |
|---|---|---|
| Chat template `<think>` rendering | Injects empty default when `reasoning_content` absent | Extracts inline from `content`; passthrough is fine |
| Tool call grammar | `<\|tool_call_begin\|>` / `<\|tool_call_argument_begin\|>` single-token specials | `<tool_call>{json}</tool_call>` text |
| Tool call id format | `functions.NAME:N` (pretraining prior) | `call_N` |
| Per-turn reminder | Canonical-format echo (lands the 5 special-token IDs in context) | Indicator only |

## Code trace — one turn

```
1. Build next-turn prompt from chat_history
   main_fleet_tinker.py:794    build_model_input_chunks(tokenizer, env.chat_history, tools=...)

2. Sample from Tinker
   main_fleet_tinker.py:856    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                                # 5 Kimi tool tokens (special: False) survive as readable text;
                                # <|im_end|> etc. stripped. <think>...</think> preserved.

3. Pass raw output to env
   main_fleet_tinker.py:869    env.step_async(output_text)
   │
   └─► env.py:883    self.turns += 1
       env.py:891    tool_call = parse_tool_call(action)        # {"name": ..., "arguments": ...}
                                                                 # (coord-scaling + done-wrap handled in-between)
       env.py:913    family = get_family(extras["model_family"])
       env.py:915    assistant_msg = family.build_assistant_message(action, tool_call, self.turns)
       env.py:927    self.chat_history.append(assistant_msg)

       (MCP executes tool_call → tool_result; observation built)

       env.py:1091   scaffold = family.per_turn_reminder(self.turns, self.max_turns)
       env.py:1139   obs_content = family.reject_message()      # only when parse failed

4. Loop back to step 1.
```

## What the family adapter returns

`family.build_assistant_message(action, tool_call, turn)` for **Kimi**:

```python
{
  "role": "assistant",
  "content": "",                                    # text outside <think> + tool-call section
  "reasoning_content": "I should click the menu.",  # joined <think> bodies
  "tool_calls": [{
    "id": "functions.computer:5",                   # canonical Kimi id
    "type": "function",
    "function": {"name": "computer",
                 "arguments": '{"action":"left_click","coordinate":[60,28]}'},
  }],
}
```

Renders as ONE clean `<think>` block + canonical tool-call section in
the next-turn prompt. The 5 Kimi tool special-token IDs land in
context with the canonical id (`functions.computer:5`) the model was
pretrained on.

For **Qwen** the adapter returns `{"role":"assistant", "content": action,
"tool_calls": [...]}` — passthrough; Qwen's template parses inline
`<think>` itself.

For an unknown family (no `model_family` in `extras` or family not in
`_REGISTRY`), env.py falls back to the raw-content shape it had before
the adapter shipped. Byte-identical to pre-PR behavior; preserves any
caller that doesn't set `model_family`.

## Adding a new family

1. Define a class in [`families.py`](../../../skyrl-gym/skyrl_gym/envs/fleet_task/families.py)
   implementing the `Family` protocol (`name`, `canonical_tool_call`,
   `build_assistant_message`, `per_turn_reminder`, `reject_message`).
2. Register it in `_REGISTRY` (families.py:204).
3. Add a prefix branch to `family_for_model` (families.py:219).
4. Add tests in `skyrl-gym/tests/test_fleet_task_families.py` covering
   the Kimi-style round-trip through `tokenizer.apply_chat_template`.

## Anchors

| File | What it owns |
|---|---|
| [`families.py`](../../../skyrl-gym/skyrl_gym/envs/fleet_task/families.py) | `Family` protocol + `Kimi` / `Qwen` adapters + `_REGISTRY` + `get_family` + `family_for_model` |
| [`env.py`](../../../skyrl-gym/skyrl_gym/envs/fleet_task/env.py) | Family-agnostic loop. Three call sites: `915` build_assistant_message, `1091` per_turn_reminder, `1139` reject_message |
| [`main_fleet_tinker.py`](main_fleet_tinker.py) | Tinker entrypoint. Sets `extras["model_family"]` from `family_for_model(tokenizer.name_or_path)`. Per-phase trace-job rotation. |
| [`skyrl_gym_generator.py`](../../../skyrl/train/generators/skyrl_gym_generator.py) | SkyRL entrypoint. `_inject_fleet_model_family` plumbs the same family name into `env_extras` from `self.model_name`. |
