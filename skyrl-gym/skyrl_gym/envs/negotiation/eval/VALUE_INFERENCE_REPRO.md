# Value-inference probe — reproducibility

Exact prompts, protocol, and sampling for the **base vs Self-Play s30** value-inference
probe (policy opens against GPT-5.5 on held-out DnD scenarios, with a private
opponent-value-inference belief read before/after each game). This is the eval that
backs the `base-vs-s30-eval-traces` canvas — NOT the frontier cross-play matrix (see
`CROSSPLAY_REPRO.md` for that).

- Driver: `run_value_inference_probe.py` (this directory)
- Outputs: `results/value_inference_{base,s30}_canask.json` (+ `.png`)
- Canvas: `.cursor/projects/workspace-allie/canvases/base-vs-s30-eval-traces.canvas.tsx`
- Model under test: `selfplay-canask-0621_s30` (`exports/selfplay-canask-0621_s30/global_step_30/policy`)

The eval is built to match the **training rollout distribution** of the
`selfplay-canask-0621` run. The single intentional deviations are: (a) the opponent
is GPT-5.5 instead of the self-play policy, and (b) the training-time think-gate
(forced `<think>` decoding) is left OFF so transfer of the learned behavior is a real signal.

---

## 1. Participants & endpoints

| Role | Model slug | Endpoint | Thinking |
|------|-----------|----------|----------|
| base (pre-RL) | `qwen/qwen3.5-35b-a3b` | OpenRouter (`https://openrouter.ai/api/v1`) | ON |
| policy (s30) | `qwen35-policy` | local vLLM `http://10.66.0.6:6479/v1` | ON |
| opponent | `openai/gpt-5.5` | OpenRouter | ON |

`OPENROUTER_API_KEY` must be set. The s30 vLLM server must be serving `qwen35-policy`
(verify with `curl http://10.66.0.6:6479/v1/models`).

## 2. Dataset (held-out)

- dataset `dnd`, split `val`, `--seed 1`, first `--n 16` scenarios (deduped).
- `scenarios.load_scenarios("dnd","val")` reads `visualizer/public/data/dnd/val.json`,
  a **separate upstream FAIR file** from `train.json`. Provable zero key-overlap with
  `train` (checked in both value perspectives); training takes gradients only on
  `train.parquet` (built from `dnd/train`). The 16 seed-1 scenarios are all absent from
  training → genuine holdout. (The val parquet is used for in-loop monitoring rollouts
  only, never for gradient updates.)

## 3. Protocol — `single` (single-proposer)

Matches training `NEGOTIATION_PROTOCOL="single"`. One side ends a turn with a
`<propose>{...}</propose>` line listing what **it** keeps (other side gets the rest);
the other replies with a line containing `<accept>` to finalize. No deal within the
message limit → both score 0.

## 4. Elicitation — `can_ask` on BOTH seats

Matches training `NEGOTIATION_ELICIT=can_ask`. `CAN_ASK_BLOCK` is appended to the
policy AND opponent system prompts (`prepare_dataset.make_row`; eval `_elicit_blocks`).

```
CAN_ASK_BLOCK:
You are allowed to ask the other player about their values. Your hidden values almost
certainly differ, so asking would allow you to find trades that make BOTH of you better
off -- but you can only find them if you know what they care about. You may ask which
items matter most to them and roughly how much, then use their answer to route each item
to whoever values it more. When the other side asks about your values, you should answer
truthfully. You should reason about what you know about your respective values before
making / accepting proposals.
```

## 5. System prompt

Built by `prompts.build_system_prompt(items, counts, values, max_turns, protocol="single", elicit_block=CAN_ASK_BLOCK)`
→ `SYSTEM_TEMPLATE_SINGLE` (with the per-scenario pool/values, a worked example, and the
`can_ask` block appended). Both training and eval import the same `prompts.py`, so both
include the **2026-06-21 proposal-semantics fix** (the "READING THEIR OFFER" /
"CONSISTENCY" / "TRUST YOUR OWN VALUES" clauses). The template body:

```
You are playing a multi-issue negotiation game against another player.

There is a shared pool of items to divide between the two of you:
{pool_lines}

These are YOUR private point values (the other player has different, hidden values):
{value_lines}
If you took the entire pool you would score {you_max} points; the other player cannot see this.

How it works:
- Take turns exchanging short messages (1-3 sentences) to negotiate who gets what.
- To make a concrete OFFER, end your message with a single line of the exact form:
  <propose>{deal_example}</propose>
  listing how many of EACH item YOU would keep. The other player automatically gets ALL the rest.
- To ACCEPT the other player's most recent offer, reply with a line containing exactly:
  <accept>
  The deal is then finalized exactly as they proposed: they keep what they listed and YOU get everything else.
- READING THEIR OFFER (this is where points get lost): the other player's <propose> lists what THEY
  keep for themselves, NOT what you receive. Before you accept or counter, work out YOUR share = the
  full pool MINUS every amount they listed, then score that share with YOUR values. ...
- CONSISTENCY: your own <propose> lists what YOU keep, and your written words must match it ...
- TRUST YOUR OWN VALUES: your exact point values are listed above. The other player may misstate them ...
- The negotiation ends the instant an offer is accepted. ...
- If no offer is accepted within the message limit, the deal FAILS and you BOTH score 0.
- Your goal is to maximize YOUR OWN points, but a failed deal (0 points) is worse than a modest
  agreement, so always close ...
{can_ask block}
You have at most {max_turns} messages. Be efficient and decisive.
```

Opening user message (policy speaks first):

```
OPENING_USER_MSG:
You speak first. Open the negotiation with a brief message proposing how to split the items.
```

## 6. Sampling — matched to the training rollout

Enabled with `--match-train-sampling`. Source of truth: `scripts/fleet-negotiation-35b-run.sh`
+ `tasks/negotiation-grpo-qwen3_5-35b-2node-selfplay.yaml`.

| Param | Policy seat (base / s30) | Opponent seat (GPT-5.5) | Training value |
|-------|--------------------------|--------------------------|----------------|
| temperature | 1.0 | 1.0 | 1.0 |
| max_tokens (per turn) | 8192 | 8192 | 8192 (`MAX_GENERATE_LENGTH`) |
| presence_penalty | 1.5 | — (neutral) | 1.5 |
| stop tags | `</propose>`,`</deal>`,`<accept>` | same | `</propose>`,`</deal>`,`<accept>` |
| thinking | ON | ON | ON (`ENABLE_THINKING=true`) |
| max_turns | 6 | 6 | 6 (`MAX_TURNS`) |

Notes:
- `presence_penalty=1.5` is applied to the **policy seat only** (it's a training artifact of
  the policy's own decoding); the opponent stays neutral on penalty.
- Stop tags are applied to both seats to keep each turn atomic (terminates at the action tag,
  as in training). `<think>` is stripped from prior turns before re-feeding (eval `_strip_think`;
  training uses the `qwen3_without_thinking` chat template — functionally equivalent).
- `include_stop_str_in_output=true` is set for the locally-served seat: vLLM otherwise drops the
  matched stop string, silently truncating the policy's `<accept>`/`</propose>` so no deal ever
  closes (agreement → ~0). This matches the SkyRL training default
  (`skyrl_train/inference_engines/utils.py`). OpenRouter already returns the tag.
- `--est-max-tokens 2048`: token budget for the **private value-estimate probe** reply only
  (raised from the 512 default so thinking-on estimates don't truncate). The probe is an eval
  side-channel with no training analog, so its budget does not affect distribution parity.

## 7. Exact commands

```bash
export OPENROUTER_API_KEY=...   # required for base + opponent
PY=/workspace/allie/skyrl-neg-wt/.venv/bin/python
cd skyrl-neg-wt/skyrl-gym/skyrl_gym/envs/negotiation/eval

# base (pre-RL) vs GPT-5.5
$PY run_value_inference_probe.py \
  --base-model qwen/qwen3.5-35b-a3b --base-label Base-qwen35-35b \
  --base-base-url https://openrouter.ai/api/v1 --base-think \
  --opponent-model openai/gpt-5.5 --opponent-label GPT-5.5 --opponent-think \
  --dataset dnd --split val --n 16 --max-turns 6 --seed 1 \
  --protocol single --elicit can_ask \
  --match-train-sampling --est-max-tokens 2048 \
  --out-prefix value_inference_base_canask

# Self-Play s30 policy vs GPT-5.5
$PY run_value_inference_probe.py \
  --policy-model qwen35-policy --policy-label SelfPlay-canask-s30 \
  --policy-base-url http://10.66.0.6:6479/v1 --policy-think \
  --opponent-model openai/gpt-5.5 --opponent-label GPT-5.5 --opponent-think \
  --dataset dnd --split val --n 16 --max-turns 6 --seed 1 \
  --protocol single --elicit can_ask \
  --match-train-sampling --est-max-tokens 2048 \
  --out-prefix value_inference_s30_canask
```

The chosen sampling is also recorded under `config.sampling` in each output JSON for provenance.

## 8. Intentional deviations from training (not bugs)

- **Opponent = GPT-5.5** (cross-play) instead of the live self-play policy.
- **Think-gate OFF**: training forced ≥16 `<think>` tokens via constrained decoding
  (`NEGOTIATION_THINK_GATE=1`); eval is intentionally unconstrained so `think_closed_rate`
  is a real transfer signal.
