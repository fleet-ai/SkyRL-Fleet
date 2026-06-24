# Frontier cross-play matrix — reproducibility

Exact prompts, protocol, and sampling for the **cross-play performance matrix**: the
trained policy (and the pre-RL base) negotiating against a pool of frontier models on
held-out DnD scenarios. This measures who-beats-whom, NOT the opponent value-inference
probe (see `VALUE_INFERENCE_REPRO.md` for that).

- Driver: `run_crossplay.py` (this directory)
- Outputs: `results/crossplay_matrix_{base,s30}_canask.json` (+ `_heatmap.png`)
- Model under test: `selfplay-canask-0621_s30` (`exports/selfplay-canask-0621_s30/global_step_30/policy`)
- Metric: seat-A normalized outcome (`you_norm`, no-deal = 0), plus agreement rate and
  joint efficiency, per ordered (opener, partner) cell.

Built to match the **training rollout distribution** of the `selfplay-canask-0621` run.
Intentional deviations: the opponents are frontier models (that's the whole point of
cross-play), and the training-time think-gate (forced `<think>` decoding) is OFF so
transfer of the learned behavior is a real signal.

---

## 1. Model pool & endpoints

Frontier pool (one per vendor) + the policy under test (added last, `--policy-model`):

| Label | Slug | Endpoint | Thinking |
|-------|------|----------|----------|
| GPT-5.5 | `openai/gpt-5.5` | OpenRouter | ON |
| Opus-4.8 | `anthropic/claude-opus-4.8` | OpenRouter | ON |
| Gemini-3.1-Pro | `google/gemini-3.1-pro-preview` | OpenRouter | ON |
| Llama-3.3-70B | `meta-llama/llama-3.3-70b-instruct` | OpenRouter | ON |
| Qwen3.5-9B | `qwen/qwen3.5-9b` | OpenRouter | OFF (hybrid → no-think) |
| **Base-qwen35-35b** (base run) | `qwen/qwen3.5-35b-a3b` | OpenRouter | ON |
| **SelfPlay-canask-s30** (policy run) | `qwen35-policy` | local vLLM `http://10.66.0.6:6479/v1` | ON |

`OPENROUTER_API_KEY` required; the s30 vLLM server must serve `qwen35-policy`.

## 2. Coverage — `--policy-only`

Only the policy's row + column are played (policy as opener vs each model, and each
model as opener vs the policy), skipping the frontier×frontier block (it's identical
across the base and s30 runs, so running it twice is wasted budget). With `--n 16`
that's 11 cells × 16 scenarios = **176 games per run**.

## 3. Dataset (held-out)

dataset `dnd`, split `val`, `--seed 1`, `--n 16` scenarios per cell. `dnd/val` is a
separate upstream FAIR file from `dnd/train` with provable zero key-overlap; training
takes gradients only on `train.parquet` (from `dnd/train`). Genuine holdout.

## 4. Protocol — `single` (single-proposer)

`--protocol single`, matching training `NEGOTIATION_PROTOCOL="single"`. Seat A opens;
a turn offers via `<propose>{...}</propose>` (what the proposer keeps), the other side
finalizes with a line containing `<accept>`. No accept within the message limit → both
score 0. (The legacy default of this script was `dual`; single is now required for
training parity.)

## 5. Elicitation — `can_ask` on BOTH seats

`--elicit can_ask` → `_elicit_blocks` returns `(CAN_ASK_BLOCK, CAN_ASK_BLOCK)`, injected
into seat A (`you_block`) and seat B (`them_block`) for **every** pairing, matching
training (`NEGOTIATION_ELICIT=can_ask`, both seats). Verbatim block and the
`SYSTEM_TEMPLATE_SINGLE` body are reproduced in `VALUE_INFERENCE_REPRO.md` §4–5
(same `prompts.py`, same post-2026-06-21 proposal-semantics fix). Opener uses
`OPENING_USER_MSG` ("You speak first…").

## 6. Sampling — policy seat matched to training (`--match-train-sampling`)

Per-model sampling (`_build_sampling`). The **policy model only** (base/s30, identified
as the last-added model) gets the full training rollout sampling; every other model
stays neutral. All seats get the protocol's action-tag stop sequences. `<think>` is
parsed for the action tag but stripped before re-entering context (matches training's
`qwen3_without_thinking` template).

| Param | Policy (base / s30) | Other models | Training value |
|-------|---------------------|--------------|----------------|
| temperature | 1.0 | 1.0 | 1.0 |
| max_tokens / turn | 8192 | 8192 | 8192 (`MAX_GENERATE_LENGTH`) |
| presence_penalty | 1.5 | — (none) | 1.5 |
| stop tags | `</propose>`,`</deal>`,`<accept>` | same | same |
| thinking | ON | per-model (Qwen3.5-9B OFF) | ON |
| max_turns | 6 | 6 | 6 |

`presence_penalty=1.5` is a property of the policy's own decoding, so it is applied to
the policy seat only; frontier opponents play naturally on penalty. The chosen per-model
sampling is recorded under `config.sampling` in each output JSON.

**`include_stop_str_in_output`**: the local vLLM checkpoint drops the matched stop string
from the response by default, which would strip the action tag (`</propose>`, `<accept>`)
and make every deal fail to parse/close. We set `include_stop_str_in_output: true` for the
locally-served seat (OpenRouter already returns the tag), matching the SkyRL training
default (`skyrl_train/inference_engines/utils.py`). Without it the policy's `<accept>` is
silently truncated and agreement collapses to ~0 — an eval artifact, not model behavior.

## 7. Exact commands

One-liners via the row-only config script (`crossplay_rowonly.sh`):

```bash
OPENROUTER_API_KEY=...  ./crossplay_rowonly.sh base
OPENROUTER_API_KEY=...  S30_BASE_URL=http://10.66.0.6:6479/v1  ./crossplay_rowonly.sh s30
```

Equivalent explicit invocations:

```bash
export OPENROUTER_API_KEY=...
PY=/workspace/allie/skyrl-neg-wt/.venv/bin/python
cd skyrl-neg-wt/skyrl-gym/skyrl_gym/envs/negotiation/eval

# base (pre-RL) row+column vs the frontier pool
$PY run_crossplay.py \
  --policy-model qwen/qwen3.5-35b-a3b --policy-label Base-qwen35-35b \
  --policy-base-url https://openrouter.ai/api/v1 --policy-only \
  --protocol single --elicit can_ask --match-train-sampling \
  --n 16 --max-turns 6 --seed 1 --concurrency 8 \
  --out-prefix crossplay_matrix_base_canask

# Self-Play s30 policy row+column vs the frontier pool
$PY run_crossplay.py \
  --policy-model qwen35-policy --policy-label SelfPlay-canask-s30 \
  --policy-base-url http://10.66.0.6:6479/v1 --policy-only \
  --protocol single --elicit can_ask --match-train-sampling \
  --n 16 --max-turns 6 --seed 1 --concurrency 8 \
  --out-prefix crossplay_matrix_s30_canask
```

Re-render a heatmap from an existing JSON without replaying:
`$PY run_crossplay.py --render-only --out-prefix crossplay_matrix_s30_canask`

## 8. Reading the output

- `seatA_outcome[i][j]` = mean normalized score for model *i* as opener vs model *j* as
  partner (no-deal = 0). Row mean = how the opener does vs the field (higher = stronger);
  column mean = how much it concedes as partner.
- `agreement[i][j]`, `joint_efficiency[i][j]` = deal rate and achieved/best joint score.
- Under `--policy-only`, frontier×frontier cells are `null` (only the policy row/col are filled).

## 9. Intentional deviations from training (not bugs)

- **Opponents = frontier models** (cross-play) rather than the self-play policy.
- **Think-gate OFF**: training forced ≥16 `<think>` tokens via constrained decoding
  (`NEGOTIATION_THINK_GATE=1`); the matrix is unconstrained so behavior transfer is a real signal.
