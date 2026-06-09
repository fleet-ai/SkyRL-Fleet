# Experiment Plan: Judge-Guided Branching for Credit Assignment

> Companion to `research-plan-judge-guided-branching.md`. That doc is the research framing; this is the execution plan — what to run, in what order, on what data, with what success criteria.

---

## Datasets

| dataset | env_key | tasks | why |
|---|---|---|---|
| `tool_use/booking` | `booking` | 1,000 | Primary. DB-diff verifier, natural decision structure (search → evaluate → book), fast iteration. |
| `tool_use/reddit` | `reddit` | 943 | Secondary. Different task structure (read/write social content), good for generalization check. |

All tasks are in `~/Work/data/fleet/v7/openenv/all_tool_use.json`. Filter by `env_key`.

---

## Phase 0 — Get Multi-Rollout Traces (prerequisite for everything)

**Goal**: For each task in booking + reddit, collect ≥4 rollouts from the current policy (or a strong base model). Store as JSONL in `~/Work/data/fleet/traces/`.

**Format** (one JSON per line):
```json
{
  "task_key": "task_abc123",
  "env_key": "booking",
  "chat_history": [...],
  "reward": 0.75
}
```

**Steps**:

1. Pull existing rollouts from Supabase (`sessions` table, filter by `env_key IN ('booking', 'reddit')`). Script: `integrations/fleet/entrypoints/pull_supabase_rollouts.py` (to be written; see note below on network access).
2. If Supabase pull is insufficient (< 2 rollouts/task on average), top up with live rollouts using the Fleet API against the OpenEnv tasks.
3. Target: **≥ 200 tasks with ≥ 2 rollouts each** per env before proceeding. That's the minimum to get meaningful divergence scores.

**Output files**:
- `data/fleet/traces/booking_tool_use_claude-opus-4.6_rollouts.jsonl`
- `data/fleet/traces/reddit_tool_use_claude-opus-4.6_rollouts.jsonl`


> **Note**: Supabase is not on the Cowork network allowlist. Run the pull script locally:
> `python -m integrations.fleet.entrypoints.pull_supabase_rollouts --env booking reddit --output ~/Work/data/fleet/traces/`

---

## Phase 1 — Calibration Baseline (Exp 1a)

**Goal**: Measure how well the divergence judge and direct LLM judge predict reward variance. This validates the judge before using it to guide branching.

**Run** (executed against the Supabase-pulled `claude-opus-4.6` rollouts; GPT judge served via OpenRouter, model `openai/gpt-4o-mini`, key `OPENROUTER_API_KEY`):
```bash
# Divergence judge (no API key needed)
python -m integrations.fleet.entrypoints.analyze_traces \
  --traces ~/Work/data/fleet/traces/booking_tool_use_claude-opus-4.6_rollouts.jsonl \
  --method divergence \
  --output ~/Work/data/fleet/analysis/booking_judge_results.json

# Both judges (GPT via OpenRouter)
python -m integrations.fleet.entrypoints.analyze_traces \
  --traces ~/Work/data/fleet/traces/booking_tool_use_claude-opus-4.6_rollouts.jsonl \
  --method both \
  --model openai/gpt-4o-mini \
  --base-url https://openrouter.ai/api/v1 \
  --max-tasks 100 \
  --output ~/Work/data/fleet/analysis/booking_judge_results_both.json \
  --csv ~/Work/data/fleet/analysis/booking_calibration_table.csv

# Repeat for reddit
python -m integrations.fleet.entrypoints.analyze_traces \
  --traces ~/Work/data/fleet/traces/reddit_tool_use_claude-opus-4.6_rollouts.jsonl \
  --method both \
  --model openai/gpt-4o-mini \
  --base-url https://openrouter.ai/api/v1 \
  --max-tasks 100 \
  --output ~/Work/data/fleet/analysis/reddit_judge_results_both.json \
  --csv ~/Work/data/fleet/analysis/reddit_calibration_table.csv
```

> **Note**: The pulled traces use the native OpenAI tool-calling schema (`tool_calls` arrays + `role: "tool"` observations) and key on `base_task_key`, not the `<tool_call>`-in-content / `task_key` format the original plan assumed. `trace_judge.parse_steps` and `analyze_traces.py` were updated to handle both formats. A turn issuing parallel tool calls is collapsed to a single step (names joined with `+`, args kept as an ordered list).

**Success criteria**:
- Spearman correlation between max divergence score and reward variance: **≥ 0.3** on booking. (Smoke test gives ~0.5 on synthetic data; real data will be noisier.)
- If correlation < 0.2, the divergence judge is not informative enough. Investigate whether tool_name_weight vs. tool_args_weight tuning helps before proceeding.

**Key output**: per-task table of `(task_key, max_div_score, max_llm_score, reward_variance, n_rollouts)`, saved to `~/Work/data/fleet/analysis/calibration_table.csv` (combined) plus per-env `booking_calibration_table.csv` / `reddit_calibration_table.csv`.

### Results (run 2026-06-05, claude-opus-4.6 traces)

| env | judge | tasks | Spearman(max_score, reward_var) | verdict |
|---|---|---|---|---|
| booking | divergence | 437 | **0.130** | below 0.2 abort line |
| booking | GPT `gpt-4o-mini` | 100 | **0.329** | meets ≥0.3 |
| reddit | divergence | 743 | **0.135** | below 0.2 abort line |
| reddit | GPT `gpt-4o-mini` | 100 | **-0.242** | anti-correlated |

**Takeaways**:
- The **divergence judge is not informative** on real data (ρ≈0.13 on both envs, below the 0.2 abort threshold). Mean reward variance is very low (~0.035): many task groups are *consistently* solved or *consistently* failed, so cross-rollout tool-call divergence is high even when the outcome never changes.
- The **GPT judge clears the bar on booking (0.33) but is negative on reddit (-0.24)** — it does not generalize across task structure. Booking has a clean search→evaluate→book decision spine that the LLM reasons about well; reddit's read/write social actions don't map onto the judge's "branching point" framing.
- **Decision**: do not proceed to Phase 2 on the strength of the divergence judge alone. Before Phase 2, either (a) tune `tool_name_weight`/`tool_args_weight` and restrict calibration to high-reward-variance tasks (the only ones where the metric is meaningful), or (b) use the GPT judge but only on booking, and revisit the reddit prompt. The cross-env instability is the main risk to flag.

### Results — math sample (run 2026-06-05, DAPO aime-2024, qwen2.5-72b)

Sanity check of the same calibration on a **non-agentic math** domain to see whether the
judges generalize beyond tool-use. Math completions are single-turn, so they were first
**chunked into reasoning steps** (paragraph boundaries → ~11 steps/rollout median) and
scored with a **text-content divergence** variant (token-set Jaccard distance across
rollouts at each step index) since there are no tool calls to diff. New code:
`trace_judge.chunk_math_completion` / `parse_math_steps` / `math_divergence_judge`, and an
`analyze_traces.py --format dapo-math` path. Sample: 30 AIME-2024 problems × 8 rollouts
(binary reward ±1; **only 7/30 problems have mixed rewards** — the rest are consistently
right or wrong).

```bash
python -m integrations.fleet.entrypoints.analyze_traces \
  --traces ~/Work/data/dapo/aime-2024_traces_qwen2.5-72b.jsonl \
  --format dapo-math --method both \
  --model openai/gpt-4o-mini --base-url https://openrouter.ai/api/v1 --max-tasks 30 \
  --output ~/Work/data/fleet/analysis/dapo_aime2024_judge_results_both.json \
  --csv ~/Work/data/fleet/analysis/dapo_aime2024_calibration_table.csv
```

| env | judge | tasks | Spearman(max_score, reward_var) | verdict |
|---|---|---|---|---|
| aime-2024 | text-divergence (token-Jaccard) | 30 | **0.342** | meets ≥0.3 |
| aime-2024 | GPT `gpt-4o-mini` | 30 | **0.433** | meets ≥0.3 |

**Takeaways**:
- Both judges clear the 0.3 bar on math — notably the **divergence judge does *better* here
  (0.34) than on booking/reddit (0.13)**. Chunking + text divergence surfaces real signal:
  the two highest-variance problems (`problem_0`, `problem_25`) are also the highest-scored.
- **Strong caveat**: with only 7/30 problems having reward variance > 0, the Spearman is
  computed over 23 zero-variance ties, so it is noisy/optimistic. And the **`max_score`
  metric saturates** (~0.8–1.0 for nearly every problem) because deep-tail steps where only
  2 rollouts remain are almost always fully divergent (Jaccard≈1). `max_score` therefore has
  little spread; a mean-divergence over well-supported steps (≥half the rollouts present)
  would discriminate better. To trust this result, re-run on a larger, more
  variance-balanced math sample and switch the calibration statistic off `max`.
- Outputs: `dapo_aime2024_judge_results_both.json`, `dapo_aime2024_calibration_table.csv`.

### Results — DAPO-17k sample (run 2026-06-06, dapo-math-17k 1k-sample, qwen2.5-72b)

Re-ran on a **much larger, variance-balanced** math sample to settle the AIME caveats:
996 problems × 4 rollouts (3,969 traces), **389/996 (~39%) with mixed rewards** vs. 7/30
on AIME. This is a real calibration set, not 23 zero-variance ties.

```bash
python -m integrations.fleet.entrypoints.analyze_traces \
  --traces ~/Work/data/dapo/dapo-17k-sample1000_traces_qwen2.5-72b.jsonl \
  --format dapo-math --method both \
  --model openai/gpt-4o-mini --base-url https://openrouter.ai/api/v1 --max-tasks 200 \
  --output ~/Work/data/fleet/analysis/dapo_17k_sample1000_judge_results_both.json \
  --csv ~/Work/data/fleet/analysis/dapo_17k_sample1000_calibration_table.csv
```

| env | judge | tasks | Spearman(score, reward_var) | verdict |
|---|---|---|---|---|
| dapo-17k | text-divergence — max over steps | 993 | **0.012** | dead |
| dapo-17k | text-divergence — mean over all steps | 993 | **-0.000** | dead |
| dapo-17k | text-divergence — mean over well-supported steps (≥½ rollouts) | 993 | **-0.000** | dead |
| dapo-17k | text-divergence — divergence at step 0 / early-mean | 993 | **-0.01 / -0.06** | dead |
| dapo-17k | GPT `gpt-4o-mini` | 200 | _(running)_ | TBD |

**Takeaways**:
- **The text-divergence judge is dead for math.** On a properly variance-balanced set,
  *no* aggregation statistic correlates with reward variance — max, mean, mean-over-
  well-supported-steps, step-0, and early-mean are all ρ≈0 (|ρ| ≤ 0.02). The AIME 0.34 was
  an artifact of 7 variance points + `max_score` saturation, exactly as flagged.
- **Why it fails**: in math, rollouts diverge *textually everywhere* (different wording /
  ordering of the same reasoning) regardless of outcome, so token-Jaccard divergence is
  uniformly high (~0.8) whether a problem is consistently solved, consistently failed, or
  mixed. There is no relationship between "where the text diverges" and "whether the outcome
  varies." This is a stronger negative than booking/reddit (ρ≈0.13) — text divergence is the
  wrong signal for free-form reasoning. A semantic/answer-level divergence (e.g. divergence
  of intermediate numeric results, not surface tokens) would be needed to recover signal.
- The GPT judge result on this set is the real test of whether *any* cheap judge predicts
  math credit assignment; pending.
- Outputs: `dapo_17k_sample1000_judge_results.json` (divergence, all 996),
  `dapo_17k_sample1000_calibration_table.csv`.

---

## Phase 2 — Compute Ground-Truth Branch Values (prerequisite for Exp 1b, 3, 4)

**Goal**: For a subset of tasks (start with 50 booking tasks with high reward variance), compute MC value estimates at each step by rolling out K=8 continuations from each step.
(you can use the same claude model with anthropic API)

This is the most compute-expensive phase. It produces the ground-truth `V(s)` labels needed to measure judge quality and compare training strategies.

**Implementation** (`integrations/fleet/entrypoints/branch_rollouts.py`, to be written):
1. Load anchor traces for selected tasks.
2. For each step `s` in each anchor trace:
   - If `judge_score(s) > threshold` (use top-30% by divergence score to limit compute), sample K=8 continuations from `s` to end.
   - Record terminal rewards.
3. `V(s) = mean(rewards)`, `Var(s) = var(rewards)`.
4. Save per-step value estimates alongside judge scores.

**Compute estimate**: 50 tasks × ~8 steps/task × 8 rollouts = ~3,200 rollouts. At 30s/rollout: ~26 GPU-hours. Run on a single 8-GPU node via the Fleet API.

**Output**: `~/Work/data/fleet/analysis/branch_values_booking_n50.jsonl`

---

## Phase 3 — Judge Criterion Ablation (Exp 1b)

**Goal**: With ground-truth V(s) from Phase 2, measure whether judge scores predict per-step value variance better than uniform step coverage.

**Compare**:

| criterion | implementation |
|---|---|
| Divergence (cross-rollout) | `divergence_judge()` in `trace_judge.py` |
| Direct LLM (gpt-4o-mini) | `direct_judge()` in `trace_judge.py` |
| Uniform (every step) | baseline — no judge |
| Oracle | top steps by `Var(V(s))` from Phase 2 ground truth |

**Metric**: rank correlation between judge score and ground-truth `Var(V(s))` per step.

**Decision gate**: if divergence judge ≥ 0.7 × oracle correlation, it's good enough to use without an LLM call (cheaper, no API dependency). If not, use direct judge for Phase 4.

---

## Phase 4 — Training Signal Comparison (Exp 3)

**Goal**: Use branch-point value estimates to improve RL training signal. Compare training strategies on a small run.

**Setup**: 200 booking tasks, 500 training steps per strategy, Qwen3.5-9B. Track reward improvement per rollout generated (efficiency metric).

**Strategies**:

1. **Baseline**: standard GRPO with terminal reward only.
2. **Advantage correction**: use MC value estimates from branch points to reweight per-step advantages. No extra training pairs.
3. **Step-level DPO**: at each branch point with variance > threshold, form (best_rollout, worst_rollout) contrastive pairs. Train with DPO from that prefix.
4. **Judge-gated advantage correction**: only correct advantages at steps where `judge_score > 0.5`. Tests whether judge filtering improves over uniform correction.

**Implementation path**: strategies 1–2 modify advantage computation in `fsdp_worker.py`. Strategy 3 requires a DPO loss path in the SkyRL trainer.

**Compute estimate**: 4 strategies × 500 steps × ~16 rollouts/step ≈ 32k rollouts total. Use `fleet-9b-run.sh` config.

---

## Phase 5 — Scale Check

If Phase 4 shows any strategy outperforming GRPO baseline by > 10% in reward-per-rollout, scale to:
- Full booking (1,000 tasks) + reddit (943 tasks)
- Qwen3.5-35B via `fleet-35b-run.sh`
- Compare against Math-Shepherd and OmegaPRM baselines

---

## Metrics Tracked in All Phases

| metric | description |
|---|---|
| `spearman_score_vs_var` | Spearman ρ between judge score and reward variance (calibration) |
| `rollouts_to_X_reward` | Rollouts needed to reach threshold reward (efficiency) |
| `branch_efficiency` | Fraction of rollout budget spent on high-variance steps |
| `value_mse` | MSE of MC value estimates vs. large-K ground truth |
| `pass_at_1` | Final task success rate |

Log all training experiments to W&B under project `skyrl-agu-vf`. Tag runs by `phase` + `strategy`.

---

## File Map

```
integrations/fleet/
├── trace_judge.py                       # judge implementations (done)
├── entrypoints/
│   ├── analyze_traces.py                # Phase 1 CLI (done)
│   ├── pull_supabase_rollouts.py        # Phase 0 (TODO — run locally)
│   └── branch_rollouts.py              # Phase 2 (TODO)
└── experiment.md                        # this file

~/Work/data/fleet/
├── v7/openenv/all_tool_use.json         # task definitions
├── traces/
│   ├── booking_rollouts.jsonl           # Phase 0 output
│   └── reddit_rollouts.jsonl
└── analysis/
    ├── booking_judge_results.json       # Phase 1 output
    ├── reddit_judge_results.json
    ├── calibration_table.csv
    └── branch_values_booking_n50.jsonl  # Phase 2 output
```

---

## Immediate Next Steps

1. **Write `pull_supabase_rollouts.py`** and run it locally. Supabase schema: `sessions` table has `env_key`, `metadata.verifier_score`, and a join to get `chat_history`. See `integrations/fleet/auto_train/discovery.py` for existing Supabase client pattern.
2. **Run Phase 1** once traces are available — < 30 min with gpt-4o-mini on 200 tasks.
3. **Check calibration** — if Spearman ≥ 0.3, proceed to Phase 2. If not, tune `tool_name_weight` / `tool_args_weight` in `divergence_judge()` and re-run.
