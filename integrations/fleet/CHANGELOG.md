# Fleet Integration Changelog

## 2026-07-20: NeMo ATOF enabled by default

Scope: shared ATOF setup, native and negotiation SkyRL launch scripts, and
Tinker launch scripts.

Both native SkyRL and Tinker now install and start the NeMo ATOF exporter
across all task configs. Launchers always set `SKYRL_ATOF_ENABLED=1`.
Exporter setup and event sends remain fail-open, so an ATOF failure does not
stop training.

## 2026-07-17: Tinker GRPO — optional MUST-conjunction (binary) training rewards + request-level judge model

Scope: `main_fleet_tinker.py`, `fleet-tinker-tool-use-run.sh`.

Also in this change: `--judge-model <model>` (env: `JUDGE_MODEL`) forces
every verifier's judge call onto a specific model via a client-side rewrite
of the tasks file (the trainer executes verifier code it ships, so no task
cloning or server-side default change is needed). The injection point is the
openclaw verifier template's `base_kwargs = dict(submission=raw,` line,
validated across 90 re-grades in the Sonnet-vs-Opus judge ablation; smoked
against all 177 lev177 verifiers (177/177 injected). Applies to training
and held-out evals alike so both are graded by the same judge.

Five clean flat runs (two models, LRs 5e-7..2.5e-5, holdout evals unmoved)
plus the judge self-agreement measurement (Opus regrade disagreement
mean |d| = 0.092 per transcript) point at the reward signal: partial-credit
rewards inject per-criterion judge noise directly into GRPO advantages,
and within-task reward variance is dominated by it.

`--binary-reward` (wired as `BINARY_REWARD=1` through the run script)
thresholds the verifier's partial-credit reward at 1.0 before advantage
computation. Because the returned reward is the MUST-criteria fraction
(NICE criteria are logged only), this is exactly "all MUST criteria
passed" conjunction scoring — the signal only flips when the judge flips
an entire conjunction, not any single criterion. The raw partial-credit
mean stays logged (`reward/raw_partial_mean`) and held-out evals are
never shaped, so cross-run comparisons stay in the same units. Default
off; no behavior change unless the flag is set.

## 2026-07-15: Tinker GRPO — trainer-side sequence-length guard

Scope: `main_fleet_tinker.py` only.

Tinker's sampling and training endpoints can enforce different max sequence
lengths for the same model: Qwen3.6-35B-A3B samples at 131072 but rejects
training-side requests (`forward`, `forward_backward`) above 65536. A config
sized for the sampling limit therefore passes rollout collection and then
kills the whole run the first time an episode exceeds the trainer cap
(observed: lev177 run dead at step 8/60 on a 77192-token episode, inside
`recompute_behavior_logprobs`'s forward probe — `forward_backward` would hit
the same wall).

Key constraint (verified empirically against live Tinker): a rejected
request POISONS its TrainingClient — Tinker refuses all further operations
on it ("create a new TrainingClient from a checkpoint if you wish to
continue"). So catch-and-retry on the real client is impossible; the cap
must be resolved before the training client ever sees an oversized batch.

Fix, two layers:

1. `--max-train-sequence-length` (optional): when set, batches are truncated
   to `min(max_sequence_length, max_train_sequence_length)` in
   `prepare_training_data`, so overlong episodes are truncated instead of
   fatal.
2. Startup probe (`discover_trainer_seqlen_cap`): when the flag is unset, a
   single max_sequence_length-token `forward` runs on a DISPOSABLE LoRA
   client before training. Pass -> no cap; rejection -> cap parsed from the
   error text and applied as the pre-truncation bound for every step. The
   poisoned probe client is simply discarded. Logged to wandb as
   `seqlen_guard/trainer_cap`.

If a mid-run rejection still somehow occurs, the step logs an actionable
error (restart with `--max-train-sequence-length` + `--load-state`) before
re-raising — the client is poisoned at that point and no in-place recovery
exists.

## 2026-07-13: Tinker GRPO — task-keyed advantage groups + temperature-consistent importance ratios

Scope: `main_fleet_tinker.py` only.

Two correctness fixes to the hosted-Tinker training loop:

**1. Advantage groups keyed by task_key (was: positional slices).**
`compute_advantages_grpo` sliced the valid-rollout list into contiguous
chunks of `n_samples_per_prompt`. Collection order is task-major, but the
valid-rollout filter (error / empty response / reward=None) removes entries
before grouping — so one invalid rollout shifts every later group boundary
and "group mean" becomes a mean over rollouts of DIFFERENT tasks. With
verifier-execution failures excluded by design (23% of zeros in the regrade
census), this fired routinely. New `compute_advantages_grpo_by_task` keys
groups by the task that produced each rollout: invariant to filtering and
ordering; normalization semantics unchanged (per-task centering, then one
global normalize).

**2. Behavior logprobs recomputed under the raw model when temperature != 1.**
Sampling at T draws from q = softmax(logits/T) and Tinker records log q;
the hosted ppo/importance-sampling loss scores its numerator under raw p.
The ratio therefore starts at p/q != 1 before any update, with deviation
growing ~(1/T - 1)*|logp| per token (T=0.9: a p=1% token carries ratio
~1.7) — positive-advantage rare tokens get clipped, negative-advantage rare
tokens over-penalized, a systematic anti-rare-token bias from step 0.
`recompute_behavior_logprobs` runs one extra `forward(loss_fn=
"cross_entropy")` under the pre-update weights and splices the raw-p
logprobs into `loss_fn_inputs["logprobs"]` at loss positions, so the
first-step ratio is exactly 1 and clipping responds only to genuine drift.
Gated on `temperature != 1.0` (recorded and recomputed logprobs already
match to ~2e-3 at T=1); opt out with `--no-fix-behavior-logprobs`. Emits
`logprob_fix/{mean,max}_abs_delta` + `time/logprob_recompute` to wandb.

## 2026-07-07: ATOF phase 2 — hint-synthesis events (generator-side)

Scope: `atof_events.py` (additive method), **core file edit**
(`skyrl_gym_generator.py`, preserve on upstream merges).

Hint synthesis is the one non-policy LLM callsite in training (litellm ->
OpenRouter in `hint_synthesizer.synthesize_hint`, audited 2026-07-06). Each
LLM-synthesized hint now emits one standalone `helper_llm_call` mini-trace
(`call_site="hint_synthesis"`, hint model as model_name, instance_id +
global_step/phase metadata). Fallback hints (static / llm_failed) made no
LLM call and emit nothing.

All shaping lives in the shared module (`AtofEmitter.hint_synthesis_events`
looping requests x results); the core generator file carries exactly one
`_atof_emit` call after `synthesize_hints_batch`, keeping the
preserve-on-upstream-merge surface minimal. skyrl-gym untouched: request
records the hint inputs the generator holds (task_prompt sliced to the same
5000 chars synthesize_hint sends, verifier feedback; chat_history excluded —
it already ships with the rollout). Tradeoff accepted: no token usage / raw
OpenRouter response metadata; if per-call cost visibility is ever needed,
thread an on_llm_call callback through hint_synthesizer instead. Fail-open
at both layers, payloads size-guarded like rollout events.

## 2026-07-07: Tinker run script — install nemo-relay wheel when ATOF is enabled

Scope: `fleet-tinker-tool-use-run.sh`.

The hosted trainer image (fleet-research-api Dockerfile) never runs
`fleet-common-setup.sh`, so `SKYRL_ATOF_ENABLED=1` on the hosted path hit
init_atof's fail-open "nemo_relay wheel not installed" warning and every
run silently produced zero trace events. The run script now mirrors the
setup script: when the flag is set and `nemo_relay` isn't importable, pull
the wheel from `s3://fleet-nemo-relay-artifacts/wheels/latest/` and pip
install it at job start (the Batch job has AWS creds; the image build does
not). Install failure warns and continues — fail-open, matching init_atof.

## 2026-07-07: ATOF — production MSK broker/tenant defaults in code

Scope: `atof_events.py`, `fleet-common-run.sh`.

`SKYRL_ATOF_ENABLED=1` is now the only knob on every path: brokers and
tenant_id default in `_component_config` (`DEFAULT_MSK_BROKERS`,
`DEFAULT_TENANT_ID`), joining the existing bucket/topic/client_id defaults.
The `THESEUS_ATOF_*` env vars remain as overrides; setting one explicitly
empty still disables with a warning. The now-redundant gated exports in
`fleet-common-run.sh` are removed (single source of truth for the broker
list), keeping only the `export SKYRL_ATOF_ENABLED` so the flag reliably
reaches Ray workers. This also removes the tinker-path failure mode where
the flag was set but the run produced zero events because the broker vars
weren't hand-exported.

## 2026-07-07: Tinker harness — ATOF rollout observability (item 3)

Scope: **Tinker harness only** (`main_fleet_tinker.py`).

### What

`collect_fleet_rollout()` emits the same one-trace-per-rollout event contract
as the SkyRL generator hooks (#95): trace open after env init, an LLM event
per turn (watermarked new messages + response + stop reason), a tool event
per env step, and a final mark carrying total reward, turn count, and the
tinker-specific `tool_calls`/`tool_errors` counters. `init_atof`/`drain_atof`
live in `main()` (plain asyncio, no Ray — unlike the SkyRL entrypoints where
init must happen inside the Ray task).

Metadata mapping: `run_name` = the resolved `wandb_name`,
`global_step` = train step / eval `step_index`, `phase` reuses the exact
`_rotate_trace_job` labels (`train_step_N`, `eval_step_N`, `eval_pre`,
`eval_final`, `eval_only`) — so the Tinker path gets the pre/final phase
refinement the SkyRL path lists as a known gap. `sample_idx` = the existing
per-task sample index.

### Behavior guarantees

- All hooks go through a module-level `_atof_emit()` that swallows
  exceptions: collect_single_rollout's catch-all would otherwise convert an
  observability bug into a zero-reward RolloutOutput.
- `atof_emitter` defaults to None on both collect functions; unset env =
  today's behavior exactly.
- `_force_verifier`'s internal `<done>` step is not instrumented; its reward
  and forced stop_reason land in the final mark.
- A rollout that raises mid-loop ships its events but gets no final mark
  (same accepted limitation as the SkyRL side).

## 2026-07-07: Launch scripts — ATOF enablement via `SKYRL_ATOF_ENABLED=1`

Scope: **launch scripts only** (`fleet-common-setup.sh`, `fleet-common-run.sh`,
the five fleet-training task YAMLs).

### What

Enabling ATOF on any fleet-training run is now one flag:
`--env SKYRL_ATOF_ENABLED=1` (declared as `"0"` in the openenv-fleet YAMLs).

- Setup: downloads the nemo-relay wheel from
  `s3://fleet-nemo-relay-artifacts/wheels/latest/` into a `mktemp -d` dir
  (shared multi-tenant `/tmp`; a reused dir could glob-match a stale wheel)
  and installs it into the venv. Fails loudly under `set -e` — a broken
  install must not degrade into a silently event-less run.
- Run: exports `THESEUS_ATOF_MSK_BROKERS` / `THESEUS_ATOF_TENANT_ID`
  defaults (`${VAR:-default}`, explicit launch values win) before
  `ray start` on every rank — the emitter initializes inside Ray tasks,
  which inherit env from the raylet. Bucket/topic/client_id default in
  `atof_events.py`; region comes from `AWS_REGION` (code default `us-east-1`).
- Unset flag = exactly the previous behavior; preflight needs no new vars.

### Verified before shipping

- skyrl-ci can read the wheels bucket (`AmazonS3FullAccess` identity policy;
  bucket policy has no Deny) and `wheels/latest/` holds exactly one wheel.
- SkyPilot merges CLI `--env` keys without requiring YAML declaration
  (`Task.update_envs`), so undeclared YAMLs (CI, task-gen) can still enable
  ATOF ad hoc.

## 2026-07-06: SkyRL generator — ATOF rollout observability hooks

Scope: **core file edit** (`skyrl/train/generators/skyrl_gym_generator.py`), not in
upstream SkyRL. Preserve on upstream merges.

### What

`agent_loop()` emits one ATOF trace per rollout through
`integrations/fleet/atof_events.py` (NeMo Relay -> MSK `atof.received` ->
ClickHouse): trace open at rollout start, an LLM event per turn (the turn's new
input messages + response + stop reason), a tool event per env step
(action/observation/reward/done), and a final mark (summed reward, turn count).
`batch_metadata` is now threaded into `agent_loop()` (and
`_run_hint_augmentation`) so events carry global_step/phase.

### Behavior guarantees

- `self.atof_emitter` defaults to None and only the Fleet entrypoints install
  one; without it every hook is a no-op and the loop is unchanged.
- All emitter calls go through `_atof_emit()`, which swallows exceptions: an
  observability bug must never surface as a zero-reward trajectory via
  generate()'s catch-all (that would silently corrupt training signal).
- The batched single-turn path (`generate_batched`) is not instrumented; all
  fleet training paths use `agent_loop`.

## 2026-07-05: Tinker harness — resume from saved state (`--load-state` + `--start-step`)

Scope: **Tinker harness only** (`main_fleet_tinker.py`, `fleet-tinker-tool-use-run.sh`).

### Problem

Cloud Run job executions hard-cap at 24h (GCP limit). Run `986205b6` (Qwen3.6-35B-A3B,
2×538, ~40h projected) was killed mid-step-39 with healthy metrics; the fleet-research-api
kept reporting `running` and its /logs went empty (the API doesn't reconcile dead
executions — W&B heartbeat is the only trustworthy liveness signal). `save_state_every=5`
had preserved `weights/state_000040`, but nothing could consume it.

### Fix

- `--load-state <tinker://.../weights/state_N>`: resume via
  `create_training_client_from_state_with_optimizer_async` (the plain `from_state` variant
  drops Adam moments, silently resetting the effective step-size schedule).
- `--start-step N`: the per-epoch shuffle is seeded (`seed + epoch`), so the interrupted
  epoch's dataloader is rebuilt and its consumed batches discarded — the resumed run replays
  the exact planned batch order (every task keeps its visit count); rollouts themselves are
  re-sampled fresh (on-policy).
- Run script maps `LOAD_STATE` / `START_STEP` env vars to the flags (fleet-research-api
  forwards them from new TrainRequest fields).

Rule of thumb: any /train run projected >20h should either lower scope or plan on a resume
leg. Resume legs must reuse the SAME dataset_id and hyperparams or the seeded schedule
diverges.

## 2026-07-03: Tinker harness — ENV_STEP_TIMEOUT_S configurable, default 300

Scope: **Tinker harness only** (`main_fleet_tinker.py`).

### Problem

`ENV_STEP_TIMEOUT_S=120` (added 2026-06-15, fix #1 below) interacts fatally with OpenEnv's
`call_tool` retry loop: one tool call may spend up to `max_retries(8) × OPERATION_TIMEOUT_S(60)`
plus backoff (~8.5 min) before failing, and one env step executes *every* tool call in the
assistant turn. A single slow or dead tool therefore guaranteed blowing the 120s step budget,
which kills the whole rollout (`stop_reason=env_step_timeout`, force-verified at partial state).
Measured on openclaw_verifiers_v5: **61% of eval rollouts** (jobs `367bbd49`/`ba3dc295`, 442
tasks × 3 × 2 arms, mean reward 0.03 vs 0.69 for natural stops) and **32% of training rollouts**
(run `202a95bc`, mean reward 0.01) died this way. For GRPO that is randomly-punishing advantage
noise on a third of every batch.

### Fix (two-sided, pairs with OpenEnv #20)

- OpenEnv #20 bounds `call_tool` total retry time to `FLEET_CALL_TOOL_DEADLINE_S` (default 90s),
  surfacing exhaustion as a normal `tool_call_failed` observation so the rollout continues.
- Here: `ENV_STEP_TIMEOUT_S = int(os.getenv("FLEET_ENV_STEP_TIMEOUT_S", "300"))` — the step
  budget must fit several deadline-length calls in one turn; 300s ≈ 3 worst-case calls. Env
  var override for experimentation without a code change.

## 2026-06-15: Tinker harness — fleet-research-api runs, timeouts, eval parity, MCP content shape

Scope: **Tinker harness only** (`integrations/fleet/entrypoints/main_fleet_tinker.py`). SkyRL harness (`skyrl/train/trainer.py` + `skyrl_train.generators.skyrl_gym_generator`) is a separate code path on local GPUs via SkyPilot; none of these fixes touch it.

Several patches surfaced while driving Tinker training jobs through the new `fleet-research-api` HTTP service.

### Datasets exercised this session

| Dataset | Team | Modality | Model | Outcome |
|---------|------|----------|-------|---------|
| `multi-env-internal-v4` (`7d619260`) | fleet-research | tool_use | Kimi-K2.6 (128K) | First run hung at step 1 rollouts (sample_async deadlock on shared HTTP/2 stream, 4h silence, no APIConnectionError). Killed. Second run with per-await timeouts completed step 0 (85 trainable seqs after 43 timeouts), then step 1 training died on Tinker 402 billing. |
| `bi-dashboard-passk-0.2-0.6` (`c43b744a`) | Macrohard | tool_use | Qwen3.5-9B, then Kimi-K2.6 | Both runs: 0 verifier passes across all rollouts because rollouts crashed before reaching `done=True`. Root cause #4 below (MCP content shape). |

### Problems

1. **`sample_async` deadlock under concurrent rollouts.** 32 concurrent rollouts share one `SamplingClient` and likely one HTTP/2 connection. When that connection went half-open (peer-dropped without RST), all 32 in-flight streams stalled together. Tinker's built-in 120-min `progress_timeout` is global across the client and never fired (kept alive by some hidden activity); even if it had, `_APIFuture.result_async`'s cancel chain reduces to `concurrent.futures.Future.cancel()` which is a no-op on RUNNING futures.
2. **Periodic eval fired at step 0.** With `eval_every=20`, `step % eval_every == 0` was True at step 0, triggering a full 70-task eval (~2h on Kimi) on a model with only one optim_step — indistinguishable from baseline, total waste.
3. **Two separate eval blocks** (periodic inside loop + always-on final after loop) drifted from upstream SkyRL's single-expression pattern, made the logic harder to reason about, and left `last_sampling_client` plumbing only used by the second block.
4. **MCP `content` shape assumption.** `step_output["observations"][0]["content"]` was assumed to always be a string. MCP spec defines it as a list of typed content parts. Some Fleet envs (bi-dashboard `query_data_lake`, `execute_python`) pass the list through unwrapped; `tokenizer.encode(list, ...)` raises `Input must be a string, list of strings, or list of ints, got: <class 'list'>` and the rollout dies with `stop_reason=error` before the verifier ever runs.
5. **Tinker 402 billing failures are silent and expensive.** First evidence is a 60-min SDK pause then `APIStatusError 402` propagating up. The fleet-research-api correctly catches the subprocess exit, but the user wastes time + tokens waiting through wandb init + spawn just to learn the credit card needs topping up.

### Root causes and fixes

#### 1. Per-await `asyncio.wait_for` in `collect_fleet_rollout` (commits `9f16eb16`, `82487161`)

**Where:** every external await in the rollout — `_env_init`, `sampling_client.sample_async`, `_env_step`, `_env_close`.

**Fix:** Wrap each with `asyncio.wait_for` and explicit timeouts (`TINKER_SAMPLE_TIMEOUT_S=600`, `ENV_INIT_TIMEOUT_S=90`, `ENV_STEP_TIMEOUT_S=120`, `ENV_CLOSE_TIMEOUT_S=30`). On Tinker timeout, set `stop_reason="tinker_timeout"`, break the turn loop, return partial trajectory. On env timeout, similar with `env_step_timeout` / `env_init_timeout`. The orphaned httpx pool slot leaks per timeout but is bounded (max_connections=100 vs max_concurrent=32 gives headroom).

**TODO upstream:** Tinker SDK should expose a per-request timeout knob; the 120-min `progress_timeout` is global and starvation-prone under concurrent rollouts. Until then this wrap is mandatory for any multi-concurrent client.

#### 2. Skip periodic eval at step 0 (commit `dec4c6c5`, superseded by `7e6ad97e`)

**Where:** `main_fleet_tinker.py` training loop.

**Fix:** Add `step > 0` to the periodic-eval condition. For an actual untrained baseline, use `--eval-before-train` (separate, explicit, runs at step_index=-1).

#### 3. Merge periodic + final eval into one expression, upstream parity (commit `7e6ad97e`)

**Where:** training loop, previously two separate blocks.

**Fix:** Single expression matching `skyrl/train/trainer.py:374`:
```python
is_final_step = step == max_steps - 1
is_periodic = eval_every > 0 and step > 0 and step % eval_every == 0
if eval_dataset and (is_periodic or is_final_step):
    if is_final_step:
        final_sampling_path = training_client.save_weights_for_sampler(name="step_final").result().path
        eval_client = service_client.create_sampling_client(model_path=final_sampling_path)
        await _run_eval(eval_client, step_index=max_steps)
    else:
        await _run_eval(sampling_client, step_index=step)
```
Final-step eval still always runs and still uses the durable `step_final` checkpoint, so the auto-train launcher still gets `post_pass_rate`. Removed unused `last_sampling_client` tracking.

#### 4. Flatten MCP list-shaped observation content (this commit)

**Where:** `collect_fleet_rollout` observation tokenization, line ~591.

**Fix:** Detect `isinstance(obs_content, list)` and join (handling string parts and `{"text": ...}` dict parts) before passing to `tokenizer.encode`. Strictly additive: envs where `content` is already a string are unaffected. SkyRL harness (`skyrl_gym_generator.SkyrlGymGenerator`) likely has the same latent bug if pointed at the same envs — needs a parallel fix there.

#### 5. (Not yet shipped) Billing preflight in fleet-research-api

**Suggested:** Before spawning the training subprocess, fleet-research-api should do a cheap `service_client.create_lora_training_client_async()` probe with a 30s timeout. On `APIStatusError 402`, mark the job `failed` / `stage=billing_blocked` immediately so the user learns about a missing credit card in seconds instead of after wandb init + subprocess spawn.

## 2026-03-29: Multi-node 35B training parity with old SkyRL fork

Fixes for 2-node (16 GPU) Qwen3.5-35B GRPO training on GCP H200. Ported from fleet-ai/SkyRL PR #328 and PR #333, plus new fixes for SkyRL-v2-specific issues.

### Problems

2-node training crashed with:
1. `cudaErrorIllegalAddress` during FSDP ref model offload/backload (multi-node race)
2. OOM / Xid 31 FAULT_PDE during policy training forward+backward (missing chunked lm_head)
3. OOM / Xid 31 at 97K sequence length — SDPA too memory-hungry, flash_attn triggers GatedDeltaNet crash
4. `AssertionError: data batch size must be divisible by mini_batch_size, got 160 and 128` (hint augmentation)

### Root causes and fixes

#### 1. Synchronous ref offload + barrier (`fsdp_worker.py`)

**Where:** `FSDPRefWorkerBase.offload_to_cpu()` and `backload_to_gpu()`

**Problem:** With colocated models, the trainer cycles: ref on GPU → ref offload to CPU → policy on GPU. With `non_blocking=True`, the CPU←GPU transfer is *queued* but returns immediately. On a single node, CUDA stream ordering serializes this naturally. Across nodes, there's no shared CUDA context — node 0's policy worker can start touching GPU memory while node 1's ref worker is still mid-transfer. Result: `cudaErrorIllegalAddress`.

**Fix:** `non_blocking=False` (wait for transfer) + `torch.distributed.barrier()` (all ranks synchronize). Guarantees every GPU finishes offloading before any policy worker starts backloading.

**Why the old fork doesn't need this:** Designed for single-node where all workers share the same CUDA context and stream ordering prevents races.

#### 2. Port chunked lm_head forward (`model_wrapper.py`, `fsdp_worker.py`)

**Where:** `HFModelWrapper.forward()` and `HFModelWrapper._chunked_lm_head_forward()`

**Problem:** SkyRL-v2's `HFModelWrapper` was missing `loss_chunk_size` support entirely — the parameter existed in config but was never passed through `fsdp_worker.py` to the model wrapper. Without it, the model materializes the full `(B, S, 131072)` logits tensor during forward pass (~10 GB for 97K-length sequences on Qwen3.5-35B with vocab_size=131072). This consumed so much GPU memory that the subsequent training forward pass (with gradients enabled) hit OOM or Xid 31 FAULT_PDE when FSDP tried to unshard parameters.

**Fix:** Ported the chunked lm_head implementation from the old fork:
- Added `loss_chunk_size` parameter to `HFModelWrapper.__init__`
- Pass `loss_chunk_size` from `fsdp_worker.py` for both policy and ref model init
- During forward, replace `lm_head` with an identity module so the model returns hidden states `(B, S, 8192)` instead of logits `(B, S, 131072)` — 16x smaller
- Compute logits in chunks of 4096 tokens with gradient checkpointing, never materializing full logits

**Why the old fork doesn't have this problem:** It already has `loss_chunk_size` support and passes it correctly.

#### 3. `empty_cache` before backward (`worker.py`)

**Where:** `PolicyWorkerBase._forward_backward_micro()` (both SFT and RL paths) and `CriticWorkerBase._forward_backward_micro()`

**Problem:** After the forward pass, freed intermediate tensors stay in PyTorch's CUDA cache as scattered blocks. The backward pass needs large contiguous allocations for gradients. On the 35B model with tight GPU memory margins, the fragmented cache can't satisfy these allocations → OOM, even though total free memory is sufficient.

**Fix:** `torch.cuda.empty_cache()` before `strategy.backward()`. Returns cached blocks to CUDA which coalesces them into contiguous allocations. This is especially important because `expandable_segments:True` cannot be used (see fix #4).

**Why the old fork doesn't need this:** Targets smaller models (8B) with enough GPU headroom that fragmentation doesn't matter.

#### 4. Reduce sequence length to 72K and disable `expandable_segments` (`fleet-35b-run.sh`)

**Where:** `fleet-35b-run.sh` — `MAX_INPUT_LENGTH` and `--no-pytorch-alloc-conf` flag.

**Problem:** At 97K sequences (96000 input + 4096 generate), memory was too tight even with chunked lm_head and `empty_cache`:
- `flash_attn=false` (SDPA): OOM requesting 5.95 GiB during backward — SDPA's O(n²) attention memory is too large at 97K.
- `flash_attn=true`: Xid 31 FAULT_PDE in GatedDeltaNet layers during ref model forward — reproduced at both 97K and 72K. Not a memory issue; vLLM 0.18.0's CuMemAllocator corrupts CUDA memory mappings that FSDP2 DTensor operations later touch.
- `expandable_segments:True` would help with fragmentation but conflicts with vLLM 0.18.0's `CuMemAllocator` (`cuMemCreate`/`cuMemMap`).

**Fix:** Reduce `MAX_INPUT_LENGTH` from 96000 to 72000 (total seq ~76K) and use `flash_attn=false` (SDPA). At 72K, SDPA's O(n²) memory is ~55% of what it was at 97K — enough to fit with chunked lm_head + `empty_cache`. The `--no-pytorch-alloc-conf` flag passed to `fleet-common-run.sh` skips the default `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, avoiding the vLLM 0.18.0 CuMemAllocator conflict. The 9B VL script (`fleet-vl-run.sh`) also passes this flag for the same reason.

**Verified working:** 10 steps completed on GCP spot 2×H200:8 (asia-south1-b) with zero GPU errors over 12 hours. Step timing: generation ~7 min, ref forward ~8 min, policy backward ~44 min, total step ~70 min avg. Checkpoint saved to S3 at step 10. SDPA is slower than flash_attn but stable. WandB: `fleet_qwen35_35b_tool_use_2c0e13b7` (run ID `f6kw15p2`).

#### 5. Dynamic mini_batch_size for hint augmentation (`dispatch.py`)

**Where:** `MeshDispatch.stage_chunks()`

**Problem:** `mini_batch_size` is computed as `policy_mini_batch_size * n_samples_per_prompt` (e.g., 16 × 8 = 128). But hint augmentation appends extra samples: 16 prompts × 2 hints = 32 additional, total batch = 160. The `stage_chunks` method asserted `160 % 128 == 0` → crash.

The old fork's manual loop (`num_mini_batches = len(data) // mini_batch_size`) silently dropped the 32 hint samples — no crash, but hint training was wasted.

**Fix:** When batch size isn't divisible by mini_batch_size, step down mini_batch_size (by `dp_size` increments to stay DP-divisible) until it divides evenly. For 160 samples with dp_size=16: adjusts from 128 → 80, giving 2 mini-batches of 80. All 160 samples (including hints) are trained on.

**Why upstream SkyRL doesn't have this:** Upstream uses a simple `for` loop with `//` division (no `stage_chunks` optimization). The `stage_chunks` pre-staging is a SkyRL-v2 optimization that added a strict assert the old code path never had.

### Files changed

| File | Change |
|------|--------|
| `skyrl/backends/skyrl_train/workers/model_wrapper.py` | Port chunked lm_head forward (loss_chunk_size) |
| `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py` | Pass loss_chunk_size to HFModelWrapper; synchronous ref offload + barrier |
| `skyrl/backends/skyrl_train/workers/worker.py` | empty_cache before backward (3 sites) |
| `scripts/fleet-35b-run.sh` | Reduce seq length to 72K, flash_attn=false, --no-pytorch-alloc-conf, wandb project rename |
| `skyrl/backends/skyrl_train/distributed/dispatch.py` | Dynamic mini_batch_size adjustment |
