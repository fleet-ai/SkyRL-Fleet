# RUN_ON_CLUSTER — collect sensory traces with Qwen3.5-9B (VL)

**You are a coding agent on a GPU box.** Your job: stand up the `falmart` env +
a vLLM-served `Qwen3.5-9B`, implement the two driver seams, and collect a few
hundred matched **sensory-on / sensory-off** trajectories as auto-labeled
training examples. The data-engine core is already written and tested — do **not**
reimplement the sense attribution or the outcome taxonomy.

## Background (read first)

- Concept + design: `sensory-sft/README.md`.
- The env's sense-log: `~/theseus-falmart/server/src/sense/README.md` and
  `SCHEMA_EXPLAINED.md`. Labels come from `GET /api/sense/log?since=<cursor>`
  (returns `{records, next, dropped, text}`); attribution is a cursor diff.
- Outcome taxonomy (the label): `new-write`, `new-read-or-route`, `ok-false`
  (incl. 501 stubs), `empty-delta`. `empty-delta` and `ok-false` are RARE and the
  most valuable — do not let them get crowded out.
- The two arms use the **same Qwen3.5-9B weights** (it's natively multimodal).
  `sensory_off` = screenshot only (vision-only RL baseline). `sensory_on` =
  screenshot **+** the env's rendered sense `text`.

## Step 0 — sanity check the core (no GPU, ~5s)

```bash
cd sensory-sft
python3 tests/test_sense.py        # expect "9 tests passed."
```

## Step 1 — bring up falmart with the sense-log ON

```bash
# Toolchain: node 20, pnpm, git-lfs.
corepack enable && corepack prepare pnpm@latest --activate
git lfs install

git clone https://github.com/<org>/theseus-falmart ~/theseus-falmart  # or copy it
cd ~/theseus-falmart
git lfs pull            # fetches data/seed.sqlite (~267 MB; REQUIRED — no programmatic seed)
pnpm install
SENSE_LOG=true pnpm dev # client :5173, server :3001, mcp :3003
```

Verify the sense surface is live (it only exists when `SENSE_LOG=true`):

```bash
curl -s 'http://localhost:5173/api/sense/log?since=0'   # -> {"records":[...],"next":N,"dropped":0,"text":...}
```

If that 404s, the flag didn't take — restart the server with `SENSE_LOG=true`.

## Step 2 — serve Qwen3.5-9B (multimodal) with vLLM

```bash
pip install "vllm>=0.17"     # >=0.17 required for Qwen3.5 / GatedDeltaNet + FA4
# GDN JIT can hang on some hosts; this is the safe fallback (see fleet-negotiation-9b-run.sh):
export VLLM_GDN_PREFILL_BACKEND=triton
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B --port 8000 --trust-remote-code   # add --limit-mm-per-prompt image=1
```

Confirm vision works: POST a tiny `image_url` chat request to
`http://localhost:8000/v1/chat/completions` and check it returns. One H200
(TP=1) is enough for the 9B.

## Step 3 — implement the two seams in `sensory_sft/drivers.py`

Both are stubbed with the exact contract in their docstrings. Implement:

1. **`VLLMQwenPolicy.act`** — format system + history + current observation
   (text **and** the screenshot as an `image_url` base64 block), call the vLLM
   OpenAI endpoint, return the assistant text. The reply must end in a
   `<tool_call>{"name":"computer","arguments":{...}}</tool_call>` or `<done>`.
   Reuse the system prompt + tool schema and the `[0,1000]→pixel` coordinate
   convention from `skyrl-gym/skyrl_gym/envs/fleet_task/env.py`
   (`_adapt_computer_tool_for_qwen`, `_convert_qwen_coordinates`).

2. **`PlaywrightDriver`** — `reset()` loads `base_url` and screenshots;
   `execute(action)` parses the `computer` action, converts coords to pixels,
   performs it, **waits for `networkidle`**, and screenshots. The wait matters:
   `run_episode` reads the sense diff right after `execute` returns, so the
   action's server calls must have landed first.

```bash
pip install playwright && playwright install chromium
```

## Step 4 — collect

```bash
cd sensory-sft
python collect.py --arms both --n-episodes 100 --max-steps 20 \
  --falmart-url http://localhost:5173 \
  --vllm-url http://localhost:8000/v1 \
  --registry ~/theseus-falmart/server/src/sense/schema-registry.json \
  --out ./traces
```

`--n-episodes` is **per arm**, so `both × 100 = 200` trajectories →
`traces/falmart_sensory_on.jsonl` and `traces/falmart_sensory_off.jsonl`. Each
line is one `(context, action) → outcome` example.

## Guardrails (already on — don't remove)

`RolloutConfig` defaults: `max_steps=20`, stop after 4 consecutive `empty-delta`,
stop after 3 identical actions. These exist because falmart's homepage hero is a
920px promo grid whose carousel **auto-advances every 4s** — a vision-only agent
sees the screen "change" after a no-op click and doom-loops, burning tokens.
Keying the loop-break off the **sense delta** (not pixels) defeats that. Leave
the guardrails on; the rare `empty-delta`/`ok-false` they catch are kept in the
data, just not allowed to dominate a single episode.

## Done when

- [ ] `tests/test_sense.py` passes.
- [ ] `curl .../api/sense/log` returns records (sense is on).
- [ ] vLLM answers an image chat request.
- [ ] `traces/falmart_sensory_{on,off}.jsonl` each have ≥100 episodes' worth of lines.
- [ ] `collect.py`'s end-of-run distribution shows a non-trivial share of
      `ok-false` + `empty-delta` (if it's ~all `new-read-or-route`, loosen the
      prompts in `sensory_sft/prompts.py` or raise `max_steps` slightly).

## Gotchas

- Sense routes are dual-mounted (`/api/sense/*` and `/sense/*`); use `/api/...`.
- `dropped > 0` in a delta ⇒ the cursor fell behind; those examples are flagged
  `reliable=false` — drop them downstream.
- Run one episode end-to-end first and eyeball the JSONL before scaling to 100.
