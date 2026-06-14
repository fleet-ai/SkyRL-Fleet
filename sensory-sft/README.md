# sensory-sft — trace collection (data engine)

Collects auto-labeled `(context, action) → outcome` traces from the `falmart`
env's sense-log, for the Sensory Predictive-Feedback SFT work trial. Produces
**matched sensory-on / sensory-off** trajectories with `Qwen3.5-9B`.

## The idea in one paragraph

`falmart` taps every server interaction and, via `GET /api/sense/log?since=<n>`,
hands back the records a click caused **plus** an env-rendered observation
`text`. We attribute each action by cursor diff (snapshot `next` → act → read
`since=next`) and reduce the records to one of four canonical outcomes —
`new-write`, `new-read-or-route`, `ok-false` (incl. 501 stubs), `empty-delta`
(pure UI / dead button) — using the env's own enforced vocabulary
(`schema-registry.json`). That label is **free ground truth**: no human, no judge.

## sensory on vs off (the two arms)

Both arms read the log and store the same label. The only difference is whether
the env's rendered `text` is injected into what the agent sees:

| arm           | agent observation            | label recorded |
|---------------|------------------------------|----------------|
| `sensory_on`  | screenshot **+ sense text**  | yes            |
| `sensory_off` | screenshot only (vision-only)| yes            |

`sensory_off` is the vision-only RL baseline; `sensory_on` is vision + the
predictive-feedback signal. `Qwen3.5-9B` is natively multimodal, so **the same
weights** drive both — the comparison is purely about the observation.

## What's built (route-agnostic core — runs with no env, no deps)

- `sensory_sft/registry.py` — loads `schema-registry.json`; annotates each
  rpc (effect/entity/fn/flags) and route (pageType, via template matching),
  with a verb-rule fallback + `unknown` flag for vocabulary misses.
- `sensory_sft/sense.py` — `SenseClient` (stdlib cursor loop) + `classify_delta`
  → `Outcome`. Surfaces `reliable=False` when the cursor dropped records.
- `sensory_sft/rollout.py` — `run_episode`: guarded loop with the on/off switch,
  `(context, action) → outcome` example construction, and `JsonlWriter`.
- `sensory_sft/prompts.py` — loose, exploration-friendly task prompts.
- `tests/test_sense.py` — offline taxonomy test (9 cases). `python3 tests/test_sense.py`.

## Doom-loop guardrails (why they're here)

The homepage hero (`PromoSection.svelte`) is a 920px banner grid whose center
carousel **auto-advances every 4s**, and several banners link nowhere. A
vision-only agent sees the screen "change" after a no-op click and re-clicks
forever — wasting tokens. Guardrails in `RolloutConfig` stop this, keyed off the
**sense delta** (not pixels, which the carousel defeats):

- `max_steps` (default 20, not 50/80),
- `loop_break_consecutive_empty` — stop after N `empty-delta` actions in a row,
- `loop_break_repeat_action` — stop after N identical actions in a row.

(Note: `empty-delta` / `ok-false` are the rare, high-value labels the trial says
to oversample — so we keep them, we just don't let one repeated click dominate.)

## What's NOT wired yet (next step, route-dependent)

Inject a `Policy` and a `Driver` into `run_episode`:

- **Policy** — `Qwen3.5-9B`. tool_use → OpenRouter (`qwen/qwen3.5-9b`, text).
  VL/computer_use → vLLM-served multimodal (needs a GPU; OpenRouter likely won't
  expose vision for it).
- **Driver** — local: Playwright against `pnpm dev` (with `SENSE_LOG=true`);
  or a Fleet-hosted computer-use instance (provides screenshots).

## Local env prerequisites (not yet satisfied on this machine)

`pnpm` and `git-lfs` aren't installed, and `data/seed.sqlite` (267 MB, git-LFS,
externally provided) is missing. To run falmart locally:

```bash
# install pnpm + git-lfs (corepack/brew), then in theseus-falmart/:
git lfs pull               # fetch data/seed.sqlite
pnpm install
SENSE_LOG=true pnpm dev    # client :5173, server :3001, MCP :3003
open http://localhost:5173/api/sense/debug   # live sense viewer
```
