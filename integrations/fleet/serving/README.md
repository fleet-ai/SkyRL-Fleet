# fleet-tinker-shim

OpenAI-compatible HTTP shim in front of a Tinker LoRA checkpoint.

Tinker is SDK-only — no public HTTP endpoint. Most downstream tools
(fleet-env-bench, BFCL, tau2, OpenRouter clients, anything calling
`/v1/chat/completions`) want an OAI-compatible URL. The shim is the
impedance match: one cheap CPU VM running FastAPI + uvicorn, forwarding
each chat completion to `tinker.SamplingClient.sample_async`. Inference
happens on Tinker's cloud and is billed per token; the shim itself just
routes HTTP.

## Endpoints

| Method | Path | What |
|---|---|---|
| `GET`  | `/health` | `{"ok": true, "model": "tinker://..."}` once the sampler + tokenizer are bound |
| `GET`  | `/v1/models` | one served model id (the checkpoint path) |
| `POST` | `/v1/chat/completions` | non-stream OpenAI chat completion, optional `tools[]` |

No streaming (`stream: true` returns 400). No auth.

## Provisioning

```bash
cd ~/repos/skyrl-fleet
sky launch integrations/fleet/serving/tinker_shim.yaml -y \
    -c tinker-shim \
    --env FLEET_TINKER_CHECKPOINT="tinker://<run-id>:train:0/sampler_weights/step_final" \
    --env FLEET_TINKER_BASE_MODEL="moonshotai/Kimi-K2.6:peft:131072" \
    --env TINKER_API_KEY="$TINKER_API_KEY" \
    -d

SHIM_IP=$(sky status --ip tinker-shim | tail -1)
until curl -sf http://$SHIM_IP:8000/health > /dev/null; do sleep 5; done
```

Cold start (provision + uv pip install + tokenizer load + sampler bind)
takes ~5-7 min on the on-demand 4 vCPU GCP VM.

## Boot env

| Var | Required | Default | What |
|---|---|---|---|
| `FLEET_TINKER_CHECKPOINT` | yes | — | `tinker://<run-id>:train:0/sampler_weights/step_*` |
| `FLEET_TINKER_BASE_MODEL` | yes | — | HF model id of the base, e.g. `moonshotai/Kimi-K2.6:peft:131072` |
| `TINKER_API_KEY` | yes | — | Tinker SDK auth |
| `TINKER_SHIM_HOST` | no | `0.0.0.0` | uvicorn bind host |
| `TINKER_SHIM_PORT` | no | `8000` | uvicorn bind port |

## Switching checkpoints without re-provisioning

The cluster persists across `sky exec` calls. Re-point at a different
checkpoint without re-doing the 7-min cold start:

```bash
sky cancel tinker-shim --all -y
sky exec tinker-shim integrations/fleet/serving/tinker_shim.yaml \
    --env FLEET_TINKER_CHECKPOINT="$NEW_CKPT" \
    --env FLEET_TINKER_BASE_MODEL="moonshotai/Kimi-K2.6:peft:131072" \
    --env TINKER_API_KEY="$TINKER_API_KEY" -d
```

If `sky exec` reports "cluster does not exist", the VM was preempted or
torn down. Re-launch with `sky launch` (the YAML now defaults to
on-demand, not spot, to avoid preemption mid-bench).

## Supported base models

Anything Tinker supports — query via SDK:

```python
import tinker
print(tinker.ServiceClient().get_server_capabilities().supported_models)
```

Tinker currently supports (as of writing):
moonshotai Kimi-K2.5 / K2.6 (+ `:peft:131072`),
nvidia Nemotron-3 Nano-30B / Super-120B / Ultra-550B (+ `:peft:262144`),
Qwen3 / Qwen3.5 / Qwen3.6 family,
openai gpt-oss-20b / 120b (+ `:peft:131072`),
deepseek DeepSeek-V3.1,
meta-llama Llama-3.2-3B.

The shim itself is base-agnostic: the tokenizer is loaded via
`AutoTokenizer.from_pretrained(<base>, trust_remote_code=True)` at boot.
Kimi-K2.6 specifically needs `tiktoken + blobfile` which the YAML installs
explicitly.

## Tool-call format support

The shim parses the model's output and re-emits structured `tool_calls[]`
in the OpenAI message shape so OAI clients work transparently. Two
delimiter formats are recognized:

**Kimi-K2** (observed in Kimi-K2.6 sampling output):

```
<|tool_calls_section_begin|>
  <|tool_call_begin|>functions.NAME:IDX
  <|tool_call_argument_begin|>{"arg":"val"}<|tool_call_end|>
<|tool_calls_section_end|>
```

**Hermes / Qwen3 / gpt-oss** (fallback):

```
<tool_call>{"name": "NAME", "arguments": {"arg":"val"}}</tool_call>
```

A leading `</think>` reasoning prefix (Kimi emits the closer; the opener
is consumed by the special-token decoder) is stripped before parsing.
If both content and tool_calls would be empty (model emitted only a
`</think>` and no usable text or call section), the shim falls back to
the raw decoded text so OAI clients like tau2 don't reject
`{content: null, tool_calls: null}`.

For other tokenizer families that use a different delimiter, extend
`_parse_tool_calls()` in `tinker_shim.py` with a new regex branch.

## Resources / cost

| | |
|---|---|
| VM | GCP `n4-standard-4` on-demand (was spot; switched after overnight preemption) |
| Cost | ~$0.10/hr idle; inference happens on Tinker so model-side cost is per-token |
| Cold start | ~5-7 min (provision + setup + tokenizer load + sampler bind) |
| Latency | ~3 s p50 per call (Tinker round-trip) |
| Throughput | flat to at least concurrency=8 (~2.4 req/s); no 429s observed |

## File layout

```
integrations/fleet/serving/
├── README.md          (this file)
├── __init__.py        (empty)
├── tinker_shim.py     (FastAPI + parser + sampler binder)
└── tinker_shim.yaml   (SkyPilot)
```

The `fleet-tinker-shim` console script is registered in the top-level
`pyproject.toml`'s `[project.scripts]`.
