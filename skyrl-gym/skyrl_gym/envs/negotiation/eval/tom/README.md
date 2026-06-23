# Offline ToM Eval Harness

This directory contains an **offline Theory-of-Mind evaluation harness** for trained negotiation checkpoints, separate from the RL training loop.

Benchmark:
- **FanToM** (Kim et al., 2023): multiparty, information-asymmetry ToM — the closest
  ToM probe to negotiation (reasoning about who knows what), chosen as the single ToM
  signal for the elicitation experiment. (HiToM and ToMi were dropped to avoid
  redundant same-role evals.)

The `fantom_loader.py` loader handles data preparation and first-run download.  
Note: FanToM is eval-only licensed.

## Evaluate a local checkpoint (vLLM)

Serve your HF-format checkpoint:

```bash
vllm serve <ckpt_dir> --port 8000
```

Run eval against the OpenAI-compatible endpoint:

```bash
OPENAI_API_KEY=dummy python3 run_tom_eval.py \
  --model <ckpt_path_or_served_name> \
  --base-url http://localhost:8000/v1 \
  --no-think
```

## Evaluate an OpenRouter baseline

```bash
OPENROUTER_API_KEY=<your_key> python3 run_tom_eval.py \
  --model openai/gpt-4o-mini \
  --base-url https://openrouter.ai/api/v1
```

## Output files

- Per-task results: `results/<sanitized_model>_<task>_n<N>.json`
- Combined task summary: `results/<sanitized_model>_summary.json`

## Sanity check without endpoint calls

Use dry run to verify loader wiring/data availability and scoring logic without making API calls:

```bash
python3 run_tom_eval.py --model dummy --dry-run
```
