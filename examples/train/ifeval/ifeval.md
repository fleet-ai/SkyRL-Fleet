# IFEval RLVR Training

## What is IFEval?

[IFEval](https://arxiv.org/abs/2311.07911) (Instruction-Following Evaluation) is a benchmark from Google Research (2023) for measuring how well large language models follow **verifiable** natural-language instructions. Each prompt includes one or more constraints that can be checked with deterministic rules—word counts, required keywords, JSON formatting, language restrictions, and similar—without human raters or an LLM judge.

The public dataset is [`google/IFEval`](https://huggingface.co/datasets/google/IFEval) on Hugging Face (~541 examples). After preprocessing with `ifeval_dataset.py`, you get ~400 training examples and 100 validation examples (default split).

## Why IFEval is a strong RLVR signal

- **Deterministic rewards**: Constraints are verified with rule-based checkers in `skyrl_gym.envs.ifeval.ifeval_utils`—no judge model, no latency, no judge bias.
- **Partial credit (0–1)**: Reward is the fraction of instructions satisfied on a prompt, not all-or-nothing. This gives a richer gradient when models satisfy some but not all constraints.
- **Single-turn, batched generation**: No tool calls or multi-turn loops; ideal for efficient GRPO rollouts with `generator.batched=true`.
- **Diverse constraint types**: 24 instruction families exercise format, length, keywords, language, and structure—useful for improving general instruction-following without task-specific rubrics.

## Quick start

### 1. Install dependencies

From the SkyRL-Fleet repo root:

```bash
uv sync --extra fsdp
```

Optional (improves `language:response_language` checks):

```bash
uv pip install langdetect
```

### 2. Prepare data

```bash
uv run examples/train/ifeval/ifeval_dataset.py --output_dir $HOME/data/ifeval
```

This writes `train.parquet`, `validation.parquet`, and JSON mirrors under `$HOME/data/ifeval`.

### 3. Launch training

```bash
export WANDB_API_KEY=<your_key>

# Qwen2.5-7B (default 8 GPUs)
bash examples/train/ifeval/run_ifeval.sh

# Llama-3.1-8B
bash examples/train/ifeval/run_ifeval_llama.sh
```

Override GPU count or logging:

```bash
NUM_GPUS=4 LOGGER=console bash examples/train/ifeval/run_ifeval.sh
```

## Model options

| Script | Model | Default run name | Checkpoint dir |
|--------|-------|------------------|----------------|
| `run_ifeval.sh` | `Qwen/Qwen2.5-7B-Instruct` | `ifeval_qwen25_7b` | `$HOME/ckpts/ifeval_qwen25_7b_ckpt` |
| `run_ifeval_llama.sh` | `meta-llama/Llama-3.1-8B-Instruct` | `ifeval_llama31_8b` | `$HOME/ckpts/ifeval_llama31_8b_ckpt` |

Both scripts use colocated FSDP2 + vLLM GRPO with `tensor_parallel_size=1` and `num_engines=NUM_GPUS` (default 8). On 4 GPUs, use `NUM_GPUS=4` and optionally `tensor_parallel_size=2` with `num_engines=2` (pass extra Hydra overrides after the script args).

## GPU requirements

- **Recommended**: 8× NVIDIA A100 or H100 (40GB+) for 7–8B models with FSDP2 colocated training and vLLM generation (`TP=1`, one engine per GPU).
- **Minimum (tighter)**: 4× 40GB+ GPUs with `tensor_parallel_size=2` and `num_engines=2` (half the rollout parallelism of the 8-GPU setup).
- Prompts are short (&lt;200 tokens typical); generation budget is 1024 tokens per response.

## Reward breakdown: 24 instruction types

Reward for a rollout = (number of satisfied constraints) / (total constraints on that prompt). The checkers implemented in SkyRL-Fleet are:

| ID | Description |
|----|-------------|
| `keywords:existence` | Response must include all listed keywords |
| `keywords:frequency` | A keyword must appear N times (at least / at most / exactly) |
| `keywords:forbidden_words` | Response must not contain forbidden words |
| `keywords:letter_frequency` | A letter must appear N times |
| `language:response_language` | Entire response must be in a specified language |
| `length_constraints:number_sentences` | Sentence count constraint |
| `length_constraints:number_paragraphs` | Paragraph count constraint |
| `length_constraints:number_words` | Word count constraint |
| `length_constraints:nth_paragraph_first_word` | Nth paragraph must start with a given word |
| `detectable_content:number_placeholders` | Bracketed placeholders `[...]` count |
| `detectable_content:postscript` | Postscript marker (e.g. P.S.) required |
| `detectable_format:number_bullet_lists` | Number of bullet lists |
| `detectable_format:constrained_response` | Response must be one of allowed options |
| `detectable_format:number_highlighted_sections` | Markdown-highlighted sections count |
| `detectable_format:multiple_sections` | Response split into labeled sections |
| `detectable_format:json_format` | Valid JSON output |
| `detectable_format:title` | Title wrapped in `<<...>>` |
| `combination:two_responses` | Exactly two responses separated by `******` |
| `combination:repeat_prompt` | Must repeat the user prompt verbatim |
| `startend:end_checker` | Response must end with a given phrase |
| `change_case:capital_word_frequency` | ALL-CAPS words frequency |
| `change_case:english_capital` | Entire response in uppercase |
| `change_case:english_lowercase` | Entire response in lowercase |
| `punctuation:no_comma` | No commas anywhere in the response |

## Tips for training stability

- **KL coefficient (`kl_loss_coef=0.01`)**: Slightly higher than code/math RLVR defaults to limit format hacking while still allowing instruction-following gains.
- **Temperature (`0.6`) and top-p (`0.95`)**: Enough diversity for GRPO with `n_samples_per_prompt=8` without collapsing to greedy decoding.
- **Batch sizes**: Small dataset (~400 train rows)—use `train_batch_size=256`, `epochs=50`, and `eval_batch_size=100` so each epoch sees the full set multiple times without oversized eval batches.
- **Generation**: `generator.batched=true` for single-turn IFEval; `max_generate_length=1024` so long formatted answers are not truncated mid-constraint.
- **Samples per prompt**: 8 rollouts help GRPO when rewards are fractional (0, 0.5, 1.0, etc.) on multi-constraint prompts.

## Monitoring (Weights & Biases)

Watch these metrics during training:

- **`reward/mean`** (or equivalent env reward mean): Should trend upward as instruction-following improves.
- **`reward/std`**: High variance early is normal; very low std with flat mean may indicate mode collapse or insufficient exploration.
- **`kl_loss`**: Should stay bounded; spikes often precede incoherent or off-distribution text—consider raising `kl_loss_coef` slightly if reward rises but eval quality degrades.

Eval runs every 5 steps (`trainer.eval_interval=5`) with `trainer.eval_before_train=true` on the 100-example validation split.

## References

- Paper: [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) (Zhou et al., 2023)
- Dataset: [google/IFEval](https://huggingface.co/datasets/google/IFEval)
- SkyRL env: `skyrl_gym/envs/ifeval/`
