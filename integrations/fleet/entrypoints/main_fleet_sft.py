"""Supervised fine-tuning on harvested Fleet trajectories via Tinker.

Trains a LoRA policy with cross-entropy loss on verified-successful rollouts
(RFT / rejection fine-tuning). Dataset: JSONL where each record carries
OpenAI-style `messages`, plus provenance (`task_key`, `reward`, `session_id`).
Tool schemas are passed separately (--tools-file) and rendered through the
chat template's native `tools=` channel, matching the GRPO trainer's
`use_tools_channel=True` rollout rendering.

Loss is applied per assistant turn: each turn becomes its own datum whose
prompt is the chat template's render of the history (reasoning-stripped, as
the policy saw it at rollout) and whose completion is that turn's raw
generation. See build_turn_datums for why a whole-conversation render
cannot express this on reasoning models.

Evaluation is deliberately out of scope: run the GRPO entrypoint with
--eval-only --from-checkpoint <sampler_path> against any eval parquet.

Usage:
  python -m integrations.fleet.entrypoints.main_fleet_sft \
    --dataset-file rft.jsonl --tools-file tools.json \
    --model-name Qwen/Qwen3.6-35B-A3B --learning-rate 1e-5 \
    --num-epochs 3 --batch-size 8 --results-out results.json

  --validate tokenizes the dataset and reports viability without training.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer

from integrations.fleet.entrypoints.main_fleet_tinker import discover_trainer_seqlen_cap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def build_turn_datums(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    max_sequence_length: int,
    rng: random.Random,
    max_turns_per_traj: int,
) -> List[types.Datum]:
    """Tokenize one trajectory into per-assistant-turn Datums.

    One datum per assistant turn: prompt = the chat template's render of the
    history plus generation prompt — which on reasoning models (Qwen3.6)
    strips `<think>` blocks from earlier assistant turns, exactly as the
    policy saw the context at rollout — and completion = that turn's raw
    generation (reasoning + answer + tool calls + stop). A single render of
    the whole conversation can NOT express this: the template keeps only the
    final turn's reasoning, so earlier reasoning tokens simply do not exist
    in it. Prompt and completion are tokenized separately, reproducing the
    rollout's token seam at the generation boundary.

    Long trajectories are subsampled to `max_turns_per_traj` assistant turns
    (always keeping the final turn) to bound cost — history prefixes repeat
    across turns, so training every turn of a 30-turn rollout is quadratic.
    """
    idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not idxs or len(messages) < 3:
        return []
    if max_turns_per_traj and len(idxs) > max_turns_per_traj:
        sampled = rng.sample(idxs[:-1], max_turns_per_traj - 1)
        idxs = sorted(set(sampled) | {idxs[-1]})

    def render(msgs: List[Dict[str, Any]], gen_prompt: bool) -> str:
        kwargs: Dict[str, Any] = {"add_generation_prompt": gen_prompt, "tokenize": False}
        if tools:
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(msgs, **kwargs)

    datums: List[types.Datum] = []
    for i in idxs:
        prompt_text = render(messages[:i], True)
        turn_text = render(messages[: i + 1], False)
        if not turn_text.startswith(prompt_text):
            continue
        completion_text = turn_text[len(prompt_text) :]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        tokens = list(prompt_ids) + list(completion_ids)
        if len(tokens) > max_sequence_length or len(completion_ids) < 2:
            continue
        weights = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(tokens[1:], dtype=torch.long)),
                    "weights": TensorData.from_torch(torch.tensor(weights[1:], dtype=torch.float32)),
                },
            )
        )
    return datums


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Read the harvest JSONL; one record per trajectory."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_datums(
    records: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    tools: Optional[List[Dict[str, Any]]],
    max_sequence_length: int,
    seed: int,
    max_turns_per_traj: int,
) -> tuple:
    """Tokenize every record into per-turn datums; return (datums, stats)."""
    rng = random.Random(seed)
    datums: List[types.Datum] = []
    stats = {"trajectories": 0, "trajectories_empty": 0, "turn_datums": 0, "tokens_total": 0, "loss_tokens_total": 0}
    for rec in records:
        ds = build_turn_datums(tokenizer, rec["messages"], tools, max_sequence_length, rng, max_turns_per_traj)
        if not ds:
            stats["trajectories_empty"] += 1
            continue
        stats["trajectories"] += 1
        stats["turn_datums"] += len(ds)
        for d in ds:
            n = len(d.loss_fn_inputs["target_tokens"].to_torch())
            stats["tokens_total"] += n
            stats["loss_tokens_total"] += int(d.loss_fn_inputs["weights"].to_torch().sum().item())
        datums.extend(ds)
    return datums, stats


async def main(args: argparse.Namespace) -> None:
    hf_model_name = args.model_name.split(":peft:")[0]
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

    tools = None
    if args.tools_file:
        with open(args.tools_file) as f:
            tools = json.load(f)

    records = load_dataset(args.dataset_file)
    logger.info(f"Loaded {len(records)} trajectories from {args.dataset_file}")

    datums, stats = prepare_datums(
        records, tokenizer, tools, args.max_sequence_length, args.seed, args.max_turns_per_traj
    )
    logger.info(
        f"Tokenized: {stats['turn_datums']} turn-datums from {stats['trajectories']} trajectories "
        f"({stats['trajectories_empty']} empty) | {stats['tokens_total']:,} tokens, "
        f"{stats['loss_tokens_total']:,} loss tokens"
    )
    if args.validate:
        print(json.dumps(stats, indent=1))
        return
    if not datums:
        raise SystemExit("no usable datums after tokenization")

    service_client = tinker.ServiceClient()

    cap = await discover_trainer_seqlen_cap(
        service_client,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        max_sequence_length=args.max_sequence_length,
    )
    if cap is not None and cap < args.max_sequence_length:
        before = len(datums)
        datums = [d for d in datums if len(d.loss_fn_inputs["target_tokens"].to_torch()) + 1 <= cap]
        logger.info(f"Trainer seqlen cap {cap}: kept {len(datums)}/{before} datums")

    if args.load_state:
        logger.info(f"Resuming training client from state: {args.load_state}")
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(args.load_state)
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.model_name, rank=args.lora_rank
        )
    adam_params = types.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    wandb_run = None
    if os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_DISABLED"):
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
        )

    rng = random.Random(args.seed)
    order = list(range(len(datums)))
    step = 0
    total_steps = args.num_epochs * ((len(datums) + args.batch_size - 1) // args.batch_size)
    saved_states: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(args.num_epochs):
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            batch = [datums[j] for j in order[i : i + args.batch_size]]
            fb = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
            await training_client.optim_step_async(adam_params)
            step += 1

            loss_sum, w_sum = 0.0, 0.0
            for datum, out in zip(batch, fb.loss_fn_outputs):
                elementwise = out["elementwise_loss"].to_torch() if "elementwise_loss" in out else None
                w = datum.loss_fn_inputs["weights"].to_torch()
                if elementwise is not None:
                    loss_sum += float((elementwise * w).sum().item())
                w_sum += float(w.sum().item())
            mean_loss = loss_sum / max(w_sum, 1.0)
            logger.info(
                f"step {step}/{total_steps} (epoch {epoch + 1}): loss/token={mean_loss:.4f} "
                f"batch={len(batch)} elapsed={time.time() - t0:.0f}s"
            )
            if wandb_run:
                wandb_run.log({"sft/loss_per_token": mean_loss, "sft/epoch": epoch + 1}, step=step)

            if args.save_state_every and step % args.save_state_every == 0:
                state = await training_client.save_state_async(f"sft_step_{step:06d}")
                saved_states.append({"step": step, "path": state.path})
                logger.info(f"saved state: {state.path}")

    final_state = await training_client.save_state_async("sft_final")
    sampler = await training_client.save_weights_for_sampler_async("sft_final_sampler")
    logger.info(f"final state: {final_state.path}")
    logger.info(f"final sampler: {sampler.path}")

    if args.results_out:
        with open(args.results_out, "w") as f:
            json.dump(
                {
                    "model_name": args.model_name,
                    "dataset_file": args.dataset_file,
                    "n_trajectories": stats["trajectories"],
                    "n_turn_datums": stats["turn_datums"],
                    "loss_tokens": stats["loss_tokens_total"],
                    "steps": step,
                    "final_state": final_state.path,
                    "final_sampler": sampler.path,
                    "intermediate_states": saved_states,
                },
                f,
                indent=1,
            )
    if wandb_run:
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-file", required=True)
    p.add_argument("--tools-file", default=None)
    p.add_argument("--model-name", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-sequence-length", type=int, default=65536)
    p.add_argument("--load-state", default=None, help="tinker:// state to SFT on top of (default: fresh LoRA)")
    p.add_argument("--save-state-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--max-turns-per-traj",
        type=int,
        default=6,
        help="assistant turns sampled per trajectory (0 = all; final turn always kept)",
    )
    p.add_argument("--wandb-project", default="fleet-tinker-grpo")
    p.add_argument("--wandb-name", default="fleet-sft")
    p.add_argument("--results-out", default=None)
    p.add_argument("--validate", action="store_true", help="tokenize + report stats, no training")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
