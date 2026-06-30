"""Re-run the witness SFT as a LoRA adapter on hosted Qwen3.5-9B (the tinker warm-start).

Cloud tinker is LoRA-only and CANNOT load our merged SFT checkpoint, so the warm-start
for the tinker RL track is produced by re-running the SAME witness SFT data as a LoRA
adapter here. The output `tinker://` checkpoint becomes `load_checkpoint_path` for train_rl.py.

  TINKER_API_KEY=… python -m tinker_backend.train_sft \
      model_name=Qwen/Qwen3.5-9B \
      sft_jsonl=/path/to/witness_sft_v6mt.jsonl \
      num_epochs=2 lora_rank=32

SFT data format (one JSON object per line) — the seam to confirm against the v6mt dataset:
  {"messages": [{"role":"system","content":...},{"role":"user","content":...},
                {"role":"assistant","content":...}]}
(If v6mt is stored as {prompt, completion}, set sft_format=prompt_completion.)

Mirrors tinker_cookbook/recipes/chat_sl/train.py + supervised.train.Config (verified signature).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
import datasets
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _read_chat_jsonl(path: str, fmt: str) -> list[list[dict]]:
    convos: list[list[dict]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if fmt == "messages":
                convos.append(obj["messages"])
            else:  # prompt_completion
                convos.append([
                    {"role": "user", "content": obj["prompt"]},
                    {"role": "assistant", "content": obj["completion"]},
                ])
    return convos


@chz.chz
class WitnessSFTDatasetBuilder(SupervisedDatasetBuilder):
    sft_jsonl: str
    sft_format: str = "messages"           # messages | prompt_completion
    model_name_for_tokenizer: str = "Qwen/Qwen3.5-9B"
    renderer_name: str = "qwen3_5"
    batch_size: int = 8
    max_length: int | None = None

    def __call__(self):
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer)
        convos = _read_chat_jsonl(self.sft_jsonl, self.sft_format)
        hf = datasets.Dataset.from_list([{"messages": c} for c in convos])
        max_len = self.max_length

        def _map(row):
            # LAST_ASSISTANT_MESSAGE: our SFT examples have exactly one assistant turn, so this
            # is equivalent to ALL_ASSISTANT_MESSAGES but avoids the renderer extension-property
            # warning (which only bites multi-assistant-turn conversations).
            return conversation_to_datum(
                row["messages"], renderer, max_length=max_len,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
        # supervised.train.main does: `dataset, maybe_test = config.dataset_builder()` → return a tuple.
        return SupervisedDatasetFromHFDataset(hf, batch_size=self.batch_size, map_fn=_map), None


@chz.chz
class CLIConfig:
    sft_jsonl: str
    sft_format: str = "messages"
    model_name: str = "Qwen/Qwen3.5-9B"
    renderer_name: str | None = None
    learning_rate: float = 1e-4
    num_epochs: int = 3                       # matches SkyRL v6 SFT (EPOCHS=3); LoRA doesn't justify fewer
    lora_rank: int = 32
    batch_size: int = 8
    save_every: int = 50
    wandb_project: str | None = "arc-agi-3"
    wandb_name: str | None = None
    log_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(c: CLIConfig) -> train.Config:
    renderer_name = c.renderer_name or model_info.get_recommended_renderer_name(c.model_name)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run = c.wandb_name or f"witness-tinker-sft-lora{c.lora_rank}-{stamp}"
    log_path = c.log_path or f"/tmp/tinker-witness-sft/{run}"
    dataset_builder = WitnessSFTDatasetBuilder(
        sft_jsonl=c.sft_jsonl, sft_format=c.sft_format,
        model_name_for_tokenizer=c.model_name, renderer_name=renderer_name,
        batch_size=c.batch_size,
    )
    return train.Config(
        log_path=log_path,
        model_name=c.model_name,
        recipe_name="witness_sft_tinker",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=c.learning_rate,
        num_epochs=c.num_epochs,
        lora_rank=c.lora_rank,
        save_every=c.save_every,
        wandb_project=c.wandb_project,
        wandb_name=run,
    )


if __name__ == "__main__":
    cli = chz.entrypoint(CLIConfig)
    config = build_config(cli)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    asyncio.run(train.main(config))
    print("SFT-LoRA done. Use the printed tinker:// checkpoint path as train_rl.py load_checkpoint_path=…")
