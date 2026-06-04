# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the IFEval dataset to parquet format
"""

import argparse
import json
import os

import datasets

SYSTEM_MESSAGE = (
    "You are a helpful, respectful and honest assistant. Always reply to the "
    "instructions carefully and follow all constraints exactly."
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/ifeval")
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        help="Number of examples from the end of the dataset to use for validation.",
    )

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "google/IFEval"

    try:
        dataset = datasets.load_dataset(data_source, "default")
    except Exception:
        dataset = datasets.load_dataset(data_source)

    full_dataset = dataset["train"]
    total = len(full_dataset)
    val_size = min(args.val_size, total)
    train_size = total - val_size

    train_dataset = full_dataset.select(range(train_size))
    val_dataset = full_dataset.select(range(train_size, total))

    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE,
                    },
                    {
                        "role": "user",
                        "content": example["prompt"],
                    },
                ],
                "env_class": "ifeval",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": json.dumps(
                        {
                            "instruction_id_list": example["instruction_id_list"],
                            "kwargs": example["kwargs"],
                        }
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "key": example["key"],
                    "instruction_id_list": example["instruction_id_list"],
                },
            }
            return data

        return process_fn

    column_names = full_dataset.column_names
    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=column_names,
    )
    val_dataset = val_dataset.map(
        function=make_map_fn("validation"),
        with_indices=True,
        remove_columns=column_names,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_parquet = os.path.join(output_dir, "train.parquet")
    val_parquet = os.path.join(output_dir, "validation.parquet")
    train_json = os.path.join(output_dir, "train.json")
    val_json = os.path.join(output_dir, "validation.json")

    train_dataset.to_parquet(train_parquet)
    val_dataset.to_parquet(val_parquet)
    train_dataset.to_json(train_json)
    val_dataset.to_json(val_json)

    print(f"Total dataset size: {total}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Train parquet: {train_parquet}")
    print(f"Validation parquet: {val_parquet}")
    print(f"Train json: {train_json}")
    print(f"Validation json: {val_json}")
