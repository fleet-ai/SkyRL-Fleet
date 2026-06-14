#!/usr/bin/env python3
"""Collect matched sensory-on / sensory-off trajectories from falmart.

Orchestration is complete; it drives the tested data-engine core. The only
things that must be implemented first are the two seams in
`sensory_sft/drivers.py` (VLLMQwenPolicy, PlaywrightDriver). See
`RUN_ON_CLUSTER.md`.

Example:
    python collect.py --arms both --n-episodes 100 --max-steps 20 \
        --falmart-url http://localhost:5173 \
        --vllm-url http://localhost:8000/v1 \
        --registry ~/theseus-falmart/server/src/sense/schema-registry.json \
        --out ./traces
"""

from __future__ import annotations

import argparse
import os
import uuid
from collections import Counter

from sensory_sft.drivers import PlaywrightDriver, VLLMQwenPolicy
from sensory_sft.prompts import exploratory_tasks
from sensory_sft.registry import Registry
from sensory_sft.rollout import JsonlWriter, RolloutConfig, run_episode
from sensory_sft.sense import SenseClient


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--falmart-url", default="http://localhost:5173")
    p.add_argument("--vllm-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--registry",
        default=os.path.expanduser(
            "~/theseus-falmart/server/src/sense/schema-registry.json"
        ),
    )
    p.add_argument("--arms", choices=["both", "on", "off"], default="both")
    p.add_argument("--n-episodes", type=int, default=100,
                   help="episodes PER ARM (so 'both' x 100 = 200 trajectories)")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--out", default="./traces")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    registry = Registry.load(args.registry)
    tasks = exploratory_tasks(args.n_episodes)

    arms = (
        [("on", True), ("off", False)] if args.arms == "both"
        else [(args.arms, args.arms == "on")]
    )

    for arm_name, sensory_on in arms:
        cfg = RolloutConfig(sensory_on=sensory_on, max_steps=args.max_steps)
        out_path = os.path.join(args.out, f"falmart_sensory_{arm_name}.jsonl")
        counts: Counter = Counter()
        print(f"\n=== arm: sensory_{arm_name} -> {out_path} ===", flush=True)

        with JsonlWriter(out_path) as writer:
            for i, task in enumerate(tasks):
                # Fresh policy/driver/sense per episode keeps cursors clean and
                # state isolated. Driver exposes the env origin for the reader.
                policy = VLLMQwenPolicy(base_url=args.vllm_url, model=args.model)
                driver = PlaywrightDriver(base_url=args.falmart_url)
                sense = SenseClient(args.falmart_url, registry)
                try:
                    result = run_episode(
                        policy, driver, sense, task, cfg,
                        episode_id=f"{arm_name}-{i:03d}-{uuid.uuid4().hex[:8]}",
                    )
                    writer.write_episode(result)
                    counts.update(result.outcome_counts)
                    print(
                        f"  ep {i:03d} [{task['key']}] steps={result.steps} "
                        f"stop={result.stop_reason} {dict(result.outcome_counts)}",
                        flush=True,
                    )
                finally:
                    driver.close()

        total = sum(counts.values())
        print(f"\n  {arm_name}: {total} labeled actions across {len(tasks)} episodes")
        for k, v in counts.most_common():
            print(f"    {k:20s} {v:6d}  ({100*v/total:.1f}%)" if total else k)


if __name__ == "__main__":
    main()
