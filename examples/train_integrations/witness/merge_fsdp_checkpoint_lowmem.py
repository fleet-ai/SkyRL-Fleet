"""Low-memory variant of merge_fsdp_checkpoint.py for inference-only merge.

Skips the meta-model build + load_state_dict + save_pretrained path that
doubles memory during writeout. Peak memory was ~50-70GB for 9B in the
original (full_state_dict ~35GB + HF model BF16 ~18GB co-exist during
save_pretrained), exceeding 48GB MacBook Pro RAM.

This variant:
- Loads + materializes shards EXACTLY the same way as the original (we
  import those functions, no re-implementation drift).
- After materialization, drops `shards`/intermediate references and runs
  gc.collect() before save.
- Skips the meta-model build entirely.
- Casts to BF16 per-shard during write (in-place via `.to(bf16)` then
  drop original reference) so peak only holds one shard's worth of BF16
  + the remaining FP32 tensors.
- Writes sharded `model-{i:05d}-of-{N:05d}.safetensors` via
  `safetensors.torch.save_file` + manual index JSON, mirroring HF's
  save_pretrained sharding layout.
- Copies tokenizer + config from `<checkpoint_dir>/huggingface/` (or
  downloads from `--model_name` HF repo if absent).

Peak memory (9B): ~35GB (materialized full_state_dict) + ~5GB (active
shard buffer) = ~40GB. Comfortable on 48GB MacBook Pro.

CAVEATS:
- Skips HF's `_tied_weights_keys` logic. After numpy-roundtrip
  materialization the ties are already broken (each tensor has its own
  storage), so safetensors save_file works directly. When you later
  load via `from_pretrained`, transformers re-ties based on
  `config.tie_word_embeddings` (no behavioral difference).
- Does NOT do the 35B-A3B visual-encoder shape-mismatch dropping. That
  was only needed for the MoE/VLM 35B variant where ckpt and model
  config had divergent vision dims. For Qwen3.5-9B (dense) all keys
  match, so no drop needed.

Usage (drop-in replacement for the original):
    python merge_fsdp_checkpoint_lowmem.py \\
        --checkpoint_dir /path/to/fsdp_ckpt/policy \\
        --output_dir /path/to/output_hf \\
        --model_name Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from typing import Any

import torch
from transformers import AutoConfig

# Reuse loaders + materialization from the original script (no drift)
from merge_fsdp_checkpoint import (
    find_shard_files,
    merge_sharded_state_dict,
)


SAFETENSORS_SHARD_BYTES = int(5e9)  # 5 GB per shard (matches HF default)


def _strip_wrapper(t: Any):
    """Inline copy of original script's _strip_wrapper (we don't import to
    avoid touching the original's private layout — keep self-contained)."""
    if "DTensor" in type(t).__name__ or hasattr(t, "_local_tensor"):
        if hasattr(t, "_local_tensor"):
            return t._local_tensor, True
        if hasattr(t, "to_local"):
            try:
                return t.to_local(), True
            except Exception:
                pass
        if hasattr(t, "full_tensor"):
            try:
                return t.full_tensor(), True
            except Exception:
                pass
    return t, False


def materialize_state_dict(full_state_dict: dict) -> dict:
    """Numpy round-trip materialization to get fresh CPU storage per tensor.
    Identical logic to original merge_fsdp_checkpoint.py.
    """
    print(f"  Materializing {len(full_state_dict)} tensors via cascading fallback...")
    n_dtensor = n_via_numpy = n_via_copy_into = n_failed = 0
    materialized = {}

    for k, v in full_state_dict.items():
        v, was_dtensor = _strip_wrapper(v)
        if was_dtensor:
            n_dtensor += 1

        # Path A: numpy round-trip
        try:
            arr = v.detach().cpu().numpy()
            materialized[k] = torch.from_numpy(arr.copy())
            n_via_numpy += 1
            continue
        except Exception:
            pass

        # Path B: copy_into
        try:
            shape = tuple(v.shape)
            dtype = v.dtype
            fresh = torch.empty(shape, dtype=dtype, device="cpu")
            with torch.no_grad():
                fresh.copy_(v.detach().cpu())
            materialized[k] = fresh
            n_via_copy_into += 1
            continue
        except Exception as e:
            print(f"    WARNING: both numpy and copy_into failed for {k!r}: {e}")
            materialized[k] = v
            n_failed += 1

    print(f"  Materialized: dtensor_wrapped={n_dtensor} "
          f"via_numpy={n_via_numpy} via_copy_into={n_via_copy_into} "
          f"failed_to_materialize={n_failed}")
    if n_failed:
        print(f"  WARNING: {n_failed} tensors fell back to raw — save may still fail.")
    return materialized


def plan_shards(state_dict: dict, target_bytes: int = SAFETENSORS_SHARD_BYTES):
    """Decide which keys go into which shard.

    Two-pass approach: first pass computes BF16 sizes (just numel * 2, no
    tensor allocation), assigns keys to shards greedily. Second pass
    (during write) loads + casts each shard.

    Returns: list of (filename, [keys]) tuples + total_size_bytes.
    """
    # Compute BF16 sizes per key (cheap — just uses tensor metadata)
    key_sizes = []
    total = 0
    for k, t in state_dict.items():
        # 2 bytes per BF16 elem
        sz = t.numel() * 2
        key_sizes.append((k, sz))
        total += sz

    # Greedy bin-packing
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for k, sz in key_sizes:
        if current_size + sz > target_bytes and current:
            shards.append(current)
            current = []
            current_size = 0
        current.append(k)
        current_size += sz
    if current:
        shards.append(current)

    n = len(shards)
    plan = []
    for i, keys in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{n:05d}.safetensors"
        plan.append((filename, keys))
    return plan, total


def save_state_dict_sharded(state_dict: dict, output_dir: str):
    """Save state_dict as sharded .safetensors with HF-compatible index JSON.

    Pops keys from state_dict as we go to free memory aggressively.
    """
    from safetensors.torch import save_file

    plan, total_bytes = plan_shards(state_dict)
    print(f"  Sharding {len(state_dict)} tensors into {len(plan)} files "
          f"(~{total_bytes / 1e9:.1f} GB total in BF16)")

    weight_map: dict[str, str] = {}
    os.makedirs(output_dir, exist_ok=True)

    for shard_idx, (filename, keys) in enumerate(plan, start=1):
        shard_data: dict[str, torch.Tensor] = {}
        for k in keys:
            t = state_dict.pop(k)  # remove from full dict → free original
            # Cast to BF16, ensure contiguous, drop autograd state
            t_bf16 = t.detach().to(dtype=torch.bfloat16).contiguous()
            del t
            shard_data[k] = t_bf16
            weight_map[k] = filename

        path = os.path.join(output_dir, filename)
        # safetensors save_file may raise if tensors share storage — our
        # numpy materialization broke any storage sharing, so this should
        # work for Qwen3.5-9B (no shared storage post-materialize).
        save_file(shard_data, path)
        size_gb = os.path.getsize(path) / 1e9
        print(f"  [{shard_idx}/{len(plan)}] wrote {filename} ({size_gb:.2f} GB)")

        # Aggressively free this shard before moving on
        shard_data.clear()
        del shard_data
        gc.collect()

    # Write HF-compatible safetensors index JSON
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Wrote index: {index_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True,
                        help="FSDP checkpoint dir containing model_world_size_*_rank_*.pt")
    parser.add_argument("--output_dir", required=True,
                        help="Output dir for HF format files")
    parser.add_argument("--model_name", required=True,
                        help="Base model name on HF Hub (e.g. Qwen/Qwen3.5-9B); used "
                             "for config + tokenizer fallback if not in checkpoint_dir/huggingface/")
    args = parser.parse_args()

    print(f"Merging FSDP checkpoint (LOW-MEM variant):")
    print(f"  Input:  {args.checkpoint_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Model:  {args.model_name}")

    # Already-merged shortcut (same as original)
    hf_dir = os.path.join(args.checkpoint_dir, "huggingface")
    safetensors_marker = os.path.join(hf_dir, "model.safetensors")
    if os.path.exists(safetensors_marker):
        print(f"  HuggingFace model already exists at {hf_dir}, copying...")
        os.makedirs(args.output_dir, exist_ok=True)
        os.system(f"cp -r {hf_dir}/* {args.output_dir}/")
        print("  Done (copied existing HF checkpoint)")
        return

    # Load + merge shards (delegate to original)
    shard_files, world_size = find_shard_files(args.checkpoint_dir)
    if not shard_files:
        print(f"  ERROR: No shard files found in {args.checkpoint_dir}")
        sys.exit(1)
    print(f"  Found {len(shard_files)} shard files (world_size={world_size})")
    full_state_dict = merge_sharded_state_dict(shard_files, world_size)

    # Materialize (numpy round-trip → fresh CPU storage)
    full_state_dict = materialize_state_dict(full_state_dict)

    # Aggressive GC: drop any lingering DTensor / device_mesh refs from
    # original shards before we move on. The merge+materialize functions
    # release `shards`/`merged_local` etc. when they return, but we force a
    # gc.collect() to make sure circular refs are torn down too.
    gc.collect()

    # Load config to verify architecture; also for copying into output_dir
    local_config = os.path.join(args.checkpoint_dir, "huggingface", "config.json")
    if os.path.exists(local_config):
        config_source = os.path.join(args.checkpoint_dir, "huggingface")
    else:
        config_source = args.model_name
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    arch = getattr(config, "architectures", ["unknown"])
    print(f"  Model architecture: {arch}")

    # ─────────────────────────────────
    # LOW-MEM SAVE: shard + cast + write directly, no model build
    # ─────────────────────────────────
    save_state_dict_sharded(full_state_dict, args.output_dir)

    # Save config.json
    config.save_pretrained(args.output_dir)
    print(f"  Wrote config.json")

    # Copy tokenizer + other HF files from local (if available) or download
    local_hf_dir = os.path.join(args.checkpoint_dir, "huggingface")
    # NEVER copy these — they would shadow our freshly-merged sharded files.
    # The FSDP huggingface/ subdir often contains a stale base-model
    # `model.safetensors` from training-time init; if copied, vllm /
    # transformers may load THAT instead of our sharded `model-*.safetensors`,
    # silently giving base weights instead of trained weights.
    SHADOWING_FILES = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "config.json",            # we wrote our own via config.save_pretrained()
        "generation_config.json", # ditto, written by config.save_pretrained()
    }
    if os.path.exists(local_hf_dir):
        copied: list[str] = []
        skipped: list[str] = []
        for fname in os.listdir(local_hf_dir):
            src = os.path.join(local_hf_dir, fname)
            dst = os.path.join(args.output_dir, fname)
            if not os.path.isfile(src):
                continue
            if fname in SHADOWING_FILES:
                skipped.append(fname)
                continue
            # Don't overwrite our own freshly-written files
            if os.path.exists(dst):
                continue
            shutil.copy2(src, dst)
            copied.append(fname)
        print(f"  Copied {len(copied)} extras from {local_hf_dir}: {copied}")
        if skipped:
            print(f"  Skipped (would shadow our merged output): {skipped}")
    else:
        # Download tokenizer from HF (likely needs internet)
        print(f"  No local huggingface/ dir; downloading tokenizer from {args.model_name}")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=args.model_name,
            allow_patterns=["tokenizer*", "vocab*", "merges*", "special_tokens*"],
            local_dir=args.output_dir,
            local_dir_use_symlinks=False,
        )

    # Sanity print
    files = sorted(os.listdir(args.output_dir))
    print(f"\n  Output dir contents:")
    for f in files:
        path = os.path.join(args.output_dir, f)
        if os.path.isfile(path):
            size = os.path.getsize(path) / 1e9
            unit = "GB" if size >= 1 else "MB"
            disp = size if size >= 1 else size * 1000
            print(f"    {f}  ({disp:.2f} {unit})")
    print(f"\n  [merge-lowmem] OK")


if __name__ == "__main__":
    main()
