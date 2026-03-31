"""
Merge FSDP sharded checkpoint into a single HuggingFace model.

FSDP saves one file per GPU rank: model_world_size_8_rank_0.pt ... rank_7.pt
This script loads all shards, reconstructs the full state dict, and saves
in standard HuggingFace format (model.safetensors + config.json + tokenizer).

Usage:
  python merge_fsdp_checkpoint.py \\
    --checkpoint_dir ~/checkpoint_sharded/global_step_100/policy \\
    --output_dir ~/checkpoint_merged/ \\
    --model_name Qwen/Qwen3.5-9B
"""

import argparse
import glob
import os
import re
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Qwen3.5 is a VLM (Qwen3_5ForConditionalGeneration), not a CausalLM
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


def find_shard_files(checkpoint_dir: str):
    """Find all model shard files and determine world_size."""
    pattern = os.path.join(checkpoint_dir, "model_world_size_*_rank_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        return [], 0

    # Extract world_size from filename
    match = re.search(r"world_size_(\d+)_rank_(\d+)", os.path.basename(files[0]))
    if match:
        world_size = int(match.group(1))
    else:
        world_size = len(files)

    return files, world_size


def merge_sharded_state_dict(shard_files: list, world_size: int) -> dict:
    """
    Load FSDP sharded state dicts and merge into a full state dict.

    FSDP ShardedStateDictConfig saves each rank's shard of each parameter.
    We concatenate them along the sharding dimension to reconstruct the full tensor.
    """
    print(f"  Loading {len(shard_files)} shards (world_size={world_size})...")

    # Load all shards
    shards = []
    for f in sorted(shard_files):
        print(f"    Loading {os.path.basename(f)}...")
        shard = torch.load(f, map_location="cpu", weights_only=False)
        shards.append(shard)

    if not shards:
        raise ValueError("No shards loaded")

    # Get all parameter names from first shard
    param_names = list(shards[0].keys())
    print(f"  Found {len(param_names)} parameters")

    full_state_dict = {}
    for name in param_names:
        tensors = [s[name] for s in shards if name in s]

        if len(tensors) == 1:
            # Not sharded — same across all ranks
            full_state_dict[name] = tensors[0]
        elif all(t.shape == tensors[0].shape for t in tensors):
            # All same shape — likely replicated, take first
            full_state_dict[name] = tensors[0]
        else:
            # Different shapes — try to concatenate along sharding dim
            # FSDP typically shards along dim 0
            try:
                full_state_dict[name] = torch.cat(tensors, dim=0)
            except RuntimeError:
                # Try dim 1
                try:
                    full_state_dict[name] = torch.cat(tensors, dim=1)
                except RuntimeError:
                    print(f"    WARNING: Cannot merge {name}, using first shard")
                    full_state_dict[name] = tensors[0]

    return full_state_dict


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP shards to HuggingFace")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing model_world_size_X_rank_Y.pt files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for merged HuggingFace model")
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-9B",
                        help="Base model name (for config and tokenizer)")
    args = parser.parse_args()

    print(f"Merging FSDP checkpoint:")
    print(f"  Input:  {args.checkpoint_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Model:  {args.model_name}")

    # Check if HuggingFace format already exists
    hf_dir = os.path.join(args.checkpoint_dir, "huggingface")
    safetensors = os.path.join(hf_dir, "model.safetensors")
    if os.path.exists(safetensors):
        print(f"  HuggingFace model already exists at {hf_dir}, copying...")
        os.makedirs(args.output_dir, exist_ok=True)
        os.system(f"cp -r {hf_dir}/* {args.output_dir}/")
        print("  Done (copied existing HF checkpoint)")
        return

    # Find shard files
    shard_files, world_size = find_shard_files(args.checkpoint_dir)
    if not shard_files:
        print(f"  ERROR: No shard files found in {args.checkpoint_dir}")
        print(f"  Looking for: model_world_size_*_rank_*.pt")
        print(f"  Contents: {os.listdir(args.checkpoint_dir)[:20]}")
        sys.exit(1)

    print(f"  Found {len(shard_files)} shard files (world_size={world_size})")

    # Merge shards
    full_state_dict = merge_sharded_state_dict(shard_files, world_size)

    # Load config only (no weights) and create empty model
    print(f"  Loading model config...")

    local_config = os.path.join(args.checkpoint_dir, "huggingface", "config.json")
    if os.path.exists(local_config):
        config_source = os.path.join(args.checkpoint_dir, "huggingface")
    else:
        config_source = args.model_name

    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    arch = getattr(config, 'architectures', ['unknown'])
    print(f"  Creating empty model from config ({arch})...")

    # Detect model type: VLM (Qwen3_5ForConditionalGeneration) vs CausalLM
    is_vlm = any("Conditional" in a or "ImageText" in a for a in (arch or []))
    AutoModelClass = AutoModelForImageTextToText if (is_vlm and AutoModelForImageTextToText) else AutoModelForCausalLM

    # Create model without loading any weights
    with torch.device("meta"):
        model = AutoModelClass.from_config(config, trust_remote_code=True)

    # Move to CPU and load merged weights
    model = model.to_empty(device="cpu")
    print("  Loading merged state dict into model...")
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

    # Save in HuggingFace format
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer_source = os.path.join(args.checkpoint_dir, "huggingface")
    if os.path.exists(os.path.join(tokenizer_source, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"  Merged checkpoint saved: {args.output_dir}")
    print(f"  Size: {sum(os.path.getsize(os.path.join(args.output_dir, f)) for f in os.listdir(args.output_dir)) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
