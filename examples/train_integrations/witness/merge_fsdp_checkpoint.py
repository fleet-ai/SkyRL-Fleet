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

    # NUMPY ROUND-TRIP MATERIALIZATION
    # ─────────────────────────────────
    # Tensors loaded from FSDP ShardedStateDictConfig may be DTensor wrappers
    # whose process-group is no longer bound (we load offline without a PG),
    # OR plain tensors whose storage handle is invalidated. Either way,
    # `tensor.storage().data_ptr()` raises "Attempted to access the data pointer
    # on an invalid python storage", which crashes safetensors' shared-tensor
    # detection (`_find_shared_tensors`).
    #
    # Going through numpy guarantees we end up with a torch.Tensor whose
    # storage was freshly allocated by torch.from_numpy (not inherited from
    # any FSDP/DTensor lineage). `.numpy()` requires a plain CPU Tensor — so
    # we first peel off any DTensor wrapper via `_local_tensor`.
    print(f"  Materializing {len(full_state_dict)} tensors via cascading fallback...")
    n_dtensor = n_via_numpy = n_via_copy_into = n_failed = 0
    materialized = {}

    def _peel_dtensor(t):
        """Strip DTensor wrapper, returning best-effort plain Tensor."""
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

    for k, v in full_state_dict.items():
        v, was_dtensor = _peel_dtensor(v)
        if was_dtensor:
            n_dtensor += 1

        # Cascading materialization. Each path produces a fresh-storage tensor.
        # Path A: numpy round-trip (fastest, requires .numpy() to work)
        try:
            arr = v.detach().cpu().numpy()
            materialized[k] = torch.from_numpy(arr.copy())
            n_via_numpy += 1
            continue
        except Exception:
            pass

        # Path B: allocate a zero tensor then copy_ into it (avoids numpy entirely;
        # copy_ goes through the source tensor's element accessor which doesn't
        # depend on storage().data_ptr())
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
            print(f"    WARNING: both numpy and copy_into failed for {k!r}: {e}; "
                  f"using raw tensor (save may fail)")
            materialized[k] = v
            n_failed += 1

    full_state_dict = materialized
    print(f"  Materialized: dtensor_wrapped={n_dtensor} "
          f"via_numpy={n_via_numpy} via_copy_into={n_via_copy_into} "
          f"failed_to_materialize={n_failed}")
    if n_failed:
        print(f"  WARNING: {n_failed} tensors fell back to raw — save may still crash. "
              f"Will use --safe_serialization=False as last resort if save fails.")

    # Load config
    print(f"  Loading model config...")

    local_config = os.path.join(args.checkpoint_dir, "huggingface", "config.json")
    if os.path.exists(local_config):
        config_source = os.path.join(args.checkpoint_dir, "huggingface")
    else:
        config_source = args.model_name

    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    arch = getattr(config, 'architectures', ['unknown'])
    print(f"  Model architecture: {arch}")

    # MODEL-BASED SAVE
    # ────────────────
    # We use HF's `model.save_pretrained(safe_serialization=True)` rather than
    # raw `safetensors.save_file(state_dict)`. HF's path uses
    # `_tied_weights_keys` config metadata to handle tied embeddings
    # (Qwen3.5's lm_head <-> embed_tokens), bypassing safetensors'
    # `_find_shared_tensors` (which calls .storage().data_ptr() and is
    # what crashed the previous attempt).
    #
    # Building the model on the meta device first means we don't allocate
    # 18 GB just to overwrite it with our state_dict.
    print(f"  Building empty model on meta device + loading merged state_dict...")
    is_vlm = hasattr(config, "vision_config") and getattr(config, "vision_config") is not None
    if is_vlm and AutoModelForImageTextToText is not None:
        model_class = AutoModelForImageTextToText
        print(f"  (VLM detected — using AutoModelForImageTextToText)")
    else:
        model_class = AutoModelForCausalLM

    with torch.device("meta"):
        model = model_class.from_config(config, trust_remote_code=True)

    # to_empty allocates real (empty) storage on cpu, then load_state_dict fills it
    model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
    print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    first missing keys: {missing[:5]}")
    if unexpected:
        print(f"    first unexpected keys: {unexpected[:5]}")

    os.makedirs(args.output_dir, exist_ok=True)
    # Try safetensors first (preferred — smaller, safer, faster load).
    # Fall back to torch.save (.bin) if any tensor still has storage issues
    # — HF's from_pretrained tries .safetensors first then .bin.
    saved_via = None
    try:
        print(f"  Saving via model.save_pretrained(safe_serialization=True) ...")
        model.save_pretrained(args.output_dir, safe_serialization=True)
        saved_via = "safetensors"
    except Exception as e:
        print(f"  WARNING: safetensors save failed ({type(e).__name__}: {e}); "
              f"falling back to torch.save (.bin)")
        try:
            print(f"  Saving via model.save_pretrained(safe_serialization=False) ...")
            model.save_pretrained(args.output_dir, safe_serialization=False)
            saved_via = "torch_bin"
        except Exception as e2:
            print(f"  WARNING: even .bin save failed ({type(e2).__name__}: {e2}); "
                  f"writing raw state dict via torch.save")
            torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
            # Save config / generation_config manually
            model.config.save_pretrained(args.output_dir)
            saved_via = "raw_torch_save"
    print(f"  Save successful via: {saved_via}")

    # Copy tokenizer from local or download from HF
    import shutil
    local_hf_dir = os.path.join(args.checkpoint_dir, "huggingface")
    if os.path.exists(local_hf_dir):
        # Copy any tokenizer files not produced by save_pretrained
        for fname in os.listdir(local_hf_dir):
            src = os.path.join(local_hf_dir, fname)
            dst = os.path.join(args.output_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"  Copied tokenizer from {local_hf_dir}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  Downloaded tokenizer from {args.model_name}")

    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if os.path.isfile(os.path.join(args.output_dir, f))
    )
    print(f"  Merged checkpoint saved: {args.output_dir} ({total_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
