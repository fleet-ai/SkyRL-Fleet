from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from narc.checksum import stable_hash, tensor_hash
from narc.env import (
    collect_fingerprint,
    prepare_deterministic_environment,
    slurm_context,
)
from narc.schema import SCHEMA_VERSION, ProbeConfig, ProbeResult


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def dump_json(outfile: Any, payload: dict[str, Any]) -> None:
    json.dump(payload, outfile, indent=2, sort_keys=True)
    outfile.write("\n")


def resolve_device(torch: Any, device: str, logical_device: int) -> Any:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        if logical_device >= torch.cuda.device_count():
            raise RuntimeError(
                f"logical CUDA device {logical_device} is out of range; "
                f"torch sees {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(logical_device)
        return torch.device(f"cuda:{logical_device}")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {device}")


def resolve_dtype(torch: Any, dtype: str, device: Any) -> Any:
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")


def configure_torch(
    torch: Any,
    *,
    seed: int,
    deterministic: bool,
    allow_tf32: bool,
) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision") and not allow_tf32:
        torch.set_float32_matmul_precision("highest")


def fixed_inputs(torch: Any, config: ProbeConfig, device: Any) -> tuple[Any, Any]:
    token_count = config.batch_size * config.sequence_length
    base = torch.arange(token_count, dtype=torch.long, device=device).view(
        config.batch_size,
        config.sequence_length,
    )
    input_ids = (base * 17 + 23) % config.vocab_size
    labels = (base * 31 + 7) % config.vocab_size
    return input_ids, labels


def manual_cross_entropy(torch: Any, logits: Any, labels: Any) -> Any:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    flat_logits = shifted_logits.view(-1, shifted_logits.shape[-1])
    flat_labels = shifted_labels.view(-1)
    log_denominator = torch.logsumexp(flat_logits.float(), dim=-1)
    selected = flat_logits.float().gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    return (log_denominator - selected).mean()


def manual_sgd_step(model: Any, *, learning_rate: float) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.add_(parameter.grad, alpha=-learning_rate)


def hash_named_tensors(tensors: list[tuple[str, Any]]) -> str:
    return stable_hash(
        [
            {
                "name": name,
                "hash": tensor_hash(tensor),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in tensors
        ]
    )


def parameter_hash(model: Any) -> str:
    return hash_named_tensors(list(model.named_parameters()))


def gradient_hash(model: Any) -> str:
    named_gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    return hash_named_tensors(named_gradients)


def cuda_memory(torch: Any, device: Any) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_training_steps(
    torch: Any,
    config: ProbeConfig,
    *,
    device: Any,
    dtype: Any,
    seed: int,
    steps: int,
    timed: bool,
) -> dict[str, Any]:
    from narc.tiny_lm import build_model

    model = build_model(config, seed=seed).to(device=device, dtype=dtype)
    input_ids, labels = fixed_inputs(torch, config, device)
    learning_rate = 1e-3
    losses: list[str] = []
    step_seconds: list[float] = []
    final_logits = None
    final_grad_hash = None

    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True) if timed else None
            end_event = torch.cuda.Event(enable_timing=True) if timed else None
            if start_event is not None:
                start_event.record()
        else:
            start_time = time.perf_counter() if timed else None

        logits = model(input_ids)
        loss = manual_cross_entropy(torch, logits, labels)
        loss.backward()
        final_grad_hash = gradient_hash(model)
        with torch.no_grad():
            manual_sgd_step(model, learning_rate=learning_rate)

        if device.type == "cuda":
            if end_event is not None and start_event is not None:
                end_event.record()
                torch.cuda.synchronize(device)
                step_seconds.append(start_event.elapsed_time(end_event) / 1000.0)
        elif start_time is not None:
            step_seconds.append(time.perf_counter() - start_time)

        losses.append(f"{float(loss.detach().cpu()):.17g}")
        final_logits = logits.detach()

    if final_logits is None or final_grad_hash is None:
        raise RuntimeError("probe ran zero training steps")

    selected = (
        final_logits.reshape(-1)[: min(16, final_logits.numel())]
        .float()
        .detach()
        .cpu()
        .tolist()
    )
    output = {
        "losses": losses,
        "final_loss": losses[-1],
        "logits_hash": tensor_hash(final_logits.float()),
        "grad_hash": final_grad_hash,
        "parameter_hash": parameter_hash(model),
        "selected_logits": [f"{value:.9g}" for value in selected],
    }
    output["output_hash"] = stable_hash(output)
    if step_seconds:
        token_count = config.batch_size * config.sequence_length * len(step_seconds)
        elapsed = sum(step_seconds)
        output["timing"] = {
            "step_seconds": step_seconds,
            "mean_step_seconds": elapsed / len(step_seconds),
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed else None,
        }
    return output


def run_correctness(torch: Any, config: ProbeConfig, *, device: Any, dtype: Any) -> dict:
    runs = [
        run_training_steps(
            torch,
            config,
            device=device,
            dtype=dtype,
            seed=config.seed,
            steps=config.steps,
            timed=False,
        )
        for _ in range(config.repeat)
    ]
    output_hashes = [run["output_hash"] for run in runs]
    return {
        "repeat_match": len(set(output_hashes)) == 1,
        "output_hash": output_hashes[0],
        "runs": runs,
    }


def run_performance(torch: Any, config: ProbeConfig, *, device: Any, dtype: Any) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if config.warmup_steps:
        run_training_steps(
            torch,
            config,
            device=device,
            dtype=dtype,
            seed=config.seed,
            steps=config.warmup_steps,
            timed=False,
        )
    synchronize(torch, device)
    measured = run_training_steps(
        torch,
        config,
        device=device,
        dtype=dtype,
        seed=config.seed,
        steps=config.steps,
        timed=True,
    )
    memory = cuda_memory(torch, device)
    timings = measured.get("timing", {})
    step_seconds = timings.get("step_seconds", [])
    variance = None
    if step_seconds:
        mean = timings["mean_step_seconds"]
        variance = sum((value - mean) ** 2 for value in step_seconds) / len(
            step_seconds
        )
    return {
        "output_hash": measured["output_hash"],
        "losses": measured["losses"],
        "timing": timings,
        "step_seconds_variance": variance,
        "memory": memory,
    }


def default_config(args: argparse.Namespace, dtype_name: str) -> ProbeConfig:
    profile = args.profile
    if profile == "correctness":
        defaults = {
            "batch_size": 2,
            "sequence_length": 16,
            "vocab_size": 128,
            "d_model": 32,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "steps": 3,
            "warmup_steps": 0,
            "repeat": args.repeat,
        }
    else:
        defaults = {
            "batch_size": 8,
            "sequence_length": 128,
            "vocab_size": 1024,
            "d_model": 256,
            "num_layers": 4,
            "num_heads": 8,
            "mlp_ratio": 4,
            "steps": 10,
            "warmup_steps": 5,
            "repeat": 1,
        }
    return ProbeConfig(
        profile=profile,
        seed=args.seed,
        batch_size=args.batch_size or defaults["batch_size"],
        sequence_length=args.sequence_length or defaults["sequence_length"],
        vocab_size=args.vocab_size or defaults["vocab_size"],
        d_model=args.d_model or defaults["d_model"],
        num_layers=args.num_layers or defaults["num_layers"],
        num_heads=args.num_heads or defaults["num_heads"],
        mlp_ratio=args.mlp_ratio or defaults["mlp_ratio"],
        steps=args.steps or defaults["steps"],
        warmup_steps=args.warmup_steps
        if args.warmup_steps is not None
        else defaults["warmup_steps"],
        dtype=dtype_name,
        repeat=defaults["repeat"],
        deterministic=not args.no_deterministic,
        allow_tf32=args.allow_tf32,
    )


def run_probe(args: argparse.Namespace) -> ProbeResult:
    deterministic_env_applied = prepare_deterministic_environment()

    import torch

    started_at = utc_now()
    run_id = args.run_id or uuid.uuid4().hex
    device = resolve_device(torch, args.device, args.logical_device)
    dtype = resolve_dtype(torch, args.dtype, device)
    dtype_name = str(dtype).removeprefix("torch.")
    config = default_config(args, dtype_name)
    configure_torch(
        torch,
        seed=config.seed,
        deterministic=config.deterministic,
        allow_tf32=config.allow_tf32,
    )

    fingerprint = collect_fingerprint(
        torch,
        device_type=device.type,
        logical_device=args.logical_device,
    )
    config_hash = stable_hash(config.to_dict())
    fingerprint_hash = stable_hash(fingerprint)

    errors: list[dict[str, Any]] = []
    checks: dict[str, Any] = {
        "expected_hash": args.expected_hash,
        "expected_hash_match": None,
    }
    measurements: dict[str, Any]
    status = "pass"

    try:
        if config.profile == "correctness":
            measurements = run_correctness(torch, config, device=device, dtype=dtype)
            checks["repeat_match"] = measurements["repeat_match"]
            checks["output_hash"] = measurements["output_hash"]
            if not measurements["repeat_match"]:
                status = "fail"
            if args.expected_hash:
                expected_match = measurements["output_hash"] == args.expected_hash
                checks["expected_hash_match"] = expected_match
                if not expected_match:
                    status = "fail"
        else:
            measurements = run_performance(torch, config, device=device, dtype=dtype)
            checks["output_hash"] = measurements["output_hash"]
            if args.expected_hash:
                expected_match = measurements["output_hash"] == args.expected_hash
                checks["expected_hash_match"] = expected_match
                if not expected_match:
                    status = "fail"
    except Exception as error:
        status = "fail"
        errors.append(
            {
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        measurements = {}

    finished_at = utc_now()
    result = ProbeResult(
        schema_version=SCHEMA_VERSION,
        status=status,  # type: ignore[arg-type]
        profile=config.profile,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm=slurm_context(),
        command={
            "argv": sys.argv,
            "device": args.device,
            "logical_device": args.logical_device,
            "deterministic_env_applied": deterministic_env_applied,
        },
        probe_config=config.to_dict(),
        probe_config_hash=config_hash,
        fingerprint=fingerprint,
        fingerprint_hash=fingerprint_hash,
        checks=checks,
        measurements=measurements,
        errors=errors,
    )
    return result


def default_output_path(result: ProbeResult, out_dir: Path) -> Path:
    rank = result.slurm.get("slurm_procid") or "local"
    local_rank = result.slurm.get("slurm_localid") or "0"
    filename = (
        f"{result.hostname}-rank{rank}-local{local_rank}-pid{result.pid}-"
        f"{result.run_id}.json"
    )
    return out_dir / filename


def handle_run_local(args: argparse.Namespace) -> None:
    result = run_probe(args)
    payload = result.to_dict()
    if args.out_dir:
        output_path = default_output_path(result, Path(args.out_dir))
        payload["output_path"] = str(output_path)
        write_json(output_path, payload)

    context = nullcontext(args.outfile) if args.outfile is sys.stdout else args.outfile
    with context as outfile:
        dump_json(outfile, payload)

    if result.status == "fail":
        raise SystemExit(1)


def generate_run_local_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a deterministic PyTorch probe on the assigned local device",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        choices=("correctness", "performance"),
        default="correctness",
        help="Probe profile to run.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device family to probe.",
    )
    parser.add_argument(
        "--logical-device",
        type=int,
        default=0,
        help="Logical CUDA device index when --device=cuda.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Probe seed.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Fresh repeated runs for the correctness profile.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="fp32",
        help="Computation dtype. Use auto for bf16/fp16 performance on CUDA.",
    )
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--sequence-length", type=int, help="Override sequence length.")
    parser.add_argument("--vocab-size", type=int, help="Override vocabulary size.")
    parser.add_argument("--d-model", type=int, help="Override model width.")
    parser.add_argument("--num-layers", type=int, help="Override layer count.")
    parser.add_argument("--num-heads", type=int, help="Override attention head count.")
    parser.add_argument("--mlp-ratio", type=int, help="Override MLP expansion ratio.")
    parser.add_argument("--steps", type=int, help="Override measured training steps.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="Override warmup steps for the performance profile.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow TF32 matmul/cudnn kernels. Disabled by default for strictness.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Disable torch deterministic algorithm enforcement.",
    )
    parser.add_argument(
        "--expected-hash",
        type=str,
        help="Expected probe output hash for this execution fingerprint.",
    )
    parser.add_argument("--run-id", type=str, help="Override generated run id.")
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Directory to write the per-device JSON result.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file for JSON result.",
    )
    parser.set_defaults(func=handle_run_local)
    return parser
