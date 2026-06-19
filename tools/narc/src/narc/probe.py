from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from narc.checksum import stable_hash, tensor_hash
from narc.env import (
    collect_fingerprint,
    prepare_deterministic_environment,
    slurm_context,
)
from narc.files import (
    ResultLocation,
    is_s3_uri,
    parse_s3_uri,
    require_s3_object_uri,
    s3_uri,
    write_json_report,
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


def safe_filename_component(value: Any) -> str:
    text = str(value)
    safe = "".join(
        character if character.isalnum() or character == "-" else "-"
        for character in text
    )
    return safe.strip("-") or "unknown"


def output_device_id(result: ProbeResult) -> Any:
    device = result.fingerprint.get("device")
    if not isinstance(device, dict):
        return result.run_id
    accelerator_id = device.get("accelerator_id")
    if accelerator_id:
        return accelerator_id
    pci_bus_id = device.get("pci_bus_id")
    if pci_bus_id:
        return pci_bus_id
    device_type = device.get("type")
    logical_index = device.get("logical_index")
    if device_type is not None and logical_index is not None:
        return f"{device_type}-{logical_index}"
    return result.run_id


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


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


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
    generator = torch.Generator()
    generator.manual_seed(config.input_seed)
    shape = (config.batch_size, config.sequence_length)
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=shape,
        dtype=torch.long,
        device="cpu",
        generator=generator,
    ).to(device=device)
    labels = torch.randint(
        low=0,
        high=config.vocab_size,
        size=shape,
        dtype=torch.long,
        device="cpu",
        generator=generator,
    ).to(device=device)
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


def input_hash(input_ids: Any, labels: Any) -> str:
    return hash_named_tensors(
        [
            ("input_ids", input_ids),
            ("labels", labels),
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


def timed_call(
    torch: Any,
    device: Any,
    timed: bool,
    operation: Any,
) -> tuple[Any, float | None]:
    if not timed:
        return operation(), None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        value = operation()
        end_event.record()
        torch.cuda.synchronize(device)
        return value, start_event.elapsed_time(end_event) / 1000.0
    start_time = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - start_time


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def timing_summary(
    step_timings: list[dict[str, float]],
    config: ProbeConfig,
) -> dict[str, Any]:
    step_seconds = [entry["step_seconds"] for entry in step_timings]
    elapsed = sum(step_seconds)
    token_count = config.batch_size * config.sequence_length * len(step_seconds)
    timing: dict[str, Any] = {
        "steps": step_timings,
        "step_seconds": step_seconds,
        "mean_step_seconds": mean(step_seconds),
        "elapsed_seconds": elapsed,
        "tokens_per_second": token_count / elapsed if elapsed else None,
    }
    for key in (
        "forward_seconds",
        "loss_seconds",
        "backward_seconds",
        "optimizer_step_seconds",
    ):
        values = [entry[key] for entry in step_timings]
        timing[key] = values
        timing[f"mean_{key}"] = mean(values)
    return timing


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
    generated_input_hash = input_hash(input_ids, labels)
    learning_rate = 1e-3
    losses: list[str] = []
    step_timings: list[dict[str, float]] = []
    final_logits = None
    final_grad_hash = None

    for _ in range(steps):
        model.zero_grad(set_to_none=True)

        logits, forward_seconds = timed_call(
            torch,
            device,
            timed,
            lambda: model(input_ids),
        )
        loss, loss_seconds = timed_call(
            torch,
            device,
            timed,
            lambda: manual_cross_entropy(torch, logits, labels),
        )
        if not bool(torch.isfinite(loss).detach().cpu().item()):
            raise RuntimeError(f"loss is not finite: {loss.detach().cpu().item()}")
        _, backward_seconds = timed_call(
            torch,
            device,
            timed,
            loss.backward,
        )
        final_grad_hash = gradient_hash(model)

        def optimizer_step() -> None:
            with torch.no_grad():
                manual_sgd_step(model, learning_rate=learning_rate)

        if timed:
            _, optimizer_step_seconds = timed_call(
                torch,
                device,
                timed,
                optimizer_step,
            )
            step_timing = {
                "forward_seconds": forward_seconds or 0.0,
                "loss_seconds": loss_seconds or 0.0,
                "backward_seconds": backward_seconds or 0.0,
                "optimizer_step_seconds": optimizer_step_seconds or 0.0,
            }
            step_timing["step_seconds"] = sum(step_timing.values())
            step_timings.append(step_timing)
        else:
            optimizer_step()

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
    output["input_hash"] = generated_input_hash
    if step_timings:
        output["timing"] = timing_summary(step_timings, config)
    return output


def argument_or_default(value: int | None, default: int) -> int:
    return default if value is None else value


def validate_config(config: ProbeConfig) -> None:
    positive_fields = {
        "batch_size": config.batch_size,
        "d_model": config.d_model,
        "mlp_ratio": config.mlp_ratio,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "repeat": config.repeat,
        "steps": config.steps,
    }
    for name, value in positive_fields.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if config.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if config.vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be at least 0")
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")


def run_measurements(
    torch: Any,
    config: ProbeConfig,
    *,
    device: Any,
    dtype: Any,
) -> dict:
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runs = [
        run_training_steps(
            torch,
            config,
            device=device,
            dtype=dtype,
            seed=config.seed,
            steps=config.steps,
            timed=True,
        )
        for _ in range(config.repeat)
    ]
    output_hashes = [run["output_hash"] for run in runs]
    input_hashes = [run["input_hash"] for run in runs]
    return {
        "repeat_match": len(set(output_hashes)) == 1,
        "output_hash": output_hashes[0],
        "input_repeat_match": len(set(input_hashes)) == 1,
        "input_hash": input_hashes[0],
        "memory": cuda_memory(torch, device),
        "runs": runs,
    }


def default_config(args: argparse.Namespace, dtype_name: str) -> ProbeConfig:
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
    return ProbeConfig(
        seed=args.seed,
        input_seed=args.input_seed,
        batch_size=argument_or_default(args.batch_size, defaults["batch_size"]),
        sequence_length=argument_or_default(
            args.sequence_length,
            defaults["sequence_length"],
        ),
        vocab_size=argument_or_default(args.vocab_size, defaults["vocab_size"]),
        d_model=argument_or_default(args.d_model, defaults["d_model"]),
        num_layers=argument_or_default(args.num_layers, defaults["num_layers"]),
        num_heads=argument_or_default(args.num_heads, defaults["num_heads"]),
        mlp_ratio=argument_or_default(args.mlp_ratio, defaults["mlp_ratio"]),
        steps=argument_or_default(args.steps, defaults["steps"]),
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
    validate_config(config)
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
        measurements = run_measurements(torch, config, device=device, dtype=dtype)
        checks["repeat_match"] = measurements["repeat_match"]
        checks["output_hash"] = measurements["output_hash"]
        checks["input_repeat_match"] = measurements["input_repeat_match"]
        checks["input_hash"] = measurements["input_hash"]
        if not measurements["repeat_match"]:
            status = "fail"
        if not measurements["input_repeat_match"]:
            status = "fail"
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
    device_id = safe_filename_component(output_device_id(result))
    rank = safe_filename_component(result.slurm.get("slurm_procid") or "local")
    local_rank = safe_filename_component(result.slurm.get("slurm_localid") or "0")
    run_id = safe_filename_component(result.run_id)
    filename = (
        f"{device_id}-rank{rank}-local{local_rank}-pid{result.pid}-"
        f"{run_id}.json"
    )
    return out_dir / filename


def default_output_filename(result: ProbeResult) -> str:
    return default_output_path(result, Path(".")).name


def default_output_location(
    result: ProbeResult,
    out_dir: ResultLocation,
) -> ResultLocation:
    filename = default_output_filename(result)
    if is_s3_uri(out_dir):
        bucket, prefix = parse_s3_uri(str(out_dir))
        key = f"{prefix.rstrip('/')}/{filename}" if prefix else filename
        return s3_uri(bucket, key)
    return Path(out_dir) / filename


def write_probe_result(location: ResultLocation, payload: dict[str, Any]) -> None:
    if is_s3_uri(location):
        write_json_report(location, payload)
        return
    write_json(Path(location), payload)


def write_result_to_outfile(outfile: str, payload: dict[str, Any]) -> None:
    if outfile == "-":
        dump_json(sys.stdout, payload)
        return
    if is_s3_uri(outfile):
        write_json_report(outfile, payload)
        return
    write_json(Path(outfile), payload)


def validate_run_outfile(outfile: str) -> None:
    if outfile == "-":
        return
    if is_s3_uri(outfile):
        require_s3_object_uri(outfile)
        return
    output_path = Path(outfile)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("outfile must not be an existing directory")
    parent = output_path.parent
    if parent.exists() and not parent.is_dir():
        raise ValueError("outfile parent must be a directory")


def handle_run(args: argparse.Namespace) -> None:
    validate_run_outfile(args.outfile)
    result = run_probe(args)
    payload = result.to_dict()
    if args.out_dir:
        output_location = default_output_location(result, args.out_dir)
        payload["output_path"] = str(output_location)
        write_probe_result(output_location, payload)

    write_result_to_outfile(args.outfile, payload)

    if result.status == "fail":
        raise SystemExit(1)


def generate_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a deterministic PyTorch probe on the assigned local device",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device family to probe.",
    )
    parser.add_argument(
        "--logical-device",
        type=non_negative_int,
        default=0,
        help="Logical CUDA device index when --device=cuda.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Probe seed.")
    parser.add_argument(
        "--input-seed",
        type=int,
        default=0,
        help="Seed for deterministic randomized input token IDs and labels.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Fresh repeated measured runs.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="fp32",
        help="Computation dtype. Use auto for bf16/fp16 on CUDA.",
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
        help="Warmup steps to run before the measured repeats.",
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
        help="Local directory or S3 prefix to write the per-device JSON result.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="-",
        help="Output file, S3 URI, or '-' for stdout.",
    )
    parser.set_defaults(func=handle_run)
    return parser
