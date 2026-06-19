from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SCHEMA_VERSION = 1

ProbeStatus = Literal["pass", "fail", "warn"]
ProbeProfile = Literal["correctness", "performance"]


@dataclass(frozen=True)
class ProbeConfig:
    profile: ProbeProfile
    seed: int
    input_seed: int
    batch_size: int
    sequence_length: int
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    mlp_ratio: int
    steps: int
    warmup_steps: int
    dtype: str
    repeat: int
    deterministic: bool
    allow_tf32: bool
    optimizer: str = "manual_sgd"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeResult:
    schema_version: int
    status: ProbeStatus
    profile: ProbeProfile
    run_id: str
    started_at: str
    finished_at: str
    hostname: str
    pid: int
    slurm: dict[str, Any]
    command: dict[str, Any]
    probe_config: dict[str, Any]
    probe_config_hash: str
    fingerprint: dict[str, Any]
    fingerprint_hash: str
    checks: dict[str, Any]
    measurements: dict[str, Any]
    errors: list[dict[str, Any]] = field(default_factory=list)
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
