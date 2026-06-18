from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass, asdict
from io import StringIO
from typing import Any


@dataclass(frozen=True)
class SlurmNode:
    name: str
    state: str
    cpus: str
    memory_mb: str
    gres: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_command(args: list[str]) -> str:
    return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()


def list_nodes() -> list[SlurmNode]:
    output = run_command(
        [
            "sinfo",
            "--Node",
            "--noheader",
            "--Format=%N|%T|%c|%m|%G",
        ]
    )
    nodes: list[SlurmNode] = []
    reader = csv.reader(StringIO(output), delimiter="|")
    for row in reader:
        if len(row) != 5:
            continue
        nodes.append(
            SlurmNode(
                name=row[0].strip(),
                state=row[1].strip(),
                cpus=row[2].strip(),
                memory_mb=row[3].strip(),
                gres=row[4].strip(),
            )
        )
    return nodes
