from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def tensor_hash(tensor: Any) -> str:
    data = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(data.shape)).encode("utf-8"))
    digest.update(str(data.dtype).encode("utf-8"))
    digest.update(str(data.device.type).encode("utf-8"))
    digest.update(bytes(data.untyped_storage()))
    return digest.hexdigest()
