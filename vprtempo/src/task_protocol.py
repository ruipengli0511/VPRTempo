"""MIT licensed helpers for query roles and immutable Linux experiment caches."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import torch


def query_ids_from_config(cfg: dict) -> list[int]:
    """Resolve explicit map IDs; retained queries require a later locked evaluator."""
    n = int(cfg['n_places'])
    if cfg.get('query_role', 'development') != 'development':
        raise ValueError('training/search accepts development queries only; retained evaluation is locked')
    ids = cfg.get('query_ids')
    if ids is None:
        start = int(cfg.get('query_start', 0))
        ids = list(range(start, start + int(cfg.get('query_count', n))))
    if not ids or any(type(i) is not int or not 0 <= i < n for i in ids):
        raise ValueError('query IDs must be nonempty integer map IDs inside the reference map')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate query ID')
    return list(ids)


def cached_tensor(path: Path, builder, read_only: bool = False):
    """Atomically publish one cache under an interprocess Linux file lock."""
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True)
    if read_only:
        raise FileNotFoundError(f'input cache missing; run the preparation stage first: {path}')
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=True)
        value = builder()
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            torch.save(value, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return value
