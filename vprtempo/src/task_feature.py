"""Shared single-frame task-feature experiment on the VPRTempo abstract layer.

This module deliberately leaves the original VPRTempo training path untouched.
Only the method-specific update lives on the two experimental branches.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from vprtempo.src import blitnet as bn
from vprtempo.src.dataset import CustomImageDataset, ProcessImage
from vprtempo.src.task_protocol import cached_tensor


SEASONS = ("spring", "fall", "summer")
CONV_STATE = Path("IDEA1-covstdp/handoff/conv_only_state_b5_500__seed0.json")


def digest_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def git_identity(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain=v1", "-uno"))}


def _datasets(root: Path, data_dir: Path, n_places: int, filter_step: int, query_ids=None):
    query_ids = list(range(n_places)) if query_ids is None else list(query_ids)
    transform = ProcessImage([28, 28], 7, patch_norm=True)
    datasets = {}
    manifest = []
    csv_hashes = {}
    for season in SEASONS:
        csv_path = root / "vprtempo" / "dataset" / f"nordland-{season}.csv"
        csv_hashes[season] = digest_file(csv_path)
        dataset = CustomImageDataset(str(csv_path), str(data_dir), [season],
                                     transform=transform, filter=filter_step,
                                     skip=0, max_samples=n_places, test=True)
        if len(dataset) != n_places:
            raise ValueError(f"{season}: expected {n_places} samples, got {len(dataset)}")
        ids = query_ids if season == 'summer' else list(range(n_places))
        dataset.img_labels = dataset.img_labels.iloc[ids].reset_index(drop=True)
        datasets[season] = dataset
        for place, (_, row) in zip(ids, dataset.img_labels.iterrows()):
            path = Path(row["file_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            manifest.append({"place": place, "season": season,
                             "file": path.name, "csv_index": int(row.iloc[1]),
                             "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    records = {(row['season'], row['place']): row for row in manifest}
    query_set = set(query_ids)
    for place in range(n_places):
        aligned_seasons = SEASONS if place in query_set else ('spring', 'fall')
        names = [records[s, place]['file'] for s in aligned_seasons]
        indices = [records[s, place]['csv_index'] for s in aligned_seasons]
        if len(set(names)) != 1 or len(set(indices)) != 1:
            raise ValueError(f"season alignment fails at place {place}: {names}/{indices}")
    return datasets, manifest, csv_hashes


def load_pixels(root: Path, data_dir: Path, n_places: int, filter_step: int,
                cache_dir: Path, query_ids=None, read_only=False) -> tuple[dict[str, torch.Tensor], dict]:
    query_ids = list(range(n_places)) if query_ids is None else list(query_ids)
    datasets, manifest, csv_hashes = _datasets(root, data_dir, n_places, filter_step, query_ids)
    identity = {"dataset": "Nordland", "data_dir": str(data_dir.resolve()),
                "n_places": n_places, "filter": filter_step, "dims": [28, 28],
                "patches": 7, "patch_norm": True, "csv_sha256": csv_hashes,
                "manifest_sha256": digest_json(manifest), "manifest": manifest,
                "query_ids": query_ids, "query_role": "development"}
    source_hash = digest_json({p: digest_file(root / p) for p in (
        'vprtempo/src/task_feature.py', 'vprtempo/src/dataset.py')})
    key = digest_json({"identity": identity["manifest_sha256"],
                       "csv": csv_hashes, "source": source_hash, "filter": filter_step})
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"pixel_{n_places}_{key[:16]}.pt"
    def build():
        pixels = {}
        for season, dataset in datasets.items():
            items = [dataset[i][0].float() for i in tqdm(range(len(dataset)), desc=f"read {season}")]
            pixels[season] = torch.stack(items).contiguous()
        return {"pixels": pixels}
    pixels = cached_tensor(cache, build, read_only)["pixels"]
    for season in SEASONS:
        rows = len(query_ids) if season == 'summer' else n_places
        if pixels[season].shape != (rows, 784) or not torch.isfinite(pixels[season]).all():
            raise ValueError(f"invalid pixel cache for {season}: {pixels[season].shape}")
    identity["pixel_cache"] = str(cache)
    return pixels, identity


def load_gabor(root: Path, pixels: dict[str, torch.Tensor], cache_dir: Path,
               identity: dict, read_only=False) -> tuple[dict[str, torch.Tensor], dict]:
    from vprtempo.src import conv_frontend as cf

    state_path = root / CONV_STATE
    state_sha = digest_file(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))["conv_state"]
    key = digest_json({"pixel": identity["manifest_sha256"],
                       "csv": identity["csv_sha256"], "state": state_sha,
                       "source": {p: digest_file(root / p) for p in (
                           'vprtempo/src/task_feature.py', 'vprtempo/src/dataset.py',
                           'vprtempo/src/conv_frontend.py', 'IDEA1-covstdp/src/conv_snn_layer.py')}})
    cache = cache_dir / f"gabor_{identity['n_places']}_{key[:16]}.pt"
    def build():
        layer = cf.ConvSNNLayer(input_dims=[28, 28], in_channels=1,
                                out_channels=32, kernel_size=5, wta_mode="local",
                                wta_block=4, device=torch.device("cpu"),
                                inference=True, frozen=True)
        weights = torch.tensor(state["weights"]["values"], dtype=torch.float32).unsqueeze(1)
        threshold = torch.tensor(state["thresholds"]["values"], dtype=torch.float32).view(1, 32, 1, 1)
        polarity = torch.tensor(state["on_off_mask"]["values"], dtype=torch.bool)
        if weights.shape != (32, 1, 5, 5) or threshold.shape != (1, 32, 1, 1):
            raise ValueError("unexpected B5 conv-only state shape")
        with torch.no_grad():
            layer.w.weight.copy_(weights)
            layer.thr.copy_(threshold)
            layer.havconnExc.copy_(polarity)
        layer.eval()
        features = {}
        for season in SEASONS:
            # The existing ConvSNNLayer.reshape_input is intentionally single-image.
            chunks = [cf.conv_forward(layer, sample.unsqueeze(0)).cpu() for sample in
                      tqdm(pixels[season], desc=f"gabor {season}")]
            features[season] = torch.cat(chunks).contiguous()
        return {"features": features}
    features = cached_tensor(cache, build, read_only)["features"]
    for season in SEASONS:
        rows = len(identity.get('query_ids', range(identity['n_places']))) if season == 'summer' else identity['n_places']
        if features[season].shape != (rows, 1152) or not torch.isfinite(features[season]).all():
            raise ValueError(f"invalid Gabor cache for {season}")
    return features, {**identity, "gabor_state_sha256": state_sha,
                      "gabor_cache": str(cache), "wta_block": 4}


class TaskFeature(nn.Module):
    """One VPRTempo SNNLayer with a fixed threshold and no output layer."""

    def __init__(self, input_dim: int, width: int, seed: int, device: torch.device):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.layer = bn.SNNLayer(dims=[input_dim, width], thr_range=[0, 0],
                                 fire_rate=[0, 0], ip_rate=0, stdp_rate=0,
                                 p=[0.1, 0.5], device=device, inference=False)
        self.layer.thr.requires_grad_(False)
        self.register_buffer("exc_mask", self.layer.havconnCombinedExc.detach().clone())
        self.register_buffer("inh_mask", self.layer.havconnCombinedInh.detach().clone())
        self.register_buffer("connection_mask", self.exc_mask | self.inh_mask)
        self.to(device)
        self.project()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = self.layer.w(x) - self.layer.thr
        h = pre.clamp(0.0, 0.9)
        norm = h.norm(dim=1, keepdim=True)
        valid = norm[:, 0] > 1e-8
        desc = h / norm.clamp_min(1e-8)
        return desc, h, pre, valid

    @torch.no_grad()
    def project(self) -> dict[str, int]:
        w = self.layer.w.weight
        clipped = int(((w[self.exc_mask] < 1e-6) | (w[self.exc_mask] > 10)).sum()
                      + ((w[self.inh_mask] < -10) | (w[self.inh_mask] > -1e-6)).sum())
        w[self.exc_mask] = w[self.exc_mask].clamp(1e-6, 10.0)
        w[self.inh_mask] = w[self.inh_mask].clamp(-10.0, -1e-6)
        w[~self.connection_mask] = 0.0
        return {"projected_connections": clipped}

    def initial_state(self) -> dict:
        return {"weight": self.layer.w.weight.detach().cpu().clone(),
                "threshold": self.layer.thr.detach().cpu().clone(),
                "exc_mask": self.exc_mask.cpu().clone(),
                "inh_mask": self.inh_mask.cpu().clone()}


@torch.no_grad()
def descriptors(model: TaskFeature, x: torch.Tensor, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor, dict]:
    descs, valid = [], []
    activity, saturation = [], []
    device = model.layer.w.weight.device
    for part in x.split(batch_size):
        d, h, pre, v = model.encode(part.to(device))
        descs.append(d.cpu())
        valid.append(v.cpu())
        activity.append(float((h > 0).float().mean()))
        saturation.append(float((pre >= 0.9).float().mean()))
    return torch.cat(descs), torch.cat(valid), {
        "active_fraction": float(np.mean(activity)),
        "saturated_fraction": float(np.mean(saturation))}


def schedule(n_places: int, epochs: int, seed: int,
             initial: dict[str, torch.Tensor], negative_min_gap: int = 50) -> tuple[list[list[tuple[int, int, int]]], str]:
    """Each tuple is (anchor season: 0 spring/1 fall, place, negative place)."""
    if negative_min_gap < 1 or n_places < 2 * negative_min_gap:
        raise ValueError("need at least 2 * negative_min_gap places")
    rng = random.Random(seed)
    all_epochs = []
    for _ in range(epochs):
        anchors = [(season, place) for season in (0, 1) for place in range(n_places)]
        rng.shuffle(anchors)
        triples = []
        for index, (season, place) in enumerate(anchors):
            # Same sorted legal set and RNG draws without materializing O(N) per triplet.
            left = max(0, place - negative_min_gap + 1)
            right = min(n_places, place + negative_min_gap)
            count = left + n_places - right
            to_place = lambda rank: rank if rank < left else right + rank - left
            if index % 2 == 0:
                negative = to_place(rng.choice(range(count)))
            else:
                candidates = [to_place(i) for i in rng.sample(range(count), min(16, count))]
                other = "fall" if season == 0 else "spring"
                anchor = initial["spring" if season == 0 else "fall"][place]
                negative = max(candidates, key=lambda p: (float(anchor @ initial[other][p]), -p))
            triples.append((season, place, negative))
        all_epochs.append(triples)
    return all_epochs, digest_json(all_epochs)


def _rank(db: torch.Tensor, q: torch.Tensor, n_places: int, query_ids=None) -> dict:
    query_ids = list(range(n_places)) if query_ids is None else list(query_ids)
    if db.shape[0] != 2 * n_places or q.shape[0] != len(query_ids):
        raise ValueError('reference/query shape does not match map IDs')
    if not query_ids or len(set(query_ids)) != len(query_ids) or any(not 0 <= i < n_places for i in query_ids):
        raise ValueError('invalid or duplicate query ground truth IDs')
    db = db.float()
    q = q.float()
    db_norm = db.norm(dim=1)
    q_norm = q.norm(dim=1)
    valid_db = db_norm > 1e-8
    valid_q = q_norm > 1e-8
    scores = (db / db_norm[:, None].clamp_min(1e-8)) @ (q / q_norm[:, None].clamp_min(1e-8)).T
    array = scores.numpy()
    array[~valid_db.numpy(), :] = -np.inf
    place_scores = np.maximum(array[:n_places], array[n_places:])
    predictions = []
    image_hits = {1: 0, 5: 0}
    place_hits = {1: 0, 5: 0}
    tie_count = 0
    for query, gt in enumerate(query_ids):
        if not bool(valid_q[query]):
            predictions.append({"query": query, "query_place_id": gt, "gt": gt, "invalid_query": True,
                                "image_top5": [], "place_top5": []})
            continue
        image_order = np.argsort(-array[:, query], kind="stable")
        place_order = np.argsort(-place_scores[:, query], kind="stable")
        image_order = [int(i) for i in image_order if np.isfinite(array[i, query])]
        place_order = [int(i) for i in place_order if np.isfinite(place_scores[i, query])]
        if place_order:
            tie_count += int(np.sum(place_scores[:, query] == place_scores[place_order[0], query]) > 1)
        for k in (1, 5):
            image_hits[k] += int(any(i % n_places == gt for i in image_order[:k]))
            place_hits[k] += int(gt in place_order[:k])
        predictions.append({"query": query, "query_place_id": gt, "gt": gt, "invalid_query": False,
                            "image_top5": [{"row": i, "place": i % n_places,
                                             "score": float(array[i, query])} for i in image_order[:5]],
                            "place_top5": [{"place": i, "score": float(place_scores[i, query])}
                                            for i in place_order[:5]]})
    return {"place_R@1": place_hits[1] / len(query_ids), "place_R@5": place_hits[5] / len(query_ids),
            "image_R@1": image_hits[1] / len(query_ids), "image_R@5": image_hits[5] / len(query_ids),
            "n_queries": len(query_ids),
            "invalid_reference_rows": int((~valid_db).sum()),
            "invalid_queries": int((~valid_q).sum()), "top1_tied_queries": tie_count,
            "predictions": predictions}


def evaluate(model: TaskFeature | None, inputs: dict[str, torch.Tensor], n_places: int, query_ids=None) -> dict:
    if model is None:
        features = {k: v for k, v in inputs.items()}
        diagnostics = {}
    else:
        batches = {k: descriptors(model, v) for k, v in inputs.items()}
        features = {k: value[0] for k, value in batches.items()}
        diagnostics = {k: value[2] for k, value in batches.items()}
    db = torch.cat((features["spring"], features["fall"]))
    result = _rank(db, features["summer"], n_places, query_ids)
    result["activity"] = diagnostics
    return result
