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


def _datasets(root: Path, data_dir: Path, n_places: int, filter_step: int):
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
        datasets[season] = dataset
        for place, (_, row) in enumerate(dataset.img_labels.iterrows()):
            path = Path(row["file_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            manifest.append({"place": place, "season": season,
                             "file": path.name, "csv_index": int(row.iloc[1]),
                             "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    for place in range(n_places):
        names = [manifest[i * n_places + place]["file"] for i in range(3)]
        indices = [manifest[i * n_places + place]["csv_index"] for i in range(3)]
        if len(set(names)) != 1 or len(set(indices)) != 1:
            raise ValueError(f"season alignment fails at place {place}: {names}/{indices}")
    return datasets, manifest, csv_hashes


def load_pixels(root: Path, data_dir: Path, n_places: int, filter_step: int,
                cache_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    datasets, manifest, csv_hashes = _datasets(root, data_dir, n_places, filter_step)
    identity = {"dataset": "Nordland", "data_dir": str(data_dir.resolve()),
                "n_places": n_places, "filter": filter_step, "dims": [28, 28],
                "patches": 7, "patch_norm": True, "csv_sha256": csv_hashes,
                "manifest_sha256": digest_json(manifest), "manifest": manifest}
    source_hash = digest_file(Path(__file__))
    key = digest_json({"identity": identity["manifest_sha256"],
                       "csv": csv_hashes, "source": source_hash, "filter": filter_step})
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"pixel_{n_places}_{key[:16]}.pt"
    if cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=True)
        pixels = data["pixels"]
    else:
        pixels = {}
        for season, dataset in datasets.items():
            items = [dataset[i][0].float() for i in tqdm(range(n_places), desc=f"read {season}")]
            pixels[season] = torch.stack(items).contiguous()
        torch.save({"pixels": pixels}, cache)
    for season in SEASONS:
        if pixels[season].shape != (n_places, 784) or not torch.isfinite(pixels[season]).all():
            raise ValueError(f"invalid pixel cache for {season}: {pixels[season].shape}")
    identity["pixel_cache"] = str(cache)
    return pixels, identity


def load_gabor(root: Path, pixels: dict[str, torch.Tensor], cache_dir: Path,
               identity: dict, seed: int = 0, conv_epochs: int = 4) -> tuple[dict[str, torch.Tensor], dict]:
    """Build block2 from spring/fall only, then freeze before backend learning.

    No block4 threshold state is transplanted. This is a new recorded frontend,
    not a reproduction of an unavailable historical block2 checkpoint.
    """
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, TensorDataset
    from vprtempo.src import conv_frontend as cf

    settings = {"block": 2, "conv_epochs": conv_epochs, "seed": seed,
                "source": "fresh_block2_gabor_train_only_itp"}
    key = digest_json({"pixel": identity["manifest_sha256"],
                       "csv": identity["csv_sha256"], "settings": settings,
                       "source": digest_file(Path(__file__))})
    cache = cache_dir / f"gabor_block2_{identity['n_places']}_{key[:16]}.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        features, frozen_state = saved["features"], saved["frontend_state"]
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cpu")
        spec = SimpleNamespace(frontend="gabor", conv_channels=32, conv_kernel=5,
                               wta_mode="local", wta_block=2)
        layer = cf.build_conv_layer(spec, [28, 28], device, inference=False)
        original_weight = layer.w.weight.detach().clone()
        training = torch.cat((pixels["spring"], pixels["fall"]))
        loader = DataLoader(TensorDataset(training, torch.zeros(len(training))),
                            batch_size=1, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
        cf.train_conv_layer(loader, layer,
                            SimpleNamespace(device=device, conv_epoch=conv_epochs))
        if not torch.equal(layer.w.weight, original_weight):
            raise RuntimeError("fixed Gabor weights changed during threshold adaptation")
        layer.requires_grad_(False)
        layer.eval()
        frozen_state = {name: value.detach().cpu().clone()
                        for name, value in layer.state_dict().items()}
        features = {}
        for season in SEASONS:
            chunks = [cf.conv_forward(layer, sample.unsqueeze(0)).cpu() for sample in
                      tqdm(pixels[season], desc=f"block2 gabor {season}")]
            features[season] = torch.cat(chunks).contiguous()
        torch.save({"features": features, "frontend_state": frozen_state,
                    "settings": settings}, cache)
    for season in SEASONS:
        if features[season].shape != (identity["n_places"], 4608):
            raise ValueError(f"invalid block2 Gabor cache for {season}")
    hasher = hashlib.sha256()
    for name, tensor in sorted(frozen_state.items()):
        hasher.update(name.encode())
        hasher.update(tensor.contiguous().numpy().tobytes())
    return features, {**identity, "gabor_state_sha256": hasher.hexdigest(),
                      "gabor_cache": str(cache), "wta_block": 2,
                      "gabor_state_source": settings["source"],
                      "gabor_conv_epochs": conv_epochs, "gabor_frontend_seed": seed}


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
            legal = [p for p in range(n_places) if abs(p - place) >= negative_min_gap]
            if index % 2 == 0:
                negative = rng.choice(legal)
            else:
                candidates = rng.sample(legal, min(16, len(legal)))
                other = "fall" if season == 0 else "spring"
                anchor = initial["spring" if season == 0 else "fall"][place]
                negative = max(candidates, key=lambda p: (float(anchor @ initial[other][p]), -p))
            triples.append((season, place, negative))
        all_epochs.append(triples)
    return all_epochs, digest_json(all_epochs)


def _rank(db: torch.Tensor, q: torch.Tensor, n_places: int) -> dict:
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
    for query in range(n_places):
        if not bool(valid_q[query]):
            predictions.append({"query": query, "gt": query, "invalid_query": True,
                                "image_top5": [], "place_top5": []})
            continue
        image_order = np.argsort(-array[:, query], kind="stable")
        place_order = np.argsort(-place_scores[:, query], kind="stable")
        image_order = [int(i) for i in image_order if np.isfinite(array[i, query])]
        place_order = [int(i) for i in place_order if np.isfinite(place_scores[i, query])]
        if place_order:
            tie_count += int(np.sum(place_scores[:, query] == place_scores[place_order[0], query]) > 1)
        for k in (1, 5):
            image_hits[k] += int(any(i % n_places == query for i in image_order[:k]))
            place_hits[k] += int(query in place_order[:k])
        predictions.append({"query": query, "gt": query, "invalid_query": False,
                            "image_top5": [{"row": i, "place": i % n_places,
                                             "score": float(array[i, query])} for i in image_order[:5]],
                            "place_top5": [{"place": i, "score": float(place_scores[i, query])}
                                            for i in place_order[:5]]})
    return {"place_R@1": place_hits[1] / n_places, "place_R@5": place_hits[5] / n_places,
            "image_R@1": image_hits[1] / n_places, "image_R@5": image_hits[5] / n_places,
            "invalid_reference_rows": int((~valid_db).sum()),
            "invalid_queries": int((~valid_q).sum()), "top1_tied_queries": tie_count,
            "predictions": predictions}


def evaluate(model: TaskFeature | None, inputs: dict[str, torch.Tensor], n_places: int) -> dict:
    if model is None:
        features = {k: v for k, v in inputs.items()}
        diagnostics = {}
    else:
        batches = {k: descriptors(model, v) for k, v in inputs.items()}
        features = {k: value[0] for k, value in batches.items()}
        diagnostics = {k: value[2] for k, value in batches.items()}
    db = torch.cat((features["spring"], features["fall"]))
    result = _rank(db, features["summer"], n_places)
    result["activity"] = diagnostics
    return result
