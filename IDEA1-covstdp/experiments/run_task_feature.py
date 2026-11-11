#!/usr/bin/env python3
"""Train and evaluate one task-feature update rule on aligned Nordland places.

The imported task_update module is the only method-specific file on each branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vprtempo.src import task_update  # noqa: E402
from vprtempo.src.task_feature import (TaskFeature, descriptors, evaluate,  # noqa: E402
                                       git_identity, load_gabor, load_pixels, schedule)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint(path: Path, model: TaskFeature, metadata: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "metadata": metadata}, temporary)
    os.replace(temporary, path)


def evaluate_epoch(run_dir: Path, epoch: int, model: TaskFeature,
                   inputs: dict, n_places: int) -> dict:
    result = evaluate(model, inputs, n_places)
    predictions = result.pop("predictions")
    write_json(run_dir / f"predictions_epoch_{epoch}.json", predictions)
    result["epoch"] = epoch
    print(f"epoch={epoch} place R@1={result['place_R@1']:.4f} "
          f"R@5={result['place_R@5']:.4f} image R@1={result['image_R@1']:.4f}", flush=True)
    return result


def args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frontend", choices=("gabor", "gray"), required=True)
    parser.add_argument("--n-places", type=int)
    parser.add_argument("--negative-min-gap", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-runtime-s", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-dirty", action="store_true", help="only for implementation smoke")
    return parser.parse_args()


def resolve_config(cli: argparse.Namespace) -> dict:
    cfg = json.loads(cli.config.read_text(encoding="utf-8"))
    for key in ("n_places", "negative_min_gap", "epochs", "batch_size", "lr", "margin", "seed",
                "max_updates", "max_runtime_s", "device"):
        value = getattr(cli, key)
        if value is not None:
            cfg[key] = value
    cfg["frontend"] = cli.frontend
    cfg["run_id"] = cli.run_id
    cfg["method"] = task_update.METHOD
    cfg["allow_dirty"] = cli.allow_dirty
    cfg["data_dir"] = str(Path(cfg["data_dir"]).expanduser().resolve())
    cfg["n_places"] = int(cfg["n_places"])
    cfg["feature_width"] = int(cfg["feature_width"])
    if not (7 <= cfg["n_places"] <= 500 and cfg["feature_width"] == 2304):
        raise ValueError("pilot requires 7..500 places and 2304 feature units")
    if cfg["batch_size"] < 1 or cfg["lr"] <= 0 or cfg["margin"] <= 0:
        raise ValueError("invalid batch size, learning rate or margin")
    return cfg


def main() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    cli = args_from_cli()
    cfg = resolve_config(cli)
    version = git_identity(ROOT)
    if version["dirty"] and not cli.allow_dirty:
        raise RuntimeError("commit source changes before a recorded run; use --allow-dirty for smoke")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    run_root = ROOT / "runs" / "task_feature"
    run_dir = run_root / cli.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + cfg["max_runtime_s"]
    write_json(run_dir / "effective_config.json", {"config": cfg, "git": version})

    cache_dir = run_root / "cache"
    pixels, identity = load_pixels(ROOT, Path(cfg["data_dir"]), cfg["n_places"],
                                   cfg["filter"], cache_dir)
    if cfg["frontend"] == "gabor":
        inputs, identity = load_gabor(ROOT, pixels, cache_dir, identity)
    else:
        inputs = pixels
    write_json(run_dir / "data_identity.json", identity)
    n = cfg["n_places"]
    in_dim = int(inputs["spring"].shape[1])
    cfg["effective_input_dim"] = in_dim
    cfg["gabor_block"] = 4 if cfg["frontend"] == "gabor" else None
    write_json(run_dir / "effective_config.json", {"config": cfg, "git": version})
    model = TaskFeature(in_dim, cfg["feature_width"], cfg["seed"], device)
    initial = model.initial_state()
    w0_sha = hashlib.sha256(initial["weight"].numpy().tobytes()).hexdigest()
    torch.save(initial, run_dir / "initial_weight_and_masks.pth")

    with torch.no_grad():
        d_spring = descriptors(model, inputs["spring"])[0]
        d_fall = descriptors(model, inputs["fall"])[0]
    triples, schedule_sha = schedule(n, cfg["epochs"], cfg["seed"],
                                     {"spring": d_spring, "fall": d_fall},
                                     negative_min_gap=cfg["negative_min_gap"])
    write_json(run_dir / "triplet_schedule.json", triples)
    write_json(run_dir / "run_identity.json", {"method": task_update.METHOD,
               "git": version, "initial_weight_sha256": w0_sha,
               "schedule_sha256": schedule_sha, "data_manifest_sha256": identity["manifest_sha256"],
               "gabor_state_sha256": identity.get("gabor_state_sha256")})

    results = {"method": task_update.METHOD, "frontend": cfg["frontend"],
               "seed": cfg["seed"], "n_places": n, "epoch_results": [],
               "update_diagnostics": [], "status": "running"}
    if cfg["frontend"] == "gabor":
        direct = evaluate(None, inputs, n)
        write_json(run_dir / "predictions_gabor_direct.json", direct.pop("predictions"))
        results["gabor_direct"] = direct
    results["random_feature"] = evaluate_epoch(run_dir, 0, model, inputs, n)
    results["epoch_results"].append(results["random_feature"])
    checkpoint(run_dir / "checkpoint_epoch_0.pth", model, {"epoch": 0, "config": cfg,
               "git": version, "initial_weight_sha256": w0_sha, "schedule_sha256": schedule_sha})
    write_json(run_dir / "summary.json", results)

    optimizer = SGD([model.layer.w.weight], lr=cfg["lr"], momentum=0.0, weight_decay=0.0)
    total_updates = 0
    stopped = False
    for epoch, epoch_triples in enumerate(triples, 1):
        totals = {"loss_sum": 0.0, "active_count": 0, "valid_count": 0,
                  "triplet_count": 0, "steps": 0, "update_norm_sum": 0.0,
                  "projected_connections": 0}
        for start in tqdm(range(0, len(epoch_triples), cfg["batch_size"]), desc=f"epoch {epoch}"):
            if time.monotonic() >= deadline or (cfg["max_updates"] is not None and
                                                total_updates >= cfg["max_updates"]):
                stopped = True
                break
            batch = epoch_triples[start:start + cfg["batch_size"]]
            anchor = torch.stack([inputs["spring" if s == 0 else "fall"][p]
                                  for s, p, _ in batch]).to(device)
            positive = torch.stack([inputs["fall" if s == 0 else "spring"][p]
                                    for s, p, _ in batch]).to(device)
            negative = torch.stack([inputs["fall" if s == 0 else "spring"][q]
                                    for s, _, q in batch]).to(device)
            report = task_update.step(model, anchor, positive, negative, optimizer,
                                      cfg["margin"])
            count = report["valid_count"]
            totals["loss_sum"] += report["loss"] * count
            totals["active_count"] += report["active_count"]
            totals["valid_count"] += count
            totals["triplet_count"] += len(batch)
            totals["steps"] += 1
            totals["update_norm_sum"] += report["update_norm"]
            totals["projected_connections"] += report["projected_connections"]
            total_updates += 1
        if totals["steps"]:
            totals["mean_loss"] = totals["loss_sum"] / max(1, totals["valid_count"])
            totals["active_fraction"] = totals["active_count"] / max(1, totals["valid_count"])
            totals["valid_fraction"] = totals["valid_count"] / max(1, totals["triplet_count"])
            totals["mean_update_norm"] = totals["update_norm_sum"] / totals["steps"]
            totals["epoch"] = epoch
            results["update_diagnostics"].append(totals)
            results["epoch_results"].append(evaluate_epoch(run_dir, epoch, model, inputs, n))
            checkpoint(run_dir / f"checkpoint_epoch_{epoch}.pth", model,
                       {"epoch": epoch, "config": cfg, "git": version,
                        "initial_weight_sha256": w0_sha, "schedule_sha256": schedule_sha})
            results["elapsed_s"] = time.monotonic() - started
            results["max_gpu_memory_bytes"] = (torch.cuda.max_memory_allocated()
                                               if device.type == "cuda" else 0)
            write_json(run_dir / "summary.json", results)
        if stopped:
            break
    best = max(results["epoch_results"], key=lambda value: (value["place_R@1"],
                                                           -value["epoch"]))
    results["selected_epoch"] = best["epoch"]
    results["selected_place_R@1"] = best["place_R@1"]
    results["elapsed_s"] = time.monotonic() - started
    results["status"] = "limited" if stopped else "complete"
    write_json(run_dir / "summary.json", results)
    print(f"RUN_DIR={run_dir} STATUS={results['status']} BEST_EPOCH={best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
