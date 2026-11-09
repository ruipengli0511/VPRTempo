#!/usr/bin/env python
import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.io import read_image

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "IDEA1-covstdp" / "experiments"))

from run_exp import load_config
from vprtempo.VPRTempo import VPRTempo
from vprtempo.src import blitnet as bn
from vprtempo.src.dataset import CustomImageDataset, ProcessImage
from vprtempo.src.gt import build_ground_truth_multiseason

CONFIG = REPO / "IDEA1-covstdp" / "phase2" / "configs" / "b5_500.json"
CHECKPOINT = REPO / "vprtempo" / "models" / "b5_500__seed0.pth"
PLACE_IDS = [0, 71, 142, 213, 284, 355, 426, 499]
SEASONS = ["spring", "fall", "summer"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_stats(x):
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": bool(torch.isfinite(x).all()),
        "min": float(x.min()),
        "max": float(x.max()),
        "zero_fraction": float((x == 0).float().mean()),
        "l2_norm": float(torch.linalg.vector_norm(x.float())),
    }


def state_snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_difference(before, after):
    keys = sorted(set(before) | set(after))
    per_key = {}
    for key in keys:
        if key not in before or key not in after:
            per_key[key] = {"presence_changed": True}
            continue
        b, a = before[key], after[key]
        per_key[key] = {
            "equal": bool(torch.equal(b, a)),
            "changed_elements": int((b != a).sum()),
            "max_abs_diff": float((b.float() - a.float()).abs().max()),
        }
    return {
        "all_equal": all(v.get("equal", False) for v in per_key.values()),
        "per_key": per_key,
    }


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    cfg = load_config(str(CONFIG), 0)
    dims = [int(x) for x in cfg["dims"].split(",")]
    logger = logging.getLogger("p0_forward_acceptance")
    logger.addHandler(logging.NullHandler())
    model = VPRTempo(
        argparse.Namespace(**cfg), dims, logger, num_modules=1,
        output_folder=str(OUT), out_dim=cfg["database_places"],
        out_dim_remainder=cfg["database_places"],
    )
    model.load_model([model], str(CHECKPOINT))
    model.eval()
    model.to(torch.device("cpu"))
    before = state_snapshot(model)

    transform = ProcessImage(dims, cfg["patches"], patch_norm=cfg["patch_norm"] == "on")
    datasets = {}
    identities = []
    for place_id in PLACE_IDS:
        item = {"place_id": place_id, "seasons": {}}
        for season in SEASONS:
            if season not in datasets:
                datasets[season] = CustomImageDataset(
                    annotations_file=str(REPO / "vprtempo" / "dataset" / f"nordland-{season}.csv"),
                    base_dir=cfg["data_dir"], img_dirs=[season], transform=transform,
                    filter=cfg["filter"], skip=cfg["skip"], test=True,
                    max_samples=cfg["database_places"],
                )
            ds = datasets[season]
            row = ds.img_labels.iloc[place_id]
            rel_path = str(Path(cfg["data_dir"]) / season / row.iloc[0])
            item["seasons"][season] = {
                "role": "development_query" if season == "summer" else "training_reference",
                "dataset_index": place_id,
                "raw_csv_zero_based_row": place_id * cfg["filter"],
                "label": int(row.iloc[1]),
                "file_name": row.iloc[0],
                "relative_to_data_dir": f"{season}/{row.iloc[0]}",
                "file_readable": os.access(rel_path, os.R_OK),
            }
        identities.append(item)

    def encode(place_id, season):
        spikes, label = datasets[season][place_id]
        spikes = spikes.unsqueeze(0).float()
        with torch.no_grad():
            conv_out = model.conv_layer(model.conv_layer.reshape_input(spikes))
            conv = conv_out.pooled_flat
            feature_raw = model.feature_layer.w(conv)
            feature = bn.clamp_spikes(feature_raw, model.feature_layer)
            track_a = model.output_layer.w(feature_raw)
        return {
            "label": int(label),
            "spikes": spikes,
            "conv_pre_wta": conv_out.pre_wta,
            "conv_post_wta": conv_out.post_wta,
            "conv": conv,
            "feature_raw": feature_raw,
            "feature": feature,
            "track_a_output": track_a,
        }

    records = {}
    order = [(p, s) for p in PLACE_IDS for s in SEASONS]
    for place_id, season in order:
        records[f"{season}/{place_id}"] = encode(place_id, season)
    for place_id, season in reversed(order):
        key = f"{season}/{place_id}"
        reverse = encode(place_id, season)
        for tensor_name in ["spikes", "conv_pre_wta", "conv_post_wta", "conv", "feature_raw", "feature", "track_a_output"]:
            assert torch.equal(records[key][tensor_name], reverse[tensor_name]), (key, tensor_name)
        records[key]["order_reverse_equal"] = True

    repeat_key = "spring/0"
    repeated = encode(0, "spring")
    repeat = {}
    for tensor_name in ["spikes", "conv_pre_wta", "conv_post_wta", "conv", "feature_raw", "feature", "track_a_output"]:
        repeat[tensor_name] = {
            "equal": bool(torch.equal(records[repeat_key][tensor_name], repeated[tensor_name])),
            "max_abs_diff": float((records[repeat_key][tensor_name] - repeated[tensor_name]).abs().max()),
        }

    gt = build_ground_truth_multiseason(
        cfg["database_places"], cfg["query_places"], 2,
        skip=cfg["skip"], filter=cfg["filter"], tolerance=cfg["GT_tolerance"],
    )
    gt_checks = []
    for p in PLACE_IDS:
        gt_checks.append({
            "place_id": p,
            "spring_positive": int(gt[p, p]),
            "fall_positive": int(gt[p + cfg["database_places"], p]),
            "adjacent_negative": int(gt[(p + 1) % cfg["database_places"], p]),
            "query_label_matches_reference": all(
                identities[PLACE_IDS.index(p)]["seasons"][s]["label"] == p * cfg["filter"]
                for s in SEASONS
            ),
        })

    per_sample = []
    for p in PLACE_IDS:
        for s in SEASONS:
            r = records[f"{s}/{p}"]
            per_sample.append({
                "place_id": p,
                "season": s,
                "returned_label": r["label"],
                "spikes": tensor_stats(r["spikes"]),
                "conv_pre_wta": tensor_stats(r["conv_pre_wta"]),
                "conv": tensor_stats(r["conv"]),
                "feature": tensor_stats(r["feature"]),
                "track_a_output": tensor_stats(r["track_a_output"]),
            })

    after = state_snapshot(model)
    first = records["spring/0"]
    result = {
        "schema_version": 1,
        "status": "completed",
        "generated_at": datetime.now().astimezone().isoformat(),
        "purpose": "P0 short forward acceptance only; no training or threshold update",
        "repository": {
            "path": str(REPO),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "executable": sys.executable,
        },
        "config": {
            "source": str(CONFIG),
            "effective": cfg,
            "model_name_used": CHECKPOINT.name,
        },
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": sha256(CHECKPOINT),
            "loaded_with_weights_only": True,
            "modules": 1,
            "checkpoint_modified": False,
        },
        "sampling": {
            "rule": "eight evenly spaced, fixed place IDs from the existing 500-place development configuration",
            "place_ids": PLACE_IDS,
            "reference_seasons": ["spring", "fall"],
            "development_query_season": "summer",
            "winter_images_opened": False,
            "independent_final_query_opened": False,
        },
        "samples": identities,
        "forward_contract": {
            "reference_and_query_same_transform_and_encoder": True,
            "track_b_conv_bypasses_feature_and_output": True,
            "track_b_feature_applies_linear_then_threshold_clamp": True,
            "track_a_output_uses_feature_raw_without_threshold_clamp": True,
            "batch_size": 1,
        },
        "shapes_from_one_sample": {
            k: list(first[k].shape) for k in
            ["spikes", "conv_pre_wta", "conv_post_wta", "conv", "feature_raw", "feature", "track_a_output"]
        },
        "repeat_same_image": repeat,
        "reverse_order": {
            "all_samples_all_recorded_tensors_exactly_equal": True,
            "tensor_names": ["spikes", "conv_pre_wta", "conv_post_wta", "conv", "feature_raw", "feature", "track_a_output"],
        },
        "model_state_before_after": state_difference(before, after),
        "non_state_flags": {
            "conv_frozen": bool(model.conv_layer.frozen),
            "conv_itp_on_frozen": bool(model.conv_layer.itp_on_frozen),
            "inference_feature_has_x_state": hasattr(model.feature_layer, "x"),
            "inference_feature_has_eta_ip": hasattr(model.feature_layer, "eta_ip"),
        },
        "validity": {
            "all_sample_tensors_finite": all(
                x["finite"]
                for row in per_sample for k, x in row.items()
                if k in ["spikes", "conv_pre_wta", "conv", "feature", "track_a_output"]
            ),
            "all_conv_descriptors_nonempty": all(row["conv"]["l2_norm"] > 0 for row in per_sample),
            "all_feature_descriptors_nonempty": all(row["feature"]["l2_norm"] > 0 for row in per_sample),
            "all_returned_labels_match_identity": all(
                row["returned_label"] == row["place_id"] * cfg["filter"] for row in per_sample
            ),
        },
        "gt_checks": gt_checks,
        "per_sample": per_sample,
        "visual_check": {
            "contact_sheet": str(OUT / "P0_样本三联图.png"),
            "status": "pending_manual_view",
            "notes": None,
        },
        "invalid_items": [],
        "limitations": [
            "Eight-place forward acceptance is not a retrieval performance evaluation.",
            "Only the b5_500 seed0 single-module checkpoint was exercised.",
            "Static and dynamic results bind to the recorded current HEAD; they do not prove the historical run SHA.",
            "No training, evaluation entry point, checkpoint save, or winter image read was performed.",
        ],
    }
    with (OUT / "样本与前向验收.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(len(PLACE_IDS), 3, figsize=(9, 16))
    for i, p in enumerate(PLACE_IDS):
        for j, season in enumerate(SEASONS):
            rel = identities[i]["seasons"][season]["relative_to_data_dir"]
            img = read_image(str(Path(cfg["data_dir"]) / rel)).permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"place {p} | {season}\n{Path(rel).name}", fontsize=8)
    fig.suptitle("P0 fixed sample triplets (spring/fall references; summer development)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "P0_样本三联图.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "status": result["status"],
        "places": len(PLACE_IDS),
        "images_per_order": len(order),
        "state_all_equal": result["model_state_before_after"]["all_equal"],
        "repeat_all_equal": all(x["equal"] for x in repeat.values()),
        "finite": result["validity"]["all_sample_tensors_finite"],
        "json": str(OUT / "样本与前向验收.json"),
        "contact_sheet": str(OUT / "P0_样本三联图.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
