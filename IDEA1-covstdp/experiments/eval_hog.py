#!/usr/bin/env python
# ================================================================================
# eval_hog.py — IDEA1 S3.6：HOG 手工描述子直通检索基线（外部参照，不进 SNN）
#
# 功能：绕开整个 SNN，用经典 HOG 描述子做最近邻检索，补齐从未实现的纯外部基线：
#   1. 数据通路复用 main.py 的构建方式（CustomImageDataset + ProcessImage），
#      但 patch_norm='off'（HOG 自带块归一化）；灰度+gamma+resize 前三步与 SNN
#      输入逐比特一致（ProcessImage patch_norm=off 的输出 = uint8/255 灰度图）；
#   2. hog_descriptor（src/hog_descriptor.py）→ L2 → cosine 相似度矩阵 S；
#   3. GT 用与轨 B 同一实现（vprtempo/src/gt.py 多季节纵向平铺）；
#   4. recallAtK + recall@100%precision + P@100%R，落盘 results/hog_<places>/。
#
# 用法：pixi run python IDEA1-covstdp/experiments/eval_hog.py --config <config.json> --places 100 --seed 0
# ================================================================================
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "IDEA1-covstdp" / "src"))

from run_exp import load_config, RESULTS_DIR  # noqa: E402


def extract_hog_features(dataset, device):
    """逐样本提取 HOG 描述子，返回 [N, D] 的 torch 张量（CPU）"""
    from hog_descriptor import hog_descriptor

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    feats = []
    with torch.no_grad():
        for spikes, _ in loader:
            # ProcessImage(patch_norm=off) 输出 [1, H*W] 的 [0,1] 灰度图（batch=1）
            feats.append(torch.from_numpy(
                hog_descriptor(spikes.float()).copy()))
    return torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser(description="IDEA1 S3.6：HOG 直通检索基线（无 SNN）")
    ap.add_argument("--config", required=True, help="配置 JSON（复用数据通路字段）")
    ap.add_argument("--places", type=int, default=None,
                    help="覆盖 database/query 地点数（冒烟小档用，如 100）")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
    os.environ["PYTHONHASHSEED"] = str(cli.seed)
    cfg = load_config(cli.config, cli.seed)

    # ---- 地点数覆盖（冒烟档）----
    if cli.places is not None:
        cfg["database_places"] = cli.places
        cfg["query_places"] = cli.places
        cfg["max_module"] = cli.places

    from vprtempo.src.dataset import CustomImageDataset, ProcessImage
    from vprtempo.src.gt import build_ground_truth_multiseason
    from vprtempo.src.metrics import recallAtK, createPR

    dims = [int(x) for x in cfg["dims"].split(",")]
    # HOG 自带块归一化 → patch_norm='off'；灰度+gamma+resize 与 SNN 输入前三步一致
    transform = ProcessImage(dims, cfg["patches"], patch_norm=False)

    # ---- 数据库数据集：多季节合并（test=False），顺序 = [季节0 全部地点, 季节1 ...] ----
    skip_db = cfg.get("skip_db", cfg["skip"])
    db_csvs = [os.path.join('./vprtempo/dataset', f'{cfg["dataset"]}-{d}.csv')
               for d in cfg["database_dirs"].split(',')]
    db_dataset = CustomImageDataset(
        annotations_file=db_csvs, base_dir=cfg["data_dir"],
        img_dirs=cfg["database_dirs"].split(','), transform=transform,
        filter=cfg["filter"], skip=skip_db, test=False,
        max_samples=cfg["database_places"])
    n_seasons = len(db_csvs)

    # ---- 查询数据集（test=True，与轨 B eval_retrieval 同构）----
    q_csv = os.path.join('./vprtempo/dataset', f'{cfg["dataset"]}-{cfg["query_dir"]}.csv')
    q_dataset = CustomImageDataset(
        annotations_file=q_csv, base_dir=cfg["data_dir"],
        img_dirs=[cfg["query_dir"]], transform=transform,
        filter=cfg["filter"], skip=cfg["skip"], test=True,
        max_samples=cfg["query_places"])

    t0 = time.time()
    F_db = extract_hog_features(db_dataset, 'cpu')
    F_q = extract_hog_features(q_dataset, 'cpu')
    wall = time.time() - t0

    # ---- cosine 相似度矩阵 ----
    F_db = torch.nn.functional.normalize(F_db, dim=1)
    F_q = torch.nn.functional.normalize(F_q, dim=1)
    S = (F_db @ F_q.T).numpy()  # [n_seasons*db_places, query_places]

    # ---- GT：与轨 B 同一实现，多季节纵向平铺 ----
    GT = build_ground_truth_multiseason(
        cfg["database_places"], cfg["query_places"], n_seasons,
        skip=cfg["skip"], filter=cfg["filter"], tolerance=cfg["GT_tolerance"])

    # ---- 指标（口径与 eval_retrieval.py 逐行一致）----
    Ks = [1, 5, 10, 15, 20, 25]
    recalls = {k: round(recallAtK(S, GT, K=k), 4) for k in Ks}
    P, R = createPR(S, GT, matching='single', n_thresh=100)
    P_arr, R_arr = np.array(P), np.array(R)
    r100 = float(np.max(R_arr[P_arr == 1.0])) if np.any(P_arr == 1.0) else 0.0
    p100 = float(np.max(P_arr[R_arr >= 1.0 - 1e-9])) if np.any(R_arr >= 1.0 - 1e-9) else 0.0

    print(f"[eval_hog] HOG dim = {F_db.shape[1]} | db_rows={S.shape[0]} query={S.shape[1]}")
    print(f"[eval_hog] Recall@K = {recalls}")
    print(f"[eval_hog] R@100%P = {r100:.4f} | P@100%R = {p100:.4f}")

    # ---- 落盘：results/hog_<places>/，命名风格与 eval_retrieval 一致 ----
    places = cfg["database_places"]
    exp_id = f"hog_{places}"
    out_dir = RESULTS_DIR / exp_id / f"seed_{cli.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "exp_id": exp_id, "seed": cli.seed, "phase": "trackB",
        "descriptor": "hog", "feature_point": "hog_direct",
        "feature_dim": int(F_db.shape[1]),
        "n_db_rows": int(S.shape[0]), "n_query": int(S.shape[1]),
        "n_seasons": n_seasons,
        "wall_time_s": round(wall, 2), "device": "cpu",
        "config": cfg,
        "recallAtK": recalls, "recallAt100precision": round(r100, 4),
        "precisionAt100recall": round(p100, 4),
    }
    out_file = out_dir / f"{exp_id}__seed{cli.seed}__trackB_hog.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.save(out_dir / f"{exp_id}__seed{cli.seed}__S_trackB_hog.npy", S)
    print(f"[eval_hog] 结果已写入 {out_file}")


if __name__ == "__main__":
    main()
