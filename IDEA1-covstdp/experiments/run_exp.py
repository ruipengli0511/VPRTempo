#!/usr/bin/env python
# ================================================================================
# run_exp.py — 配置驱动训练与原版地点输出评价
#
# 用法：pixi run python IDEA1-covstdp/experiments/run_exp.py <config.json> --train --eval --seed 0
# 必须从仓库根目录运行。已有同名模型时拒绝训练覆盖；重跑需使用新的 run 名称。
# ================================================================================
import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

# 【S2.5】评估阶段的 plt.show()（VPRTempo.py PR 曲线绘制）在交互式后端下会阻塞进程，
# 统一强制 Agg 后端（无头环境），防止 eval 挂死。
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "IDEA1-covstdp" / "phase1" / "configs"
RESULTS_DIR = REPO_ROOT / "IDEA1-covstdp" / "results"

# main.py parser 的默认值镜像（main.py:388-477）。新增 IDEA1 字段（seed/patch_norm/frontend
# 等）不在此列——Namespace 直接携带，VPRTempoTrain 会把 vars(args) 全部拷贝为属性，无害。
DEFAULTS = {
    "dataset": "nordland",
    "data_dir": "./vprtempo/dataset/",
    "database_places": 500,
    "query_places": 500,
    "max_module": 500,
    "database_dirs": "spring,fall",
    "query_dir": "summer",
    "GT_tolerance": 0,
    "skip": 0,
    "filter": 8,
    "epoch": 4,
    "patches": 15,
    "dims": "56,56",
    "train_new_model": False,
    "quantize": False,
    "model_name": None,
    "PR_curve": False,
    "sim_mat": False,
    "run_demo": False,
}


def load_config(config_path: str, seed: int) -> dict:
    """读配置 → 应用本机覆盖 → 应用命令行 seed → 补齐默认值"""
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)

    # 本机覆盖（gitignore，不进版本库）
    override_path = CONFIGS_DIR / "local_override.json"
    if override_path.exists():
        with open(override_path) as f:
            override = json.load(f)
        override.pop("_comment", None)
        for k, v in override.items():
            if cfg.get(k) is None or k not in cfg:
                cfg[k] = v

    cfg["seed"] = seed
    merged = dict(DEFAULTS)
    merged.update(cfg)
    return merged


def make_namespace(cfg: dict, train: bool) -> argparse.Namespace:
    ns = argparse.Namespace(**cfg)
    ns.train_new_model = train
    # 模型名：配置显式给出（train/eval 分离口径，如会议 3300 地）时尊重配置；
    # 否则用唯一名 <exp_id>__seed<n>，规避 check_pretrained_model 的交互式询问
    if not cfg.get("model_name"):
        ns.model_name = f"{cfg['exp_id']}__seed{cfg['seed']}.pth"
    elif not ns.model_name.endswith(".pth"):
        ns.model_name += ".pth"
    return ns


def reset_root_logger():
    """清理 root logger 的 handlers，防止多次函数调用时日志重复"""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()


def collect_results(exp_id: str, seed: int, phase: str, wall_s: float,
                    cfg: dict, out_dir: Path):
    """从最新 output 文件夹提取 Recall 表与 PR 数据，写结果 JSON"""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = sorted(glob.glob(str(REPO_ROOT / "vprtempo" / "output" / "*")),
                         key=os.path.getmtime)
    result = {
        "exp_id": exp_id, "seed": seed, "phase": phase,
        "wall_time_s": round(wall_s, 2), "device": "unknown",
        "config": cfg,
    }
    if output_dirs:
        latest = Path(output_dirs[-1])
        # Recall 表：logfile.log 里 PrettyTable 的 Recall 行
        logfile = latest / "logfile.log"
        if logfile.exists():
            text = logfile.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Recall\s*\|((?:\s*[\d.]+\s*\|){6})", text)
            if m:
                vals = [float(x) for x in re.findall(r"[\d.]+", m.group(1))]
                result["recallAtK"] = dict(zip([1, 5, 10, 15, 20, 25], vals))
            dev = re.search(r"Current device is:\s*(\w+)", text)
            if dev:
                result["device"] = dev.group(1)
        # PR 曲线数据 → R@100%P 与 P@100%R（LoCS-Net 口径）
        pr_file = latest / "PR_curve_data.json"
        if pr_file.exists():
            with open(pr_file) as f:
                pr = json.load(f)
            r100 = max((r for p, r in zip(pr["Precision"], pr["Recall"]) if p == 1.0),
                       default=0.0)
            p100 = max((p for p, r in zip(pr["Precision"], pr["Recall"]) if r >= 1.0 - 1e-9),
                       default=0.0)
            result["recallAt100precision"] = r100
            result["precisionAt100recall"] = p100
        # 原始日志一并归档
        if logfile.exists():
            shutil.copy(logfile, out_dir / f"logfile_{phase}.log")

    out_file = out_dir / f"{exp_id}__seed{seed}__{phase}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[run_exp] 结果已写入 {out_file}")
    if "recallAtK" in result:
        print(f"[run_exp] Recall@K = {result['recallAtK']}")


def main():
    ap = argparse.ArgumentParser(description="IDEA1 实验驱动器")
    ap.add_argument("config", help="配置文件路径（IDEA1-covstdp/phase1/configs/ 下）")
    ap.add_argument("--train", action="store_true", help="训练模式")
    ap.add_argument("--eval", action="store_true", help="推理/评估模式")
    ap.add_argument("--seed", type=int, default=0, help="实验 seed（覆盖配置）")
    cli = ap.parse_args()

    if not cli.train and not cli.eval:
        cli.eval = True  # 默认评估

    # 坑 2：相对路径要求 cwd 为仓库根；同时把仓库根加入 sys.path 以便 import main
    assert Path.cwd() == REPO_ROOT, f"必须从仓库根目录运行（当前 {Path.cwd()}）"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    cfg = load_config(cli.config, cli.seed)
    exp_id = cfg["exp_id"]
    os.environ["PYTHONHASHSEED"] = str(cli.seed)

    out_dir = RESULTS_DIR / exp_id / f"seed_{cli.seed}"
    dims = [int(x) for x in cfg["dims"].split(",")]

    # 推迟到断言后再 import（main 的 import 链会初始化 logging）
    from main import initialize_and_run_model

    for phase, train in (("train", True), ("eval", False)):
        if (train and not cli.train) or (not train and not cli.eval):
            continue
        reset_root_logger()
        ns = make_namespace(cfg, train)
        if train:
            model_path = REPO_ROOT / "vprtempo" / "models" / ns.model_name
            if model_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing model: {model_path}. "
                    "Use a new exp_id, seed, or model_name.")
        t0 = time.time()
        initialize_and_run_model(ns, dims)
        wall = time.time() - t0
        print(f"[run_exp] {phase} 完成，墙钟 {wall:.1f}s")
        if not train:
            collect_results(exp_id, cli.seed, phase, wall, cfg, out_dir)


if __name__ == "__main__":
    main()
