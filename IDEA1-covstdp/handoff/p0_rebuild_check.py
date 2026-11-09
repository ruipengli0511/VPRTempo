# MIT License
#
# Copyright (c) 2023 Adam Hines, Peter Stratton, Michael Milford, Tobias Fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
================================================================================
p0_rebuild_check.py — P0 样本输出对拍：原 checkpoint conv 前端 vs
                      conv_only_state JSON 重建前端（D1–D6 修订裁决追加验收）

路径 A（原前端）：build_inference_models(phase2 b5_500.json, seed=0) 加载
  vprtempo/models/b5_500__seed0.pth 的 conv_layer（权重来自 checkpoint）。
路径 B（JSON 重建）：ConvSNNLayer 以 conv_stdp 随机初始化构造后，
  w / thr / havconnExc 全部由 conv_only_state_b5_500__seed0.json 覆写
  —— 即前端状态的唯一来源是工件 JSON，不经过 gabor_frontend 代码重生成。

对 P0 样本清单（handoff/P0_2026-11-08/样本与前向验收.json）的 8 个地点
× 春/秋/夏三季共 24 张图，两路前向输出（pooled_flat，1152 维）逐元素比对，
要求 max|Δ| = 0（同代码路径、同状态，应为比特级一致）。

产出 IDEA1-covstdp/handoff/p0_rebuild_check.json（canonical self_sha256），
并由 make_artifact_index.py 登记进 artifact_index.json。

用法（仓库根目录）：
    pixi run python IDEA1-covstdp/handoff/p0_rebuild_check.py
================================================================================
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "IDEA1-covstdp" / "experiments"))

HANDOFF_DIR = REPO_ROOT / "IDEA1-covstdp" / "handoff"
MANIFEST = HANDOFF_DIR / "P0_2026-11-08" / "样本与前向验收.json"
CONV_JSON = HANDOFF_DIR / "conv_only_state_b5_500__seed0.json"
OUT_JSON = HANDOFF_DIR / "p0_rebuild_check.json"
CFG = REPO_ROOT / "IDEA1-covstdp" / "phase2" / "configs" / "b5_500.json"


def canonical_sha256(obj: dict) -> str:
    placeholder = dict(obj)
    placeholder["self_sha256"] = ""
    blob = json.dumps(placeholder, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    from run_exp import load_config
    from eval_retrieval import build_inference_models
    from vprtempo.src import conv_frontend as cf
    from vprtempo.src.dataset import ProcessImage

    # ---- 路径 A：原 checkpoint 的 conv 层 ----
    cfg = load_config(str(CFG), 0)
    models_a = build_inference_models(cfg)
    layer_a = models_a[0].conv_layer
    print(f"[A] checkpoint 前端：frontend={models_a[0].frontend} "
          f"wta_block={layer_a.wta_block} thr[0,3]={layer_a.thr.flatten()[:4].tolist()}")

    # ---- 路径 B：conv_stdp 随机初始化 + JSON 全量覆写 ----
    st = json.loads(CONV_JSON.read_text(encoding="utf-8"))
    ns = argparse.Namespace(**{**cfg, "frontend": "conv_stdp"})
    dims = [int(x) for x in cfg["dims"].split(",")]
    layer_b = cf.build_conv_layer(ns, dims, torch.device("cpu"), inference=True)
    w = torch.tensor(st["conv_state"]["weights"]["values"], dtype=torch.float32).unsqueeze(1)
    thr = torch.tensor(st["conv_state"]["thresholds"]["values"], dtype=torch.float32).view(1, -1, 1, 1)
    mask = torch.tensor(st["conv_state"]["on_off_mask"]["values"], dtype=torch.bool)
    assert tuple(w.shape) == (32, 1, 5, 5) and tuple(thr.shape) == (1, 32, 1, 1) and tuple(mask.shape) == (32,)
    with torch.no_grad():
        layer_b.w.weight.copy_(w)
        layer_b.thr.copy_(thr)
        layer_b.havconnExc.copy_(mask)
    print(f"[B] JSON 重建前端：wta_block={layer_b.wta_block} thr[0,3]={layer_b.thr.flatten()[:4].tolist()}")

    # ---- 样本前向对拍 ----
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data_dir = manifest["config"]["effective"]["data_dir"]
    patch_norm = cfg.get("patch_norm", "on") == "on"
    transform = ProcessImage(dims, cfg["patches"], patch_norm=patch_norm)

    from torchvision.io import read_image
    per_sample = []
    overall_max = 0.0
    for sample in manifest["samples"]:
        place = sample["place_id"]
        for season in ("spring", "fall", "summer"):
            entry = sample["seasons"][season]
            rel = entry["relative_to_data_dir"]
            img_path = Path(data_dir) / rel
            if not entry.get("file_readable", False) or not img_path.is_file():
                per_sample.append({"place_id": place, "season": season, "file": rel,
                                   "skipped": "manifest 标记不可读或文件缺失"})
                continue
            spikes = transform(read_image(str(img_path))).unsqueeze(0)  # [1, H*W]
            out_a = cf.conv_forward(layer_a, spikes)
            out_b = cf.conv_forward(layer_b, spikes)
            d = float((out_a - out_b).abs().max())
            overall_max = max(overall_max, d)
            per_sample.append({
                "place_id": place, "season": season, "file": rel,
                "out_dim": int(out_a.shape[1]),
                "max_abs_diff": d, "match": d == 0.0,
            })

    checked = [p for p in per_sample if not p.get("skipped")]
    all_match = all(p["match"] for p in checked)
    result: dict = {
        "artifact": "p0_rebuild_check",
        "schema_version": 1,
        "purpose": "原 checkpoint conv 前端 vs conv_only_state JSON 重建前端的 P0 样本输出对拍",
        "inputs": {
            "manifest": "IDEA1-covstdp/handoff/P0_2026-11-08/样本与前向验收.json",
            "conv_state": "IDEA1-covstdp/handoff/conv_only_state_b5_500__seed0.json",
            "checkpoint": "vprtempo/models/b5_500__seed0.pth",
            "config": "IDEA1-covstdp/phase2/configs/b5_500.json（block=4 口径）",
        },
        "path_a": "build_inference_models 加载 checkpoint 的 conv_layer",
        "path_b": "ConvSNNLayer(conv_stdp 随机初始化) 的 w/thr/havconnExc 全部由 conv_only_state JSON 覆写",
        "n_samples": len(checked),
        "tolerance": 0.0,
        "all_match": all_match,
        "max_abs_diff_overall": overall_max,
        "per_sample": per_sample,
        "limitations": "仅证明登记配置下 JSON 状态可比特级重建 conv 前向；不证明历史训练运行等价",
        "self_sha256": "",
        "self_sha256_scope": ("SHA256(UTF-8 字节 of canonical JSON)：整对象中本字段取空串占位、"
                              "sort_keys=True、separators=(',',':')、ensure_ascii=False"),
    }
    result["self_sha256"] = canonical_sha256(result)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                        encoding="utf-8")

    print(f"[OK] {len(checked)} 样本对拍：all_match={all_match} max|Δ|={overall_max}")
    print(f"     写出 {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"     self_sha256 = {result['self_sha256']}")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
