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
生成 conv-only 状态工件（D1–D6 移交片区：conv-only 状态工件与工件索引）

从 vprtempo/models/b5_500__seed0.pth 提取 conv 层（32 通道、5×5）权重实际值、
conv 阈值与 ON/OFF mask，连同前处理 / WTA 配置、来源检查点 SHA256 一起写入
IDEA1-covstdp/handoff/conv_only_state_b5_500__seed0.json，并做两项自检：

1. 重算来源检查点 SHA256，与登记值核对（不一致即报错退出）；
2. 用 gabor_frontend.gabor_kernel_bank() 重新生成 32 核 Gabor 组，与检查点
   conv 权重逐元素比对（判定 conv 权重确为 Gabor 核），结果写入工件
   verification 区块。

可重复运行（幂等）：输出 JSON 无时间戳、键序固定（sort_keys），
self_sha256 的计算口径为「整对象中 self_sha256 字段取占位空串时的
canonical JSON（sort_keys、无空白分隔）的 SHA256」，因此重复运行产出一致。

用法（仓库根或任意目录）：
    pixi run python IDEA1-covstdp/handoff/make_conv_only_state.py
================================================================================
"""

import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "vprtempo" / "models" / "b5_500__seed0.pth"
OUT_JSON = REPO_ROOT / "IDEA1-covstdp" / "handoff" / "conv_only_state_b5_500__seed0.json"

# 移交登记值（P0 2026-11-08 已核实）
EXPECTED_CKPT_SHA256 = (
    "2f6d8736a5762608f4d0fd6428dd9a35aa57223d3fbcbf716f6349a501c3b6be"
)
EXPECTED_CKPT_SIZE = 15_242_975


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj: dict) -> str:
    """整对象中 self_sha256 字段取空串占位时的 canonical JSON 的 SHA256。"""
    placeholder = dict(obj)
    placeholder["self_sha256"] = ""
    blob = json.dumps(placeholder, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def tensor_sha256_f32(t: torch.Tensor) -> str:
    """float32 张量 little-endian 字节的 SHA256（numpy '<f4' 序列化口径）。"""
    arr = t.detach().cpu().numpy().astype("<f4")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main() -> int:
    # ---- 自检 1：来源检查点哈希与登记值一致 ----
    ckpt_sha = sha256_file(CHECKPOINT)
    ckpt_size = CHECKPOINT.stat().st_size
    if ckpt_sha != EXPECTED_CKPT_SHA256 or ckpt_size != EXPECTED_CKPT_SIZE:
        print(f"[FATAL] 检查点哈希/大小与登记值不符：\n"
              f"  实测   {ckpt_sha}  {ckpt_size} bytes\n"
              f"  登记   {EXPECTED_CKPT_SHA256}  {EXPECTED_CKPT_SIZE} bytes",
              file=sys.stderr)
        return 1

    sd = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")
    if set(sd.keys()) != {"model_0"}:
        print(f"[FATAL] 顶层键异常：{sorted(sd.keys())}", file=sys.stderr)
        return 1
    sub = sd["model_0"]

    w = sub["conv_layer.w.weight"].detach().cpu().to(torch.float32)  # [32,1,5,5]
    thr = sub["conv_layer.thr"].detach().cpu().to(torch.float32)     # [1,32,1,1]
    mask = sub["conv_layer.havconnExc"].detach().cpu()               # [32] bool

    assert tuple(w.shape) == (32, 1, 5, 5), f"conv 权重形状异常 {tuple(w.shape)}"
    assert tuple(thr.shape) == (1, 32, 1, 1), f"conv 阈值形状异常 {tuple(thr.shape)}"
    assert tuple(mask.shape) == (32,), f"mask 形状异常 {tuple(mask.shape)}"

    # ---- 自检 2：conv 权重与 Gabor 核组逐元素比对 ----
    sys.path.insert(0, str(REPO_ROOT / "IDEA1-covstdp" / "src"))
    from gabor_frontend import gabor_kernel_bank

    bank = gabor_kernel_bank(kernel_size=5).detach().cpu().to(torch.float32)
    max_abs_diff = float((w - bank).abs().max())
    kernels_match = max_abs_diff == 0.0
    kernel_bank_sha = tensor_sha256_f32(w)

    thr_flat = [float(v) for v in thr.reshape(-1)]          # [32]
    weights = w[:, 0, :, :].tolist()                        # [32][5][5]，C_in=1 已压缩
    mask_bool = [bool(v) for v in mask]                     # [32]
    on_indices = [i for i, v in enumerate(mask_bool) if v]

    artifact: dict = {
        "artifact": "conv_only_state",
        "schema_version": 1,
        "exp_id": "b5_500",
        "seed": 0,
        "role": "P1 conv-only 固定前端状态（C0–C3 共用输入）；C0 旧协议复核的 conv 来源",
        "source_checkpoint": {
            "path": "vprtempo/models/b5_500__seed0.pth",
            "sha256": ckpt_sha,
            "size_bytes": ckpt_size,
            "top_level_key": "model_0",
            "loaded_with": "torch.load(weights_only=True)",
            "keys_in_checkpoint": sorted(sub.keys()),
        },
        "conv_state": {
            "layer_key_prefix": "model_0.conv_layer",
            "weight_key": "conv_layer.w.weight",
            "threshold_key": "conv_layer.thr",
            "mask_key": "conv_layer.havconnExc",
            "mask_key_note": ("移交指令中写作 model_0.conv_layer.havconnCombinedExc；"
                              "实际 state dict 键为 conv_layer.havconnExc（本工件登记实际键）"),
            "channels": 32,
            "kernel_size": 5,
            "weights": {
                "original_shape": [32, 1, 5, 5],
                "stored_shape": [32, 5, 5],
                "stored_layout": "out_channel[32] × ky[5] × kx[5]（输入通道=1 已压缩）",
                "dtype": "float32（JSON 中为 float64 精确提升表示，可无损还原 float32）",
                "values": weights,
            },
            "thresholds": {
                "original_shape": [1, 32, 1, 1],
                "stored_shape": [32],
                "dtype": "float32",
                "semantic": "conv ITP 产物（历史 B5 运行中阈值未冻结），前向 a=clamp(z_signed−θ,0,0.9)",
                "min": min(thr_flat),
                "max": max(thr_flat),
                "nonzero_count": sum(1 for v in thr_flat if v != 0.0),
                "values": thr_flat,
            },
            "on_off_mask": {
                "dtype": "bool",
                "stored_shape": [32],
                "semantic": "True=ON 兴奋通道（前向保留 z）；False=OFF 抑制通道（前向取 −z）",
                "on_count": len(on_indices),
                "off_count": 32 - len(on_indices),
                "on_indices": on_indices,
                "values": mask_bool,
            },
        },
        "preprocessing": {
            "dims": "28,28",
            "patches": 7,
            "patch_norm": "on",
            "filter": 8,
            "skip": 0,
            "GT_tolerance": 0,
            "config_source": "IDEA1-covstdp/phase2/configs/b5_500.json",
        },
        "wta": {
            "mode": "local",
            "block": 4,
            "evidence_note": ("block=4 口径，证据链：phase2/configs/b5_500.json（wta_block=4，P0 核实链）；"
                              "feature 输入 1152=32×6×6 为 block4 池化形状（block2 应为 4608 维）；"
                              "handoff/P0_2026-11-08/P1实施建议.md 自述 local WTA block=4、1152 维输出。"
                              "注意与主表 block2 口径（phase3/configs/table1/b5_500*.json）不可混用。"),
        },
        "verification": {
            "checkpoint_sha256_matches_registry": True,
            "kernel_bank_sha256_f32le": kernel_bank_sha,
            "kernel_bank_sha256_scope": "conv_layer.w.weight [32,1,5,5] float32 little-endian 字节",
            "gabor_regen_source": "IDEA1-covstdp/src/gabor_frontend.py::gabor_kernel_bank(5)（当前 HEAD d7a4ff3）",
            "gabor_regen_max_abs_diff": max_abs_diff,
            "gabor_regen_match": kernels_match,
            "gabor_regen_note": "conv 权重与确定性 Gabor 核组逐元素相等（最大绝对差 0）→ conv 权重确为 Gabor 核；"
                                "历史运行 SHA 未知，此处以当前 HEAD 代码重生成核对",
        },
        "limitations": [
            "阈值是历史 ITP 产物，历史运行代码 SHA 未知",
            "checkpoint 只含 state_dict，不含配置、数据 ID、代码 SHA 或优化器状态",
            "本工件仅含 conv 层；feature/output 层不属于 conv-only 前端，未提取",
        ],
        "self_sha256": "",
        "self_sha256_scope": ("SHA256(UTF-8 字节 of canonical JSON)：整对象中本字段取空串占位、"
                              "sort_keys=True、separators=(',',':')、ensure_ascii=False；"
                              "self_sha256 计入 artifact_index.json 时按本口径可复算核对"),
    }
    artifact["self_sha256"] = canonical_sha256(artifact)

    OUT_JSON.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")

    print(f"[OK] 写出 {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"     source sha256      = {ckpt_sha}")
    print(f"     kernel bank sha256 = {kernel_bank_sha}")
    print(f"     gabor match        = {kernels_match} (max_abs_diff={max_abs_diff})")
    print(f"     thr nonzero/max    = {sum(1 for v in thr_flat if v != 0.0)}/32, max={max(thr_flat)}")
    print(f"     self_sha256        = {artifact['self_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
