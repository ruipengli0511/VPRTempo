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
生成 handoff 工件索引 artifact_index.json（D1–D6 移交片区：conv-only 状态工件与工件索引）

登记两个工件：
  ① vprtempo/models/b5_500__seed0.pth —— P1 conv-only 来源 / C0 旧协议复核
  ② IDEA1-covstdp/handoff/conv_only_state_b5_500__seed0.json —— 由
    make_conv_only_state.py 从 ① 提取的 conv-only 状态（含来源检查点哈希）

自检（不满足即非零退出）：
  - ① 的文件 SHA256 / 字节数与登记值一致；
  - ② 已存在且其内部登记的 source_checkpoint.sha256 与 ① 一致。

可重复运行（幂等）：输出无时间戳、键序固定（sort_keys）、列表顺序固定；
self_sha256 口径同 make_conv_only_state.py（整对象中本字段取空串占位时的
canonical JSON 的 SHA256）。

用法（仓库根或任意目录）：
    pixi run python IDEA1-covstdp/handoff/make_artifact_index.py
================================================================================
"""

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_DIR = REPO_ROOT / "IDEA1-covstdp" / "handoff"
CHECKPOINT = REPO_ROOT / "vprtempo" / "models" / "b5_500__seed0.pth"
CONV_JSON = HANDOFF_DIR / "conv_only_state_b5_500__seed0.json"
OUT_JSON = HANDOFF_DIR / "artifact_index.json"

MACHINE = "工位机"
CKPT_SHA256 = "2f6d8736a5762608f4d0fd6428dd9a35aa57223d3fbcbf716f6349a501c3b6be"
CKPT_SIZE = 15_242_975


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj: dict) -> str:
    placeholder = dict(obj)
    placeholder["self_sha256"] = ""
    blob = json.dumps(placeholder, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    # ---- 自检 1：检查点哈希/大小与登记值一致 ----
    ckpt_sha = sha256_file(CHECKPOINT)
    ckpt_size = CHECKPOINT.stat().st_size
    if ckpt_sha != CKPT_SHA256 or ckpt_size != CKPT_SIZE:
        print(f"[FATAL] 检查点哈希/大小与登记值不符：\n"
              f"  实测   {ckpt_sha}  {ckpt_size} bytes\n"
              f"  登记   {CKPT_SHA256}  {CKPT_SIZE} bytes", file=sys.stderr)
        return 1

    # ---- 自检 2：conv JSON 已存在且来源哈希一致 ----
    if not CONV_JSON.is_file():
        print(f"[FATAL] 缺少 {CONV_JSON.relative_to(REPO_ROOT)}，"
              f"请先运行 make_conv_only_state.py", file=sys.stderr)
        return 1
    conv_doc = json.loads(CONV_JSON.read_text(encoding="utf-8"))
    conv_src_sha = conv_doc.get("source_checkpoint", {}).get("sha256")
    if conv_src_sha != ckpt_sha:
        print(f"[FATAL] conv JSON 的来源检查点哈希与实测不一致：\n"
              f"  conv JSON 登记 {conv_src_sha}\n"
              f"  实测           {ckpt_sha}", file=sys.stderr)
        return 1
    conv_sha = sha256_file(CONV_JSON)
    conv_size = CONV_JSON.stat().st_size
    conv_self_sha = conv_doc.get("self_sha256")

    artifacts = [
        {
            "id": "b5_500__seed0.pth",
            "type": "checkpoint",
            "machine": MACHINE,
            "path": "vprtempo/models/b5_500__seed0.pth",
            "size_bytes": ckpt_size,
            "sha256": ckpt_sha,
            "purpose": "P1 conv-only 来源 / C0 旧协议复核",
            "related_configs": [
                "IDEA1-covstdp/phase2/configs/b5_500.json"
                "（P0 核实的对应配置：local WTA block=4，28×28/patches=7/patch_norm=on）",
                "IDEA1-covstdp/phase3/configs/table1/b5_500.json"
                "（主表 block=2 口径配置，不对应本 checkpoint 的 1152 维形状）",
                "IDEA1-covstdp/handoff/P0_2026-11-08/（P0 材料）",
            ],
            "limitations": "历史运行代码 SHA 未知（checkpoint 无运行级元数据）",
        },
        {
            "id": "conv_only_state_b5_500__seed0.json",
            "type": "derived_artifact",
            "machine": MACHINE,
            "path": "IDEA1-covstdp/handoff/conv_only_state_b5_500__seed0.json",
            "size_bytes": conv_size,
            "sha256": conv_sha,
            "self_sha256_canonical": conv_self_sha,
            "sha256_scope": ("sha256 为整个文件字节的 SHA256；"
                             "self_sha256_canonical 为工件内部登记的 canonical-JSON 口径，"
                             "两者算法与语义不同，各自标注"),
            "purpose": "P1 conv-only 固定前端状态（conv W/阈值/ON-OFF mask 实际值）"
                       " / C0 旧协议复核",
            "source_checkpoint_sha256": ckpt_sha,
            "generated_by": "IDEA1-covstdp/handoff/make_conv_only_state.py",
            "related_configs": [
                "IDEA1-covstdp/phase2/configs/b5_500.json",
                "IDEA1-covstdp/phase3/configs/table1/b5_500.json",
                "IDEA1-covstdp/handoff/P0_2026-11-08/（P0 材料）",
            ],
            "limitations": "历史运行代码 SHA 未知；conv 阈值为历史 ITP 产物；"
                           "仅含 conv 层，不含 feature/output",
        },
    ]

    # ---- 自检 3（可选）：P0 重建对拍结果存在时登记 ----
    check_json = HANDOFF_DIR / "p0_rebuild_check.json"
    if check_json.is_file():
        check_doc = json.loads(check_json.read_text(encoding="utf-8"))
        artifacts.append({
            "id": "p0_rebuild_check.json",
            "type": "verification_result",
            "machine": MACHINE,
            "path": "IDEA1-covstdp/handoff/p0_rebuild_check.json",
            "size_bytes": check_json.stat().st_size,
            "sha256": sha256_file(check_json),
            "self_sha256_canonical": check_doc.get("self_sha256"),
            "purpose": "P0 样本输出对拍：原 checkpoint conv 前端 vs conv_only_state JSON 重建前端",
            "scope": f"{check_doc.get('n_samples')} 个 P0 样本（8 地点 × 春/秋/夏），"
                     f"all_match={check_doc.get('all_match')}，"
                     f"max_abs_diff_overall={check_doc.get('max_abs_diff_overall')}",
            "generated_by": "IDEA1-covstdp/handoff/p0_rebuild_check.py",
            "source_checkpoint_sha256": ckpt_sha,
            "limitations": "仅证明在登记配置下 JSON 状态可无损重建 conv 前向；"
                           "不证明历史训练运行等价",
        })

    index: dict = {
        "index": "idea1_covstdp_handoff_artifacts",
        "schema_version": 1,
        "scope": "D1–D6 移交片区：conv-only 状态工件与工件索引（仅登记本片区产出的工件"
                 "及其来源检查点，不替代其他片区的索引）",
        "generated_by": "IDEA1-covstdp/handoff/make_artifact_index.py",
        "artifacts": artifacts,
        "self_sha256": "",
        "self_sha256_scope": ("SHA256(UTF-8 字节 of canonical JSON)：整对象中本字段取空串占位、"
                              "sort_keys=True、separators=(',',':')、ensure_ascii=False"),
    }
    index["self_sha256"] = canonical_sha256(index)

    OUT_JSON.write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")

    print(f"[OK] 写出 {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"     artifacts          = {len(artifacts)}")
    print(f"     checkpoint  sha256 = {ckpt_sha}")
    print(f"     conv JSON  sha256  = {conv_sha} ({conv_size} bytes)")
    print(f"     self_sha256        = {index['self_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
