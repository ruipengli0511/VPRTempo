#!/usr/bin/env bash
# ================================================================================
# run_f0f3.sh — IDEA1 S3.6 feature 层可塑性分解（F0–F3）+ HOG 直通基线 批量跑批
#
# 矩阵：F0/F1/F2/F3（B5 Gabor 前端 block2 口径，feature_point=feature_layer）
#       × 2 规模（500 / 3300 地）× 3 seeds（0/1/2）= 24 格；
#       外加 HOG 直通 2 格（500/3300，无训练、确定性单跑）。
#       配置：IDEA1-covstdp/phase3/configs/f{0,1,2,3}_b5_{500,3300}.json
#             3300 评估段用 f{0,1,2,3}_b5_3300_eval.json（skip=4800 避开训练段）。
#
# 用法（在 GPU 工作站上，tmux 里跑；本机 CPU 调试设 PIXI_ENV=""）：
#   tmux new -s f0f3
#   cd /root/VPRTempo
#   export PATH="$HOME/.pixi/bin:$PATH"   # 工作站非交互 shell 不带 pixi，必须先加
#   bash IDEA1-covstdp/experiments/run_f0f3.sh            # 双卡（GPU 0/1）
#   NUM_GPUS=1 bash IDEA1-covstdp/experiments/run_f0f3.sh # 单卡调试
#
# 特性（对齐 run_table1.sh 约定）：
#   - 双卡队列：每格绑定一张卡，格间启动 sleep 2（防 logger 秒级时间戳撞目录）；
#   - 断点续跑：轨 A eval.json 与轨 B __trackB_feature_layer.json 都存在的格跳过；
#   - HOG 格最先调度（便宜且确定性，失败不影响 F 组训练格）；
#   - 进度写 results/f0f3_progress.log；每格 stdout/stderr 存 results/f0f3_logs/；
#   - 失败的格标 FAIL，下次运行自动重试。
#
# 纪律：本脚本只负责执行；何时点火由母会话审定。
# ================================================================================
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG_DIR="IDEA1-covstdp/phase3/configs"
RESULTS="IDEA1-covstdp/results"
PROGRESS_LOG="$RESULTS/f0f3_progress.log"
CELL_LOG_DIR="$RESULTS/f0f3_logs"
mkdir -p "$CELL_LOG_DIR"

VARIANTS=(f0 f1 f2 f3)
SCALES=(500 3300)
SEEDS=(0 1 2)

NUM_GPUS=${NUM_GPUS:-2}        # 双卡队列；单卡设 NUM_GPUS=1
PIXI_ENV=${PIXI_ENV:-cuda}     # 工作站 cuda 环境；本机 CPU 调试设 PIXI_ENV=""
if [ -n "${PY_CMD:-}" ]; then
    PY=($PY_CMD)
elif [ -n "$PIXI_ENV" ]; then
    PY=(pixi run --environment "$PIXI_ENV" python)
else
    PY=(pixi run python)
fi

log_progress() {  # log_progress <status> <exp> <seed> <gpu> <wall_s>
    echo "$(date '+%F %T') | $1 | $2 | seed$3 | gpu$4 | wall=${5}s" >> "$PROGRESS_LOG"
}

tracka_done() {  # tracka_done <exp> <seed>
    [ -f "$RESULTS/$1/seed_$2/$1__seed$2__eval.json" ]
}

trackb_done() {  # trackb_done <exp> <seed>
    [ -f "$RESULTS/$1/seed_$2/$1__seed$2__trackB_feature_layer.json" ]
}

hog_done() {  # hog_done <places>
    [ -f "$RESULTS/hog_$1/seed_0/hog_${1}__seed0__trackB_hog.json" ]
}

run_hog() {  # run_hog <gpu>
    local gpu=$1 t0=$SECONDS rc=0
    {
        export CUDA_VISIBLE_DEVICES=$gpu
        export MPLBACKEND=Agg
        if ! hog_done 500; then
            "${PY[@]}" IDEA1-covstdp/experiments/eval_hog.py \
                --config "$CFG_DIR/f3_b5_500.json" || rc=1
            sleep 2
        fi
        if [ $rc -eq 0 ] && ! hog_done 3300; then
            "${PY[@]}" IDEA1-covstdp/experiments/eval_hog.py \
                --config "$CFG_DIR/f3_b5_3300_eval.json" || rc=1
        fi
    } > "$CELL_LOG_DIR/hog.log" 2>&1
    local wall=$((SECONDS - t0))
    if [ $rc -eq 0 ] && hog_done 500 && hog_done 3300; then
        log_progress DONE "hog(500+3300)" "-" "$gpu" "$wall"
    else
        log_progress "FAIL(rc=$rc)" "hog(500+3300)" "-" "$gpu" "$wall"
    fi
    return $rc
}

run_cell() {  # run_cell <variant> <scale> <seed> <gpu>
    local v=$1 scale=$2 seed=$3 gpu=$4
    local exp="${v}_b5_${scale}"
    local cell_log="$CELL_LOG_DIR/${exp}__seed${seed}.log"
    local t0=$SECONDS rc=0
    {
        echo "===== $(date '+%F %T') cell ${exp} seed${seed} on GPU ${gpu} ====="
        export CUDA_VISIBLE_DEVICES=$gpu
        export MPLBACKEND=Agg
        if ! tracka_done "$exp" "$seed"; then
            if [ "$scale" = "500" ]; then
                "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                    "$CFG_DIR/${v}_b5_500.json" --train --eval --seed "$seed" || rc=1
            else
                "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                    "$CFG_DIR/${v}_b5_3300.json" --train --seed "$seed" || rc=1
                if [ $rc -eq 0 ]; then
                    sleep 2   # train/eval 两进程间留 logger 时间戳间隔
                    "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                        "$CFG_DIR/${v}_b5_3300_eval.json" --eval --seed "$seed" || rc=1
                fi
            fi
        else
            echo "[run_f0f3] 轨 A 已存在，跳过训练+评估"
        fi
        if [ $rc -eq 0 ] && ! trackb_done "$exp" "$seed"; then
            local eval_cfg="$CFG_DIR/${v}_b5_500.json"
            [ "$scale" = "3300" ] && eval_cfg="$CFG_DIR/${v}_b5_3300_eval.json"
            "${PY[@]}" IDEA1-covstdp/experiments/eval_retrieval.py \
                "$eval_cfg" --seed "$seed" || rc=1
        else
            [ $rc -ne 0 ] || echo "[run_f0f3] 轨 B 已存在，跳过"
        fi
    } > "$cell_log" 2>&1
    local wall=$((SECONDS - t0))
    if [ $rc -eq 0 ] && tracka_done "$exp" "$seed" && trackb_done "$exp" "$seed"; then
        log_progress DONE "$exp" "$seed" "$gpu" "$wall"
    else
        log_progress "FAIL(rc=$rc)" "$exp" "$seed" "$gpu" "$wall"
    fi
    return $rc
}

# ---------------- 双卡调度队列 ----------------
# HOG 优先（便宜确定性）；随后 F 组按 500 优先排序（先出迭代档结论）。
declare -A SLOT_PID=()
declare -A SLOT_CELL=()

hog_pending=0
if ! (hog_done 500 && hog_done 3300); then
    hog_pending=1
fi

pending=()
for scale in "${SCALES[@]}"; do          # 500 整批排在 3300 之前
    for v in "${VARIANTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp="${v}_b5_${scale}"
            if tracka_done "$exp" "$seed" && trackb_done "$exp" "$seed"; then
                echo "[run_f0f3] SKIP（已完成）$exp seed$seed"
            else
                pending+=("$v $scale $seed")
            fi
        done
    done
done
total=${#pending[@]}
echo "[run_f0f3] 待跑 $total 训练格 + HOG=$hog_pending，GPU 数 $NUM_GPUS，进度日志 $PROGRESS_LOG"
log_progress "BATCH_START" "remaining=$total hog=$hog_pending" "-" "-" 0

idx=0
hog_started=0
while [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ] || { [ "$hog_pending" = 1 ] && [ "$hog_started" = 0 ]; }; do
    # 回收已完成槽位
    for gpu in "${!SLOT_PID[@]}"; do
        if ! kill -0 "${SLOT_PID[$gpu]}" 2>/dev/null; then
            wait "${SLOT_PID[$gpu]}" 2>/dev/null
            echo "[run_f0f3] 槽位 gpu$gpu 完成：${SLOT_CELL[$gpu]}"
            unset "SLOT_PID[$gpu]" "SLOT_CELL[$gpu]"
        fi
    done
    # HOG 优先占空位
    if [ "$hog_pending" = 1 ] && [ "$hog_started" = 0 ]; then
        for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
            [ -n "${SLOT_PID[$gpu]:-}" ] && continue
            run_hog "$gpu" &
            SLOT_PID[$gpu]=$!
            SLOT_CELL[$gpu]="hog(500+3300)"
            echo "[run_f0f3] 启动 HOG（500+3300）→ gpu$gpu"
            hog_started=1
            break
        done
        [ "$hog_started" = 1 ] && hog_pending=0
    fi
    # 填空位
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        [ $idx -ge $total ] && break
        [ -n "${SLOT_PID[$gpu]:-}" ] && continue
        read -r v scale seed <<< "${pending[$idx]}"
        sleep 2   # 防 logger 秒级时间戳撞目录
        run_cell "$v" "$scale" "$seed" "$gpu" &
        SLOT_PID[$gpu]=$!
        SLOT_CELL[$gpu]="${v}_b5_${scale} seed$seed"
        echo "[run_f0f3] 启动 $((idx + 1))/$total：${v}_b5_${scale} seed$seed → gpu$gpu"
        idx=$((idx + 1))
    done
    if [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; then
        sleep 10
    fi
done

done_n=$(grep -c "| DONE |" "$PROGRESS_LOG" 2>/dev/null || echo 0)
fail_n=$(grep -c "| FAIL" "$PROGRESS_LOG" 2>/dev/null || echo 0)
log_progress "BATCH_END" "done_cells=$done_n fail_cells=$fail_n" "-" "-" 0
echo "[run_f0f3] 全部格子调度结束：累计 DONE=$done_n FAIL=$fail_n（见 $PROGRESS_LOG）"
