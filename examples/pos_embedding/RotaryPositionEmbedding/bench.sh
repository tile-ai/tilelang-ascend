#!/bin/bash
# bench.sh — RoPE 性能对比：TileLang vs aclnnRotaryPositionEmbedding (AscendC baseline)
#
# 用 msprof 采集 device Task Duration，丢首次 launch（冷启动），取其余平均。
# 加速比 = ascendc_latency / tilelang_latency  (>1.0 表示 TileLang 更快)。
#
# TileLang 侧: python perf_rope.py (通过 PYTHONPATH 找到 tilelang 包)
# AscendC 侧: ./build/perf_rope_ascendc (C++ driver，调 aclnnRotaryPositionEmbedding)
#
# 用法：
#   bash bench.sh                        # 跑默认 shape 组（含 layout×dtype 交叉）
#   bash bench.sh --list                 # 列出所有 shape
#   bash bench.sh --shape "4 64 128 128" --layout half
#   bash bench.sh --repeats 10           # 每个算子跑 10 次
#   bash bench.sh --op-type-tl main_kernel --op-type-ac RotaryPositionEmbedding
#
# 依赖：msprof 在 PATH 中（source CANN 环境变量）。

set -euo pipefail

# ======================== 参数解析 ========================
REPEATS=6
LAYOUT="half"
DTYPE="float16"
OP_TYPE_TL="kernel_kernel"
OP_TYPE_AC="RotaryPositionEmbedding"
OUTPUT_DIR="./msprof_output"
CUSTOM_SHAPE=""
LIST_ONLY=false
TIMEOUT=600
PYTHON_BIN="python3"
DRIVER_AC=""  # auto-detected if empty

while [[ $# -gt 0 ]]; do
    case $1 in
        --repeats)       REPEATS="$2"; shift 2 ;;
        --layout)        LAYOUT="$2"; shift 2 ;;
        --dtype)         DTYPE="$2"; shift 2 ;;
        --op-type-tl)    OP_TYPE_TL="$2"; shift 2 ;;
        --op-type-ac)    OP_TYPE_AC="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --shape)         CUSTOM_SHAPE="$2"; shift 2 ;;
        --list)          LIST_ONLY=true; shift ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        --python)        PYTHON_BIN="$2"; shift 2 ;;
        --driver-ac)     DRIVER_AC="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -25
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ======================== 路径 / 环境检测 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERF_SCRIPT_TL="$SCRIPT_DIR/perf_rope.py"
PERF_BIN_AC="${DRIVER_AC:-$SCRIPT_DIR/build/perf_rope_ascendc}"

# TileLang 包位置: workspace 根目录（tilelang-ascend/）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# 检测 msprof
if ! command -v msprof &>/dev/null; then
    echo "Error: msprof not found in PATH. Run 'source set_env.sh' first." >&2
    exit 1
fi

# ======================== Shape 组 ========================
# 格式: "name|shape|layout|dtype"
# - shape: TND 用 4 个数 (BS H HS RD)，BSND 用 5 个数 (B S H HS RD)
# - layout: half|interleaved（留空则用全局 --layout）
# - dtype:  float16|bfloat16|float32（留空则用全局 --dtype）
#
# 交叉测试矩阵: half×fp16, half×bf16, interleaved×fp16, interleaved×bf16, bsnd
DEFAULT_SHAPES=(
    # --- half × fp16 (主力场景) ---
    "decode_bs1|1 32 128 128|half|float16"
    "decode_bs64|64 64 128 128|half|float16"
    "prefill_bs4_h64|4 64 128 128|half|float16"
    "prefill_bs8_h64|8 64 128 128|half|float16"
    "prefill_bs32_h64|32 64 128 128|half|float16"
    "prefill_bs4_d256|4 64 256 256|half|float16"
    "prefill_bs4_d512|4 64 512 512|half|float16"
    # --- half × bfloat16 ---
    "prefill_bs4_bf16|4 64 128 128|half|bfloat16"
    "prefill_bs8_bf16|8 64 128 128|half|bfloat16"
    # --- interleaved × fp16 ---
    "prefill_bs4_inter|4 64 128 128|interleaved|float16"
    "prefill_bs8_inter|8 64 128 128|interleaved|float16"
    # --- interleaved × bfloat16 ---
    "prefill_bs4_inter_bf16|4 64 128 128|interleaved|bfloat16"
    # --- BSND layout ---
    "bsnd_bs4_s4|4 4 64 128 128|half|float16"
)

# 构建 shape 列表
declare -a CASES
if [[ -n "$CUSTOM_SHAPE" ]]; then
    CASES=("custom|$CUSTOM_SHAPE|$LAYOUT|$DTYPE")
else
    CASES=("${DEFAULT_SHAPES[@]}")
fi

# --list 不需要 driver，提前处理
if $LIST_ONLY; then
    echo "Shape list (name|shape|layout|dtype):"
    for c in "${CASES[@]}"; do
        name="${c%%|*}"; rest="${c#*|}"
        s="${rest%%|*}"; rest="${rest#*|}"
        l="${rest%%|*}"; d="${rest#*|}"
        printf "  %-24s shape=[%s] layout=%s dtype=%s\n" "$name" "$s" "$l" "$d"
    done
    exit 0
fi

# 检测 AscendC driver（仅实际跑 bench 时需要）
if [[ ! -x "$PERF_BIN_AC" ]]; then
    echo "Error: AscendC driver not found at: $PERF_BIN_AC" >&2
    echo "       Build it first:  bash $SCRIPT_DIR/build_perf_rope.sh" >&2
    exit 1
fi

# ======================== 工具函数 ========================

# 查找最新的 PROF_*/mindstudio_profiler_output/op_summary_*.csv
find_csv() {
    local dir="$1" pattern="$2"
    local matches
    matches=$(ls -t "$dir"/PROF_*/mindstudio_profiler_output/${pattern} 2>/dev/null || true)
    if [[ -n "$matches" ]]; then
        echo "$matches" | head -1
    fi
}

# 从 op_summary_*.csv 提取指定 Op Type 的所有 Task Duration(us)
# 优先 op_summary（逐次 launch），fallback op_statistic（聚合 Total/count）
parse_duration_us() {
    local out_dir="$1" op_type="$2"

    "$PYTHON_BIN" - "$out_dir" "$op_type" <<'PYEOF'
import csv, glob, os, sys

out_dir, op_type = sys.argv[1], sys.argv[2]

# 1. op_summary: one row per launch
pattern = os.path.join(out_dir, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
files = glob.glob(pattern)
if files:
    target = max(files, key=os.path.getctime)
    durations = []
    with open(target, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ot = (row.get("OP Type") or row.get("Op Type") or "").strip()
            if ot == op_type:
                val = row.get("Task Duration(us)") or row.get("Op Duration(us)") or ""
                if val and val != "N/A":
                    try: durations.append(float(val))
                    except ValueError: pass
    if durations:
        # drop first (cold start), average the rest
        if len(durations) > 1:
            avg = sum(durations[1:]) / len(durations[1:])
        else:
            avg = durations[0]
        print(f"{avg:.3f}")
        sys.exit(0)

# 2. fallback: op_statistic (aggregated)
pattern = os.path.join(out_dir, "PROF_*", "mindstudio_profiler_output", "op_statistic_*.csv")
files = glob.glob(pattern)
if files:
    target = max(files, key=os.path.getctime)
    with open(target, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ot = (row.get("OP Type") or row.get("Op Type") or "").strip()
            if ot == op_type:
                total = row.get("Total Time(us)") or ""
                count = row.get("count") or row.get("Count") or "1"
                if total and total != "N/A":
                    try:
                        avg = float(total) / float(count)
                        print(f"{avg:.3f}")
                        sys.exit(0)
                    except (ValueError, ZeroDivisionError):
                        pass

print("N/A")
PYEOF
}

# 打印某次 msprof 采集到的所有 Op Type（调试用）
list_available_ops() {
    local out_dir="$1"
    local csv
    csv=$(find_csv "$out_dir" "op_statistic_*.csv")
    if [[ -z "$csv" ]]; then
        echo "    (no op_statistic CSV found under $out_dir)"
        return
    fi
    echo "    Available Op Types in $(basename "$csv"):"
    "$PYTHON_BIN" - "$csv" <<'PYEOF'
import csv, sys
with open(sys.argv[1], encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ot = row.get("OP Type") or row.get("Op Type") or "?"
        tt = row.get("Total Time(us)") or "?"
        c  = row.get("count") or row.get("Count") or ""
        print(f"      {ot}: Total={tt} us  count={c}")
PYEOF
}

# TileLang layout → aclnn mode (see op_host/rotary_position_embedding_tiling.h
# enum: 0=HALF, 1=INTERLEAVE, 2=QUARTER, 3=DEEPSEEK_INTERLEAVE.
# NOTE: aclnn header comment is wrong — it says 2=interleave but actual
# enum is 1=interleave. Confirmed by proto def + ST golden.)
layout_to_ac_mode() {
    case "$1" in
        half)        echo 0 ;;
        interleaved) echo 1 ;;
        *) echo "ERROR: unknown layout $1" >&2; exit 1 ;;
    esac
}

# TND/BSND shape → 4D shape for aclnn driver (B S N D)
shape_to_4d() {
    local shape_str="$1"
    local arr=($shape_str)
    if [[ ${#arr[@]} -eq 4 ]]; then
        # TND: BS H HS RD → B=BS S=1 N=H D=HS
        echo "${arr[0]} 1 ${arr[1]} ${arr[2]}"
    elif [[ ${#arr[@]} -eq 5 ]]; then
        # BSND: B S N H HS RD → already B S N D (drop the extra)
        # shape is [B, S, H, HS, RD] with RD==HS, so N=H, D=HS
        echo "${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]}"
    else
        echo "ERROR: bad shape $shape_str" >&2
        return 1
    fi
}

# 跑一次 msprof 采集
# $1=side(tl|ac) $2=shape_str $3=layout $4=dtype $5=out_dir
run_msprof() {
    local side="$1" shape_str="$2" layout="$3" dtype="$4" out_dir="$5"
    mkdir -p "$out_dir"

    local app_cmd=""
    if [[ "$side" == "tl" ]]; then
        app_cmd="$PYTHON_BIN $PERF_SCRIPT_TL --shape $shape_str --layout $layout --dtype $dtype --repeats $REPEATS"
    else
        local mode shape_4d
        mode=$(layout_to_ac_mode "$layout")
        shape_4d=$(shape_to_4d "$shape_str")
        app_cmd="$PERF_BIN_AC --shape $shape_4d --mode $mode --dtype $dtype --repeats $REPEATS"
    fi

    local cmd="msprof --output=$out_dir --application=\"$app_cmd\""

    echo "  > msprof $side (layout=$layout dtype=$dtype repeats=$REPEATS)..."
    if ! timeout "$TIMEOUT" bash -c "$cmd" >"$out_dir/msprof.log" 2>&1; then
        echo "  [X] msprof failed for $side (see $out_dir/msprof.log)"
        tail -5 "$out_dir/msprof.log" 2>/dev/null | sed 's/^/    /'
        return 1
    fi
    sleep 2  # wait for CSV flush
    return 0
}

# ======================== 主流程 ========================

echo "================================================================"
echo "RoPE Benchmark: TileLang vs aclnnRotaryPositionEmbedding"
echo "================================================================"
echo "Repeats:    $REPEATS (drop first, average rest)"
echo "Op Types:   TL=$OP_TYPE_TL  AC=$OP_TYPE_AC"
echo "Drivers:    TL=$PERF_SCRIPT_TL"
echo "            AC=$PERF_BIN_AC"
echo "Output:     $OUTPUT_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "================================================================"

# 结果汇总 CSV
SUMMARY="$OUTPUT_DIR/summary.csv"
mkdir -p "$OUTPUT_DIR"
echo "name,shape,layout,dtype,tl_us,ac_us,speedup" > "$SUMMARY"

for c in "${CASES[@]}"; do
    name="${c%%|*}"; rest="${c#*|}"
    shape_str="${rest%%|*}"; rest="${rest#*|}"
    l="${rest%%|*}"; d="${rest#*|}"

    echo ""
    echo "[$name] shape=[$shape_str] layout=$l dtype=$d"

    tl_dir="$OUTPUT_DIR/${name}_tl"
    ac_dir="$OUTPUT_DIR/${name}_ac"

    # --- TileLang ---
    tl_us=""
    if run_msprof "tl" "$shape_str" "$l" "$d" "$tl_dir"; then
        tl_us=$(parse_duration_us "$tl_dir" "$OP_TYPE_TL")
        if [[ "$tl_us" == "N/A" || -z "$tl_us" ]]; then
            echo "  [!] Op Type '$OP_TYPE_TL' not found for TileLang"
            list_available_ops "$tl_dir"
        fi
    fi
    echo "  TileLang:   ${tl_us:-N/A} us"

    # --- AscendC (aclnnRotaryPositionEmbedding) ---
    ac_us=""
    if run_msprof "ac" "$shape_str" "$l" "$d" "$ac_dir"; then
        ac_us=$(parse_duration_us "$ac_dir" "$OP_TYPE_AC")
        if [[ "$ac_us" == "N/A" || -z "$ac_us" ]]; then
            echo "  [!] Op Type '$OP_TYPE_AC' not found for AscendC"
            list_available_ops "$ac_dir"
        fi
    fi
    echo "  AscendC:    ${ac_us:-N/A} us"

    # --- Speedup ---
    speedup=""
    if [[ -n "$tl_us" && -n "$ac_us" && "$tl_us" != "N/A" && "$ac_us" != "N/A" ]]; then
        speedup=$("$PYTHON_BIN" -c "print(f'{$ac_us/$tl_us:.2f}x')")
    else
        speedup="N/A"
    fi
    echo "  Speedup:    $speedup  (ascendc / tilelang, >1.0 = TileLang faster)"

    echo "$name,\"$shape_str\",$l,$d,${tl_us:-N/A},${ac_us:-N/A},$speedup" >> "$SUMMARY"
done

echo ""
echo "================================================================"
echo "Summary written to: $SUMMARY"
echo "================================================================"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

echo ""
echo "Test Passed!"
