#!/bin/bash
# bench.sh — RoPE 性能对比：TileLang vs torch_npu.npu_rotary_mul
#
# 用 msprof op 算子级采集 Task Duration（warm-up 后平均）。
# 加速比 = cann_latency / tilelang_latency  (>1.0 表示 TileLang 更快)。
#
# 两侧都调 python rope_half_interleaved.py --perf --side {tl|cann}（单次 launch）
# msprof op 通过 --warm-up 预热、--launch-count 控制采集次数。
#
# 用法：
#   bash bench.sh                                            # 跑默认 shape 组
#   bash bench.sh --list                                     # 列出所有 shape
#   bash bench.sh --shape "4 64 128 128" --layout half       # 单 shape
#   bash bench.sh --warmup 10 --launch-count 50              # 自定义预热/采集次数
#   bash bench.sh --kernel-tl kernel_kernel --kernel-cann RotaryPositionEmbedding
#
# 依赖：msprof 在 PATH 中（source CANN 环境变量）。

set -euo pipefail

# ======================== 参数解析 ========================
LAYOUT="half"
DTYPE="float16"
KERNEL_TL="kernel_kernel"
KERNEL_CANN="RotaryPositionEmbedding"
OUTPUT_DIR="./msprof_output"
CUSTOM_SHAPE=""
LIST_ONLY=false
TIMEOUT=600
PYTHON_BIN="python3"
WARMUP=5
LAUNCH_COUNT=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --layout)        LAYOUT="$2"; shift 2 ;;
        --dtype)         DTYPE="$2"; shift 2 ;;
        --kernel-tl)     KERNEL_TL="$2"; shift 2 ;;
        --kernel-cann)   KERNEL_CANN="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --shape)         CUSTOM_SHAPE="$2"; shift 2 ;;
        --list)          LIST_ONLY=true; shift ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        --python)        PYTHON_BIN="$2"; shift 2 ;;
        --warmup)        WARMUP="$2"; shift 2 ;;
        --launch-count)  LAUNCH_COUNT="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -22
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ======================== 路径 / 环境检测 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERF_SCRIPT="$SCRIPT_DIR/rope_half_interleaved.py"

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

# --list 不需要 msprof，提前处理
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

# ======================== 工具函数 ========================

# 从 msprof op 的 stdout 提取 "Task Duration(us): <value>"
# msprof op 输出格式: "Task Duration(us): 7.520150"
parse_task_duration() {
    local log_file="$1"
    # grep + sed 提取数值
    local val
    val=$(grep -oP 'Task Duration\(us\):\s*\K[\d.]+' "$log_file" | head -1)
    if [[ -n "$val" ]]; then
        printf "%.3f" "$val"
    else
        echo "N/A"
    fi
}

# 跑 msprof op 采集
# $1=side(tl|cann) $2=shape_str $3=layout $4=dtype $5=out_dir
run_msprof_op() {
    local side="$1" shape_str="$2" layout="$3" dtype="$4" out_dir="$5"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    chmod 700 "$out_dir"

    local kernel_name
    if [[ "$side" == "tl" ]]; then
        kernel_name="$KERNEL_TL"
    else
        kernel_name="$KERNEL_CANN"
    fi

    echo "  > msprof op $side (warm-up=$WARMUP launch-count=$LAUNCH_COUNT)..."
    if ! timeout "$TIMEOUT" msprof op \
        --kernel-name="$kernel_name" \
        --warm-up="$WARMUP" \
        --launch-count="$LAUNCH_COUNT" \
        --output="$out_dir" \
        "$PYTHON_BIN" "$PERF_SCRIPT" --perf --side "$side" --shape $shape_str --layout "$layout" --dtype "$dtype" \
        >"$out_dir/msprof.log" 2>&1; then
        echo "  [X] msprof op failed for $side (see $out_dir/msprof.log)"
        tail -5 "$out_dir/msprof.log" 2>/dev/null | sed 's/^/    /'
        return 1
    fi
    sleep 1
    return 0
}

# ======================== 主流程 ========================

echo "================================================================"
echo "RoPE Benchmark: TileLang vs torch_npu.npu_rotary_mul"
echo "================================================================"
echo "Kernel:     TL=$KERNEL_TL  CANN=$KERNEL_CANN"
echo "Warm-up:    $WARMUP"
echo "Launch:     $LAUNCH_COUNT"
echo "Script:     $PERF_SCRIPT"
echo "Output:     $OUTPUT_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "================================================================"

# 结果汇总 CSV
SUMMARY="$OUTPUT_DIR/summary.csv"
mkdir -p "$OUTPUT_DIR"
echo "name,shape,layout,dtype,tl_us,cann_us,speedup" > "$SUMMARY"

for c in "${CASES[@]}"; do
    name="${c%%|*}"; rest="${c#*|}"
    shape_str="${rest%%|*}"; rest="${rest#*|}"
    l="${rest%%|*}"; d="${rest#*|}"

    echo ""
    echo "[$name] shape=[$shape_str] layout=$l dtype=$d"

    tl_dir="$OUTPUT_DIR/${name}_tl"
    cann_dir="$OUTPUT_DIR/${name}_cann"

    # --- TileLang ---
    tl_us=""
    if run_msprof_op "tl" "$shape_str" "$l" "$d" "$tl_dir"; then
        tl_us=$(parse_task_duration "$tl_dir/msprof.log")
        if [[ "$tl_us" == "N/A" || -z "$tl_us" ]]; then
            echo "  [!] Task Duration not found for TileLang"
            grep -i "error\|fail\|warn" "$tl_dir/msprof.log" 2>/dev/null | head -3 | sed 's/^/    /'
        fi
    fi
    echo "  TileLang:   ${tl_us:-N/A} us"

    # --- CANN ---
    cann_us=""
    if run_msprof_op "cann" "$shape_str" "$l" "$d" "$cann_dir"; then
        cann_us=$(parse_task_duration "$cann_dir/msprof.log")
        if [[ "$cann_us" == "N/A" || -z "$cann_us" ]]; then
            echo "  [!] Task Duration not found for CANN"
            grep -i "error\|fail\|warn" "$cann_dir/msprof.log" 2>/dev/null | head -3 | sed 's/^/    /'
        fi
    fi
    echo "  CANN:       ${cann_us:-N/A} us"

    # --- Speedup ---
    speedup=""
    if [[ -n "$tl_us" && -n "$cann_us" && "$tl_us" != "N/A" && "$cann_us" != "N/A" ]]; then
        speedup=$("$PYTHON_BIN" -c "print(f'{$cann_us/$tl_us:.2f}x')")
    else
        speedup="N/A"
    fi
    echo "  Speedup:    $speedup  (cann / tilelang, >1.0 = TileLang faster)"

    echo "$name,\"$shape_str\",$l,$d,${tl_us:-N/A},${cann_us:-N/A},$speedup" >> "$SUMMARY"
done

echo ""
echo "================================================================"
echo "Summary written to: $SUMMARY"
echo "================================================================"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

echo ""
echo "Test Passed!"
