import os
import subprocess
import itertools
import glob
import csv
import time
import argparse
import re
import sys
from datetime import datetime

def get_total_time(output_path, op_type_name):
    """
    解析 CSV 并仅提取 Total Time(us)
    """
    # 等待文件刷盘
    time.sleep(2.0)

    # 搜索最新的 profiler 输出
    search_pattern = os.path.join(output_path, "PROF_*", "mindstudio_profiler_output", "op_statistic_*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        return None

    # 获取最新的 CSV 文件
    target_csv = max(csv_files, key=os.path.getctime)

    try:
        with open(target_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 兼容不同版本的 msprof CSV 表头
                op_type = row.get('OP Type') or row.get('Op Type')
                if op_type == op_type_name:
                    val = row.get('Total Time(us)')
                    return float(val) if val and val != 'N/A' else None
    except Exception as e:
        print(f"    [!] CSV 解析警告 ({target_csv}): {e}")

    return None

def parse_error_info(full_output):
    """
    从完整的输出日志中提取 AssertionError 和 Mismatched 信息
    """
    # 1. 优先提取 Mismatch 信息
    match_mismatch = re.search(r"(Mismatched elements:.*)", full_output)
    if match_mismatch:
        return match_mismatch.group(1).strip()

    # 2. 如果没找到具体数值，但有 AssertionError
    if "AssertionError" in full_output:
        return "AssertionError"

    # 3. 兜底：Traceback
    if "Traceback (most recent call last):" in full_output:
        lines = full_output.strip().split('\n')
        # 提取最后几行有意义的报错
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and not line.startswith('[') and not line.startswith('Fri ') and not line.startswith('Traceback'):
                return f"RuntimeError: {line[:120]}..." 

    return ""

def run_and_analyze(cmd, output_path, kernel_name, desc):
    """
    执行命令，处理日志，返回 (耗时, 状态, 错误信息)
    """
    print(f"  > 正在执行 {desc}...")

    # 使用 errors='replace' 防止因编码问题导致的崩溃
    result = subprocess.run(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        errors='replace'
    )

    full_output = result.stdout
    exec_time = get_total_time(output_path, kernel_name)

    status = "Pass"
    error_info = ""

    # 失败判定逻辑
    has_error_keyword = ("Traceback (most recent call last):" in full_output) or \
                        ("AssertionError" in full_output) or \
                        ("Mismatched elements:" in full_output)

    # 只有当返回码非0 或者 发现了明确的错误关键字，才认为是 Fail
    if result.returncode != 0 or has_error_keyword:
        status = "Fail"
        error_info = parse_error_info(full_output)

        if not error_info:
            if result.returncode != 0:
                error_info = f"Crashed(Code:{result.returncode})"
            else:
                error_info = "UnknownError (Check Log)"

        print(f"    [X] {desc} 执行失败! 错误: {error_info}")

        print("-" * 30 + " 完整报错日志 " + "-" * 30)
        print(full_output)
        print("-" * 74)

    return exec_time, status, error_info

def generate_test_cases(args):
    """
    根据参数模式生成 B, S, H, D 的组合列表
    """
    b_list, s_list, h_list, d_list = args.B, args.S, args.H, args.D

    cases = []

    if args.iter_mode == 'product':
        # 全排列组合 (Cartesian Product)
        cases = list(itertools.product(b_list, s_list, h_list, d_list))
        print(f">>> 模式: Product (组合数量: {len(b_list)}x{len(s_list)}x{len(h_list)}x{len(d_list)} = {len(cases)})")
    else:
        # 索引对应 (Zip)，默认模式
        # 检查长度是否一致
        lengths = [len(b_list), len(s_list), len(h_list), len(d_list)]
        if len(set(lengths)) != 1:
            print("[错误] Zip 模式下，B/S/H/D 参数列表长度必须一致。")
            print(f"当前长度: B={len(b_list)}, S={len(s_list)}, H={len(h_list)}, D={len(d_list)}")
            sys.exit(1)

        cases = list(zip(b_list, s_list, h_list, d_list))
        print(f">>> 模式: Zip (测试用例数量: {len(cases)})")

    return cases

def run_benchmark():
    parser = argparse.ArgumentParser(description="Performance and Simulator Comparison Script")

    # === 核心配置参数 (B S H D) ===
    parser.add_argument("--B", type=int, nargs='+', default=[4], help="Batch sizes list (e.g. --B 1 2 4)")
    parser.add_argument("--S", type=int, nargs='+', default=[4096], help="Sequence lengths list")
    parser.add_argument("--H", type=int, nargs='+', default=[16], help="Num heads list")
    parser.add_argument("--D", type=int, nargs='+', default=[128], help="Head dim list")

    # === 遍历模式 ===
    parser.add_argument("--iter-mode", type=str, choices=['zip', 'product'], default='zip',
                        help="Mode to iterate dimensions. 'zip': index-wise (default), 'product': all combinations")

    # === 其他配置 ===
    parser.add_argument("--sim", action="store_true", help="Enable msprof op simulator mode")
    parser.add_argument("--log", type=str, default="./log", help="Root directory for logs")

    # TileLang 脚本相关
    parser.add_argument("--tl", type=str, 
                        default="./flash_attn_bhsd_cc_sync_auto_pipeline_h32_d512.py",
                        help="Path to TileLang python script")
    parser.add_argument("--kernel-tl", type=str, default="main_kernel", help="Kernel name for TileLang")

    # AscendC 脚本相关
    parser.add_argument("--ascendc", type=str, 
                        default="./flash_attn_bhsd_ascendc.py",
                        help="Path to AscendC python script")
    parser.add_argument("--kernel-ascendc", type=str, default="FlashAttentionScore", help="Kernel name for AscendC")

    parser.add_argument("--soc-version", type=str, default="Ascend910_9382", help="SoC version for msprof simulator")

    args = parser.parse_args()

    # 生成测试用例
    test_cases = generate_test_cases(args)

    mode_str = "SIMULATOR" if args.sim else "PERFORMANCE"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_cmp_{mode_str.lower()}_{timestamp}"
    session_dir = os.path.join(args.log, run_folder_name)
    os.makedirs(session_dir, exist_ok=True)

    summary_file = os.path.join(session_dir, f"comparison_report_{timestamp}.csv")

    headers = [
        'B', 'S', 'H', 'D', 'Mode', 
        'AC_Time(us)', 'TL_Time(us)', 'Ratio(AC/TL %)', 'Result_Status'
    ]

    rows_to_write = []

    print(">>> 对比评测启动！")
    print(f">>> 会话目录: {session_dir}")

    for b, s, h, d in test_cases:
        combo_name = f"B{b}_S{s}_H{h}_D{d}"
        combo_dir = os.path.join(session_dir, combo_name)
        print(f"\n[测试组合] {combo_name}")

        # --- 1. 运行 TileLang ---
        path_tl = os.path.join(combo_dir, "tilelang")
        os.makedirs(path_tl, exist_ok=True)
        app_cmd_tl = f"python {args.tl} --B {b} --S {s} --H {h} --D {d}"

        if args.sim:
            cmd_tl = f'msprof op simulator --soc-version={args.soc_version} --kernel-name="{args.kernel_tl}" --output={path_tl} --application="{app_cmd_tl}"'
        else:
            cmd_tl = f'msprof --output={path_tl} --application="{app_cmd_tl}"'

        tl_time, tl_status, tl_error = run_and_analyze(cmd_tl, path_tl, args.kernel_tl, "TileLang")

        # --- 2. 运行 AscendC ---
        path_ac = os.path.join(combo_dir, "ascendc")
        os.makedirs(path_ac, exist_ok=True)
        app_cmd_ac = f"python {args.ascendc} --B {b} --S {s} --H {h} --D {d}"

        if args.sim:
            cmd_ac = f'msprof op simulator --soc-version={args.soc_version} --kernel-name="{args.kernel_ascendc}" --output={path_ac} --application="{app_cmd_ac}"'
        else:
            cmd_ac = f'msprof --output={path_ac} --application="{app_cmd_ac}"'

        ac_time, ac_status, ac_error = run_and_analyze(cmd_ac, path_ac, args.kernel_ascendc, "AscendC")

        # --- 3. 汇总逻辑 ---
        ratio_str = "N/A"
        if tl_time and ac_time and tl_time > 0:
            ratio_str = f"{(ac_time / tl_time) * 100:.2f}%"

        # 生成 Result_Status
        if tl_status == "Pass" and ac_status == "Pass":
            final_status = "Pass"
        elif tl_status == "Fail" and ac_status == "Pass":
            final_status = f"TL_Fail: {tl_error}"
        elif tl_status == "Pass" and ac_status == "Fail":
            final_status = f"AC_Fail: {ac_error}"
        else:
            final_status = "Both_Fail"

        rows_to_write.append([
            b, s, h, d, mode_str,
            ac_time if ac_time else "N/A", 
            tl_time if tl_time else "N/A", 
            ratio_str,
            final_status
        ])

        # 终端简略输出
        print(f"  [结果] {final_status} | 比例: {ratio_str}")

    # 4. 保存到对比 CSV
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows_to_write)

    print(f"\n[任务结束] 对比报告已生成: {summary_file}")

if __name__ == "__main__":
    run_benchmark()
