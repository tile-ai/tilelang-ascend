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
    time.sleep(2.0)

    search_pattern = os.path.join(output_path, "PROF_*", "mindstudio_profiler_output", "op_statistic_*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        return None

    target_csv = max(csv_files, key=os.path.getctime)

    try:
        with open(target_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_type = row.get("OP Type") or row.get("Op Type")
                if op_type == op_type_name:
                    val = row.get("Total Time(us)")
                    return float(val) if val and val != "N/A" else None
    except Exception as e:
        print(f"    [!] CSV parse warning ({target_csv}): {e}")

    return None


def parse_error_info(full_output):
    match_mismatch = re.search(r"(Mismatched elements:.*)", full_output)
    if match_mismatch:
        return match_mismatch.group(1).strip()

    if "AssertionError" in full_output:
        return "AssertionError"

    if "Traceback (most recent call last):" in full_output:
        lines = full_output.strip().split("\n")
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and not line.startswith("[") and not line.startswith("Fri ") and not line.startswith("Traceback"):
                return f"RuntimeError: {line[:120]}..."

    return ""


import signal


def run_and_analyze(cmd, output_path, kernel_name, desc, timeout=None):
    print(f"  > Running {desc} (Timeout={timeout})..." if timeout else f"  > Running {desc}...")

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )

    try:
        stdout, _ = process.communicate(timeout=timeout)
        returncode = process.returncode
        status = "Pass"
    except subprocess.TimeoutExpired:
        print(f"    [!] {desc} timeout (>{timeout}s), sending SIGINT, waiting for data generation...")
        os.killpg(os.getpgid(process.pid), signal.SIGINT)

        stdout, _ = process.communicate()

        returncode = process.returncode
        status = "Timeout"

    full_output = stdout
    exec_time = get_total_time(output_path, kernel_name)

    error_info = ""

    has_error_keyword = (
        ("Traceback (most recent call last):" in full_output)
        or ("AssertionError" in full_output)
        or ("Mismatched elements:" in full_output)
    )

    if (returncode != 0 and status != "Timeout") or has_error_keyword:
        if status != "Timeout":
            status = "Fail"
        error_info = parse_error_info(full_output)

        if not error_info:
            if returncode != 0:
                error_info = f"Crashed(Code:{returncode})"
            else:
                error_info = "UnknownError (Check Log)"

        if status == "Timeout":
            error_info = f"Timeout & {error_info}"

        print(f"    [X] {desc} execution failed! Status: {status}, Error: {error_info}")

        print("-" * 30 + " Full Error Log " + "-" * 30)
        print(full_output)
        print("-" * 74)
    elif status == "Timeout":
        print(f"    [!] {desc} timeout, attempted to capture performance data.")

    return exec_time, status, error_info


def generate_test_cases(args):
    b_list, s_list, h_list, d_list = args.B, args.S, args.H, args.D

    q_heads_list = args.q_heads or h_list
    kv_heads_list = args.kv_heads or h_list

    cases = []

    if args.iter_mode == "product":
        if args.q_heads is None and args.kv_heads is None:
            cases = [(b, s, h, h, d) for b, s, h, d in itertools.product(b_list, s_list, args.H, d_list)]
        else:
            cases = list(itertools.product(b_list, s_list, q_heads_list, kv_heads_list, d_list))
        print(f">>> Mode: Product (count: {len(cases)})")
    else:
        lengths = [
            len(b_list),
            len(s_list),
            len(q_heads_list),
            len(kv_heads_list),
            len(d_list),
        ]
        if len(set(lengths)) != 1:
            print("In Zip mode, B/S/Q_H/KV_H/D parameter lists must have equal length.")
            print(f"Current lengths: B={len(b_list)}, S={len(s_list)}, Q_H={len(q_heads_list)}, KV_H={len(kv_heads_list)}, D={len(d_list)}")
            sys.exit(1)

        cases = list(zip(b_list, s_list, q_heads_list, kv_heads_list, d_list))
        print(f">>> Mode: Zip (test cases: {len(cases)})")

    return cases


from multiprocessing import Pool


def run_performance_case(pack):
    b, s, q_h, kv_h, d, args, session_dir, mode_str = pack

    combo_name = f"B{b}_S{s}_Q{q_h}_KV{kv_h}_D{d}"
    combo_dir = os.path.join(session_dir, combo_name)
    print(f"\n[Test Case] {combo_name}")

    path_tl = os.path.join(combo_dir, "tilelang")
    os.makedirs(path_tl, exist_ok=True)
    app_cmd_tl = f"{sys.executable} {args.tl} --B {b} --S {s} --q-heads {q_h} --kv-heads {kv_h} --D {d}"

    cmd_tl = f'msprof --output={path_tl} --application="{app_cmd_tl}"'
    tl_time, tl_status, tl_error = run_and_analyze(cmd_tl, path_tl, args.kernel_tl, "TileLang", timeout=args.timeout)

    path_ac = os.path.join(combo_dir, "ascendc")
    os.makedirs(path_ac, exist_ok=True)
    app_cmd_ac = f"{sys.executable} {args.ascendc} --B {b} --S {s} --q-heads {q_h} --kv-heads {kv_h} --D {d} --no-check"

    cmd_ac = f'msprof --output={path_ac} --application="{app_cmd_ac}"'
    ac_time, ac_status, ac_error = run_and_analyze(cmd_ac, path_ac, args.kernel_ascendc, "AscendC", timeout=args.timeout)

    ratio_str = "N/A"
    if tl_time and ac_time and tl_time > 0:
        ratio_str = f"{(ac_time / tl_time) * 100:.2f}%"

    if tl_status == "Pass" and ac_status == "Pass":
        final_status = "Pass"
    elif tl_status == "Fail" and ac_status == "Pass":
        final_status = f"TL_Fail: {tl_error}"
    elif tl_status == "Pass" and ac_status == "Fail":
        final_status = f"AC_Fail: {ac_error}"
    else:
        final_status = "Both_Fail"

    row = [
        b,
        s,
        q_h,
        kv_h,
        d,
        mode_str,
        ac_time if ac_time else "N/A",
        tl_time if tl_time else "N/A",
        ratio_str,
        final_status,
    ]

    print(f"  [Result] {final_status} | Ratio: {ratio_str}")

    return (0, row)


def run_sim_task(pack):
    case_idx, task_type, b, s, q_h, kv_h, d, args, session_dir = pack

    combo_name = f"B{b}_S{s}_Q{q_h}_KV{kv_h}_D{d}"
    combo_dir = os.path.join(session_dir, combo_name)

    exec_time = None
    status = "Fail"
    error_info = "Unknown"

    if task_type == "TL":
        path_tl = os.path.join(combo_dir, "tilelang")
        os.makedirs(path_tl, exist_ok=True)
        app_cmd_tl = f"{sys.executable} {args.tl} --B {b} --S {s} --q-heads {q_h} --kv-heads {kv_h} --D {d} --no-check"
        cmd_tl = f'msprof op simulator --soc-version={args.soc_version} --kernel-name="{args.kernel_tl}" --output={path_tl} --application="{app_cmd_tl}"'
        exec_time, status, error_info = run_and_analyze(
            cmd_tl,
            path_tl,
            args.kernel_tl,
            f"TileLang [{combo_name}]",
            timeout=args.timeout,
        )

    elif task_type == "AC":
        path_ac = os.path.join(combo_dir, "ascendc")
        os.makedirs(path_ac, exist_ok=True)
        app_cmd_ac = f"{sys.executable} {args.ascendc} --B {b} --S {s} --q-heads {q_h} --kv-heads {kv_h} --D {d} --no-check"
        cmd_ac = f'msprof op simulator --soc-version={args.soc_version} --kernel-name="{args.kernel_ascendc}" --output={path_ac} --application="{app_cmd_ac}"'
        exec_time, status, error_info = run_and_analyze(
            cmd_ac,
            path_ac,
            args.kernel_ascendc,
            f"AscendC [{combo_name}]",
            timeout=args.timeout,
        )

    return (case_idx, task_type, exec_time, status, error_info)


def run_benchmark():
    parser = argparse.ArgumentParser(description="Performance and Simulator Comparison Script")

    parser.add_argument(
        "--B",
        type=int,
        nargs="+",
        default=[4],
        help="Batch sizes list (e.g. --B 1 2 4)",
    )
    parser.add_argument("--S", type=int, nargs="+", default=[4096], help="Sequence lengths list")
    parser.add_argument("--H", type=int, nargs="+", default=[16], help="Num heads list")
    parser.add_argument("--q-heads", type=int, nargs="+", default=None, help="Num Q heads list")
    parser.add_argument("--kv-heads", type=int, nargs="+", default=None, help="Num KV heads list")
    parser.add_argument("--D", type=int, nargs="+", default=[128], help="Head dim list")

    parser.add_argument(
        "--iter-mode",
        type=str,
        choices=["zip", "product"],
        default="zip",
        help="Mode to iterate dimensions. 'zip': index-wise (default), 'product': all combinations",
    )

    parser.add_argument("--sim", action="store_true", help="Enable msprof op simulator mode")
    parser.add_argument("--log", type=str, default="./log", help="Root directory for logs")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for multiprocessing",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for each msprof execution (sends SIGINT on timeout)",
    )

    parser.add_argument(
        "--tl",
        type=str,
        default="./flash_attn_bhsd_cc_sync_auto_pipeline_h32_d512.py",
        help="Path to TileLang python script",
    )
    parser.add_argument("--kernel-tl", type=str, default="main_kernel", help="Kernel name for TileLang")

    parser.add_argument(
        "--ascendc",
        type=str,
        default="./flash_attn_bhsd_ascendc.py",
        help="Path to AscendC python script",
    )
    parser.add_argument(
        "--kernel-ascendc",
        type=str,
        default="FlashAttentionScore",
        help="Kernel name for AscendC",
    )

    parser.add_argument(
        "--soc-version",
        type=str,
        default="Ascend910_9382",
        help="SoC version for msprof simulator",
    )

    args = parser.parse_args()

    test_cases = generate_test_cases(args)

    mode_str = "SIMULATOR" if args.sim else "PERFORMANCE"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_cmp_{mode_str.lower()}_{timestamp}"
    session_dir = os.path.join(args.log, run_folder_name)
    os.makedirs(session_dir, exist_ok=True)

    summary_file = os.path.join(session_dir, f"comparison_report_{timestamp}.csv")

    headers = [
        "B",
        "S",
        "Q_H",
        "KV_H",
        "D",
        "Mode",
        "AC_Time(us)",
        "TL_Time(us)",
        "Ratio(AC/TL %)",
        "Result_Status",
    ]

    print(">>> Benchmark started!")
    print(f">>> Session directory: {session_dir}")

    final_rows = []

    if args.sim:
        print(f">>> Multiprocessing Simulation mode enabled (Workers: {args.workers})")
        sim_tasks = []
        for idx, case in enumerate(test_cases):
            sim_tasks.append((idx, "TL", *case, args, session_dir))
            sim_tasks.append((idx, "AC", *case, args, session_dir))

        aggregated = {}

        with Pool(processes=args.workers) as pool:
            for res in pool.imap_unordered(run_sim_task, sim_tasks):
                c_idx, t_type, time_val, status, error = res

                if c_idx not in aggregated:
                    aggregated[c_idx] = {}
                aggregated[c_idx][t_type] = (time_val, status, error)

                if "TL" in aggregated[c_idx] and "AC" in aggregated[c_idx]:
                    b, s, q_h, kv_h, d = test_cases[c_idx]
                    tl_res = aggregated[c_idx]["TL"]
                    ac_res = aggregated[c_idx]["AC"]

                    tl_time, tl_status, tl_error = tl_res
                    ac_time, ac_status, ac_error = ac_res

                    ratio_str = "N/A"
                    if tl_time and ac_time and tl_time > 0:
                        ratio_str = f"{(ac_time / tl_time) * 100:.2f}%"

                    if tl_status == "Pass" and ac_status == "Pass":
                        final_status = "Pass"
                    elif tl_status == "Fail" and ac_status == "Pass":
                        final_status = f"TL_Fail: {tl_error}"
                    elif tl_status == "Pass" and ac_status == "Fail":
                        final_status = f"AC_Fail: {ac_error}"
                    else:
                        final_status = "Both_Fail"

                    combo_name = f"B{b}_S{s}_Q{q_h}_KV{kv_h}_D{d}"
                    print(f"\n[Test Case] {combo_name} (Completed)")
                    print(f"  [Result] {final_status} | Ratio: {ratio_str}")

                    row = [
                        b,
                        s,
                        q_h,
                        kv_h,
                        d,
                        mode_str,
                        ac_time if ac_time else "N/A",
                        tl_time if tl_time else "N/A",
                        ratio_str,
                        final_status,
                    ]
                    final_rows.append((c_idx, row))

    else:
        print(">>> Single-process Performance mode enabled")
        tasks = []
        for _, case in enumerate(test_cases):
            tasks.append((*case, args, session_dir, mode_str))

        for idx, task in enumerate(tasks):
            _, row = run_performance_case(task)
            final_rows.append((idx, row))

    final_rows.sort(key=lambda x: x[0])
    rows_to_write = [r[1] for r in final_rows]

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows_to_write)

    print(f"\n[Done] Comparison report generated: {summary_file}")


if __name__ == "__main__":
    run_benchmark()
