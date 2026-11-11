"""Bound a task-owned Linux process group and its observed descendants."""

from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_table():
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            raw = (path / "stat").read_text()
            # comm may contain spaces or ')'; remaining fields begin at state.
            fields = raw[raw.rfind(")") + 2 :].split()
            result[int(path.name)] = dict(state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]), start_ticks=int(fields[19]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def observe(root_pid, known):
    table = process_table()
    live = {pid for pid, info in known.items() if pid in table and table[pid]["start_ticks"] == info["start_ticks"]}
    if root_pid in table and (root_pid not in known or table[root_pid]["start_ticks"] == known[root_pid]["start_ticks"]):
        live.add(root_pid)
    changed = True
    while changed:
        changed = False
        for pid, info in table.items():
            if info["ppid"] in live and pid not in live:
                live.add(pid)
                changed = True
    for pid in live:
        known[pid] = table[pid]
    return table


def cleanup_owned(root_pid, known):
    """Never signal a reused PID or an unrelated process group."""
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        table = observe(root_pid, known)
        live = {
            pid
            for pid, info in known.items()
            if pid in table and table[pid]["start_ticks"] == info["start_ticks"] and table[pid]["state"] != "Z"
        }
        # start_new_session creates a unique group rooted at this child. Only
        # signal a group if every current member is one of the verified PIDs.
        groups = {table[pid]["pgid"] for pid in live}
        for group in groups:
            members = {pid for pid, info in table.items() if info["pgid"] == group and info["state"] != "Z"}
            if members and members <= live and group != os.getpgrp():
                with suppress(ProcessLookupError):
                    os.killpg(group, sig)
        for pid in sorted(live, reverse=True):
            fresh = process_table().get(pid)
            if fresh and fresh["start_ticks"] == known[pid]["start_ticks"]:
                with suppress(ProcessLookupError):
                    os.kill(pid, sig)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            current = observe(root_pid, known)
            if not any(
                pid in current and current[pid]["start_ticks"] == info["start_ticks"] and current[pid]["state"] != "Z"
                for pid, info in known.items()
            ):
                return
            time.sleep(0.1)
    raise RuntimeError("A recorded task-owned process survived SIGKILL; report and stop")


def run_owned_process(argv, *, cwd, env, stream, timeout, process_record, started):
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    started(process.pid)
    known = {}
    deadline = time.monotonic() + timeout
    failure = None
    try:
        while True:
            table = observe(process.pid, known)
            code = process.poll()
            if code is not None:
                if code:
                    failure = f"nonzero returncode {code}"
                    cleanup_owned(process.pid, known)
                    return code
                live_children = any(
                    pid != process.pid and pid in table and table[pid]["start_ticks"] == info["start_ticks"] and table[pid]["state"] != "Z"
                    for pid, info in known.items()
                )
                if not live_children:
                    return code
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            time.sleep(min(0.5, remaining))
    except BaseException as error:
        failure = repr(error)
        cleanup_owned(process.pid, known)
        raise
    finally:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1)
        process_record.write_text(
            json.dumps(
                dict(root_pid=process.pid, observed=known, timeout_seconds=timeout, failure=failure, returncode=process.poll()), indent=2
            )
            + "\n"
        )
