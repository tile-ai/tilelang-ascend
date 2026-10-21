#!/usr/bin/env python3
"""Preprocessor: .tl → .py — expand Lock sugar into T.set_flag/T.wait_flag.

LOCK name sig_id producer consumer [count]
LOCKED pipe entry, ...: statement
ACQ pipe entry, ...
REL pipe entry, ...
INIT_LOCKS name, ...
DESTROY_LOCKS name, ...
"""

import argparse
import re
import sys

L = {}  # name → (sig, prod, cons, count)


def peer(n, p):
    _, pr, co, _ = L[n]
    return co if p == pr else pr


def sid(n, i):
    s = L[n][0]
    return str(s) if not i else f"{s} + {i}"


def ents(t):
    return [(m[1], m[2] or None) for m in re.finditer(r"(\w+)(?:\(([^)]+)\))?", t)]


def acq(ind, p, es):
    return [f'{ind}T.wait_flag("{peer(n, p)}", "{p}", {sid(n, i)})' for n, i in es]


def rel(ind, p, es):
    return [f'{ind}T.set_flag("{p}", "{peer(n, p)}", {sid(n, i)})' for n, i in es]


def expand(line):
    m = re.match(r"^(\s*)LOCK\s+(\w+)\s+(\d+)\s+(\w+)\s+(\w+)(?:\s+(\d+))?\s*$", line)
    if m:
        L[m[2]] = (int(m[3]), m[4], m[5], int(m[6] or 1))
        return []
    m = re.match(r"^(\s*)LOCKED\s+(\w+)\s+(.*?):\s*(.+)$", line)
    if m:
        e = ents(m[3])
        return acq(m[1], m[2], e) + [m[1] + m[4]] + rel(m[1], m[2], e)
    m = re.match(r"^(\s*)ACQ\s+(\w+)\s+(.+)$", line)
    if m:
        return acq(m[1], m[2], ents(m[3]))
    m = re.match(r"^(\s*)REL\s+(\w+)\s+(.+)$", line)
    if m:
        return rel(m[1], m[2], ents(m[3]))
    m = re.match(r"^(\s*)(INIT|DESTROY)_LOCKS\s+(.+)$", line)
    if m:
        ind, f = m[1], "set_flag" if m[2] == "INIT" else "wait_flag"
        return [
            f'{ind}T.{f}("{L[n][2]}", "{L[n][1]}", {L[n][0] + j})' for n in (x.strip() for x in m[3].split(",")) for j in range(L[n][3])
        ]
    return [line.rstrip("\n")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output")
    a = p.parse_args()
    out = "\n".join(r for line in open(a.input) for r in expand(line)) + "\n"
    (open(a.output, "w") if a.output else sys.stdout).write(out)
