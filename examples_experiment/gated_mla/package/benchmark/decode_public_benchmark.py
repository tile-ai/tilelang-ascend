"""Expand sanitized public benchmark data. No NPU, network, or benchmark execution."""

import argparse
import json
from pathlib import Path


def expand_public(public):
    records, pools = public["records"], public["sample_pools"]
    active = set()

    def decode(node):
        if isinstance(node, dict):
            if set(node) == {"$record"}:
                record_id = node["$record"]
                assert record_id not in active, "cyclic record"
                active.add(record_id)
                result = decode(records[record_id])
                active.remove(record_id)
                return result
            if set(node) == {"$samples"}:
                return list(pools[node["$samples"]]["values"])
            if set(node) == {"$sample"}:
                pool_id, index = node["$sample"]
                return pools[pool_id]["values"][index]
            return {k: decode(v) for k, v in node.items()}
        if isinstance(node, list):
            return [decode(v) for v in node]
        return node

    return decode(public["payload"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    public = json.loads(args.source.read_text(encoding="utf-8"))
    if public["schema"] != "gmla-source-pinned-public-benchmark-v2":
        raise ValueError("unsupported schema")
    expanded = expand_public(public)
    if args.output is not None:
        if args.output.exists():
            raise ValueError("output must be new")
        args.output.write_text(json.dumps(expanded, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "trial_count": len(expanded["trial_reports"]),
                "raw_samples": sum(len(p["values"]) for p in public["sample_pools"].values()),
                "redactions_are_not_reversed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
