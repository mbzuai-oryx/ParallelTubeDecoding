#!/usr/bin/env python3
"""Report Tube Completion Latency (TCL) and Boxes Per Second (BPS)."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(path: Path, expected_decoding: str):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != 1:
                raise ValueError(f"{path}:{line_number}: unsupported schema.")
            if record.get("decoding") != expected_decoding:
                raise ValueError(
                    f"{path}:{line_number}: expected decoding={expected_decoding}."
                )
            records.append(record)

    complete = [
        record
        for record in records
        if record.get("trajectory_complete")
        and float(record.get("tcl_s", 0)) > 0
        and int(record.get("num_boxes", 0)) > 0
    ]
    if not complete:
        raise ValueError(f"No complete timed trajectories found in {path}.")
    latencies = [float(record["tcl_s"]) for record in complete]
    rates = [
        int(record["num_boxes"]) / float(record["tcl_s"])
        for record in complete
    ]
    return {
        "decoding": expected_decoding,
        "complete": len(complete),
        "excluded": len(records) - len(complete),
        "mean_tcl_s": statistics.fmean(latencies),
        "median_tcl_s": statistics.median(latencies),
        "mean_bps": statistics.fmean(rates),
        "median_bps": statistics.median(rates),
        "mean_boxes": statistics.fmean(
            int(record["num_boxes"]) for record in complete
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--ptd", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        summarize(args.quantized, "quantized"),
        summarize(args.ptd, "ptd"),
    ]
    print(
        f"{'Decoding':<12} {'Complete':>10} {'Excluded':>10} "
        f"{'Mean TCL (s)':>14} {'Median TCL':>12} "
        f"{'Mean BPS':>10} {'Median BPS':>12} {'Mean boxes':>11}"
    )
    for row in rows:
        print(
            f"{row['decoding']:<12} {row['complete']:>10} {row['excluded']:>10} "
            f"{row['mean_tcl_s']:>14.4f} {row['median_tcl_s']:>12.4f} "
            f"{row['mean_bps']:>10.4f} {row['median_bps']:>12.4f} "
            f"{row['mean_boxes']:>11.2f}"
        )


if __name__ == "__main__":
    main()
