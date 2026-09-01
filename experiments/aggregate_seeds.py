"""Aggregate multi-seed result CSVs to mean +/- std per (method, k).

Reads an input CSV produced by ``experiments/run.py --seeds ...`` and
writes a companion CSV with one row per (method, k) cell, columns
``<metric>_mean`` and ``<metric>_std`` for each metric.  Use this on
the R1 and R2 outputs to produce the numbers that go into the
review-response tables.

Example
-------
::

    python experiments/aggregate_seeds.py \\
        experiments/results/r1_milp_mild.csv \\
        --out experiments/results/r1_milp_mild_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ("psnr", "ssim", "nrmse", "hfen")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input CSV with one row per (method, k, seed)")
    p.add_argument("--out", default=None,
                   help="Output CSV; defaults to <input>_summary.csv")
    args = p.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else in_path.with_name(
        in_path.stem + "_summary.csv"
    )

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {m: [] for m in METRICS}
    )
    with in_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("method", ""), row.get("k", ""))
            for m in METRICS:
                if m in row and row[m] != "":
                    grouped[key][m].append(float(row[m]))

    fields = ["method", "k", "n_seeds"]
    for m in METRICS:
        fields += [f"{m}_mean", f"{m}_std"]

    out_rows = []
    for (method, k), metrics in grouped.items():
        out = {"method": method, "k": k,
               "n_seeds": max(len(v) for v in metrics.values())}
        for m in METRICS:
            vals = metrics[m]
            if vals:
                out[f"{m}_mean"] = f"{mean(vals):.4f}"
                out[f"{m}_std"]  = f"{(stdev(vals) if len(vals) > 1 else 0.0):.4f}"
        out_rows.append(out)

    # Sort by method then k for readable output.
    def _k_int(s):
        try:
            return int(s)
        except (ValueError, TypeError):
            return s
    out_rows.sort(key=lambda r: (r["method"], _k_int(r["k"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote {out_path} with {len(out_rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
