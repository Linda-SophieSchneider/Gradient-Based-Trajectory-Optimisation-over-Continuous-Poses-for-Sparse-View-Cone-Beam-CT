"""Emit the LaTeX body of tab:motion from trajectory_motion.csv.

Answers "how far does the continuous operator actually move the poses?" for the
Defrise flange with noise-free selection: per initialisation, reachable set,
and view budget, the great-circle displacement between the initial and the
optimised pose set.

    python figures/make_pose_motion_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "experiments/results/lof_plate_20260825/trajectory_motion.csv"

# (csv stem prefix, init label, reachable-set label)
GROUPS = [
    ("flange_greedy_adam_circle", r"Tuy-greedy", r"single-axis circle"),
    ("flange_band_bundle", r"Tuy-greedy", r"bench band"),
    ("flange_sphere_warm_bundle", r"VCLS", r"free sphere"),
    ("flange_greedy_adam_composite", r"Tuy-greedy", r"free sphere"),
    ("flange_greedy_adam_bundle_two_axis", r"Tuy-greedy", r"two-axis gantry"),
    ("flange_greedy_adam_bundle_carm", r"Tuy-greedy", r"limited C-arm"),
]
BUDGETS = (20, 40, 80)


def main():
    rows = {r["example"]: r for r in csv.DictReader(CSV.open())}
    lines = []
    for stem, init, manifold in GROUPS:
        present = [k for k in BUDGETS if f"{stem}_k{k}" in rows]
        for i, k in enumerate(present):
            r = rows[f"{stem}_k{k}"]
            head = (rf"\multirow{{{len(present)}}}{{*}}{{{init}}} & "
                    rf"\multirow{{{len(present)}}}{{*}}{{{manifold}}}"
                    if i == 0 else " & ")
            pct = 100.0 * float(r["frac_gt_1deg"])
            lines.append(
                f"{head} & {k} & {float(r['median_deg']):.2f} & "
                f"{float(r['p95_deg']):.1f} & {float(r['max_deg']):.1f} & "
                f"{pct:.0f} \\\\")
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    print("\n".join(lines))

    # numbers used in the prose
    for stem, init, manifold in GROUPS:
        vals = [rows[f"{stem}_k{k}"] for k in BUDGETS if f"{stem}_k{k}" in rows]
        if not vals:
            continue
        med = [float(v["median_deg"]) for v in vals]
        mx = [float(v["max_deg"]) for v in vals]
        pct = [100 * float(v["frac_gt_1deg"]) for v in vals]
        print(f"%% {init:11s} {manifold:16s} median {min(med):.2f}-{max(med):.2f} "
              f"max {min(mx):.1f}-{max(mx):.1f} >1deg {min(pct):.0f}-{max(pct):.0f}%")
    for name in ("flange_limited_wedge120_cov", "flange_limited_wedge120_roi_abs",
                 "flange_limited_lamino_cov", "flange_limited_lamino_roi_abs"):
        r = rows.get(name)
        if r:
            print(f"%% {name}: median {float(r['median_deg']):.2f} "
                  f"max {float(r['max_deg']):.1f} "
                  f">1deg {100*float(r['frac_gt_1deg']):.0f}%")


if __name__ == "__main__":
    main()
