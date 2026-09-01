"""Pose dumps for the two kinematics regimes missing from ``trajectory_poses``.

``render_trajectory_evolution`` already dumps the limited C-arm and the free
sphere. The pose-motion ladder of the paper also needs the single-axis circle
and the bench band at the same budget and protocol, so this script reuses that
module's start/final pair helper and its ``render`` writer and adds nothing of
its own beyond the two missing regimes.

Run from the repo root:
    python -m experiments.studies.dump_kinematics_poses
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffct")

import mlx.core as mx

from experiments.run import _load_mlx_stack, _load_phantom_pair, _resolve_geometry
from experiments.studies.kinematics import _build_cache
from experiments.studies.render_kinematics_recon import _with_band_envelope
import experiments.studies.render_trajectory_evolution as tev

K = 80
KMAX = 360


def main() -> None:
    tev.OUT.mkdir(parents=True, exist_ok=True)
    stack = _load_mlx_stack()
    spec = {"type": "milp_npy", "path": "data/lof_flange_v3.npy",
            "resolution": 384}
    geom = _resolve_geometry("milp", 384)
    vol, _ = _load_phantom_pair(spec, stack)
    mx.eval(vol)
    sid = geom["sid"]

    cand = stack["sample_unit_sphere"](KMAX, seed=tev.SEED) * sid
    pre = _build_cache(stack, vol, geom, cand, 1e-3, tev.SEED)

    s, f = tev._sphere_pair(stack, "greedy_adam_circle", K, vol, geom, pre, KMAX)
    tev.render("flange_greedy_adam_circle_k80",
               f"Flange, cold circle, single-axis, $k={K}$", s, f, sid)

    mod, saved = _with_band_envelope()
    try:
        s, f = tev._sphere_pair(stack, "greedy_adam_bundle_carm", K, vol, geom,
                                pre, KMAX)
        tev.render("flange_band_bundle_k80",
                   f"Flange, cold bundle, bench band, $k={K}$", s, f, sid)
    finally:
        mod.CArmTwoAxisGantry, mod.SmoothTwoAxisGantry = saved

    _merge_stats()
    print("Done.", flush=True)


def _merge_stats() -> None:
    """Append the new rows to the motion CSV without dropping the old ones."""
    path = Path(tev.STATS_CSV)
    rows = []
    if path.exists():
        rows = list(csv.DictReader(path.open()))
    new = {r["example"]: r for r in tev.STATS}
    rows = [r for r in rows if r["example"] not in new]
    rows += [{k: str(v) for k, v in r.items()} for r in tev.STATS]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
