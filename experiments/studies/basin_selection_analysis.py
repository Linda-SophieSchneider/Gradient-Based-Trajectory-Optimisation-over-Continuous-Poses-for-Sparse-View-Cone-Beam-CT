"""Offline selection-rule sweep over the basin-selection member dump.

`basin_selection.py` writes a per-member CSV
(`basin_selection_<phantom>_<explore>_members.csv`) with, for every explored
ensemble member, its VCL surrogate, the OED A/D components, the combined OED
score, and the member's true reconstruction PSNR (the "oracle" signal).  Because
every member was already reconstructed, ANY selection rule can be scored here
with zero new reconstructions — we only change which member's PSNR we keep.

Rules compared (all pick one member per (phantom, photon, seed) group):
  * vcls        : the discrete VCLS baseline PSNR (not a member; reference)
  * pick-VCL    : argmax VCL surrogate            (current cold picker)
  * pick-OED    : argmax (A + D)                  (Phase-A noise-aware selector)
  * A-only/D-only : argmax A / argmax D
  * top-m       : among the top-m by VCL, argmax OED score (m = 2,3,4)
  * best-alpha/beta : argmax (alpha*A + beta*D) over a grid, reported both
                  in-sample (optimistic) and cross-validated across phantoms
  * oracle      : argmax true PSNR               (deployment-infeasible ceiling)

Run:
    python -m experiments.studies.basin_selection_analysis \
        --members experiments/basin_selection/results/basin_selection_mild_vcl_members.csv \
                  experiments/basin_selection/results/basin_selection_moderate_vcl_members.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "phantom": r["phantom"], "photon": float(r["photon"]),
                    "seed": int(r["seed"]), "member": int(r["member"]),
                    "vcl": float(r["vcl_info"]), "A": float(r["oed_A"]),
                    "D": float(r["oed_D"]), "oed": float(r["oed_score"]),
                    "psnr": float(r["oracle_psnr"]), "vcls": float(r["psnr_vcls"]),
                })
    return rows


def _groups(rows):
    """(phantom, photon, seed) -> {members:[...sorted by idx], vcls:float}."""
    g = defaultdict(list)
    for r in rows:
        g[(r["phantom"], r["photon"], r["seed"])].append(r)
    out = {}
    for key, ms in g.items():
        ms = sorted(ms, key=lambda r: r["member"])
        out[key] = {"members": ms, "vcls": ms[0]["vcls"]}
    return out


def _argmax(ms, keyfn):
    best_i, best_v = 0, float("-inf")
    for i, m in enumerate(ms):
        v = keyfn(m)
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def _topm(ms, m):
    order = sorted(range(len(ms)), key=lambda i: ms[i]["vcl"], reverse=True)[:m]
    return max(order, key=lambda i: ms[i]["oed"])


def _rule_psnr(group, rule):
    ms = group["members"]
    if rule == "vcls":
        return group["vcls"]
    if rule == "oracle":
        return max(m["psnr"] for m in ms)
    if rule == "vcl":
        return ms[_argmax(ms, lambda m: m["vcl"])]["psnr"]
    if rule == "oed":
        return ms[_argmax(ms, lambda m: m["oed"])]["psnr"]
    if rule == "A":
        return ms[_argmax(ms, lambda m: m["A"])]["psnr"]
    if rule == "D":
        return ms[_argmax(ms, lambda m: m["D"])]["psnr"]
    if rule.startswith("top"):
        return ms[_topm(ms, int(rule[3:]))]["psnr"]
    if rule.startswith("ab:"):
        a, b = (float(x) for x in rule[3:].split(","))
        return ms[_argmax(ms, lambda m: a * m["A"] + b * m["D"])]["psnr"]
    raise ValueError(rule)


def _cells(groups):
    return sorted({(ph, ph_n) for (ph, ph_n, _s) in groups})


def _mean_over_seeds(groups, cell, rule):
    ph, pn = cell
    vals = [_rule_psnr(g, rule) for (p, n, _s), g in groups.items()
            if p == ph and n == pn]
    return statistics.fmean(vals)


def _mean_delta_vs_vcls(groups, rule, cells):
    ds = []
    for c in cells:
        ds.append(_mean_over_seeds(groups, c, rule) - _mean_over_seeds(groups, c, "vcls"))
    return statistics.fmean(ds)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--grid", default="0,0.25,0.5,1,2,4",
                    help="alpha/beta grid values")
    args = ap.parse_args(argv)

    groups = _groups(_load([Path(p) for p in args.members]))
    cells = _cells(groups)
    grid = [float(x) for x in args.grid.split(",")]
    fixed_rules = ["vcls", "vcl", "oed", "A", "D", "top2", "top3", "top4", "oracle"]

    print("=" * 84)
    print("Offline selection-rule sweep   (Δ PSNR vs VCLS baseline, dB)")
    print("=" * 84)
    header = f"{'rule':<10}" + "".join(f"{ph[:4]+'/'+f'{pn:.0e}':>13}" for ph, pn in cells) + f"{'mean':>9}"
    print(header)
    print("-" * len(header))
    for rule in fixed_rules:
        if rule == "vcls":
            line = f"{rule:<10}" + "".join(
                f"{_mean_over_seeds(groups, c, 'vcls'):>13.3f}" for c in cells) + f"{'(abs)':>9}"
        else:
            cellds = [_mean_over_seeds(groups, c, rule) - _mean_over_seeds(groups, c, "vcls")
                      for c in cells]
            line = f"{rule:<10}" + "".join(f"{d:>+13.3f}" for d in cellds) \
                + f"{statistics.fmean(cellds):>+9.3f}"
        print(line)

    # --- variance: per-seed paired Δ vs VCLS.  Each seed value is already
    #     averaged over the two noise realisations; std is over the 5 model
    #     seeds (paired, so it reflects the spread of the gain itself).
    print("-" * len(header))
    print("Per-cell mean ± std of per-seed Δ vs VCLS, plus pooled (n shown):")

    def seed_deltas(cell, rule):
        ph, pn = cell
        return [_rule_psnr(g, rule) - g["vcls"]
                for (p, n, _s), g in groups.items() if p == ph and n == pn]

    for rule in ["vcl", "oed", "D", "oracle"]:
        parts, pooled = [], []
        for c in cells:
            d = seed_deltas(c, rule)
            pooled += d
            m = statistics.fmean(d)
            sd = statistics.pstdev(d) if len(d) > 1 else 0.0
            parts.append(f"{c[0][:4]}/{c[1]:.0e} {m:+.2f}±{sd:.2f}")
        pm = statistics.fmean(pooled)
        psd = statistics.pstdev(pooled)
        se = psd / (len(pooled) ** 0.5) if pooled else 0.0
        npos = sum(1 for x in pooled if x > 0)
        print(f"  {rule:<7} " + " | ".join(parts))
        print(f"  {'':7} pooled {pm:+.3f} ± {psd:.3f}  (SE {se:.3f}, "
              f"{npos}/{len(pooled)} seed-cells > 0)")

    # --- alpha/beta grid: best in-sample, plus cross-validated across phantoms
    print("-" * len(header))
    ab_rules = [(a, b) for a in grid for b in grid if (a, b) != (0.0, 0.0)]
    best_ab, best_mean = None, float("-inf")
    for a, b in ab_rules:
        md = _mean_delta_vs_vcls(groups, f"ab:{a},{b}", cells)
        if md > best_mean:
            best_mean, best_ab = md, (a, b)
    cellds = [_mean_over_seeds(groups, c, f"ab:{best_ab[0]},{best_ab[1]}")
              - _mean_over_seeds(groups, c, "vcls") for c in cells]
    print(f"{'best-ab*':<10}" + "".join(f"{d:>+13.3f}" for d in cellds)
          + f"{best_mean:>+9.3f}   (in-sample alpha,beta={best_ab}; optimistic)")

    phantoms = sorted({ph for ph, _ in cells})
    if len(phantoms) >= 2:
        print(f"{'  CV note':<10} tune alpha/beta on one phantom, apply to the other:")
        for test_ph in phantoms:
            train_cells = [c for c in cells if c[0] != test_ph]
            test_cells = [c for c in cells if c[0] == test_ph]
            ba, bm = None, float("-inf")
            for a, b in ab_rules:
                md = _mean_delta_vs_vcls(groups, f"ab:{a},{b}", train_cells)
                if md > bm:
                    bm, ba = md, (a, b)
            test_d = _mean_delta_vs_vcls(groups, f"ab:{ba[0]},{ba[1]}", test_cells)
            oed_d = _mean_delta_vs_vcls(groups, "oed", test_cells)
            print(f"           tuned on !{test_ph} -> alpha,beta={ba}: "
                  f"test Δ={test_d:+.3f} dB   (vs plain OED {oed_d:+.3f})")


if __name__ == "__main__":
    main()
