"""Aggregate a multi-seed scout study into mean +/- std curves, a summary
table, and the final figure.  Reads the accumulated per-seed rows written by
``run_minimal_scout_experiment.py`` (scout_metrics.csv + scout_reference.csv)
and never recomputes anything.

Usage::

    python aggregate_scout.py --root experiments/scout_native384 --plot-k 80
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

_SCOUT_STYLE = {
    "vcls_scout": ("#1f77b4", "o", "VCLS"),
    "vcls_bundle_scout": ("#2ca02c", "s", "VCLS+bundle"),
    "greedy_bundle_scout": ("#ff7f0e", "^", "Coverage+bundle"),
    "greedy_bundle_ocr_scout": ("#d62728", "D", "OED"),
}
_REF_STYLE = {
    "uniform_bundle": "#7f7f7f", "vcls": "#1f77b4", "vcls_bundle": "#2ca02c",
    "greedy_bundle": "#ff7f0e", "greedy_bundle_ocr": "#d62728",
}
_SCOUT_TO_REF = {
    "vcls_scout": "vcls", "vcls_bundle_scout": "vcls_bundle",
    "greedy_bundle_scout": "greedy_bundle",
    "greedy_bundle_ocr_scout": "greedy_bundle_ocr",
}
_METRIC_LABEL = {"psnr": "PSNR (dB)", "hfen": "HFEN", "ssim": "SSIM",
                 "nrmse": "NRMSE"}


def _load(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _agg(rows, metric):
    vals = [float(r[metric]) for r in rows]
    if not vals:
        return None, None, 0
    return (statistics.fmean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/scout_minimal")
    ap.add_argument("--plot-k", type=int, default=80)
    ap.add_argument("--metrics", default="psnr,hfen")
    args = ap.parse_args(argv)
    root = Path(args.root)

    scout = _load(root / "results" / "scout_metrics.csv")
    ref = _load(root / "results" / "scout_reference.csv")
    if not scout:
        raise SystemExit(f"no scout_metrics.csv under {root}")
    metrics = [m.strip() for m in args.metrics.split(",")]
    phantoms = sorted({r["phantom"] for r in scout})
    scout_views = sorted({int(float(r["scout_views"])) for r in scout})
    methods = [m for m in _SCOUT_STYLE if any(r["method"] == m for r in scout)]
    n_seeds = len({r.get("seed", "0") for r in scout})

    # ---- summary table (mean +/- std over seeds) ----------------------------
    summ_path = root / "results" / "scout_summary.csv"
    fields = ["phantom", "method", "scout_views", "k", "n_seeds"] + \
             [f"{m}_{s}" for m in ("psnr", "ssim", "nrmse", "hfen")
              for s in ("mean", "std")]
    out_rows = []
    for ph in phantoms:
        for m in methods:
            for s in scout_views:
                for k in sorted({int(float(r["k"])) for r in scout}):
                    sub = [r for r in scout if r["phantom"] == ph
                           and r["method"] == m
                           and int(float(r["scout_views"])) == s
                           and int(float(r["k"])) == k]
                    if not sub:
                        continue
                    row = {"phantom": ph, "method": m, "scout_views": s,
                           "k": k, "n_seeds": len(sub)}
                    for met in ("psnr", "ssim", "nrmse", "hfen"):
                        mean, std, _ = _agg(sub, met)
                        row[f"{met}_mean"] = round(mean, 4)
                        row[f"{met}_std"] = round(std, 4)
                    out_rows.append(row)
    with summ_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {summ_path}  ({len(out_rows)} rows, n_seeds={n_seeds})")

    # ---- console table at plot_k -------------------------------------------
    print(f"\n=== mean PSNR over {n_seeds} seeds at k={args.plot_k} "
          "(oracle | scout 8..) ===")
    for ph in phantoms:
        print(f"-- {ph}")
        for m in methods:
            rname = _SCOUT_TO_REF[m]
            orc = [r for r in ref if r["phantom"] == ph
                   and r["method"] == rname
                   and int(float(r["k"])) == args.plot_k]
            om, os_, _ = _agg(orc, "psnr")
            cells = []
            for s in scout_views:
                sub = [r for r in scout if r["phantom"] == ph
                       and r["method"] == m
                       and int(float(r["scout_views"])) == s
                       and int(float(r["k"])) == args.plot_k]
                sm, ss, _ = _agg(sub, "psnr")
                cells.append(f"s{s}:{sm:.3f}" if sm is not None else f"s{s}:--")
            ostr = f"{om:.3f}+/-{os_:.3f}" if om is not None else "--"
            print(f"   {_SCOUT_STYLE[m][2]:16s} oracle={ostr} | " + "  ".join(cells))

    # ---- figure -------------------------------------------------------------
    try:
        _plot(root, scout, ref, phantoms, scout_views, methods, metrics,
              args.plot_k, n_seeds)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] figure failed: {e}")
    return 0


def _plot(root, scout, ref, phantoms, scout_views, methods, metrics, plot_k,
          n_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow, ncol = len(phantoms), len(metrics)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.7 * nrow),
                             squeeze=False)
    handles = labels = None
    for i, ph in enumerate(phantoms):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            for m in methods:
                color, marker, disp = _SCOUT_STYLE[m]
                xs, ys, es = [], [], []
                for s in scout_views:
                    sub = [r for r in scout if r["phantom"] == ph
                           and r["method"] == m
                           and int(float(r["scout_views"])) == s
                           and int(float(r["k"])) == plot_k]
                    mean, std, _ = _agg(sub, metric)
                    if mean is not None:
                        xs.append(s); ys.append(mean); es.append(std)
                if xs:
                    ax.plot(xs, ys, color=color, marker=marker, lw=1.8,
                            ms=6, label=disp)
                    if any(e > 0 for e in es):
                        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                                        [y + e for y, e in zip(ys, es)],
                                        color=color, alpha=0.15)
            for rname, color in _REF_STYLE.items():
                orc = [r for r in ref if r["phantom"] == ph
                       and r["method"] == rname
                       and int(float(r["k"])) == plot_k]
                mean, _, _ = _agg(orc, metric)
                if mean is not None:
                    ax.axhline(mean, color=color, ls="--", lw=1.1, alpha=0.7,
                               label=f"{rname} (oracle)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(scout_views)
            ax.set_xticklabels([str(s) for s in scout_views])
            ax.set_xlabel("scout views")
            ax.set_ylabel(_METRIC_LABEL.get(metric, metric))
            ax.set_title(f"{ph}  -  {_METRIC_LABEL.get(metric, metric)}  (k={plot_k})")
            ax.grid(True, which="both", alpha=0.25)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=7, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Scout-based planning robustness "
                 f"(k={plot_k}, mean$\\pm$std over {n_seeds} seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    png = root / "figures" / "scout_quality_vs_views.png"
    pdf = root / "figures" / "scout_quality_vs_views.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    raise SystemExit(main())
