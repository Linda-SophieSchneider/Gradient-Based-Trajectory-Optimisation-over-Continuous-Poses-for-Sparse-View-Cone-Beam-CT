import sys
sys.path.insert(0, ".")
sys.path.insert(0, "../Differentiable-Coverage")
import numpy as np
from scanner_io.rek2py import rek2py
from differentiable_coverage.eval import metrics as M

_, ref = rek2py("reference_reconstructions/output_circular1200_fdk_quant/reconstruction_FDK.rek", switch_order=True)
ref = np.asarray(ref, np.float32)

arms = [
    ("circular", 400, "results_final/circular_final_k0400/reconstruction.rek"),
    # circular/all3 k=100 completed their residual-selected TV sweeps on
    # 2026-08-11 (winner alpha=0.15, replacing the pre-sweep alpha=0.30
    # reconstructions) — this script now covers all eight printed cells.
    ("circular", 100, "results_final/circular_final_k0100/reconstruction.rek"),
    ("all3", 400, "results_final/all3_final_k0400/reconstruction.rek"),
    ("all3", 100, "results_final/all3_final_k0100/reconstruction.rek"),
    ("bundle", 400, "results_final/bundle_final_k0400/reconstruction.rek"),
    ("bundle", 100, "results_final/bundle_final_k0100/reconstruction.rek"),
    ("uniform", 400, "results_final/uniform_final_k0400/reconstruction.rek"),
    ("uniform", 100, "results_final/uniform_final_k0100/reconstruction.rek"),
]
rows = []
for name, k, path in arms:
    _, v = rek2py(path, switch_order=True)
    v = np.asarray(v, np.float32)
    scale = float((v * ref).sum() / (v * v).sum())
    v_s = v * scale
    row = dict(arm=name, k=k, path=path, ls_scale=round(scale, 4),
               psnr=round(float(M.psnr(v_s, ref)), 3),
               ssim=round(float(M.ssim(v_s, ref)), 4),
               nrmse=round(float(M.nrmse(v_s, ref)), 4),
               hfen=round(float(M.hfen(v_s, ref)), 3))
    rows.append(row)
    print(f"{name:10s} k={k:3d}: LS={scale:.3f}  PSNR={row['psnr']:.2f}  "
          f"SSIM={row['ssim']:.4f}  NRMSE={row['nrmse']:.4f}  HFEN={row['hfen']:.2f}", flush=True)

# Versioned provenance export (REV-P1-11): the exact artifact behind the
# printed measured-camera table.
import csv
with open("results_final/final_metrics_20260811.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Wrote results_final/final_metrics_20260811.csv")
print("METRICS_DONE")
