# Real-world robot-CT experiment — driver scripts, configs, and small result artifacts

Backs the paper's real-hardware section (`sec:real`). This directory holds
the planning/reconstruction/registration pipeline and small, non-proprietary
result artifacts (previews, CSVs, JSON). **All `.rek` reconstruction volumes
and raw `.raw` EZRT projections are excluded** — those are multi-GB each and
are available on request from the corresponding author (see the top-level
README's "Data" section and `REPRODUCIBILITY.md` below).

## Layout

- `planning/` — trajectory exports (`planned_trajectory_*_z2000/`,
  `planned_trajectory_Full_reco/`) plus the objective-term dynamics traces.
- `results_final/` — the residual-selected canonical reconstructions (preview
  PNGs only), the paper figure previews, and the final metric tables.
- `reference_reconstructions/` — circular-1200 quantitative FDK and the
  prescan prior used for planning.
- `registration/` — the shared geometry-frame pose and registration QA slices.
- `sweeps/` — per-(arm, k) TV-sweep summaries (`sweep_<arm>_k<budget>_summary.csv`).
- `pinterguss/` — the 2026-08 follow-up experiment on a second real object
  (see below).
- `EZRT_Helpers/` — vendored EZRT header/IO helpers.
- Driver scripts stay at the top level.

## Pipeline scripts (run in this rough order)

- `reconstruct_ezrt_cuda.py` — FDK/SIRT reconstruction of a raw EZRT dataset.
  Data location is read from `EZRT_DATA_DIR` / `EZRT_OUT_DIR` (env vars) so
  the script never needs editing for a different dataset or output folder.
- `plan_trajectory.py` — continuous trajectory planning (coverage / bundle /
  VCL / all-three composite). `PLAN_OBJECTIVE` selects the objective,
  `PLAN_FDK_PATH` / `PLAN_OUT_DIR` / `PLAN_N_RADON` the input volume, output
  folder, and Radon-normal lattice size.
- `fdk_scale_check.py` — synthetic radiometric validation of the FDK path
  (legacy vs. quantitative diffct >= 2.1.0).
- `sart_circular_baseline.py` — circular-subset baseline, tuned ASD-POCS
  protocol (canonical `ASD_OUTER`/`ASD_ALPHA` live here).
- `measured_scan_reco.py` — validate + reconstruct one newly-arrived measured
  scan (geometry match against a planned trajectory, projection-consistency
  check, canonical ASD-POCS reco).
- `geometry_frame_reco.py` — rigid-pose registration done in the ACQUISITION
  GEOMETRY (not on volumes): pose search/refinement + reconstruction
  variants used to validate the registration and tune the protocol.
- `uniform_band_baseline.py` — assembles a real "uniform-on-band" baseline
  from projections pooled across all measured scans (matched via
  `scipy.linear_sum_assignment` to a Fibonacci-band target set). Reads the
  measured-scan pool from `KAMERA_DATA_DIR`.
- `regularization_sweep.py` — per-(arm, k) TV-strength sweep, selected by
  data residual alone (never by comparison to the reference). Final
  canonical volumes are written as `<arm>_final_k<k>/reconstruction.rek`.
  Each run also writes `sweep_<arm>_k<budget>_summary.csv`, which records all
  tested weights, their fractional data residuals, and the selected row. This
  compact CSV is the versionable evidence for the TV-choice rule. Also reads
  `KAMERA_DATA_DIR`.
- `compute_final_metrics.py` — batch PSNR/SSIM/NRMSE/HFEN for a list of
  final reconstructions.
- `render_real_slices.py` — the axial-slice comparison figure (`fig:real_slices`).

Four one-off diagnostic scripts from the internal development history
(term-composition dynamics, coverage-lattice matching, ASD-POCS stage-2
tuning, and an early volume-warp registration superseded by
`geometry_frame_reco.py`) are not part of this release since they do not
back any reported paper value.

## Result/config artifacts kept

- `registration/geoframe_all3_N0100_pose.json` — the shared rigid registration
  pose (yaw/roll/translation) found by a convention search + coordinate-descent
  refinement; reused (with a small per-scan translation touch-up) for every
  measured arm.
- `results_final/final_metrics_20260811.csv` — the final paper values.
  `results_final/metrics_planned_arms.csv` is an earlier diagnostic table from
  intermediate reconstruction/registration stages, kept for provenance only.
- `planning/*_dynamics_*.csv/png` — Adam-iteration term traces.
- `registration/reg_check_*.png` — pre-geometry-frame volume-registration QA
  slices (superseded method, kept for the record).
- `planning/planned_trajectory_*_z2000/` — the planning outputs actually
  exported to the scanner: `trajectory_coords.csv` + `trajectory_headers.txt`
  per view budget, for every objective variant (`coverageOnly`, `bundle`,
  `vcl`, `all3`) on the dense `z=2000` Radon lattice used for the final
  planning run.
- `sweeps/sweep_<arm>_k<budget>_summary.csv` — provenance artifacts of the
  reconstruction/regularization-sweep runs (residuals, winning alpha per arm).

## What's not here

All `.rek` reconstruction volumes and raw `.raw` EZRT projections (reference,
circular/uniform/bundle/all3 arms at every stage of tuning) are not
redistributed in this repository; see the top-level README's "Data" section.
`REPRODUCIBILITY.md` gives the test-bench identity, calibration and
registration procedure, and the controlled access route for the excluded raw
projections and calibration files.

## Pinterguss follow-up experiment (2026-08)

`pinterguss/` repeats the camera protocol on a second real object (a THD
aluminium casting, 1095-view circular reference scan). Protocol deltas vs.
the camera run — everything else identical (+-30 deg band, z=2000 lattice,
tau=0.07, seed 0, per-k random init):

- **Prior = stride-9 pseudo-prescan** (every 9th circular projection, 122
  views, FDK) — there is no physical Pinterguss prescan; this keeps the
  "planning never sees the reference" separation.
- **Corrected bundle quadrature**: `plan_trajectory.py` now builds
  `BundleAbsorptionConfig(..., n_samples=512, clip_to_volume=True)`; the
  camera-era exports used the legacy full-segment@32 rule (see
  `experiments/bundle_quadrature/`).
- The runs also pick up the complete-basis VCL geometry-VJP correction and
  the Adam evaluated-iterate fix (both post-camera, see `docs/`): the
  Pinterguss acquisitions are the first measured evidence for the corrected
  implementation.
- Exported for measurement: `planned_trajectory_pinterguss_{bundle,all3}_z2000/`
  (k=50..400 each), packaged as
  `pinterguss/planned_trajectories_pinterguss_2026_08_19.zip`.
- The circular scan truncates the mounting base at the low-z field-of-view
  edge (see `pinterguss/output_pinterguss_circular_fdk/truncation_projections.png`);
  the casting proper is fully covered and 94% of the ROI weight lands on it.
