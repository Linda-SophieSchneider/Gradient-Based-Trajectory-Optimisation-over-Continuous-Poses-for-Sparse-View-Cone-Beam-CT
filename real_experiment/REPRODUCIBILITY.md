# Measured robot CT study: reproducibility record

This record describes the custom laboratory cone-beam CT test bench used for
the measured camera experiment. It deliberately distinguishes information that
is versioned with this repository from raw measurements that are not licensed
for public redistribution.

## Public hardware identity and acquisition configuration

The system is a custom robot-based cone-beam CT test bench; it has no single
commercial system-model designation. The paper therefore does not substitute a
commercial model name. The public, reproducible operating configuration is:

- detector: 3072 x 3072 pixels at 0.139 mm pitch before 4x mean pooling;
- mean focus-to-detector distance: 1995.6 mm and focus-to-object distance:
  997.1 mm;
- tube setting: 175 kV, 2000 microampere for the 1200-view reference and
  2200 microampere for the prescan, 200 ms exposure and four averages;
- source manifold: 360 degree azimuth and +/-30 degree elevation; and
- reconstruction grid: 768 cubed voxels at 0.278 mm pitch.

Component manufacturer/model and serial identifiers are not in the public
record. They are not inferred or replaced by an approximate designation. A
researcher who needs them, the per-view calibration export, or the raw
projections should contact the corresponding author; access is subject to the
source-data and institutional redistribution conditions described below.

## Geometry, registration, and reconstruction procedure

Every acquired projection carries its scanner-exported source and detector
geometry. The circular 1200-view reference and 120-view prescan share their
reference frame. The elevation-band trajectories were acquired in separate
sessions, with a rigid pose offset between each session and the reference
frame; each measured acquisition is reconstructed directly in the reference
frame after transforming its *geometry* by that pose, and reconstructed
volumes are never warped or resampled.

The shared rigid transform is found reproducibly as follows:

1. Enumerate the yaw/roll/translation sign and inverse-convention candidates.
   Score a candidate using the median Pearson correlation between measured
   projections and reprojections of the reference FDK volume.
2. Refine the selected convention by coordinate descent (yaw and roll steps
   down to 0.25 degrees; translation steps down to 0.5 mm) over ten evenly
   spaced projection views.
3. Apply a small per-scan translation touch-up, again by reprojection
   correlation, and rebuild the cone-beam operators from the transformed
   source, detector centre, and detector axes.

The committed pose for the all3 100-view arm is
`geoframe_all3_N0100_pose.json` (yaw -89.75 degrees, roll -0.50 degrees,
translation [-6.1648, -2.3324, 13.8296] mm, median correlation 0.9858). The
implementation is `geometry_frame_reco.py`; the measured-scan validation and
projection-consistency checks are in `measured_scan_reco.py`.

## Versioned artifacts and controlled data access

The repository version-controls the trajectory-coordinate/header exports,
registration pose, reconstruction/planning scripts, diagnostic metrics, and
small provenance artifacts. In particular,
`planned_trajectory_prescan_{bundle,all3}*/**/trajectory_headers.txt` provides
the executed planning geometries without exposing images.

The raw `.raw` projections, Nextcloud source archives, and `.rek`
reconstruction volumes are excluded because they are large and governed by
source-data and institutional redistribution conditions. Requests for the raw
projections, calibration export, and component identifiers should be sent to
the corresponding author with the intended use and data-protection context.

## TV-weight evidence

The one-configuration-per-arm rule is reference-free: for each `(arm, k)`,
`regularization_sweep.py` chooses the lowest fractional data residual over the
declared alpha grid. It writes
`sweep_<arm>_k<budget>_summary.csv` containing every tested alpha, residual, TV
norm, a clearly labelled non-selection PSNR diagnostic, and the selected row.
The two last-completed cells (circular k=100 and full composite k=100) were
swept on 2026-08-11 (logs `sweep_{circular,all3}_k100_20260811.log`); their
summary CSVs (`sweep_{circular,all3}_k0100_summary.csv`, archived here) were
regenerated on 2026-08-13 by re-scoring the retained per-alpha volumes
(`SWEEP_REUSE=1`), reproducing the logged residuals exactly. alpha=0.15 wins
every cell; the printed k=100 circular/full-composite metrics come from the
alpha=0.15 canonical volumes.
