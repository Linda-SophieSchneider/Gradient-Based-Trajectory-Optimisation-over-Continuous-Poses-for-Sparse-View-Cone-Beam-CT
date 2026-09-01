# Gradient-Based Trajectory Optimisation over Continuous Poses for Sparse-View Cone-Beam CT

Reference implementation accompanying the paper of the same title
(Linda-Sophie Schneider, Simon Wittl, Gabriel Herl, Andreas Maier; IEEE
Transactions on Pattern Analysis and Machine Intelligence, submitted).

> Trajectory optimisation for cone-beam computed tomography (CBCT) decides
> where the X-ray source is placed before a scan is acquired. Existing
> methods choose those poses from a fixed candidate pool, which prevents
> off-grid refinement and forces a new pool and a new object-specific
> precomputation for every acquisition manifold. We make every source pose
> of a fixed-budget scan an individual continuous variable and move all
> poses jointly by gradient ascent on the kinematic manifold of the
> scanner. The objective combines soft-Tuy plane coverage, a continuous
> View Covariance Loss information term, and an analytic attenuation-aware
> ray-bundle penalty. The same optimiser handles circular, limited C-arm,
> two-axis, and free-sphere parametrisations without rebuilding a candidate
> pool.

## What is in this repository

```
differentiable_coverage/     # the Python package
  score.py                   # soft Tuy-coverage objective and closed-form gradient
  absorption_bundle.py       # analytic bundle absorption term (noise-aware selection)
  absorption.py              # legacy finite-difference absorption path (ablation only)
  vcl_diff.py                # continuous View Covariance Loss (VCL) score + custom VJP
  vcl_backprojection.py      # per-view FBP-operator backend for the VCL score
  vcl_geometry.py            # working-grid resolution helper for the VCL score
  trajectory.py               # kinematic parametrisations (circle, helix, two-axis, C-arm, sphere)
  optimize.py                 # Adam ascent with cosine schedule, patience, Langevin noise
  roi.py, joint.py, landscape.py, runtime.py, fourier_radon.py, ramp_filter.py
  _torch_bridge.py            # MLX <-> Torch bridge for the VCL term on CUDA
  oed.py                      # photon-weighted optimal-design score; NOT part of the
                               #   paper's method (see "What was left out" below); kept
                               #   only because experiments/studies/basin_selection.py
                               #   imports it behind a non-default CLI flag
  eval/                        # benchmark utilities: phantoms, geometry, SART, metrics
    datasets/ornl_nozzle.py    # loader for the ORNL fuel-nozzle release
figures/                       # scripts that render the paper's non-tabular figures
experiments/
  run.py, run_noise_eval.py    # config-driven benchmark runner (paper1 main/ablation tables)
  precompute_caches.py, aggregate_seeds.py
  configs/paper1/{main,ablation}/      # one YAML per benchmark cell
  studies/                     # CLI-driven studies (kinematics ladder, limited angle,
                                #   scout robustness, pose-motion dump, bundle-quadrature
                                #   convergence, basin-selection sensitivity probe)
  bundle_quadrature/, coverage_multipoint/   # frozen convergence/robustness data
  results/                     # committed CSV/JSON/NPZ outputs that the paper's
                                #   tables and figures are read from directly
real_experiment/                # robot-CT reproduction pipeline (planning, geometry-frame
                                 #   reconstruction, registration, real-data figures/tables)
tests/                          # unit and integration tests for differentiable_coverage/
```

## Installation

```bash
pip install -e ".[experiments]"
```

This installs the package, [MLX](https://github.com/ml-explore/mlx),
[diffct](https://doi.org/10.20944/preprints202605.1446.v1), PyYAML, and
Matplotlib.

**Backends.** `diffct` auto-selects its backend: MLX on Apple Silicon
(Metal), Torch/CUDA everywhere else. On Linux/CUDA, install the full set
from `requirements.txt` instead (adds `torch`, `numba`, and `cma` for the
CMA-ES ablation column of the finite-difference-vs-bundle comparison):

```bash
pip install -r requirements.txt
```

`real_experiment/reconstruct_measured_cuda.py` additionally needs the
`numba-cuda` plugin package and `nvidia-cuda-nvcc` (for `libnvvm.so`).

## Quick start

```bash
# Unit and integration tests (no phantom data required)
python -m pytest tests/ -m "not integration"

# Reproduce one benchmark cell (needs the Defrise flange phantom, see "Data" below)
python -m experiments.run --config experiments/configs/paper1/main/lof_v3_384.yaml
```

## Reproducing the paper

Every number in the paper is produced by one of the config-driven runs
under `experiments/configs/paper1/`, one of the CLI studies under
`experiments/studies/`, or the real-hardware pipeline under
`real_experiment/`. The already-computed outputs are committed under
`experiments/results/lof_plate_20260825/`,
`experiments/results/flange_collar_20260829/`,
`experiments/results/basin_v4a_20260829/`, `experiments/bundle_quadrature/`,
and `experiments/coverage_multipoint/`, so tables and figures can be
regenerated from them without rerunning the optimiser.

| Paper content | Source |
|---|---|
| Kinematics ladder (circle / band / two-axis / C-arm), pose-motion figure | `experiments/studies/kinematics.py`, `kinematics_band.py`, `dump_kinematics_poses.py` -> `experiments/results/lof_plate_20260825/studies/` |
| Finite-difference-vs-bundle estimator, ORNL runs, noise table | `experiments/configs/paper1/main/{flange_384_*,ornl_512_*,noise_i0_*}.yaml` run via `experiments/run.py` / `run_noise_eval.py` |
| Pose-motion table (`tab:motion`) and figure | `figures/make_pose_motion_table.py`, `figures/render_pose_motion.py` |
| Scout robustness | `experiments/studies/scout_robustness.py` + `scout_aggregate.py` |
| Selection-cost table | `experiments/configs/paper1/ablation/cost_{lof_v3_384,ornl_512}.yaml` |
| Absorption-target ablation | `experiments/configs/paper1/ablation/term_contribution_lof_v3.yaml` and the `absorption4_*` studies under `experiments/results/lof_plate_20260825/studies/` |
| Parameter-sensitivity sweeps (z, lambda_bundle, K_max, LR, Langevin T, explorers, seeds) | `experiments/configs/paper1/ablation/{z_sweep,lambda_bundle_sweep,kmax,ablation_lr_cold,ablation_langevin_T,ablation_explorers,seeds}_lof_v3.yaml` |
| Collar control study (supplementary) | `experiments/configs/paper1/ablation/*_lof_v4a.yaml`, `term_contribution_lof_v4b.yaml` -> `experiments/results/flange_collar_20260829/`, `experiments/results/basin_v4a_20260829/` |
| Local-basins sensitivity probe (supplementary) | `experiments/studies/basin_selection.py --phantom lof_flange_v3` |
| Defrise flange / directional absorption figures | `figures/render_defrise_flange.py`, `figures/render_absorption_map.py` |
| ORNL reconstruction slices | `experiments/render_milp_slices.py` via `experiments/configs/paper1/main/ornl_slices_main.yaml` |
| Robot-CT camera results | `real_experiment/README.md` and `real_experiment/REPRODUCIBILITY.md` |

## Data

Phantom volumes are **not** committed to this repository. Place them
under `./data/` (the path each config expects is given in its
`phantom.path` field):

| Phantom | Availability |
|---|---|
| **Defrise flange** (`data/lof_flange_v3.npy`, our custom aluminium/steel specimen, `sec:setup_phantoms`) | Available on request from the corresponding author (see below). |
| **ORNL fuel nozzle** (`data/ornl_nozzle.h5`) | Publicly released by Oak Ridge National Laboratory: Ziabari, Rahman, Venkatakrishnan and Dehoff, *X-ray Computed Tomography Data of Dense Metallic Components*, 2025, [doi:10.13139/ORNLNCCS/2568789](https://doi.org/10.13139/ORNLNCCS/2568789). This is also the object used in the discrete View Covariance Loss Search (VCLS) comparison of Lin, Ziabari, Venkatakrishnan, Rahman, Buzzard and Bouman, *Tomographic Sparse View Selection Using the View Covariance Loss*, IEEE TPAMI, 2025, [doi:10.1109/TPAMI.2025.3600072](https://doi.org/10.1109/TPAMI.2025.3600072) — see `content/related.tex`/`experimental_setup.tex` of the paper for the exact comparison protocol. |
| **Digital camera** (measured robot-CT projections, `real_experiment/`) | Real hardware scans; not redistributed here. Available on request from the corresponding author, subject to the source data and institutional redistribution conditions. |

Any cubic reference volume can be substituted by pointing a config's
`phantom.path` at it; absolute PSNR/SSIM numbers will then differ from the
paper, but the methodology is unchanged.

## What was left out of this release

- **`differentiable_coverage/oed.py`** implements a photon-weighted
  optimal-experimental-design view score. A verification pass found it adds
  no measurable reconstruction-quality benefit over the bundle absorption
  penalty as an in-loop objective, so it is **not part of the paper's
  method**. It is kept in this release only because
  `experiments/studies/basin_selection.py` imports it behind its
  non-default `--explore-objective oed` flag; the paper's own results use
  the default `vcl` path.
- Exploratory/superseded experiment runs that predate the current Defrise
  flange phantom (early MILP mild/moderate benchmark cells, retired
  limited-angle/ROI-occlusion studies, earlier kinematics reruns) are not
  included, since none of them back a table or figure in the current
  paper. `experiments/bundle_quadrature/` is the one exception that
  intentionally keeps its original mild/moderate convergence data, since
  the paper cites that quadrature study on the phantoms it was frozen on.

## Citation

```bibtex
@article{Schneider2026Gradient,
  author  = {Schneider, Linda-Sophie and Wittl, Simon and Herl, Gabriel and Maier, Andreas},
  title   = {Gradient-Based Trajectory Optimisation over Continuous Poses
             for Sparse-View Cone-Beam {CT}},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  note    = {submitted}
}
```

The companion discrete framework:

```bibtex
@misc{Schneider2026Soft,
  author       = {Schneider, Linda-Sophie and Maier, Andreas},
  title        = {Soft {Tuy}-Completeness for Robust Projection
                  Selection in Cone-Beam {CT}},
  howpublished = {arXiv preprint arXiv:2605.24023},
  year         = {2026},
  url          = {https://arxiv.org/abs/2605.24023}
}
```

The differentiable CT operators:

```bibtex
@article{SunDiffCT2026,
  author = {Sun, Y. and Schneider, L.-S. and Ye, C. and Maier, A.},
  title  = {diffct: Differentiable {CT} Operators from Circular
            Orbits to Arbitrary Trajectories},
  year   = {2026},
  doi    = {10.20944/preprints202605.1446.v1}
}
```

## License

Apache License 2.0, see [LICENSE](LICENSE).

## Contact

Linda-Sophie Schneider, Pattern Recognition Lab, Friedrich-Alexander-Universität
Erlangen-Nürnberg. linda-sophie.schneider@fau.de
