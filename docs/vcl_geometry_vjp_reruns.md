# Corrected continuous-optimisation reruns

> This note documents the two corrections behind the paper's reported
> numbers and the fixed parameter contract they were rerun under. The
> batch-execution tooling it describes (`run_vcl_geometry_vjp_reruns.{py,sh}`
> and its manifest) drove the original rerun campaign across both retired
> and current benchmark configs and is not part of this release; use the
> per-config commands in the top-level README's "Reproducing the paper"
> table instead. The numerical contract below applies to every config under
> `experiments/configs/paper1/`.

## Why the inventory is larger than the VCL-only set

Two implementation corrections invalidate old continuous selections:

1. the VCL source gradient now finite-differences the complete normalised
   projector--filter--backprojector basis, rather than stopping the geometry
   gradient through backprojection; and
2. Adam now associates the best objective value with the same evaluated
   parameter iterate, rather than with the following update.

The first correction affects VCL-active selectors. The second affects every
paper-facing continuous Coverage, VCL, bundle, composite, and finite-difference
selector. The manifest therefore contains the complete virtual rerun set, not
only rows with `lambda_vcl > 0`. Measured/robot acquisitions are deliberately
excluded.

## Source of truth

The machine-readable inventory is
`experiments/configs/paper1/vcl_geometry_vjp_reruns.yaml`. It contains:

- 35 submission-facing virtual entries;
- 2 matched kinematics entries; and
- 4 active auxiliary diagnostics.

The 41 entries cover the noise-free, photon-noise, low-dose, kinematic,
limited-angle, scout, cost, finite-difference, qualitative-figure, optimiser,
hyperparameter, and seed-variability results that depend on a continuous
selector. Discrete rows are included only where needed to regenerate a
complete qualitative plate.

The manifest also pins the following numerical contract:

| Parameter | Value / rule |
|---|---|
| bundle-only naming | `lambda_bundle > 0`, `lambda_vcl = 0` |
| composite naming | `lambda_bundle > 0`, `lambda_vcl = 0.2` |
| kinematic continuous arms | cold coverage + 0.2 VCL + calibrated bundle |
| hidden path term | `lambda_path = 0` for VCL and composite selectors |
| VCL geometry VJP | complete-basis central difference, 0.5 mm |
| absorption FD counterfactual | complete source/detector-frame difference, 0.001 mm |
| VCL sample / ridge | fixed seed 0, `r1 = 0.001`, ridge 0.001 |
| bundle calibration | `0.2 / median(mean optical depth)`, 256 probes |
| optimiser | 100 objective evaluations unless swept; cosine schedule and patience 15 where configured; best evaluated iterate |
| physical downsampling | detector width and volume field of view preserved |

## Safe two-GPU execution

`all-2gpu` requires a fresh output root. It starts exactly two CUDA processes,
one per physical GPU, and assigns manifest entries by deterministic alternating
shards:

- GPU queue 0 receives entries `0, 2, 4, ...` (21 entries);
- GPU queue 1 receives entries `1, 3, 5, ...` (20 entries).

Each queue is sequential, so at most one experiment runs on a GPU at a time.
The two sets are disjoint and their union is the complete manifest. Each
process performs a torch/CUDA allocation test before its first experiment.
Resolved YAML files, outputs, and regenerated figures are written below the
fresh root; source configs and paper figures are never overwritten.

```bash
PYTHON_BIN=../.venv/bin/python \
DIFFCOV_GPU_IDS=0,1 \
  bash experiments/run_vcl_geometry_vjp_reruns.sh all-2gpu \
  experiments/results/corrected_continuous_2gpu_20260724
```

Logs are written to `<output-root>/logs/`. Each shard also writes a provenance
JSON below `<output-root>/provenance/` containing the Python and package
versions, backend, CUDA device, interpreter, environment contract, and exact
experiment IDs.

## Inspection without execution

These commands never start an experiment:

```bash
bash experiments/run_vcl_geometry_vjp_reruns.sh list
bash experiments/run_vcl_geometry_vjp_reruns.sh dry-run
```

CUDA can be validated independently:

```bash
PYTHON_BIN=../.venv/bin/python \
  bash experiments/run_vcl_geometry_vjp_reruns.sh cuda-check
```

## Outputs and post-processing

- filtered configs: `<output-root>/configs/`
- numerical CSVs: `<output-root>/results/`
- module studies: `<output-root>/studies/`
- qualitative plates: `<output-root>/artifacts/`
- runtime/device records: `<output-root>/provenance/`
- queue logs: `<output-root>/logs/`

Seeded config runs invoke `experiments/aggregate_seeds.py` automatically.
Limited-angle arguments explicitly request five selection seeds, three noise
realisations, and four ROIs. Scout robustness is run separately for seeds
0--2. Qualitative outputs are redirected to the fresh output root.

The cost entries will run on CUDA with the rest of the manifest, but those
times are workstation timings. They must not replace the paper's
Apple-laptop timing table unless its hardware caption is updated or the cost
entries are repeated on the stated Apple hardware.

After the run, no old continuous numerical value should be copied forward
without comparing it to its corresponding corrected output. The real-data
section is unaffected by this virtual rerun command and remains excluded.
