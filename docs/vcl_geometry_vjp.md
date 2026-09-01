# VCL geometry VJP

## Why the implementation changed

The evaluated per-view VCL basis is

\[
q_i(s_i)
=
\frac{S A_{s_i}^{\top} H A_{s_i}x}
     {\lVert S A_{s_i}^{\top} H A_{s_i}x\rVert_2+\varepsilon}.
\]

The former autograd path propagated source-position gradients through
\(A_sx\), but the matched-footprint backprojector exposed a data-only VJP.
It therefore omitted

\[
D_s(A_s^\top)\,H A_sx
\]

from

\[
D_s(A_s^\top H A_sx)
=
D_s(A_s^\top)\,H A_sx
+
A_s^\top H\,D_s(A_s)x.
\]

The resulting vector was finite and nonzero, but it was not the gradient of
the VCL value evaluated by the optimiser.

## Current default

`build_vcl_context` now defaults to:

```python
geometry_vjp_mode="full_finite_difference"
geometry_fd_step=0.5
```

The outer quadratic form
\(\gamma^\top R^{-1}\gamma\) retains its analytical custom VJP. Its cotangent
with respect to the normalised basis is contracted with a central difference
of the *complete basis row*. Each perturbation therefore includes:

- source-dependent detector construction;
- matched-footprint forward projection;
- ramp filtering;
- matched-footprint backprojection;
- stochastic voxel sampling; and
- L2 row normalisation.

Each row depends only on its corresponding source. All views are consequently
perturbed together along a coordinate axis, requiring six additional basis
evaluations per reverse pass rather than \(6k\).

The old behaviour remains available as:

```python
geometry_vjp_mode="legacy_autodiff"
```

Use it only for regression comparisons.

## Numerical validation

On the two-view \(12^3\) synthetic case in `tests/test_vcl_diff.py`, with a
0.1 source-unit central-difference step:

| Gradient | Cosine against scalar central difference | Relative error |
|---|---:|---:|
| Former partial VJP | 0.41 | 1.83 |
| Complete-basis VJP | 0.9999999 | 0.000443 |

The test suite also verifies the row-wise VJP against an analytical toy basis
and checks that the legacy and complete modes evaluate the same VCL value while
producing different gradients.

On the same case, the default 0.5-mm step has cosine similarity 0.9998 to the
0.1-mm reference gradient and a relative difference of 2.1%; 0.25, 0.5, and
1.0 mm all preserve cosine similarity above 0.998. This step-size check is a
numerical diagnostic, not evidence that one step is universally optimal.

Run the focused checks with:

```bash
python -m pytest tests/test_vcl_diff.py -q
```

## Consequences for paper experiments

The VCL value is unchanged, but every trajectory refined with
`lambda_vcl > 0` can change because the optimiser now receives the gradient of
the evaluated score. Before submission, rerun all VCL-dependent trajectories,
their timing measurements, and downstream reconstructions. Coverage-only and
bundle-only trajectories are unaffected.

The paper must keep the analytical-gradient stability claim scoped to the
bundle absorption term. VCL uses an analytical outer score VJP and a complete
finite-difference geometry VJP.
