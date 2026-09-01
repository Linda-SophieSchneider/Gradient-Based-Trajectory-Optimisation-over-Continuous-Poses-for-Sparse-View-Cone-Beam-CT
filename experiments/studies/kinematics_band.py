"""Kinematics study on the bench-matched elevation band.

The reachable set of our measured robot bench is the full azimuth range with
+-30 deg of elevation. The library's constrained two-axis parametrisation
(``CArmTwoAxisGantry``) carries the C-arm envelope as its defaults, so this
wrapper rebinds that name in the two modules that construct it and then runs
the ordinary kinematics study. Both the discrete feasible-grid comparator and
our continuous arm therefore see the band instead of the C-arm envelope; no
library code is modified.

    python -m experiments.studies.kinematics_band --phantom lof_flange_v3 \
        --resolution 384 --target-views 40 --seeds 0,1,2 --noisefree \
        --methods vcls_carm_grid,greedy_adam_bundle_carm --out-root <dir>
"""
from __future__ import annotations

import math
import sys
from functools import partial

import differentiable_coverage.eval.trajectories as _eval_traj
import experiments.studies.kinematics as _kin
from differentiable_coverage.trajectory import (CArmTwoAxisGantry,
                                               SmoothTwoAxisGantry)

D2R = math.pi / 180.0
_BAND_LIMITS = dict(theta_min=-180.0 * D2R, theta_max=180.0 * D2R,
                    phi_min=-30.0 * D2R, phi_max=30.0 * D2R)
BAND = partial(CArmTwoAxisGantry, **_BAND_LIMITS)
BAND_SMOOTH = partial(SmoothTwoAxisGantry, **_BAND_LIMITS)


def main(argv=None):
    # Rebind in both construction sites: the continuous arm and the init
    # mapping live in eval.trajectories, the discrete candidate pool in the
    # kinematics study module.
    _eval_traj.CArmTwoAxisGantry = BAND
    _eval_traj.SmoothTwoAxisGantry = BAND_SMOOTH
    _kin.CArmTwoAxisGantry = BAND
    print("[band] reachable set: azimuth +-180 deg, elevation +-30 deg",
          flush=True)
    return _kin.main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
