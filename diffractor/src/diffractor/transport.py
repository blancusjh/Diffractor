"""Energy transport — the general conservation factors of geometrical propagation.

Everything here is one statement worn three ways: the radiance invariant
L/n² is conserved along a ray tube in a lossless medium, and only the
interface response T touches it.

* along a tube:      n |U|² dA = const   →   |U| ∝ 1/sqrt(tube section)
* across a boundary: the conserved normal flux is  n cosθ |U|²  (the n cosθ
  factor is impedance × projected area — it belongs to every wave theory,
  scalar included)
* point-to-point:    for homocentric bundles the tube section is d², so the
  amplitude simply carries 1/d.

These are GENERAL propagation facts.  They compose with a scattering
coefficient (t_s from :mod:`diffractor.scattering`) but do not contain it.
"""

from __future__ import annotations

import numpy as np

from .scattering.fresnel import T_s, t_s

__all__ = ["ray_tube_amplitude", "stigmatic_pupil", "stigmatic_pupil_from_tubes"]


def ray_tube_amplitude(n1, n2, w1, w2, T, *, regularize_axis=True):
    """General ray-tube amplitude ratio across an interface.

        n1 |U1|² dΩ1 · T = n2 |U2|² dΩ2
        →   |U2|/|U1| = sqrt( T · (n1/n2) · w1/w2 )

    with w = sinθ·dθ/dparam the solid-angle densities of the in/out bundles
    (any common parametrisation).  Valid for ANY surface — aberrated or not —
    because it never assumes the outgoing bundle is homocentric.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.sqrt(T * (n1 / n2) * w1 / w2)
    if regularize_axis and P.size > 3:
        # The 0/0 on the axis is removable.  Fit a low-order polynomial to the
        # first few off-axis points and evaluate at index 0.  This handles
        # non-uniform spacing (e.g. clustered sampling) correctly, unlike the
        # previous hard-coded equal-spacing extrapolation.
        n_fit = min(6, P.size - 1)
        idx = np.arange(1, n_fit + 1)
        coeffs = np.polyfit(idx, P[1:n_fit + 1].real, min(2, n_fit - 1))
        P[0] = np.polyval(coeffs, 0)
    return P


def stigmatic_pupil(interface, g):
    """Amplitude on the exit reference sphere of a stigmatic interface.

        P(θ2) = t_s · d2/d1

    The stigmatic special case of :func:`ray_tube_amplitude`: both legs are
    exact spherical waves, so the tube Jacobians collapse via the identity
    w2/w1 = (cos i2/cos i1)(d1/d2)² and only the 1/d factors survive.

    Validation status: an earlier BOR-BEM measurement appeared to confirm this
    pupil to 0.25 %, but that measurement is RETRACTED — the closed benchmark
    body turned out to be a lossless cavity whose reverberant field
    contaminated the pupil edge (see benchmarks/cases.md and PLAN.md, fase 1).
    The prediction awaits the open-surface benchmark; what stands today is the
    machine-level identity with the general ray-tube law, asserted in the test
    suite.

    ``g`` is a geometric sample of the interface profile.
    """
    ts = t_s(interface.n1, interface.n2, g["cos_i1"], g["cos_i2"])
    return ts * g["d2"] / g["d1"]


def stigmatic_pupil_from_tubes(interface, g):
    """Same quantity from the general form — kept as a consistency check."""
    T = T_s(interface.n1, interface.n2, g["cos_i1"], g["cos_i2"])
    return ray_tube_amplitude(interface.n1, interface.n2, g["w1"], g["w2"], T)
