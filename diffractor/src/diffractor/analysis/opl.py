"""Optical path length — a measurement on ray paths, so it lives in analysis.

OPL is not a property of an interface; it is a functional of a path through a
scene.  For the two-leg path A → Q → A' through a surface point Q it is
n1|AQ| + n2|QA'|; stigmatism is the OBSERVATION that this is constant over Q,
never an input.
"""

from __future__ import annotations

import numpy as np

__all__ = ["two_leg_opl", "invariant_opl", "opd_waves"]


def two_leg_opl(n1, n2, r, z, zo, zi):
    """OPL of A(0,zo) → Q(r,z) → A'(0,zi), from coordinates alone."""
    d1 = np.hypot(r, z - zo)
    d2 = np.hypot(r, zi - z)
    return n1 * d1 + n2 * d2


def invariant_opl(n1, n2, zo, zi):
    """The axial value C = n1|zo| + n2 zi — the stigmatic constant."""
    return n1 * abs(zo) + n2 * zi


def opd_waves(interface, g, wavelength):
    """Wavefront error of a sampled profile, in waves, measured independently:
    d1 and d2 are rebuilt from the (r, z) coordinates, not taken from the
    parametrisation that may have used the equal-OPL condition to exist."""
    zo, zi = interface.profile.zo, interface.profile.zi
    opl = two_leg_opl(interface.n1, interface.n2, g["r"], g["z"], zo, zi)
    return (opl - invariant_opl(interface.n1, interface.n2, zo, zi)) / wavelength
