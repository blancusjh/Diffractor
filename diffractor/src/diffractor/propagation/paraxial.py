"""Paraxial (Fresnel) propagation — kept, but GATED.

The Fresnel propagator is a legitimate tool in its regime and a wrong answer
outside it (session evidence: at NA 0.75 the validity distance was violated
16×).  Every call therefore checks the standard condition

    z³  ≫  (π / 4λ) · r_max⁴        (Goodman)

and refuses by default when it fails.  ``policy`` controls the behaviour:
"raise" (default), "warn", or "force" for the caller who knows better.
"""

from __future__ import annotations

import warnings

import numpy as np

from .exact import fresnel_plane as _fresnel_plane_kernel

__all__ = ["fresnel_validity_distance", "fresnel_plane"]


def fresnel_validity_distance(r_max: float, wavelength: float,
                              margin: float = 3.0) -> float:
    """Minimum z for the paraxial condition, with a safety ``margin``."""
    return margin * ((np.pi / (4.0 * wavelength)) * r_max**4) ** (1.0 / 3.0)


def fresnel_plane(U, r, z, n, wavelength, rho_out, *, policy="raise", **kw):
    """Gated paraxial propagator (axisymmetric)."""
    z_min = fresnel_validity_distance(float(np.max(r)), wavelength / n)
    if abs(z) < z_min:
        msg = (f"Fresnel propagator outside its validity regime: |z| = {abs(z):.3e}"
               f" < required {z_min:.3e} (r_max = {np.max(r):.3e}). "
               f"Use propagation.exact (ASM/RS1) instead.")
        if policy == "raise":
            raise ValueError(msg)
        if policy == "warn":
            warnings.warn(msg, stacklevel=2)
        elif policy != "force":
            raise ValueError(f"unknown policy {policy!r}")
    return _fresnel_plane_kernel(U, r, z, n, wavelength, rho_out, **kw)
