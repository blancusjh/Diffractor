"""Sources.  The point source is the physical spherical wave, 1/r included."""

from __future__ import annotations

import numpy as np

__all__ = ["point_source", "point_source_normal_derivative"]


def point_source(rho, z, z_src, k, amplitude=1.0 / (4 * np.pi)):
    """ψ = A₀ e^{ikd}/d from an on-axis point at z_src (free-space Green form)."""
    d = np.hypot(rho, z - z_src)
    return amplitude * np.exp(1j * k * d) / d


def point_source_normal_derivative(rho, z, n_rho, n_z, z_src, k,
                                   amplitude=1.0 / (4 * np.pi)):
    """∂ψ/∂n of the same wave on a surface with unit normal (n_rho, n_z)."""
    d = np.hypot(rho, z - z_src)
    u = amplitude * np.exp(1j * k * d) / d
    rn = (rho * n_rho + (z - z_src) * n_z) / d
    return (1j * k - 1.0 / d) * u * rn
