"""Refracting surfaces and thin-element phase masks.

A surface is described by its sag ``z(x, y)``: the axial displacement of
the interface from its vertex plane, positive toward the second medium.
Light travels along +z from medium ``n1`` into medium ``n2``; a ray at
height ``(x, y)`` covers the extra sag distance still inside ``n1``, so
under the thin-element approximation the surface imprints the phase

    t(x, y) = exp(i k0 (n1 - n2) z(x, y)),    k0 = 2π / λ

on a field crossing it, which is what :meth:`Surface.phase_mask` returns.
With this convention a convex surface (positive sag growing off-axis)
into a denser medium (``n2 > n1``) converges the beam, as it must.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .grids import Array, Grid

__all__ = ["CartesianSurface", "ParabolicSurface", "Surface", "thin_element_phase"]


def thin_element_phase(sag: Array, wavelength: float, n1: float, n2: float) -> Array:
    """Thin-element transmission ``exp(i k0 (n1 - n2) sag)``.

    ``sag`` is measured toward the second medium, so positive sag means
    the ray travels that extra distance in ``n1`` before refracting.
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    k0 = 2.0 * np.pi / wavelength
    return np.exp(1.0j * k0 * (n1 - n2) * sag)


class Surface(ABC):
    """Axisymmetric refracting surface defined by its sag profile."""

    @abstractmethod
    def sag(self, x: Array, y: Array) -> Array:
        """Return the sag ``z(x, y)`` [m]."""

    def phase_mask(self, grid: Grid, wavelength: float, n1: float, n2: float) -> Array:
        """Thin-element phase imprinted on a field crossing the surface.

        Parameters
        ----------
        grid : (x, y)
            Spatial coordinate grids.
        wavelength : float
            Vacuum wavelength [m].
        n1, n2 : float
            Refractive indices before and after the surface.
        """
        x, y = grid
        return thin_element_phase(self.sag(x, y), wavelength, n1, n2)


@dataclass(frozen=True)
class ParabolicSurface(Surface):
    """Axisymmetric paraboloid, ``z(ρ) = z0 + ρ² / (4 f)``.

    Parameters
    ----------
    focal_length : float
        Focal length ``f`` of the parabola [m], non-zero.
    z0 : float
        Axial position of the vertex [m].
    """

    focal_length: float
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.focal_length == 0:
            raise ValueError("focal_length must be non-zero.")

    def sag_rho(self, rho: Array) -> Array:
        """Sag as a function of the radial coordinate ρ."""
        return self.z0 + rho**2 / (4.0 * self.focal_length)

    def sag(self, x: Array, y: Array) -> Array:
        return self.sag_rho(np.sqrt(x**2 + y**2))


@dataclass(frozen=True)
class CartesianSurface(Surface):
    """Stigmatic Cartesian-oval refracting surface (single diopter).

    The unique interface between media ``n1`` and ``n2`` that images the
    on-axis object point A at ``zo`` (real object: ``zo < 0``) perfectly
    onto the image point A' at ``zi > 0``: every path A → Q → A' through
    a surface point Q has the same optical length,

        n1 |AQ| + n2 |QA'| = n1 |zo| + n2 zi.

    The sag ``z(r)`` is obtained by solving this equality with Newton
    iterations on a cancellation-free residual, so it is exact to machine
    precision at any aperture (series expansions of the oval are not).

    Parameters
    ----------
    n1, n2 : float
        Refractive indices of the object and image spaces.
    zo : float
        Object distance [m], negative (real object to the left).
    zi : float
        Image distance [m], positive.
    """

    n1: float
    n2: float
    zo: float
    zi: float

    def __post_init__(self) -> None:
        if self.n1 <= 0 or self.n2 <= 0:
            raise ValueError("Refractive indices must be positive.")
        if self.n1 == self.n2:
            raise ValueError("n1 and n2 must differ for a refracting surface.")
        if not (self.zo < 0 < self.zi):
            raise ValueError("Real conjugates required: zo < 0 < zi.")

    def paraxial_curvature(self) -> float:
        """Vertex curvature 1/R from the paraxial diopter equation."""
        return (self.n2 / self.zi - self.n1 / self.zo) / (self.n2 - self.n1)

    def sag(self, x: Array, y: Array) -> Array:
        n1, n2, zo, zi = self.n1, self.n2, self.zo, self.zi
        r2 = np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2

        # Newton iterations on the equal-optical-path residual, written as
        #   f(z) = n1 (|AQ| - |zo|) + n2 (|QA'| - zi)
        # with each difference in the cancellation-free form
        #   sqrt(u^2 + w) - u = w' / (sqrt(u^2 + w) + u),
        # so the solve stays at machine precision even for |zo| >> zi
        # (e.g. collimated illumination).
        z = 0.5 * self.paraxial_curvature() * r2  # paraxial seed
        for _ in range(60):
            d1 = np.sqrt((z - zo) ** 2 + r2)
            d2 = np.sqrt((zi - z) ** 2 + r2)
            f = n1 * (z * z - 2.0 * z * zo + r2) / (d1 - zo) + n2 * (
                z * z - 2.0 * z * zi + r2
            ) / (d2 + zi)
            fp = n1 * (z - zo) / d1 - n2 * (zi - z) / d2
            step = f / fp
            z = z - step
            if np.all(np.abs(step) <= 1e-15 * (1.0 + np.abs(z))):
                break
        else:
            raise ValueError(
                "Cartesian oval solve did not converge; the requested radii "
                "likely exceed the extent of the oval for these conjugates."
            )
        if not np.all(np.isfinite(z)):
            raise ValueError(
                "Cartesian oval solve diverged; the requested radii likely "
                "exceed the extent of the oval for these conjugates."
            )
        return z
