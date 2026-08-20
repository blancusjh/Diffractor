"""Space: grids as typed samples of the plane.

A grid is not an array — it is a *sampling of space*, and the way its
coordinates address the plane is part of what it is.  So a :class:`Grid` is
the pair (axes, basis): two 1-D coordinate arrays laid out with ``'ij'``
indexing, and a :class:`~diffractor.basis.Basis` naming the coordinate system
they are written in.  ``Grid.cartesian(x, y)`` and ``Grid.polar(r, theta)``
build the two canonical cases; ``Grid.custom(axes, basis)`` accepts any basis
whose maps the caller supplies.

Two grid-level quantities keep the rest of the package basis-agnostic:

* :meth:`Grid.weights` — the quadrature weights of the area measure d²x on the
  sample nodes (dx·dy for cartesian, r·dr·dθ for polar), so that
  ``(f * grid.weights()).sum()`` is ∬ f d²x whatever the basis;
* :meth:`Grid.reciprocal` — the conjugate grid of angular spatial frequencies
  the Fourier operators transform onto by default.  For a cartesian grid this
  is the FFT's own k-grid, ``k = 2π·fftfreq``; for a polar grid it is a radial
  k-axis whose extent and count are the radial Nyquist limit and the Bessel
  oscillation-sampling rule below — both overridable, neither hidden.

Axisymmetric fields are not a special machinery: they are polar grids with a
single angular sample (``n_theta=1``), for which the angular Fourier series
degenerates to the m = 0 term and the 2-D transform to an order-zero Hankel
transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from .basis import Basis, CARTESIAN, POLAR, resolve_basis

__all__ = ["Grid", "BESSEL_SAMPLES_PER_PERIOD", "POLAR_BAND_FRACTION"]

#: Samples per oscillation period of J_m(k·r_max) that the default polar
#: reciprocal keeps on its k axis: the inverse-transform integrand oscillates
#: in k with local period 2π/r_max, so Δk = 2π/(β·r_max).  Measured on a
#: Gaussian round trip with the Gregory-corrected quadrature, the error falls
#: as β⁻⁴ and reaches a few 1e-4 at β = 5 — the older 2.5 (the legacy
#: ``n_rho = 2.5·rho_max·r_max`` heuristic) leaves it at a few 1e-3.
BESSEL_SAMPLES_PER_PERIOD: float = 5.0

#: Fraction of the radial Nyquist limit π/Δr that the default polar reciprocal
#: spans.  Beyond k = π/(2Δr) the kernel J_m(k r) has fewer than 4 radial
#: samples per period near r_max, its quadrature values stop being
#: trustworthy, and — worse — the inverse transform *collects* that noise
#: under its k dk measure (measured: extending a Gaussian's default band from
#: π/2Δr to π/Δr grows the round-trip error 20×).  Content beyond the default
#: band is a statement that Δr is too coarse, not that k_max is too small.
POLAR_BAND_FRACTION: float = 0.5

_Axes = Tuple[np.ndarray, np.ndarray]


def _as_axis(a, name: str) -> np.ndarray:
    a = np.asarray(a, float)
    if a.ndim != 1 or a.size < 1:
        raise ValueError(f"{name} axis must be a 1-D array with at least one sample")
    return a


def _uniform_spacing(axis: np.ndarray, name: str) -> float:
    """Spacing of a uniform axis, validated (raises naming the axis)."""
    if axis.size < 2:
        raise ValueError(f"{name} axis needs at least 2 samples for a spacing")
    d = np.diff(axis)
    if not np.allclose(d, d[0], rtol=1e-9, atol=0.0):
        raise ValueError(f"{name} axis must be uniformly sampled for this operation")
    return float(d[0])


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    """Composite-trapezoid weights of a (possibly non-uniform) 1-D axis."""
    if axis.size == 1:
        return np.ones(1)
    w = np.empty(axis.size)
    w[0] = (axis[1] - axis[0]) / 2.0
    w[-1] = (axis[-1] - axis[-2]) / 2.0
    w[1:-1] = (axis[2:] - axis[:-2]) / 2.0
    return w


@dataclass(frozen=True, eq=False)
class Grid:
    """Typed samples of the plane: 1-D coordinate ``axes`` in a ``basis``.

    Layout is ``'ij'``: samples of a function on this grid have shape
    ``grid.shape = (len(axes[0]), len(axes[1]))`` with ``axes[0]`` varying
    along the first array axis.  Cartesian: ``axes = (x, y)``.  Polar:
    ``axes = (r, theta)``.
    """

    axes: _Axes
    basis: Basis

    def __post_init__(self):
        object.__setattr__(self, "axes",
                           (_as_axis(self.axes[0], "first"),
                            _as_axis(self.axes[1], "second")))
        object.__setattr__(self, "basis", resolve_basis(self.basis))

    # ── constructors ─────────────────────────────────────────────────────────
    @classmethod
    def cartesian(cls, x, y) -> "Grid":
        """Grid of the cartesian axes ``(x, y)``."""
        return cls((x, y), CARTESIAN)

    @classmethod
    def polar(cls, r, theta=None, *, n_theta: Optional[int] = None) -> "Grid":
        """Grid of the polar axes ``(r, theta)``.

        ``r`` must be non-negative and ascending.  ``theta`` must be the
        uniform circle sampling ``θ_j = 2πj/n_θ`` (the angular Fourier series
        needs it); give either the axis itself, ``n_theta`` to build it, or
        neither for the axisymmetric single sample ``[0.0]``.
        """
        r = _as_axis(r, "r")
        if r[0] < 0 or np.any(np.diff(r) <= 0):
            raise ValueError("r axis must be non-negative and strictly ascending")
        if theta is not None and n_theta is not None:
            raise ValueError("give theta or n_theta, not both")
        if theta is None:
            n = 1 if n_theta is None else int(n_theta)
            if n < 1:
                raise ValueError("n_theta must be at least 1")
            theta = 2.0 * np.pi * np.arange(n) / n
        else:
            theta = _as_axis(theta, "theta")
            expected = 2.0 * np.pi * np.arange(theta.size) / theta.size
            if not np.allclose(theta, expected, rtol=0.0, atol=1e-12 * 2 * np.pi):
                raise ValueError(
                    "theta must sample the circle uniformly from 0: "
                    "θ_j = 2πj/n_θ (the angular FFT depends on it)")
        return cls((r, theta), POLAR)

    @classmethod
    def custom(cls, axes: _Axes, basis: Basis) -> "Grid":
        """Grid in a caller-supplied basis (the basis maps must be explicit)."""
        if not isinstance(basis, Basis):
            raise TypeError("a custom grid requires an explicit Basis instance")
        return cls(axes, basis)

    # ── coordinate queries ───────────────────────────────────────────────────
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.axes[0].size, self.axes[1].size)

    def meshes(self) -> _Axes:
        """The two coordinate meshes, ``'ij'`` indexed."""
        return np.meshgrid(self.axes[0], self.axes[1], indexing="ij")

    def cartesian_meshes(self) -> _Axes:
        """The sample nodes as cartesian coordinate meshes, via the basis map."""
        return self.basis.to_cartesian(*self.meshes())

    def r2(self) -> np.ndarray:
        """|x|² at every node (x² + y² cartesian, r² polar), shape ``self.shape``."""
        if self.basis is POLAR:
            return np.broadcast_to((self.axes[0] ** 2)[:, None], self.shape)
        X, Y = self.cartesian_meshes()
        return X * X + Y * Y

    def weights(self) -> np.ndarray:
        """Quadrature weights of the area measure d²x, shape ``self.shape``.

        Cartesian: trapezoid ⊗ trapezoid (non-uniform axes included).  Polar:
        (trapezoid in r) · r ⊗ (2π/n_θ) — with n_θ = 1 the angular factor is
        the full revolution, so axisymmetric integrals come out already
        revolved.  A custom basis has no generic area element; supply
        weights explicitly where they are needed.
        """
        if self.basis is CARTESIAN:
            return np.outer(_trapezoid_weights(self.axes[0]),
                            _trapezoid_weights(self.axes[1]))
        if self.basis is POLAR:
            r, theta = self.axes
            w_r = _trapezoid_weights(r) * r
            w_t = np.full(theta.size, 2.0 * np.pi / theta.size)
            return np.outer(w_r, w_t)
        raise NotImplementedError(
            f"no generic area element for basis {self.basis.name!r}; "
            f"pass explicit weights to the operation that needs them")

    def scaled(self, factor: float) -> "Grid":
        """The grid with its metric coordinates scaled by ``factor``.

        Cartesian: both axes scale.  Polar: only r scales — angles are
        dimensionless.  This is how a frequency grid becomes the Fresnel
        output plane ``x' = (λz/2π)·k``.
        """
        if self.basis is POLAR:
            return Grid((self.axes[0] * factor, self.axes[1]), POLAR)
        return Grid((self.axes[0] * factor, self.axes[1] * factor), self.basis)

    # ── the conjugate grid ───────────────────────────────────────────────────
    def reciprocal(self, *, k_max: Optional[float] = None,
                   n_k: Optional[int] = None) -> "Grid":
        """The conjugate grid of angular spatial frequencies.

        Cartesian: the FFT's own centred k-grid, ``k = 2π·fftshift(fftfreq)``
        per axis (``k_max``/``n_k`` are polar-only knobs and are rejected
        here).  Polar: ``k = linspace(0, k_max, n_k)`` with the same θ axis;
        defaults ``k_max = POLAR_BAND_FRACTION · π/Δr_min`` (the band over
        which the radial sampling still resolves the kernel — see the module
        constant) and ``n_k = ⌈BESSEL_SAMPLES_PER_PERIOD · k_max · r_max /
        2π⌉ + 1`` (the k-axis oscillation-sampling rule).  Both are honest
        knobs, not magic: band-limit ``k_max`` to your field's content (a
        propagator limits it to the propagating cone) and the quadrature
        rewards you.
        """
        if self.basis is CARTESIAN:
            if k_max is not None or n_k is not None:
                raise ValueError(
                    "k_max/n_k choose the polar radial frequency axis; a "
                    "cartesian grid's reciprocal is fixed by its own sampling")
            kx = 2.0 * np.pi * np.fft.fftshift(
                np.fft.fftfreq(self.axes[0].size,
                               _uniform_spacing(self.axes[0], "x")))
            ky = 2.0 * np.pi * np.fft.fftshift(
                np.fft.fftfreq(self.axes[1].size,
                               _uniform_spacing(self.axes[1], "y")))
            return Grid.cartesian(kx, ky)
        if self.basis is POLAR:
            r = self.axes[0]
            if r.size < 2:
                raise ValueError("polar reciprocal needs at least 2 radial samples")
            if k_max is None:
                k_max = POLAR_BAND_FRACTION * np.pi / float(np.min(np.diff(r)))
            if n_k is None:
                n_k = int(np.ceil(BESSEL_SAMPLES_PER_PERIOD * k_max * r[-1]
                                  / (2.0 * np.pi))) + 1
            return Grid((np.linspace(0.0, float(k_max), int(n_k)), self.axes[1]),
                        POLAR)
        raise NotImplementedError(
            f"no default reciprocal for basis {self.basis.name!r}; "
            f"pass an explicit kgrid to FT2")
