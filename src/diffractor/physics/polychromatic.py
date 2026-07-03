"""Polychromatic (broadband / white-light) propagation and rendering.

A broadband field is just many monochromatic fields. This module propagates a
source at each wavelength onto a **shared** output grid and composites the
per-wavelength intensities into an sRGB image with
:mod:`diffractor.viz.colorimetry`.

Using a shared, wavelength-independent output grid is what makes this clean:
``fresnel_zoom_propagator`` and ``asm_propagator(..., output_grid=...)`` both
evaluate onto a grid you choose, so every wavelength lands on the same physical
screen and dispersive features (e.g. grating orders at ``x_m = m λ z / d``)
appear at their correct positions with no resampling. The single-FFT
``fresnel_propagator`` / ``fraunhofer_propagator`` are deliberately not offered
here: their output grid scales with ``λ``, so channels would not share a grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from ..viz.colorimetry import spectrum_to_srgb
from .field import Field
from ..mathutils.grids import Array, Grid
from .longitudinal import longitudinal_field
from .monochromatic import MonochromaticField
from .propagation import asm_propagator, fresnel_zoom_propagator

__all__ = [
    "PolychromaticField",
    "PolychromaticImage",
    "RGBLongitudinalSection",
    "propagate_polychromatic",
    "propagate_polychromatic_longitudinal",
]

FieldOf = Union[Field, Callable[[float], Field]]


def _as_builder(field_of: FieldOf) -> Callable[[float], Field]:
    if isinstance(field_of, Field):
        return lambda _lam: field_of
    if callable(field_of):
        return field_of
    raise TypeError("field_of must be a Field or a callable wavelength -> Field.")


def propagate_polychromatic(
    field_of: FieldOf,
    wavelengths: Sequence[float],
    z: float,
    *,
    weights: Optional[Sequence[float]] = None,
    n: float = 1.0,
    propagator: str = "fresnel_zoom",
    output_grid: Optional[Grid] = None,
    output_half_width: Optional[float] = None,
    output_samples: int = 512,
    gamut: str = "desaturate",
    saturation: float = 1.0,
    stretch: float = 1.0,
    brightness: float = 1.0,
    pad_factor: int = 2,
) -> Tuple[Array, Grid]:
    """Propagate a broadband field and render it as an sRGB image.

    Parameters
    ----------
    field_of : Field or callable
        The input field. Pass a single ``Field`` for a wavelength-independent
        source (an amplitude mask), or a callable ``λ [m] -> Field`` for a
        chromatic element (e.g. a :func:`~diffractor.physics.gratings.phase_grating`).
    wavelengths : sequence of float
        Wavelengths [m] to sample across the band.
    z : float
        Propagation distance [m].
    weights : sequence of float, optional
        Spectral power of the source at each wavelength (e.g.
        ``d65_weights(np.array(wavelengths) * 1e9)``). Defaults to flat.
    n : float
        Refractive index of the propagation medium.
    propagator : {"fresnel_zoom", "asm"}
        Which shared-output-grid propagator to use.
    output_grid : Grid, optional
        Shared output grid for every wavelength. For ``"fresnel_zoom"`` it may
        be given via ``output_half_width``/``output_samples`` instead. For
        ``"asm"`` it defaults to the source grid.
    output_half_width, output_samples :
        Convenience square output window for ``"fresnel_zoom"``.
    gamut : {"desaturate", "clip"}
        Out-of-gamut handling in :func:`~diffractor.viz.colorimetry.xyz_to_srgb`.
    pad_factor : int
        ``pad_factor`` for the ASM propagator (ignored by fresnel_zoom).

    Returns
    -------
    (rgb, grid) : (array (H, W, 3) in [0, 1], Grid)
        The rendered image and the shared output grid it lives on.
    """
    build = _as_builder(field_of)
    wavelengths = [float(w) for w in wavelengths]
    if not wavelengths:
        raise ValueError("wavelengths is empty.")

    frames = []
    out_grid = output_grid
    for lam in wavelengths:
        field = build(lam)
        if propagator == "fresnel_zoom":
            out = fresnel_zoom_propagator(
                field, z=z, wavelength=lam, n=n,
                output_grid=out_grid,
                output_half_width=None if out_grid is not None else output_half_width,
                output_samples=output_samples,
            )
        elif propagator == "asm":
            grid = out_grid if out_grid is not None else field.grid
            # Take the fast native-FFT path whenever the target coordinates
            # coincide with the field's own grid — also when a per-wavelength
            # callable rebuilt the Field on an equal but distinct Grid object.
            native = grid is field.grid or (
                grid.shape == field.grid.shape
                and np.array_equal(grid.x, field.grid.x)
                and np.array_equal(grid.y, field.grid.y)
            )
            out = asm_propagator(
                field, z=z, wavelength=lam, n=n,
                output_grid=None if native else grid,
                pad_factor=pad_factor,
            )
        else:
            raise ValueError("propagator must be 'fresnel_zoom' or 'asm'.")

        if out_grid is None:
            out_grid = out.grid
        frames.append(np.abs(out.values) ** 2)

    stack = np.stack(frames, axis=0)  # (K, H, W)
    # colorimetry wants wavelengths in nm; weights combine spectral power + Δλ.
    rgb = spectrum_to_srgb(
        np.array(wavelengths) * 1e9,
        stack,
        weights=None if weights is None else np.asarray(weights, dtype=float),
        gamut=gamut,
        saturation=saturation,
        stretch=stretch,
        brightness=brightness,
    )
    return rgb, out_grid


@dataclass(frozen=True)
class RGBLongitudinalSection:
    """An axial sRGB cross-section, produced by :func:`propagate_polychromatic_longitudinal`.

    Attributes
    ----------
    rgb : Array
        sRGB image of shape ``(n_z, n_t, 3)`` in ``[0, 1]`` — one propagated
        plane per row, the transverse line along the columns, three color
        channels last.
    z : Array
        Propagation distances [m], one per row (length ``n_z``).
    t : Array
        Transverse coordinate [m] along the sliced line (length ``n_t``).
    axis : str
        Which transverse axis was sampled, ``"x"`` or ``"y"``.
    """

    rgb: Array
    z: Array
    t: Array
    axis: str


def propagate_polychromatic_longitudinal(
    field_of: FieldOf,
    wavelengths: Sequence[float],
    zs: Sequence[float],
    *,
    n: float = 1.0,
    axis: str = "x",
    offset: float = 0.0,
    output_half_width: Optional[float] = None,
    output_samples: int = 512,
    pad_factor: int = 1,
    bandlimit: bool = True,
    weights: Optional[Sequence[float]] = None,
    gamut: str = "desaturate",
    saturation: float = 1.0,
    stretch: float = 1.0,
    brightness: float = 1.0,
) -> RGBLongitudinalSection:
    """Broadband axial (``x–z``/``y–z``) cross-section, composited to sRGB.

    The polychromatic counterpart of
    :func:`~diffractor.physics.longitudinal.longitudinal_field`: propagates the
    (possibly chromatic) field at every wavelength through the same z-sweep and
    transverse line, then composites the per-wavelength ``|U|²`` maps into a
    true-color longitudinal image with :mod:`diffractor.viz.colorimetry` — e.g. a
    white-light focusing cone, where each wavelength's own diffraction-limited
    scale (∝ λ) shows up as color fringing, or a grating's colored Talbot
    carpet.

    Parameters mirror :func:`~diffractor.physics.longitudinal.longitudinal_field`
    (``n``, ``axis``, ``offset``, ``output_half_width``, ``output_samples``,
    ``pad_factor``, ``bandlimit``) plus :func:`propagate_polychromatic`'s color
    controls (``weights``, ``gamut``, ``saturation``, ``stretch``,
    ``brightness``).

    Parameters
    ----------
    field_of : Field or callable
        The input field at ``z = 0``. Pass a single ``Field`` for a
        wavelength-independent source, or a callable ``λ [m] -> Field`` for a
        chromatic element (e.g. a lens or grating rebuilt per wavelength).
    wavelengths : sequence of float
        Wavelengths [m] to sample across the band.
    zs : sequence of float
        Propagation distances [m] (the horizontal axis of the map).

    Returns
    -------
    RGBLongitudinalSection
        The composited image and its ``z`` / ``t`` coordinates.
    """
    build = _as_builder(field_of)
    wavelengths = [float(w) for w in wavelengths]
    if not wavelengths:
        raise ValueError("wavelengths is empty.")

    frames = []
    z_arr = t_arr = None
    for lam in wavelengths:
        field = build(lam)
        sec = longitudinal_field(
            field, lam, zs,
            n=n, axis=axis, offset=offset,
            output_half_width=output_half_width, output_samples=output_samples,
            pad_factor=pad_factor, bandlimit=bandlimit,
            normalize=False,  # raw |U|^2: colorimetry needs relative brightness
        )
        if z_arr is None:
            z_arr, t_arr = sec.z, sec.t
        frames.append(sec.intensity)

    stack = np.stack(frames, axis=0)  # (K, n_z, n_t)
    rgb = spectrum_to_srgb(
        np.array(wavelengths) * 1e9,
        stack,
        weights=None if weights is None else np.asarray(weights, dtype=float),
        gamut=gamut,
        saturation=saturation,
        stretch=stretch,
        brightness=brightness,
    )
    return RGBLongitudinalSection(rgb=rgb, z=z_arr, t=t_arr, axis=axis)


@dataclass(frozen=True)
class PolychromaticImage:
    """A rendered broadband image: an sRGB array on its shared output grid.

    Returned by :meth:`PolychromaticField.propagate`. ``plot`` draws it on a
    matplotlib axis via :func:`~diffractor.viz.plotting.plot_rgb`.
    """

    rgb: Array
    grid: Grid

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the RGB array ``(H, W, 3)``."""
        return self.rgb.shape

    def plot(self, ax=None, **kwargs):
        """Draw the sRGB image on a matplotlib axis (creates one if None)."""
        from ..viz.plotting import plot_rgb  # lazy: matplotlib optional

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(constrained_layout=True)
        plot_rgb(ax, self.rgb, self.grid, **kwargs)
        return ax


class PolychromaticField:
    """A broadband field with the same fluent building API as
    :class:`~diffractor.physics.monochromatic.MonochromaticField`.

    Where a ``MonochromaticField`` carries a single ``wavelength``, a
    ``PolychromaticField`` carries an array of ``wavelengths`` (and optional
    spectral ``weights``). The ``add_*`` builder methods are **recorded** and
    replayed on a fresh ``MonochromaticField`` at *each* wavelength when you
    propagate, so wavelength-dependent elements (``add_lens``, ``add_surface``,
    ``add_phase_grating``) automatically get their correct per-λ phase while
    amplitude masks stay wavelength-independent. Propagation composites the
    per-wavelength intensities to an sRGB image through the CIE 1931
    color-matching functions.

    Parameters
    ----------
    grid : Grid
        Sampling grid the field lives on.
    source : callable, scalar, array or Field
        What defines the field on the grid, as for ``MonochromaticField``.
    wavelengths : sequence of float
        Wavelengths [m] to sample across the band.
    weights : sequence of float, optional
        Spectral power at each wavelength (e.g. ``d65_weights(wl * 1e9)``).
        Defaults to flat.
    n : float
        Refractive index of the medium the field currently sits in.

    Examples
    --------
    >>> img = (PolychromaticField(grid, 1.0, wavelengths=wl, weights=d65_weights(wl * 1e9))
    ...        .add_aperture(polygon_aperture, 6, 0.5e-3, antialiased=True)
    ...        .propagate(0.3, method="fresnel_zoom", output_half_width=2e-3))
    >>> img.plot(ax)
    """

    def __init__(
        self,
        grid: Grid,
        source=1.0,
        *,
        wavelengths: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        n: float = 1.0,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive.")
        wl = np.asarray([float(w) for w in wavelengths], dtype=float)
        if wl.size == 0:
            raise ValueError("wavelengths is empty.")
        if np.any(wl <= 0):
            raise ValueError("wavelengths must be positive.")
        self.grid = grid
        self.wavelengths = wl
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.n = n
        self._source = source
        self._ops: list = []  # recorded (method_name, args, kwargs) to replay per-λ

    # -- recording -------------------------------------------------------------
    def _record(self, name: str, args: tuple, kwargs: dict) -> "PolychromaticField":
        self._ops.append((name, args, kwargs))
        return self

    def field_at(self, wavelength: float) -> Field:
        """The built (pre-propagation) :class:`Field` at a single wavelength."""
        mf = MonochromaticField(self.grid, self._source, wavelength=float(wavelength), n=self.n)
        for name, args, kwargs in self._ops:
            getattr(mf, name)(*args, **kwargs)
        return mf.to_field()

    # -- field building (mirror MonochromaticField; chainable) ----------------
    def add_aperture(self, aperture, *params, **kwargs) -> "PolychromaticField":
        """Record an aperture mask (see ``MonochromaticField.add_aperture``)."""
        return self._record("add_aperture", (aperture, *params), kwargs)

    def add_grating(self, grating, *params, **kwargs) -> "PolychromaticField":
        """Record an amplitude grating (see ``MonochromaticField.add_grating``)."""
        return self._record("add_grating", (grating, *params), kwargs)

    def add_phase_grating(self, **kwargs) -> "PolychromaticField":
        """Record a chromatic phase grating, evaluated per wavelength."""
        return self._record("add_phase_grating", (), kwargs)

    def add_lens(self, focal_length: float, **kwargs) -> "PolychromaticField":
        """Record a thin lens, whose phase scales with 1/λ per wavelength."""
        return self._record("add_lens", (focal_length,), kwargs)

    def add_surface(self, surface, **kwargs) -> "PolychromaticField":
        """Record a refracting surface, evaluated per wavelength."""
        return self._record("add_surface", (surface,), kwargs)

    def add_phase(self, phase) -> "PolychromaticField":
        """Record a (wavelength-independent) phase screen ``exp(i·phase)``."""
        return self._record("add_phase", (phase,), {})

    def multiply(self, mask) -> "PolychromaticField":
        """Record a generic (wavelength-independent) multiplicative mask."""
        return self._record("multiply", (mask,), {})

    # -- propagation (returns rendered results) -------------------------------
    def propagate(
        self,
        z: float,
        *,
        method: str = "fresnel_zoom",
        weights: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> PolychromaticImage:
        """Propagate every wavelength to ``z`` and composite to an sRGB image.

        Parameters
        ----------
        z : float
            Propagation distance [m].
        method : {"fresnel_zoom", "asm"}
            Shared-output-grid propagator (single-FFT methods are not offered:
            their output grid scales with λ). Default ``"fresnel_zoom"``.
        weights : sequence of float, optional
            Override the field's spectral weights for this call.
        **kwargs :
            Forwarded to :func:`propagate_polychromatic` (``output_half_width``,
            ``output_samples``, ``output_grid``, ``gamut``, ``saturation``,
            ``stretch``, ``brightness``, ``pad_factor``).

        Returns
        -------
        PolychromaticImage
            The rendered sRGB image and its shared output grid.
        """
        w = self.weights if weights is None else np.asarray(weights, dtype=float)
        rgb, out_grid = propagate_polychromatic(
            self.field_at, self.wavelengths, z,
            weights=w, n=self.n, propagator=method, **kwargs,
        )
        return PolychromaticImage(rgb=rgb, grid=out_grid)

    def longitudinal(
        self,
        zs: Sequence[float],
        *,
        axis: str = "x",
        weights: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> RGBLongitudinalSection:
        """Broadband axial (``x–z``/``y–z``) cross-section, composited to sRGB.

        Thin wrapper over :func:`propagate_polychromatic_longitudinal` using the
        field's stored wavelengths, weights and ``n``.
        """
        w = self.weights if weights is None else np.asarray(weights, dtype=float)
        return propagate_polychromatic_longitudinal(
            self.field_at, self.wavelengths, zs,
            n=self.n, axis=axis, weights=w, **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"PolychromaticField(shape={self.grid.shape}, "
            f"n_wavelengths={self.wavelengths.size}, n={self.n})"
        )
