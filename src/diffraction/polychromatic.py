"""Polychromatic (broadband / white-light) propagation and rendering.

A broadband field is just many monochromatic fields. This module propagates a
source at each wavelength onto a **shared** output grid and composites the
per-wavelength intensities into an sRGB image with
:mod:`diffraction.colorimetry`.

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

from .colorimetry import spectrum_to_srgb
from .field import Field
from .grids import Array, Grid
from .longitudinal import longitudinal_field
from .propagation import asm_propagator, fresnel_zoom_propagator

__all__ = [
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
        chromatic element (e.g. a :func:`~diffraction.gratings.phase_grating`).
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
        Out-of-gamut handling in :func:`~diffraction.colorimetry.xyz_to_srgb`.
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
    :func:`~diffraction.longitudinal.longitudinal_field`: propagates the
    (possibly chromatic) field at every wavelength through the same z-sweep and
    transverse line, then composites the per-wavelength ``|U|²`` maps into a
    true-color longitudinal image with :mod:`diffraction.colorimetry` — e.g. a
    white-light focusing cone, where each wavelength's own diffraction-limited
    scale (∝ λ) shows up as color fringing, or a grating's colored Talbot
    carpet.

    Parameters mirror :func:`~diffraction.longitudinal.longitudinal_field`
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
