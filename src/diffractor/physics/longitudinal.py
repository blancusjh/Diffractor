"""Longitudinal (axial) field cross-sections.

Propagate a single monochromatic field through a range of distances and slice a
transverse line at each plane to assemble an ``x–z`` (or ``y–z``) map of the
intensity. This is the standard way to *see* propagation itself: a lens's
focusing cone and focal waist, a beam's Rayleigh range, or a periodic grating's
Talbot self-imaging carpet.

The heavy lifting is done by :class:`~diffractor.physics.asm.AngularSpectrum`, which
precomputes the transfer-function machinery and the input FFT once, so every
extra plane in the sweep costs a single inverse transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .asm import AngularSpectrum
from ..mathutils.backend import asnumpy
from .field import Field
from ..mathutils.grids import Array, Grid

__all__ = ["LongitudinalSection", "longitudinal_field"]


@dataclass(frozen=True)
class LongitudinalSection:
    """An axial intensity cross-section produced by :func:`longitudinal_field`.

    Attributes
    ----------
    intensity : Array
        Real intensity map of shape ``(n_z, n_t)`` — one propagated plane per
        row, the transverse line along the columns.
    z : Array
        Propagation distances [m], one per row (length ``n_z``).
    t : Array
        Transverse coordinate [m] along the sliced line (length ``n_t``).
    axis : str
        Which transverse axis was sampled, ``"x"`` or ``"y"``.
    """

    intensity: Array
    z: Array
    t: Array
    axis: str


def longitudinal_field(
    field: Field,
    wavelength: float,
    zs: Sequence[float],
    *,
    n: float = 1.0,
    axis: str = "x",
    offset: float = 0.0,
    output_half_width: Optional[float] = None,
    output_samples: int = 512,
    pad_factor: int = 1,
    bandlimit: bool = True,
    normalize: bool = True,
) -> LongitudinalSection:
    """Assemble an axial (``x–z``/``y–z``) intensity cross-section of a field.

    The field is propagated with the angular-spectrum method to every distance
    in ``zs``; at each plane the transverse line through ``offset`` (on the
    perpendicular axis) is extracted, and the stacked ``|U|²`` lines form the
    longitudinal map.

    Parameters
    ----------
    field : Field
        Input field at ``z = 0`` (e.g. an aperture, an aperture times a lens
        phase, or a grating). Its grid sets the transverse sampling.
    wavelength : float
        Vacuum wavelength [m].
    zs : sequence of float
        Propagation distances [m] (the horizontal axis of the map). May include
        or straddle zero; negative distances back-propagate.
    n : float
        Refractive index of the medium.
    axis : {"x", "y"}
        Transverse axis to slice. ``"x"`` holds ``y = offset`` and varies ``x``.
    offset : float
        Position [m] on the perpendicular axis at which to take the line
        (default the optical axis, ``0``). With the native sampling this snaps
        to the nearest grid line; with ``output_half_width`` it is evaluated
        exactly.
    output_half_width : float, optional
        Half-width [m] of a *decoupled* transverse output window. When given,
        each plane is sampled on a line of ``output_samples`` points spanning
        ``±output_half_width`` (via :class:`~diffractor.physics.asm.AngularSpectrum`'s
        matrix-DFT output grid) instead of the input grid — e.g. to resolve a
        focal waist far finer than the input spacing without enlarging the
        input ``N``. As everywhere in the package, this decouples *output*
        sampling only: the input grid must still sample the field adequately.
    output_samples : int
        Number of transverse samples when ``output_half_width`` is given.
    pad_factor : int
        Zero-padding passed to :class:`~diffractor.physics.asm.AngularSpectrum` to
        suppress FFT wrap-around. Leave at ``1`` for a periodic grating whose
        window spans an integer number of periods (the wrap is then physical).
    bandlimit : bool
        Apply the Matsushima–Shimobaba band limit (default ``True``).
    normalize : bool
        Divide the whole map by its global maximum (default ``True``), so a
        single ``vmin/vmax`` spans the section.

    Returns
    -------
    LongitudinalSection
        The intensity map and its ``z`` / ``t`` coordinates.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'.")

    grid = field.grid
    xline = asnumpy(grid.x[0, :])
    yline = asnumpy(grid.y[:, 0])

    if output_half_width is not None:
        # Decoupled transverse sampling: a line output grid through `offset`,
        # evaluated per plane by the precomputed matrix DFT.
        if output_half_width <= 0:
            raise ValueError("output_half_width must be positive.")
        if output_samples < 2:
            raise ValueError("output_samples must be at least 2.")
        t = np.linspace(-output_half_width, output_half_width, output_samples)
        if axis == "x":
            line_grid = Grid(t[None, :], np.full((1, t.size), float(offset)))
        else:
            line_grid = Grid(np.full((t.size, 1), float(offset)), t[:, None])
        prop = AngularSpectrum(
            field, wavelength=wavelength, n=n, output_grid=line_grid,
            pad_factor=pad_factor, bandlimit=bandlimit,
        )
        take = (lambda U: U[0, :]) if axis == "x" else (lambda U: U[:, 0])
    else:
        prop = AngularSpectrum(
            field, wavelength=wavelength, n=n, pad_factor=pad_factor, bandlimit=bandlimit
        )
        if axis == "x":
            idx = int(np.argmin(np.abs(yline - offset)))
            t = xline
        else:
            idx = int(np.argmin(np.abs(xline - offset)))
            t = yline
        take = (lambda U: U[idx, :]) if axis == "x" else (lambda U: U[:, idx])

    zs = np.asarray([float(z) for z in zs], dtype=float)
    rows = np.empty((zs.size, t.size), dtype=float)
    for i, z in enumerate(zs):
        line = asnumpy(take(prop.propagate(float(z)).values))
        rows[i] = np.abs(line) ** 2

    if normalize:
        peak = rows.max()
        if peak > 0:
            rows = rows / peak

    return LongitudinalSection(intensity=rows, z=zs, t=np.asarray(t), axis=axis)
