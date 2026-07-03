"""Longitudinal (axial) field cross-sections.

Propagate a single monochromatic field through a range of distances and slice a
transverse line at each plane to assemble an ``x–z`` (or ``y–z``) map of the
intensity. This is the standard way to *see* propagation itself: a lens's
focusing cone and focal waist, a beam's Rayleigh range, or a periodic grating's
Talbot self-imaging carpet.

The heavy lifting is done by :class:`~diffraction.asm.AngularSpectrum`, which
precomputes the transfer-function machinery and the input FFT once, so every
extra plane in the sweep costs a single inverse transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .asm import AngularSpectrum
from .backend import asnumpy
from .field import Field
from .grids import Array

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
        (default the optical axis, ``0``).
    pad_factor : int
        Zero-padding passed to :class:`~diffraction.asm.AngularSpectrum` to
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

    prop = AngularSpectrum(
        field, wavelength=wavelength, n=n, pad_factor=pad_factor, bandlimit=bandlimit
    )

    if axis == "x":
        idx = int(np.argmin(np.abs(yline - offset)))
        t = xline
    else:
        idx = int(np.argmin(np.abs(xline - offset)))
        t = yline

    zs = np.asarray([float(z) for z in zs], dtype=float)
    rows = np.empty((zs.size, t.size), dtype=float)
    for i, z in enumerate(zs):
        U = prop.propagate(float(z)).values
        line = asnumpy(U[idx, :] if axis == "x" else U[:, idx])
        rows[i] = np.abs(line) ** 2

    if normalize:
        peak = rows.max()
        if peak > 0:
            rows = rows / peak

    return LongitudinalSection(intensity=rows, z=zs, t=np.asarray(t), axis=axis)
