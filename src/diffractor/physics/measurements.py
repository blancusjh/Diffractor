"""Point and line probes on rectilinear grids, in physical coordinates."""

from __future__ import annotations

import numpy as np

from ..mathutils.backend import asnumpy
from .field import Field
from .monochromatic import MonochromaticField

__all__ = ["sample_field", "sample_intensity"]


def _sample(field, x, y, *, intensity):
    if isinstance(field, MonochromaticField):
        field = field.to_field()
    if not isinstance(field, Field):
        raise TypeError("Expected a Field or MonochromaticField.")
    if field.domain != "space":
        raise ValueError("Probes require a field in the spatial domain.")
    gx, gy = asnumpy(field.grid.x), asnumpy(field.grid.y)
    if gx.ndim != 2 or gx.shape != gy.shape or min(gx.shape) < 2:
        raise ValueError("Probes require a 2D rectilinear grid with at least two samples per axis.")
    xs, ys = gx[0, :], gy[:, 0]
    if not (
        np.isfinite(gx).all() and np.isfinite(gy).all()
        and np.all(gx == xs[None, :]) and np.all(gy == ys[:, None])
        and np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)
    ):
        raise ValueError("Probes require increasing rectilinear grid coordinates.")
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Probe coordinates must be finite.")
    if np.any((x < xs[0]) | (x > xs[-1]) | (y < ys[0]) | (y > ys[-1])):
        raise ValueError("Probe coordinates lie outside the sampled grid.")
    ix = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    iy = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
    tx = (x - xs[ix]) / (xs[ix + 1] - xs[ix])
    ty = (y - ys[iy]) / (ys[iy + 1] - ys[iy])
    values = asnumpy(field.intensity if intensity else field.values)
    return (
        (1 - ty) * ((1 - tx) * values[iy, ix] + tx * values[iy, ix + 1])
        + ty * ((1 - tx) * values[iy + 1, ix] + tx * values[iy + 1, ix + 1])
    )


def sample_field(field, x, y):
    """Bilinearly sample complex amplitude at ``(x, y)`` in meters.

    Scalars give a point probe; broadcastable arrays give lines or point
    collections. Returns NumPy values (GPU data is copied to the CPU).
    Queries outside the grid raise an error. The grid must resolve the field:
    interpolation cannot recover unsampled phase oscillations.
    """
    return _sample(field, x, y, intensity=False)


def sample_intensity(field, x, y):
    """Bilinearly sample ``|U|²``, without normalization, at ``(x, y)``.

    Uses the same coordinates and bounds as :func:`sample_field`. Intensity
    is computed *before* interpolation, so opposite phases at neighboring
    samples do not create a spurious dark point in an intensity probe.
    """
    return _sample(field, x, y, intensity=True)
