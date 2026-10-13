"""Combine fields at a common observation plane without display normalization."""

from __future__ import annotations

import numpy as np

from ..mathutils.backend import asnumpy
from .monochromatic import MonochromaticField

__all__ = ["coherent_sum", "incoherent_intensity"]


def _fields(fields, *, coherent):
    fields = tuple(fields)
    if not fields:
        raise ValueError("Provide at least one field.")
    if not all(isinstance(f, MonochromaticField) for f in fields):
        raise TypeError("Expected MonochromaticField objects.")
    first = fields[0]
    for field in fields:
        if field.n != first.n:
            raise ValueError("Fields must share the same medium.")
        if coherent and field.wavelength != first.wavelength:
            raise ValueError("Coherent fields must share the same wavelength.")
        if field.to_field().domain != "space":
            raise ValueError("Expected fields in the spatial domain.")
        if field.to_field().xp is not first.to_field().xp:
            raise ValueError("Fields must use the same array backend.")
        if field.grid is not first.grid and not (
            np.array_equal(asnumpy(field.grid.x), asnumpy(first.grid.x))
            and np.array_equal(asnumpy(field.grid.y), asnumpy(first.grid.y))
        ):
            raise ValueError("Fields must share the same observation grid.")
    return fields


def coherent_sum(fields, *, coefficients=None) -> MonochromaticField:
    """Superpose same-wavelength fields: ``U = sum(c[j] * U[j])``.

    Complex amplitude ``coefficients`` encode relative amplitude and phase;
    they default to one. Inputs must share a grid, medium and array backend.
    The returned field owns new samples and can be propagated further.
    No input is modified and no normalization is applied.
    """
    fields = _fields(fields, coherent=True)
    c = np.ones(len(fields), complex) if coefficients is None else np.asarray(coefficients, complex)
    if c.shape != (len(fields),) or not np.isfinite(c).all():
        raise ValueError("Provide one finite complex coefficient per field.")
    values = fields[0].values * c[0]
    for field, coefficient in zip(fields[1:], c[1:]):
        values += coefficient * field.values
    first = fields[0]
    return MonochromaticField._wrap(first.to_field().like(values), first.wavelength, first.n)


def incoherent_intensity(fields, *, weights=None):
    """Sum mutually incoherent contributions: ``I = sum(w[j] * |U[j]|²)``.

    Inputs must share an observation grid, medium and array backend; their
    wavelengths may differ. Nonnegative power ``weights`` default to one.
    For an ensemble *mean*, explicitly use weights summing to one. Returns an
    intensity array on the input device, without normalization or RGB mapping.
    This is a measurement, not a complex field that can be propagated again.
    """
    fields = _fields(fields, coherent=False)
    w = np.ones(len(fields)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(fields),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Provide one finite nonnegative power weight per field.")
    result = fields[0].intensity * w[0]
    for field, weight in zip(fields[1:], w[1:]):
        result += weight * field.intensity
    return result
