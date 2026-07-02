"""The :class:`Field` — a sampled scalar field bundled with its grid.

A field is the pair ``φ = (grid, values)``: the sampling :class:`~diffraction.grids.Grid`
and the complex samples on it. Coordinate queries live on the *grid*
(``field.grid.x``, ``field.grid.spacing``), and Fourier transforms are
*operators* applied to a field (:func:`~diffraction.fourier.FT2`,
:func:`~diffraction.fourier.IFT2`) — a ``Field`` deliberately exposes neither
coordinate accessors nor transform methods of its own.

The ``domain`` tag distinguishes a spatial field (``"space"``) from a spectral
one (``"frequency"``); the Fourier operators flip it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple, Union

from .backend import array_module, to_device
from .grids import Array, Grid

__all__ = ["Field"]

_Scalar = Union[int, float, complex]


@dataclass(frozen=True, eq=False)
class Field:
    """A scalar field: samples ``values`` on a :class:`Grid`.

    Parameters
    ----------
    grid : Grid
        Sampling grid (spatial or frequency coordinates).
    values : 2D array
        Complex (or real) samples, NumPy or CuPy, matching ``grid.shape``.
    domain : {"space", "frequency"}
        Which domain the samples live in. Defaults to ``"space"``.

    Notes
    -----
    ``eq=False`` keeps the field hashable-by-identity and avoids the ambiguous
    element-wise ``==`` of the underlying arrays.
    """

    grid: Grid
    values: Array
    domain: str = "space"

    def __post_init__(self) -> None:
        if self.domain not in ("space", "frequency"):
            raise ValueError("domain must be 'space' or 'frequency'.")
        if self.values.shape != self.grid.shape:
            raise ValueError(
                f"values shape {self.values.shape} does not match grid shape "
                f"{self.grid.shape}."
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the sample array."""
        return self.values.shape

    @property
    def xp(self):
        """Array module (NumPy or CuPy) backing ``values``."""
        return array_module(self.values)

    @property
    def intensity(self) -> Array:
        """Intensity ``|values|²`` (unnormalized), on the field's device."""
        return self.xp.abs(self.values) ** 2

    def like(self, values: Array) -> "Field":
        """A new field with the same grid and domain but different ``values``."""
        return replace(self, values=values)

    def to(self, device: str) -> "Field":
        """Move the samples to ``"cpu"`` (NumPy) or ``"gpu"`` (CuPy)."""
        return replace(self, values=to_device(self.values, device))

    def _same_grid(self, other: "Field") -> bool:
        if self.grid is other.grid:
            return True
        if self.grid.shape != other.grid.shape:
            return False
        xp = self.xp
        return bool(
            xp.allclose(self.grid.x, other.grid.x)
            and xp.allclose(self.grid.y, other.grid.y)
        )

    def __mul__(self, other: Union["Field", Array, _Scalar]) -> "Field":
        """Multiply by a scalar, a bare array, or a co-gridded field."""
        if isinstance(other, Field):
            if not self._same_grid(other):
                raise ValueError("Cannot multiply fields on different grids.")
            return self.like(self.values * other.values)
        return self.like(self.values * other)

    __rmul__ = __mul__
