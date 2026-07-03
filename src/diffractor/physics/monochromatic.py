"""The :class:`MonochromaticField` — the fluent, single-wavelength API.

A :class:`MonochromaticField` is a thin, chainable *builder* around the
low-level immutable :class:`~diffractor.physics.field.Field`. It carries its own
``wavelength`` and medium index ``n``, so a whole optical setup reads as one
pipeline::

    U = (MonochromaticField(grid, 1.0, wavelength=532e-9)
         .add_aperture(circular_aperture, 0.3e-3, antialiased=True)
         .propagate(1.15, method="fresnel"))

The field itself is *defined* by the ``source`` argument: a function
``f(x, y) -> values`` evaluated on the grid, or a constant that fills the grid
uniformly (``1.0`` is a unit plane wave at normal incidence). The ``add_*``
methods mutate the field in place and return ``self`` for chaining; the
propagation methods return **new** objects on the output plane.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np

from ..mathutils.grids import Array, Grid
from .apertures import antialiased as _antialiased
from .field import Field
from .longitudinal import LongitudinalSection, longitudinal_field
from .propagation import (
    asm_propagator,
    fraunhofer_propagator,
    fresnel_propagator,
    fresnel_zoom_propagator,
)
from .surfaces import CartesianSurface, Surface, thin_lens

__all__ = ["MonochromaticField"]

Source = Union[float, complex, Array, Field, Callable[[Array, Array], Array]]
Mask = Union[float, complex, Array, Field, Callable[[Array, Array], Array]]

_PROPAGATORS = {
    "asm": asm_propagator,
    "fresnel": fresnel_propagator,
    "fraunhofer": fraunhofer_propagator,
    "fresnel_zoom": fresnel_zoom_propagator,
}


def _evaluate_source(source: Source, grid: Grid) -> Array:
    """Turn a ``source`` spec into a complex sample array on ``grid``.

    Accepts a callable ``f(x, y) -> array``, a scalar constant (broadcast to
    fill the grid uniformly), a bare array, or an existing :class:`Field`.
    """
    x, y = grid
    if isinstance(source, Field):
        values = source.values
    elif callable(source):
        values = source(x, y)
    elif np.isscalar(source):
        values = np.full(grid.shape, source)
    else:  # array-like
        values = np.asarray(source)
    values = np.asarray(values)
    if values.shape != grid.shape:
        raise ValueError(
            f"source produced shape {values.shape}, expected grid shape {grid.shape}."
        )
    return values.astype(np.complex128)


class MonochromaticField:
    """A single-wavelength scalar field with a chainable building API.

    Parameters
    ----------
    grid : Grid
        Sampling grid the field lives on.
    source : callable, scalar, array or Field
        What defines the field on the grid. A callable ``f(x, y) -> array`` is
        evaluated on the grid; a scalar constant fills the grid uniformly
        (``1.0`` = unit plane wave); an array/Field is used directly.
    wavelength : float
        Vacuum wavelength [m]. Carried by the field so the propagation methods
        need not repeat it.
    n : float
        Refractive index of the medium the field currently sits in.
    """

    def __init__(
        self,
        grid: Grid,
        source: Source = 1.0,
        *,
        wavelength: float = 532e-9,
        n: float = 1.0,
    ) -> None:
        if wavelength <= 0:
            raise ValueError("wavelength must be positive.")
        if n <= 0:
            raise ValueError("n must be positive.")
        self.wavelength = wavelength
        self.n = n
        self._field = Field(grid, _evaluate_source(source, grid))

    # -- alternate constructor -------------------------------------------------
    @classmethod
    def _wrap(cls, field: Field, wavelength: float, n: float) -> "MonochromaticField":
        """Wrap an existing low-level :class:`Field` (e.g. a propagator output)."""
        obj = cls.__new__(cls)
        obj._field = field
        obj.wavelength = wavelength
        obj.n = n
        return obj

    # -- read-only views -------------------------------------------------------
    @property
    def grid(self) -> Grid:
        """The sampling grid."""
        return self._field.grid

    @property
    def values(self) -> Array:
        """The complex samples."""
        return self._field.values

    @property
    def intensity(self) -> Array:
        """Intensity ``|values|²`` (unnormalized)."""
        return self._field.intensity

    @property
    def shape(self):
        """Shape of the sample array."""
        return self._field.shape

    def to_field(self) -> Field:
        """The underlying immutable :class:`Field` (escape hatch)."""
        return self._field

    def to(self, device: str) -> "MonochromaticField":
        """Move the samples to ``"cpu"`` or ``"gpu"`` (returns a new object)."""
        return self._wrap(self._field.to(device), self.wavelength, self.n)

    def copy(self) -> "MonochromaticField":
        """A detached copy sharing the same grid and values."""
        return self._wrap(self._field, self.wavelength, self.n)

    # -- field building (mutating, chainable) ---------------------------------
    def multiply(self, mask: Mask) -> "MonochromaticField":
        """Multiply the field by ``mask`` in place and return ``self``.

        ``mask`` may be a scalar, an array, a :class:`Field` on the same grid,
        or a callable ``f(x, y) -> array``.
        """
        x, y = self.grid
        if isinstance(mask, Field):
            self._field = self._field * mask
        elif callable(mask):
            self._field = self._field.like(self.values * np.asarray(mask(x, y)))
        else:
            self._field = self._field.like(self.values * mask)
        return self

    def add_aperture(
        self,
        aperture: Callable[..., Array],
        *params,
        antialiased: bool = False,
        factor: int = 4,
        **kwargs,
    ) -> "MonochromaticField":
        """Multiply an aperture/mask onto the field.

        Parameters
        ----------
        aperture : callable
            A mask kernel ``aperture(x, y, *params, **kwargs)`` from
            :mod:`diffractor.physics.apertures` (or any compatible function).
        *params, **kwargs :
            Positional/keyword parameters forwarded to ``aperture`` (e.g. the
            radius ``R`` of :func:`circular_aperture`).
        antialiased : bool
            Evaluate the mask with grey (area-coverage) edge pixels, suppressing
            the pixelation artifacts of hard edges. Default ``False``.
        factor : int
            Subsamples per pixel and axis when ``antialiased`` (default 4).
        """
        if antialiased:
            mask = _antialiased(aperture, self.grid, *params, factor=factor, **kwargs)
            return self.multiply(mask)
        x, y = self.grid
        return self.multiply(np.asarray(aperture(x, y, *params, **kwargs)))

    def add_grating(
        self,
        grating: Callable[..., Array],
        *params,
        antialiased: bool = False,
        factor: int = 4,
        **kwargs,
    ) -> "MonochromaticField":
        """Multiply an amplitude grating onto the field.

        Same forwarding rules as :meth:`add_aperture`. For a wavelength-
        dependent phase grating, use :meth:`multiply` with
        ``phase_grating(grid, self.wavelength, ...)`` instead.
        """
        return self.add_aperture(
            grating, *params, antialiased=antialiased, factor=factor, **kwargs
        )

    def add_phase_grating(self, **kwargs) -> "MonochromaticField":
        """Multiply by a chromatic thin phase-relief grating.

        Forwards to :func:`~diffractor.physics.gratings.phase_grating` at the
        field's own ``wavelength`` (``period=``, ``height=``, ``n1=``, ``n2=``,
        ``profile=``, ``orientation=``, ``center=``). Because it reads
        ``self.wavelength``, a :class:`PolychromaticField` replays it correctly
        per wavelength.
        """
        from .gratings import phase_grating

        return self.multiply(phase_grating(self.grid, self.wavelength, **kwargs))

    def add_phase(self, phase: Union[Array, Callable[[Array, Array], Array]]) -> "MonochromaticField":
        """Multiply by a pure phase screen ``exp(i·phase)``.

        ``phase`` is a real phase map [rad], either an array or a callable
        ``f(x, y) -> array``.
        """
        x, y = self.grid
        p = np.asarray(phase(x, y)) if callable(phase) else np.asarray(phase)
        return self.multiply(np.exp(1.0j * p))

    def add_lens(self, focal_length: float, *, n: Optional[float] = None) -> "MonochromaticField":
        """Multiply by an ideal thin-lens phase of focal length ``focal_length``."""
        n = self.n if n is None else n
        return self.multiply(thin_lens(self.grid, focal_length, self.wavelength, n=n))

    def add_surface(
        self,
        surface: Surface,
        *,
        n1: Optional[float] = None,
        n2: Optional[float] = None,
        enter: bool = True,
    ) -> "MonochromaticField":
        """Refract the field through a thin :class:`Surface`.

        Multiplies by ``surface.phase_mask(grid, wavelength, n1, n2)``. For a
        :class:`CartesianSurface` the surface's own ``n1``/``n2`` are used
        unless overridden. For other surfaces ``n1`` defaults to the field's
        current medium index and ``n2`` is required. With ``enter=True``
        (default) the field's medium index becomes ``n2`` afterwards.
        """
        if isinstance(surface, CartesianSurface):
            n1 = surface.n1 if n1 is None else n1
            n2 = surface.n2 if n2 is None else n2
        else:
            n1 = self.n if n1 is None else n1
            if n2 is None:
                raise ValueError("n2 is required for this surface.")
        self.multiply(surface.phase_mask(self.grid, self.wavelength, n1, n2))
        if enter:
            self.n = n2
        return self

    # -- propagation (returns new objects) ------------------------------------
    def propagate(self, z: float, *, method: str = "asm", **kwargs) -> "MonochromaticField":
        """Propagate the field a distance ``z`` and return the new field.

        Parameters
        ----------
        z : float
            Propagation distance [m].
        method : {"asm", "fresnel", "fraunhofer", "fresnel_zoom"}
            Propagator to use (default ``"asm"``). The field's stored
            ``wavelength`` and ``n`` are supplied automatically.
        **kwargs :
            Extra keyword arguments forwarded to the chosen propagator
            (e.g. ``pad_factor``, ``output_grid``, ``output_half_width``).
        """
        try:
            propagator = _PROPAGATORS[method]
        except KeyError:
            raise ValueError(
                f"unknown method {method!r}; choose from {sorted(_PROPAGATORS)}."
            ) from None
        wavelength = kwargs.pop("wavelength", self.wavelength)
        n = kwargs.pop("n", self.n)
        out = propagator(self._field, z, wavelength, n, **kwargs)
        return self._wrap(out, wavelength, n)

    def longitudinal(
        self,
        zs: Sequence[float],
        *,
        axis: str = "x",
        **kwargs,
    ) -> LongitudinalSection:
        """Axial ``x–z`` (or ``y–z``) intensity cross-section over ``zs``.

        Thin wrapper over
        :func:`~diffractor.physics.longitudinal.longitudinal_field` using the
        field's stored ``wavelength`` and ``n``.
        """
        return longitudinal_field(
            self._field, self.wavelength, zs, n=self.n, axis=axis, **kwargs
        )

    # -- convenience ----------------------------------------------------------
    def plot(self, ax=None, **kwargs):
        """Draw the (log-)intensity on a matplotlib axis (creates one if None)."""
        from ..viz.plotting import plot_intensity  # lazy: matplotlib optional

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(constrained_layout=True)
        plot_intensity(ax, self._field, **kwargs)
        return ax

    def __repr__(self) -> str:
        return (
            f"MonochromaticField(shape={self.shape}, "
            f"wavelength={self.wavelength:.3e} m, n={self.n})"
        )
