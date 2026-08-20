"""Fields: samples of a function of space and time.

Analytically a scalar optical field is a function of space; numerically its
samples are measured at space samples and at time samples.  A :class:`Field`
is therefore the triple (grid, values, spectrum) plus the :class:`Medium` it
lives in: ``values`` always carries a trailing spectral axis, shape
``(*grid.shape, spectrum.n)``, so one array holds the whole measurement and
every operator vectorises over wavelength instead of looping.

Monochromatic light is the one-line spectrum.  :class:`MonochromaticField`
is the companion class for that everyday case: it takes a plain 2-D array and
a wavelength, exposes them back as ``.u`` and ``.wavelength``, and — because
every operator rebuilds fields through :meth:`Field.like` — survives any
pipeline of transforms and propagators as itself.

Fields are frozen.  Applying a mask or a phase is building a new field from
new values: ``field.like(field.values * mask[..., None])``.  There is no
operator overloading — an aperture is physics, and physics is written out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .optics.media import Medium
from .space import Grid
from .spectrum import Spectrum

__all__ = ["Field", "MonochromaticField"]

_DOMAINS = ("space", "frequency")


def _over_grid(a: np.ndarray) -> np.ndarray:
    """Broadcast a grid-shaped array over the spectral axis."""
    return np.asarray(a)[..., np.newaxis]


def _over_spectrum(a) -> np.ndarray:
    """Broadcast a per-wavelength vector over the grid axes."""
    return np.reshape(np.asarray(a), (1, 1, -1))


@dataclass(frozen=True, eq=False)
class Field:
    """Sampled scalar field: complex ``values`` on a ``grid``, per spectral line.

    ``values`` has shape ``(*grid.shape, spectrum.n)``; a 2-D array is accepted
    when the spectrum is a single line and expanded to the canonical 3-D shape.
    ``domain`` says whether the samples live in space or in the reciprocal
    (angular-frequency) space — the Fourier operators flip it.
    """

    grid: Grid
    values: np.ndarray
    spectrum: Spectrum
    medium: Medium = Medium(1.0)
    domain: str = "space"

    def __post_init__(self):
        values = np.asarray(self.values, complex)
        if values.ndim == 2 and self.spectrum.n == 1:
            values = values[..., np.newaxis]
        expected = (*self.grid.shape, self.spectrum.n)
        if values.shape != expected:
            raise ValueError(
                f"values shape {values.shape} does not match "
                f"(*grid.shape, spectrum.n) = {expected}")
        if self.domain not in _DOMAINS:
            raise ValueError(f"domain must be one of {_DOMAINS}, got {self.domain!r}")
        object.__setattr__(self, "values", values)

    # ── measured aspects ─────────────────────────────────────────────────────
    @property
    def amplitude(self) -> np.ndarray:
        """|values| per node per spectral line, shape ``(*grid.shape, n_λ)``."""
        return np.abs(self.values)

    @property
    def phase(self) -> np.ndarray:
        """arg(values) per node per spectral line, shape ``(*grid.shape, n_λ)``."""
        return np.angle(self.values)

    @property
    def spectral_intensity(self) -> np.ndarray:
        """|values|² per spectral line (unweighted), shape ``(*grid.shape, n_λ)``."""
        return np.abs(self.values) ** 2

    @property
    def intensity(self) -> np.ndarray:
        """Spectrum-weighted intensity Σ_λ w_λ |U_λ|², shape ``grid.shape``."""
        return (self.spectral_intensity
                * _over_spectrum(self.spectrum.weights)).sum(axis=-1)

    def power(self) -> np.ndarray:
        """∬ |U_λ|² d²x per spectral line, shape ``(n_λ,)`` (unweighted)."""
        return (self.spectral_intensity
                * _over_grid(self.grid.weights())).sum(axis=(0, 1))

    # ── construction of relatives ────────────────────────────────────────────
    def like(self, values: np.ndarray, *, grid: Optional[Grid] = None,
             domain: Optional[str] = None,
             medium: Optional[Medium] = None) -> "Field":
        """A field of the same kind with new ``values`` (and optionally a new
        grid, domain or medium).  The spectrum always carries over, so a
        :class:`MonochromaticField` stays monochromatic through every operator.
        """
        new = object.__new__(type(self))
        object.__setattr__(new, "grid", self.grid if grid is None else grid)
        object.__setattr__(new, "values", values)
        object.__setattr__(new, "spectrum", self.spectrum)
        object.__setattr__(new, "medium", self.medium if medium is None else medium)
        object.__setattr__(new, "domain", self.domain if domain is None else domain)
        Field.__post_init__(new)
        return new


class MonochromaticField(Field):
    """The one-line field: a 2-D array of samples at a single wavelength.

    A convenience shape of :class:`Field`, not a different physics: the
    spectrum is ``Spectrum.line(wavelength)`` and the trailing spectral axis
    has length one.  ``.u`` gives the samples back as the 2-D array they came
    from; ``.wavelength`` the single vacuum wavelength.
    """

    def __init__(self, grid: Grid, values: np.ndarray, wavelength: float, *,
                 medium: Medium = Medium(1.0), domain: str = "space"):
        Field.__init__(self, grid, values, Spectrum.line(wavelength),
                       medium=medium, domain=domain)

    def __post_init__(self):
        Field.__post_init__(self)
        if self.spectrum.n != 1:
            raise ValueError("a MonochromaticField carries exactly one wavelength")

    @property
    def wavelength(self) -> float:
        """The single vacuum wavelength."""
        return float(self.spectrum.wavelengths[0])

    @property
    def u(self) -> np.ndarray:
        """The samples as a 2-D array, shape ``grid.shape``."""
        return self.values[..., 0]
