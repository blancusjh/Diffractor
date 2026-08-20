"""Time: the spectral content of a field.

Analytically a field is a function of space *and time*; numerically its time
side is sampled as a set of vacuum wavelengths with relative spectral weights.
That pair is a :class:`Spectrum`.  Monochromatic light is not a different
kind of thing — it is the one-line spectrum :meth:`Spectrum.line`.

Weights are relative spectral power; they matter when per-wavelength
intensities are combined (``Field.intensity``, colour compositing) and nowhere
else — propagation treats every line independently, which is what linearity
of the wave equation says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
from scipy.constants import c as _c, h as _h, k as _kB

__all__ = ["Spectrum"]


@dataclass(frozen=True, eq=False)
class Spectrum:
    """Sampled spectral content: vacuum ``wavelengths`` and relative ``weights``.

    Wavelengths must be positive and strictly ascending; weights non-negative,
    one per line (omitted → flat).
    """

    wavelengths: np.ndarray
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        wl = np.atleast_1d(np.asarray(self.wavelengths, float))
        if wl.ndim != 1 or wl.size == 0:
            raise ValueError("wavelengths must be a non-empty 1-D array")
        if np.any(wl <= 0):
            raise ValueError("wavelengths must be positive")
        if wl.size > 1 and np.any(np.diff(wl) <= 0):
            raise ValueError("wavelengths must be strictly ascending")
        w = (np.ones(wl.size) if self.weights is None
             else np.atleast_1d(np.asarray(self.weights, float)))
        if w.shape != wl.shape:
            raise ValueError("weights must match wavelengths, one per line")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        object.__setattr__(self, "wavelengths", wl)
        object.__setattr__(self, "weights", w)

    # ── constructors ─────────────────────────────────────────────────────────
    @classmethod
    def line(cls, wavelength: float) -> "Spectrum":
        """The monochromatic spectrum: one line, unit weight."""
        return cls(np.array([float(wavelength)]))

    @classmethod
    def flat(cls, lo: float, hi: float, n: int) -> "Spectrum":
        """``n`` equally weighted lines spanning [lo, hi]."""
        return cls(np.linspace(float(lo), float(hi), int(n)))

    @classmethod
    def blackbody(cls, lo: float, hi: float, n: int, *,
                  temperature: float) -> "Spectrum":
        """``n`` lines over [lo, hi] weighted by Planck spectral radiance.

        Weights are peak-normalised — they are *relative* spectral power, and
        every consumer of a Spectrum is homogeneous in the weights.
        """
        wl = np.linspace(float(lo), float(hi), int(n))
        radiance = ((2.0 * _h * _c**2 / wl**5)
                    / np.expm1(_h * _c / (wl * _kB * float(temperature))))
        return cls(wl, radiance / radiance.max())

    # ── queries ──────────────────────────────────────────────────────────────
    @property
    def n(self) -> int:
        """Number of spectral lines."""
        return self.wavelengths.size

    def normalized(self) -> "Spectrum":
        """The same lines with weights summing to one."""
        return Spectrum(self.wavelengths, self.weights / self.weights.sum())

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        """Iterate as ``(wavelength, weight)`` pairs."""
        return iter(zip(self.wavelengths.tolist(), self.weights.tolist()))
