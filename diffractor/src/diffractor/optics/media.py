"""Media: a homogeneous region is just an index.  A class, not a subsystem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Medium"]


@dataclass(frozen=True)
class Medium:
    """Homogeneous, lossless, isotropic medium of refractive index ``n``."""

    n: float

    def __post_init__(self):
        if self.n <= 0:
            raise ValueError("refractive index must be positive")

    def k(self, wavelength: float) -> float:
        """Wavenumber k = 2π n / λ₀ at vacuum wavelength ``wavelength``."""
        return 2.0 * np.pi * self.n / wavelength
