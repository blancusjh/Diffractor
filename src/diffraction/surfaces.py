from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParabolicSurface:
    """
    Axisymmetric parabolic surface.

    Sag equation:
        z(rho) = z0 + rho^2 / (4 f)

    where rho^2 = x^2 + y^2.
    """

    focal_length: float
    z0: float = 0.0

    def sag_rho(self, rho: np.ndarray) -> np.ndarray:
        """Return sag as a function of radial coordinate rho."""
        if self.focal_length == 0:
            raise ValueError("focal_length must be non-zero.")
        return self.z0 + (rho**2) / (4.0 * self.focal_length)

    def sag(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return sag z(x, y)."""
        rho = np.sqrt(x**2 + y**2)
        return self.sag_rho(rho)
