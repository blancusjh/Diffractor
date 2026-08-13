"""The measurement protocol used to validate Diffractor against the BEM.

1. Solve the transmission BVP (Muller BOR-BEM) for the closed body.
2. Sample the interior field on the exit reference sphere S2 about the focus.
3. Demodulate the converging carrier: A(th) = psi * R2 * exp(+i k2 R2).
4. Compare |A| and arg A against the prediction under test, ABSOLUTELY
   (the point-source amplitude 1/4pi fixes the scale; never fit it).
5. Repeat over kR and check the residual falls like 1/(kR) before
   extrapolating to macroscopic scale.
"""
import numpy as np


def measure_on_sphere(field_eval, zi, R2, th, k2):
    """Steps 2-3: sample and demodulate. `field_eval(rho, z)` evaluates psi."""
    psi = field_eval(R2 * np.sin(th), zi - R2 * np.cos(th))
    return psi * R2 * np.exp(1j * k2 * R2)
