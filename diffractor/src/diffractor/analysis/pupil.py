"""Pupil analysis on reference spheres — the measurement protocol.

Two distinct things, kept deliberately separate:

* :func:`predicted_pupil` — what theory says (P = t_s d2/d1 for a stigmatic
  interface).  A prediction.
* :func:`demodulate` — what a FIELD says: strip the converging carrier from
  wave data sampled on the reference sphere.  A measurement.

The whole session's discipline in two functions: never plot the first and call
it the second.
"""

from __future__ import annotations

import numpy as np

__all__ = ["demodulate", "predicted_pupil", "energy_through_sphere"]


def demodulate(psi_on_sphere, R2, k2):
    """Pupil A(θ) from field samples on S2:  A = ψ · R2 · e^{+i k2 R2}.

    |A| is the measured amplitude; arg A the measured phase referenced to the
    sphere's centre (for a stigmatic system it should be k0·C = const).
    """
    return psi_on_sphere * R2 * np.exp(1j * k2 * R2)


def predicted_pupil(interface, g, amplitude=1.0 / (4 * np.pi)):
    """Ray-theory pupil on S2 for a stigmatic interface:  A = A₀ t_s d2/d1.

    Composes the transport factor (propagation) for the given interface —
    the Interface object itself carries no optics."""
    from ..propagation.transport import stigmatic_pupil
    return amplitude * stigmatic_pupil(interface, g)


def energy_through_sphere(A, th):
    """Total power through S2 (up to the constant n2/η):  2π ∫|A|² sinθ dθ."""
    return 2 * np.pi * np.trapezoid(np.abs(A) ** 2 * np.sin(th), th)
