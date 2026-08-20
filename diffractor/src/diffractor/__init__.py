"""Diffractor — scalar wave optics in which the interface is taken seriously.

The order of things:

    space (Grid)      samples of the plane, typed by their basis
    time (Spectrum)   the field's spectral content
    fields (Field)    samples of a function of space and time, in a Medium
    fourier (FT2)     one transform, evaluated per basis (FFT / Hankel modes)
    propagation       ASM, Fresnel, Fraunhofer — written once on FT2/IFT2
    optics            matter and its boundaries (Medium, Interface)
    scattering        the interface response (t_s, the exact planar operator)
    transport         geometric energy conservation (ray tubes, n·cosθ flux)
    analysis          measurements (OPL/OPD, pupils, energies)

Validation contract: nothing in the physical core ships without a benchmark
against `groundtruth` (BOR-BEM ← exact scalar solutions).  See PLAN.md.
"""

from .units import m, cm, mm, um, nm
from .basis import Basis, CARTESIAN, POLAR
from .space import Grid
from .spectrum import Spectrum
from .field import Field, MonochromaticField
from .fourier import FT2, IFT2, PolarPlan
from .propagation import (angular_spectrum, fresnel, fraunhofer,
                          rayleigh_sommerfeld, fresnel_validity_distance,
                          spectral_budget)
from .optics import Medium, Interface, stigmatic_interface

__all__ = [
    "m", "cm", "mm", "um", "nm",
    "Basis", "CARTESIAN", "POLAR", "Grid", "Spectrum",
    "Field", "MonochromaticField",
    "FT2", "IFT2", "PolarPlan",
    "angular_spectrum", "fresnel", "fraunhofer", "rayleigh_sommerfeld",
    "fresnel_validity_distance", "spectral_budget",
    "Medium", "Interface", "stigmatic_interface",
]
