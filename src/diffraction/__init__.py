"""diffraction — scalar diffraction toolkit.

Fourier-optics propagation (Fresnel, Fraunhofer, angular spectrum) of scalar
fields through apertures and thin refracting surfaces. All quantities are in
SI units (meters).

Quick start
-----------
>>> from diffraction import make_grid, circular_aperture, fresnel_propagator
>>> grid = make_grid(1024, 6e-3)
>>> x, y = grid
>>> U0 = circular_aperture(x, y, 0.3e-3).astype(complex)
>>> Uz = fresnel_propagator(U0, grid, z=1.15, wavelength=532e-9)
"""

from .aberrations import (
    fit_zernikes,
    marechal_strehl,
    noll_to_nm,
    pv,
    rms,
    synthesize_zernikes,
    zernike,
    zernike_name,
)
from .apertures import (
    annular_aperture,
    antialiased,
    circular_aperture,
    elliptical_aperture,
    rectangular_aperture,
    slit_aperture,
    square_aperture,
)
from .fields import gaussian_beam, plane_wave
from .fourier import FFT2, IFFT2, frequency_grid
from .grids import Grid, grid_spacing, make_grid
from .plotting import intensity, plot_intensity
from .propagation import (
    asm_propagator,
    fraunhofer_propagator,
    fresnel_output_grid,
    fresnel_propagator,
    fresnel_zoom_propagator,
)
from .surfaces import CartesianSurface, ParabolicSurface, Surface, thin_element_phase

__version__ = "0.2.0"

__all__ = [
    "FFT2",
    "IFFT2",
    "CartesianSurface",
    "Grid",
    "ParabolicSurface",
    "Surface",
    "annular_aperture",
    "antialiased",
    "asm_propagator",
    "circular_aperture",
    "elliptical_aperture",
    "fit_zernikes",
    "fraunhofer_propagator",
    "frequency_grid",
    "fresnel_output_grid",
    "fresnel_propagator",
    "fresnel_zoom_propagator",
    "gaussian_beam",
    "grid_spacing",
    "intensity",
    "make_grid",
    "marechal_strehl",
    "noll_to_nm",
    "plane_wave",
    "plot_intensity",
    "pv",
    "rectangular_aperture",
    "rms",
    "slit_aperture",
    "square_aperture",
    "synthesize_zernikes",
    "thin_element_phase",
    "zernike",
    "zernike_name",
]
