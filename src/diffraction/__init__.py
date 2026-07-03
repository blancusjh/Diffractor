"""diffraction — scalar diffraction toolkit.

Fourier-optics propagation (Fresnel, Fraunhofer, angular spectrum) of scalar
fields through apertures and thin refracting surfaces. All quantities are in
SI units (meters).

Quick start
-----------
>>> from diffraction import make_grid, antialiased, circular_aperture, fresnel_propagator
>>> grid = make_grid(1024, 6e-3)
>>> U0 = antialiased(circular_aperture, grid, 0.3e-3)   # a Field on the grid
>>> Uz = fresnel_propagator(U0, z=1.15, wavelength=532e-9)   # a Field on the output grid
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
    lattice_aperture,
    lattice_sites,
    nslit_aperture,
    polygon_aperture,
    rectangular_aperture,
    slit_aperture,
    square_aperture,
)
from .asm import AngularSpectrum
from .backend import CUPY_AVAILABLE, array_module, asnumpy, to_device
from .colorimetry import (
    blackbody_weights,
    cie_xyz,
    d65_weights,
    spectrum_to_srgb,
    wavelength_to_rgb,
    xyz_to_srgb,
)
from .field import Field
from .fields import gaussian_beam, plane_wave
from .fourier import FFT2, FT2, IFFT2, IFT2, frequency_grid
from .gratings import (
    cross_grating,
    phase_grating,
    ronchi_grating,
    sinusoidal_amplitude_grating,
)
from .grids import Grid, grid_spacing, make_grid
from .plotting import intensity, plot_intensity, plot_rgb
from .polychromatic import propagate_polychromatic
from .propagation import (
    asm_propagator,
    fraunhofer_propagator,
    fresnel_output_grid,
    fresnel_propagator,
    fresnel_zoom_propagator,
)
from .sampling import (
    GridRecommendation,
    fresnel_max_spacing,
    fresnel_min_distance,
    next_fft_size,
    recommend_grid_convergence,
)
from .surfaces import (
    CartesianSurface,
    ParabolicSurface,
    Surface,
    thin_element_phase,
    thin_lens,
)
from .viz import animate, plot_scalar_field, plot_scalar_field_3d

__version__ = "0.6.0"

__all__ = [
    "CUPY_AVAILABLE",
    "FFT2",
    "FT2",
    "IFFT2",
    "IFT2",
    "AngularSpectrum",
    "CartesianSurface",
    "Field",
    "Grid",
    "GridRecommendation",
    "ParabolicSurface",
    "Surface",
    "animate",
    "annular_aperture",
    "antialiased",
    "array_module",
    "asm_propagator",
    "asnumpy",
    "blackbody_weights",
    "cie_xyz",
    "circular_aperture",
    "cross_grating",
    "d65_weights",
    "elliptical_aperture",
    "fit_zernikes",
    "fraunhofer_propagator",
    "frequency_grid",
    "fresnel_max_spacing",
    "fresnel_min_distance",
    "fresnel_output_grid",
    "fresnel_propagator",
    "fresnel_zoom_propagator",
    "gaussian_beam",
    "grid_spacing",
    "intensity",
    "lattice_aperture",
    "lattice_sites",
    "make_grid",
    "marechal_strehl",
    "next_fft_size",
    "noll_to_nm",
    "nslit_aperture",
    "phase_grating",
    "plane_wave",
    "plot_intensity",
    "plot_rgb",
    "plot_scalar_field",
    "plot_scalar_field_3d",
    "polygon_aperture",
    "propagate_polychromatic",
    "pv",
    "recommend_grid_convergence",
    "rectangular_aperture",
    "rms",
    "ronchi_grating",
    "sinusoidal_amplitude_grating",
    "spectrum_to_srgb",
    "square_aperture",
    "synthesize_zernikes",
    "thin_element_phase",
    "thin_lens",
    "to_device",
    "wavelength_to_rgb",
    "xyz_to_srgb",
    "zernike",
    "zernike_name",
]
