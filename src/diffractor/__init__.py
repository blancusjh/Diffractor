"""diffractor — scalar diffraction toolkit.

Fourier-optics propagation (Fresnel, Fraunhofer, angular spectrum) of scalar
fields through apertures and thin refracting surfaces. All quantities are in
SI units (meters).

The package is organized into three subpackages — :mod:`diffractor.physics`
(fields, apertures, surfaces, propagators), :mod:`diffractor.mathutils` (grids,
Fourier operators, sampling, backends) and :mod:`diffractor.viz` (plotting,
viewers, colorimetry) — but every public name is also re-exported flat here.

Quick start
-----------
>>> from diffractor import MonochromaticField, make_grid, circular_aperture
>>> grid = make_grid(1024, 6e-3)
>>> U0 = MonochromaticField(grid, 1.0, wavelength=532e-9).add_aperture(
...     circular_aperture, 0.3e-3, antialiased=True)
>>> Uz = U0.propagate(1.15, method="fresnel")   # a MonochromaticField downstream
"""

# --- mathutils (no intra-package dependencies) ------------------------------
from .mathutils.backend import (
    CUPY_AVAILABLE,
    array_module,
    asnumpy,
    resolve_module,
    to_device,
)
from .mathutils.grids import Grid, grid_spacing, make_grid
from .mathutils.fourier import FFT2, FT2, IFFT2, IFT2, frequency_grid
from .mathutils.sampling import (
    GridRecommendation,
    fresnel_max_spacing,
    fresnel_min_distance,
    next_fft_size,
    recommend_grid_convergence,
)

# --- physics ----------------------------------------------------------------
from .physics.field import Field
from .physics.monochromatic import MonochromaticField
from .physics.sources import gaussian_beam, plane_wave
from .physics.apertures import (
    annular_aperture,
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
from .physics.gratings import (
    cross_grating,
    phase_grating,
    polar_grating,
    ronchi_grating,
    sinusoidal_amplitude_grating,
)
from .physics.surfaces import (
    CartesianSurface,
    ParabolicSurface,
    Surface,
    thin_element_phase,
    thin_lens,
)
from .physics.propagation import (
    asm_propagator,
    fraunhofer_propagator,
    fresnel_output_grid,
    fresnel_propagator,
    fresnel_zoom_propagator,
)
from .physics.asm import AngularSpectrum
from .physics.longitudinal import LongitudinalSection, longitudinal_field
from .physics.polychromatic import (
    PolychromaticField,
    PolychromaticImage,
    RGBLongitudinalSection,
    propagate_polychromatic,
    propagate_polychromatic_longitudinal,
)
from .physics.aberrations import (
    fit_zernikes,
    marechal_strehl,
    noll_to_nm,
    pv,
    rms,
    synthesize_zernikes,
    zernike,
    zernike_name,
)

# --- viz --------------------------------------------------------------------
from .viz.colorimetry import (
    blackbody_weights,
    cie_xyz,
    d65_weights,
    spectrum_to_srgb,
    wavelength_to_rgb,
    xyz_to_srgb,
)
from .viz.plotting import (
    intensity,
    plot_intensity,
    plot_longitudinal,
    plot_rgb,
    plot_rgb_longitudinal,
)
from .viz.viewers import animate, plot_scalar_field, plot_scalar_field_3d

__version__ = "0.7.0"

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
    "LongitudinalSection",
    "MonochromaticField",
    "ParabolicSurface",
    "PolychromaticField",
    "PolychromaticImage",
    "RGBLongitudinalSection",
    "Surface",
    "animate",
    "annular_aperture",
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
    "longitudinal_field",
    "make_grid",
    "marechal_strehl",
    "next_fft_size",
    "noll_to_nm",
    "nslit_aperture",
    "phase_grating",
    "plane_wave",
    "plot_intensity",
    "plot_longitudinal",
    "plot_rgb",
    "plot_rgb_longitudinal",
    "plot_scalar_field",
    "plot_scalar_field_3d",
    "polar_grating",
    "polygon_aperture",
    "propagate_polychromatic",
    "propagate_polychromatic_longitudinal",
    "pv",
    "recommend_grid_convergence",
    "rectangular_aperture",
    "resolve_module",
    "rms",
    "ronchi_grating",
    "sinusoidal_amplitude_grating",
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
