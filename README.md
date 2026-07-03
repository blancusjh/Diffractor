# diffraction

A small, tested scalar-diffraction toolkit built on NumPy. It propagates
complex optical fields with Fourier-optics methods and models apertures and
thin refracting surfaces.

![Stigmatic Cartesian oval vs its osculating paraboloid at NA 0.24 — the incident beam crosses each refracting surface and propagates continuously to (and past) the focus; the oval focuses to a diffraction-limited point while the paraboloid smears into a spherical-aberration caustic.](docs/images/oval_vs_parabola_system.png)

*One continuous propagation per surface — incident beam, refractor, converging
cone, focus, and the diverging field beyond — from
[`examples/cartesian_vs_parabolic_aberration.py`](examples/cartesian_vs_parabolic_aberration.py):
the exact stigmatic Cartesian oval (left) against its paraboloid (center) at
NA 0.24, with the focal-plane PSFs below.*

A field is a first-class object — `Field(grid, values)`, the sampling `Grid`
plus its complex samples — that flows through the whole pipeline (aperture →
surface phase → propagator → plot). Coordinate queries live on the grid
(`field.grid.x`, `field.grid.spacing`) and Fourier transforms are *operators*
applied to a field (`FT2`, `IFT2`), not methods on it.

## Features

- **Field & Grid** (`diffraction.field`, `diffraction.grids`) — `Field`
  bundles a `Grid` with its samples; `Grid` owns coordinate access (`.x`,
  `.y`, `.spacing`) and still unpacks as `x, y = grid`.
- **Fourier operators** (`diffraction.fourier`) — `FFT2` / `IFFT2` (centered
  array FFTs on the native grid) and `FT2` / `IFT2` (field operators that also
  accept an explicit target grid and evaluate the transform there, via an exact
  matrix DFT or an optional chirp-Z fast path). Plus `frequency_grid`.
- **Propagators** (`diffraction.propagation`)
  - `fresnel_propagator` — single-FFT Fresnel (near-field, paraxial) method,
    with its scaled output plane given by `fresnel_output_grid`.
  - `fraunhofer_propagator` — far-field limit on the same output grid.
  - `fresnel_zoom_propagator` — the same Fresnel integral evaluated by a
    direct (matrix) Fourier transform on a user-chosen output window, for
    resolving focal spots and other compact features.
  - `asm_propagator` — angular-spectrum method (`IFT2(H · FT2(field))`), same
    grid in and out; supports back-propagation (`z < 0`), evanescent decay,
    and — via an explicit `output_grid` — decoupled output sampling (MPASM).
- **Batched / GPU propagation** (`diffraction.AngularSpectrum`) — a reusable
  angular-spectrum object that precomputes the transfer function and the input
  FFT once, so a z-sweep, movie or depth stack costs one inverse FFT per plane
  (`propagate`, `propagate_stack`, `intensity_stack`). Runs on the CPU (NumPy)
  or, with the `gpu` extra, on an NVIDIA GPU (CuPy) — chosen automatically from
  the input array or forced with `device="gpu"`.
- **Interactive viewers** (`diffraction.viz`, optional `viz` extra) — VisPy,
  GPU-rendered pan/zoom viewers for large fields: `plot_scalar_field` (2D),
  `plot_scalar_field_3d` (surface mesh) and `animate` (z-sweep movie with
  play/pause and frame-stepping).
- **Grid sizing** (`diffraction.sampling`) — `recommend_grid_convergence`
  picks an adequate near-field grid by convergence testing (the honest fix for
  input-side aliasing, which no output-grid trick can cure), plus the Fresnel
  sampling criteria `fresnel_min_distance` / `fresnel_max_spacing` and
  `next_fft_size`.
- **Polychromatic rendering** (`diffraction.polychromatic`,
  `diffraction.colorimetry`) — `propagate_polychromatic` propagates a
  broadband field wavelength-by-wavelength onto a shared output grid and
  composites the result to an sRGB image through the CIE 1931 color-matching
  functions (analytic Wyman fit — no data file), with D65 / blackbody
  illuminants and display controls (`saturation`, `stretch`, `brightness`) for
  vivid renders. `wavelength_to_rgb`, `spectrum_to_srgb`, `plot_rgb`.
- **Gratings** (`diffraction.gratings`) — amplitude (`ronchi_grating`,
  `sinusoidal_amplitude_grating`), 2D `cross_grating`, polar `polar_grating`
  (concentric rings × angular spokes), and chromatic `phase_grating`
  (sinusoidal / binary / blazed sawtooth). Orders land at `x_m = m λ z / d`.
- **Apertures** (`diffraction.apertures`) — circular, rectangular, square,
  annular, elliptical and slit masks, regular-polygon (`polygon_aperture`,
  e.g. a hexagon) and multi-slit (`nslit_aperture`, e.g. Young's double slit),
  all with adjustable centers; a `lattice_aperture` that tiles any of them on a
  square or hexagonal lattice (hole arrays); and an `antialiased` wrapper that
  evaluates any mask with area-coverage (grey) edge pixels.
- **Surfaces & lenses** (`diffraction.surfaces`) — `ParabolicSurface` and the
  stigmatic Cartesian-oval `CartesianSurface` (sag profile + thin-element
  `phase_mask`), plus an ideal `thin_lens` phase element that forms a field's
  Fraunhofer pattern at its back focal plane.
- **Sources** (`diffraction.fields`) — Gaussian beams and (tilted) plane waves.
- **Aberrations** (`diffraction.aberrations`) — Noll-indexed, RMS-normalized
  Zernike polynomials, least-squares wavefront fitting (`fit_zernikes`),
  synthesis, PV/RMS metrics and the Maréchal Strehl estimate.
- **Backends** (`diffraction.backend`) — CPU/GPU array-module resolution
  (`array_module`, `asnumpy`, `to_device`). The GPU path is `AngularSpectrum`
  (plus the `FFT2`/`IFFT2` operators), which runs unchanged on NumPy or CuPy;
  the one-shot functional propagators are NumPy-first.
- **Longitudinal fields** (`diffraction.longitudinal`) — `longitudinal_field`
  sweeps a field through a range of distances and slices a transverse line at
  each plane to build an `x–z` (or `y–z`) cross-section — a lens's focusing
  cone, a beam waist, or a grating's Talbot self-imaging carpet — drawn with
  `plot_longitudinal`. Pass `output_half_width`/`output_samples` to sample the
  line on a decoupled window (matrix-DFT), resolving e.g. a focal waist far
  finer than the input spacing without enlarging the input `N`.
- **Helpers** — `make_grid` for FFT-friendly sampling grids and
  `plot_intensity` for quick log-intensity figures.

All quantities are in SI units (meters); `wavelength` is always the vacuum
wavelength and `n` the refractive index of the propagation medium.

## Installation

```bash
git clone https://github.com/blancusjh/diffractor.git
cd diffractor
pip install -e .          # core (NumPy + matplotlib)
pip install -e ".[dev]"   # + pytest, to run the tests
pip install -e ".[viz]"   # + VisPy interactive viewers (diffraction.viz)
pip install -e ".[gpu]"   # + CuPy GPU backend (pick the wheel for your CUDA)
```

## Quick start

```python
import matplotlib.pyplot as plt
from diffraction import (
    make_grid, antialiased, circular_aperture,
    fresnel_propagator, plot_intensity,
)

grid = make_grid(2048, 6e-3)                       # 2048×2048 samples over 6 mm

U0 = antialiased(circular_aperture, grid, 0.3e-3)  # a Field on the grid
Uz = fresnel_propagator(U0, z=1.15, wavelength=532e-9)   # a Field on the scaled output grid

fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
plot_intensity(ax[0], U0, title="Aperture")
plot_intensity(ax[1], Uz, title="Fresnel pattern at z = 1.15 m")
plt.show()
```

### Fourier as an operator, on any grid

```python
from diffraction import FT2, IFT2, Grid
import numpy as np

spectrum = FT2(U0)                 # -> Field on the native frequency grid
back = IFT2(spectrum)              # -> Field, IFT2(FT2(U0)) == U0

# Sample the transform on a chosen (zoomed) frequency window instead:
fx = np.linspace(-3e4, 3e4, 256)
kx, ky = np.meshgrid(fx, fx)
zoomed = FT2(U0, kgrid=Grid(kx, ky))   # exact matrix DFT (or chirp-Z with SciPy)
```

### White light through a grating

```python
import numpy as np
from diffraction import (
    make_grid, ronchi_grating, square_aperture, Field,
    d65_weights, propagate_polychromatic, plot_rgb,
)

grid = make_grid(1024, 4e-3)
x, y = grid
grating = ronchi_grating(x, y, period=60e-6) * square_aperture(x, y, 1.2e-3)
U0 = Field(grid, grating.astype(complex))

wl = np.linspace(410e-9, 680e-9, 40)
rgb, out = propagate_polychromatic(
    U0, wl, z=0.5, weights=d65_weights(wl * 1e9), output_half_width=27e-3,
)   # -> a white 0th order flanked by dispersed rainbow orders
```

### Batched and GPU propagation

Build the propagator once, then sweep many planes cheaply — on the GPU when
CuPy is installed:

```python
import numpy as np
from diffraction import AngularSpectrum, antialiased, circular_aperture, make_grid

grid = make_grid(1024, 6e-3)
U0 = antialiased(circular_aperture, grid, 0.3e-3)

prop = AngularSpectrum(U0, wavelength=532e-9, pad_factor=2)   # device="gpu" to force CuPy
frames = prop.intensity_stack(np.linspace(5e-3, 0.12, 60))    # 60 planes, one IFFT each

from diffraction import animate            # needs the 'viz' extra
animate([np.log10(f + 1e-6) for f in frames], list(np.linspace(5e-3, 0.12, 60)))
```

### Propagating through a refracting surface

Surfaces act as thin phase elements: crossing an interface between media
`n1` and `n2` multiplies the field by `exp(i k0 (n1 - n2) z(x, y))`, where
the sag `z(x, y)` is measured toward the second medium (a ray at height
`(x, y)` covers that extra distance still inside `n1`).

```python
from diffraction import ParabolicSurface, asm_propagator

surface = ParabolicSurface(focal_length=0.12)
U_after = U0 * surface.phase_mask(grid, wavelength=532e-9, n1=1.0, n2=1.5)
Uz = asm_propagator(U_after, z=0.2, wavelength=532e-9, n=1.5)
```

## Examples

Runnable scripts live in [`examples/`](examples):

| Script | Demonstrates |
| --- | --- |
| `simple_fresnel_diffraction.py` | Fresnel pattern of a circular aperture |
| `simple_asm_diffraction.py` | Same setup with the angular-spectrum method |
| `fresnel_with_parabolic_surface_phase.py` | Focusing by a parabolic interface |
| `fresnel_with_cartesian_surface_phase.py` | Stigmatic Cartesian-oval surface, Fresnel propagation |
| `asm_with_cartesian_surface_phase.py` | Stigmatic Cartesian-oval surface, ASM propagation |
| `cartesian_vs_parabolic_aberration.py` | Very-high-NA stigmatic Cartesian oval vs its paraboloid: edge-on **system caustics** (incident beam, refracting surface drawn on the x–z map, converging cone, focus and diverging field — one continuous propagation) + 2-D focal spots in one figure, quantitative metrics (shape difference, OPD, through-focus, PSF, encircled energy) in a second (needs `scipy`) |
| `aberration_measurement.py` | Zernike decomposition of the exact OPD of both surfaces; Maréchal Strehl cross-check |
| `asm_zsweep_animation.py` | Batched `AngularSpectrum` z-sweep, GPU when available, animated with `diffraction.viz` (needs the `viz` extra to animate) |
| `gpu_focus_viewer.py` | Batched `AngularSpectrum` focus scan through a lens surface, GPU when available, viewed with `diffraction.viz`'s 2D/3D VisPy viewers (needs the `viz` extra to view) |
| `grid_decoupled_asm.py` | Resolving a focal spot with a decoupled `output_grid` (fine output sampling from a modest input `N`) |
| `adaptive_grid_selection.py` | `recommend_grid_convergence` picking an adequate near-field grid; before/after cross-hatch → clean rings |
| `polychromatic_aperture.py` | White-light (D65) circular aperture → chromatic diffraction rings |
| `diffraction_grating.py` | Monochromatic Ronchi grating far-field orders, positions verified against `m λ z / d` |
| `white_light_grating.py` | White light dispersed into a spectrum by a grating (the polychromatic + grating showcase) |
| `lattice_lens_diffraction.py` | Hexagonal hole array at a lens focal plane → reciprocal-lattice spots, plus a white-light version with each order dispersed |
| `hexagon_polychromatic.py` | Hexagonal aperture in white light — a faithful reproduction of diffractsim's flagship colored hexagonal Fresnel pattern |
| `double_slit_white_light.py` | Young's double slit and an N-slit grating in white light → colored fringes |
| `hexagon_lens_star.py` | White-light hexagon at a lens focus → a six-pointed colored star (iris starburst) |
| `grating_spectrometer.py` | Grating + lens spectrometer — a Ronchi grating giving symmetric focused spectra vs. a blazed grating steering the energy into one order |
| `cross_grating_lens.py` | Square (cross) grating at a lens focus → a centered lattice of orders at `(m λ f / d, n λ f / d)`, monochromatic and white-light (each order dispersed) |
| `polar_grating_lens.py` | Polar grating (rings × spokes) at a lens focus → a centered polar lattice of orders; the white-light ring orders disperse radially (blue in, red out) |
| `lens_longitudinal_focus.py` | The focusing cone of a lens seen edge-on (`longitudinal_field`) — converging cone, focal waist at `z = f`, diverging cone |
| `grating_talbot_carpet.py` | A grating's Talbot carpet — the `x–z` longitudinal near field showing self-images at `z_T = 2 d²/λ` and fractional revivals between |

```bash
python examples/simple_fresnel_diffraction.py
```

## Method notes

- **Single-FFT Fresnel.** The Fresnel integral is evaluated as one FFT with
  pre- and post-multiplication by quadratic phases. The output plane is
  rescaled: its coordinates are `x' = λ z fx`, so the observation window
  grows with distance (see `fresnel_output_grid`). The `dx·dy` factor turns
  the FFT into a Riemann-sum approximation of the continuous transform.
- **Angular spectrum.** The field is decomposed into plane waves,
  multiplied by the exact transfer function `exp(i kz z)` with
  `kz = 2π √(1/λₘ² − fx² − fy²)`, and recomposed. Frequencies beyond the
  propagating band are filtered out (or, with `include_evanescent=True`,
  kept with exponential decay).
- **Numerical hygiene.** Four artifact sources are handled explicitly:
  (1) the sampled ASM transfer function aliases at high frequencies for
  long distances — the Matsushima–Shimobaba band limit (`bandlimit=True`,
  the default) zeroes the undersampled band; (2) the FFT convolution is
  circular, so fields that diffract to the window edge wrap around —
  `pad_factor=2` zero-pads before propagating and crops afterwards;
  (3) hard-edged masks have pixelated edges whose spurious spectral
  content shows up as streaks in the far field — `antialiased(...)`
  evaluates masks with area-coverage edge pixels; (4) a hard aperture in
  the deep near field (high Fresnel number) develops boundary-wave ripples
  finer than a coarse `dx`, and the undersampled content folds back as a
  non-physical axis-aligned cross-hatch — this is an *input* sampling limit
  (`k_max = π/dx` is set by `dx` alone), so it is fixed only by a finer
  `dx`, which `recommend_grid_convergence` sizes for you.
- **Decoupled output grids.** An explicit `output_grid` (on `asm_propagator`,
  `AngularSpectrum` and `fresnel_zoom_propagator`, or an explicit `kgrid` on
  `FT2`/`IFT2`) samples the transform on a grid of your choice via a matrix
  DFT — resolving a focal spot at fine output sampling from a modest input
  `N`. It decouples *output* sampling from the input; it does **not** relax
  the input Nyquist requirement above, which no downstream transform can.
- **Polychromatic rendering.** A broadband field is propagated one wavelength
  at a time onto a **shared** output grid (via `fresnel_zoom_propagator` or
  `asm_propagator(..., output_grid=...)`), so every wavelength lands on the
  same physical screen and dispersive features appear at their true positions
  with no resampling. The per-wavelength intensities are integrated against
  the CIE 1931 color-matching functions to CIE XYZ and mapped to sRGB (with
  add-white gamut handling and the sRGB gamma).
- **Sampling.** For the single-FFT Fresnel method to be well sampled, the
  quadratic phase must vary slowly between samples: roughly
  `z ≳ N dx² / λ` (exposed as `fresnel_min_distance`). The ASM prefers the
  opposite regime (short distances); the two methods complement each other.

## Tests

The test suite checks the physics, not just the plumbing: Gaussian-beam
widths against the analytic beam-propagation law, the Airy first-zero
radius, energy conservation and back-propagation round trips for the ASM.

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

```
src/diffraction/
├── grids.py         # Grid class, make_grid, grid_spacing
├── field.py         # Field: grid + samples
├── fourier.py       # FFT2 / IFFT2 and the FT2 / IFT2 operators, frequency_grid
├── propagation.py   # fresnel, fraunhofer, angular-spectrum propagators
├── asm.py           # AngularSpectrum: batched CPU/GPU angular spectrum
├── sampling.py      # recommend_grid_convergence, Fresnel sampling criteria
├── backend.py       # CPU/GPU array-module resolution (NumPy / CuPy)
├── apertures.py     # aperture masks
├── gratings.py      # amplitude / phase / blazed / 2D diffraction gratings
├── fields.py        # gaussian_beam, plane_wave
├── surfaces.py      # ParabolicSurface, CartesianSurface, thin-element phase
├── aberrations.py   # Zernike fitting, PV/RMS, Maréchal Strehl
├── colorimetry.py   # CIE 1931 CMF, wavelength/spectrum -> sRGB
├── polychromatic.py # broadband propagation -> color image
├── longitudinal.py  # x-z / y-z axial field cross-sections (focus, Talbot)
├── plotting.py      # matplotlib intensity + RGB display helpers
└── viz.py           # optional VisPy interactive viewers + animation
examples/            # runnable demo scripts
tests/               # pytest suite
```
