# diffraction

A small, tested scalar-diffraction toolkit built on NumPy. It propagates
complex optical fields with Fourier-optics methods and models apertures and
thin refracting surfaces.

## Features

- **Propagators** (`diffraction.propagation`)
  - `fresnel_propagator` — single-FFT Fresnel (near-field, paraxial) method,
    with its scaled output plane given by `fresnel_output_grid`.
  - `fraunhofer_propagator` — far-field limit on the same output grid.
  - `fresnel_zoom_propagator` — the same Fresnel integral evaluated by a
    direct (matrix) Fourier transform on a user-chosen output window, for
    resolving focal spots and other compact features.
  - `asm_propagator` — angular-spectrum method (exact scalar transfer
    function), same grid in and out, supports back-propagation (`z < 0`) and
    optional evanescent-wave decay.
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
- **Fourier core** (`diffraction.fourier`) — centered `FFT2` / `IFFT2` pairs
  and `frequency_grid`, so fields on symmetric grids transform without
  phase-shift bookkeeping.
- **Apertures** (`diffraction.apertures`) — circular, rectangular, square,
  annular, elliptical and slit masks, all with adjustable centers, plus an
  `antialiased` wrapper that evaluates any mask with area-coverage (grey)
  edge pixels.
- **Surfaces** (`diffraction.surfaces`) — `ParabolicSurface` and the
  stigmatic Cartesian-oval `CartesianSurface`, each providing a sag profile
  and a thin-element `phase_mask`.
- **Sources** (`diffraction.fields`) — Gaussian beams and (tilted) plane waves.
- **Aberrations** (`diffraction.aberrations`) — Noll-indexed, RMS-normalized
  Zernike polynomials, least-squares wavefront fitting (`fit_zernikes`),
  synthesis, PV/RMS metrics and the Maréchal Strehl estimate.
- **Backends** (`diffraction.backend`) — CPU/GPU array-module resolution
  (`array_module`, `asnumpy`, `to_device`) so the numerical core runs
  unchanged on NumPy or CuPy.
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
    make_grid, circular_aperture,
    fresnel_propagator, fresnel_output_grid, plot_intensity,
)

grid = make_grid(2048, 6e-3)            # 2048×2048 samples over 6 mm
x, y = grid

U0 = circular_aperture(x, y, 0.3e-3).astype(complex)
Uz = fresnel_propagator(U0, grid, z=1.15, wavelength=532e-9)
grid_out = fresnel_output_grid(grid, z=1.15, wavelength=532e-9)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
plot_intensity(ax[0], U0, grid, title="Aperture")
plot_intensity(ax[1], Uz, grid_out, title="Fresnel pattern at z = 1.15 m")
plt.show()
```

### Batched and GPU propagation

Build the propagator once, then sweep many planes cheaply — on the GPU when
CuPy is installed:

```python
import numpy as np
from diffraction import AngularSpectrum, circular_aperture, make_grid

grid = make_grid(1024, 6e-3)
x, y = grid
U0 = circular_aperture(x, y, 0.3e-3).astype(complex)

prop = AngularSpectrum(U0, grid, wavelength=532e-9, pad_factor=2)  # device="gpu" to force CuPy
frames = prop.intensity_stack(np.linspace(5e-3, 0.12, 60))         # 60 planes, one IFFT each

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
Uz = asm_propagator(U_after, grid, z=0.2, wavelength=532e-9, n=1.5)
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
| `cartesian_vs_parabolic_aberration.py` | Spherical aberration of the paraxial (parabolic) approximation vs the exact oval (needs `scipy`) |
| `aberration_measurement.py` | Zernike decomposition of the exact OPD of both surfaces; Maréchal Strehl cross-check |
| `asm_zsweep_animation.py` | Batched `AngularSpectrum` z-sweep, GPU when available, animated with `diffraction.viz` (needs the `viz` extra to animate) |
| `gpu_focus_viewer.py` | Batched `AngularSpectrum` focus scan through a lens surface, GPU when available, viewed with `diffraction.viz`'s 2D/3D VisPy viewers (needs the `viz` extra to view) |

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
- **Numerical hygiene.** Three artifact sources are handled explicitly:
  (1) the sampled ASM transfer function aliases at high frequencies for
  long distances — the Matsushima–Shimobaba band limit (`bandlimit=True`,
  the default) zeroes the undersampled band; (2) the FFT convolution is
  circular, so fields that diffract to the window edge wrap around —
  `pad_factor=2` zero-pads before propagating and crops afterwards;
  (3) hard-edged masks have pixelated edges whose spurious spectral
  content shows up as streaks in the far field — `antialiased(...)`
  evaluates masks with area-coverage edge pixels.
- **Sampling.** For the single-FFT Fresnel method to be well sampled, the
  quadratic phase must vary slowly between samples: roughly
  `z ≳ N dx² / λ`. The ASM prefers the opposite regime (short distances);
  the two methods complement each other.

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
├── fourier.py       # centered FFT2 / IFFT2, frequency_grid
├── grids.py         # make_grid, grid_spacing
├── propagation.py   # fresnel, fraunhofer, angular-spectrum propagators
├── asm.py           # AngularSpectrum: batched CPU/GPU angular spectrum
├── backend.py       # CPU/GPU array-module resolution (NumPy / CuPy)
├── apertures.py     # aperture masks
├── fields.py        # gaussian_beam, plane_wave
├── surfaces.py      # ParabolicSurface, CartesianSurface, thin-element phase
├── aberrations.py   # Zernike fitting, PV/RMS, Maréchal Strehl
├── plotting.py      # matplotlib intensity display helpers
└── viz.py           # optional VisPy interactive viewers + animation
examples/            # runnable demo scripts
tests/               # pytest suite
```
