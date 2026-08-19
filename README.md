# Diffractor

**Scalar wave optics in which the interface is taken seriously.**

Most diffraction toolkits model a lens as a phase mask: multiply the field by
`exp(i k Δ(x,y))` and propagate. That is a boundary condition nobody derived,
and it silently violates energy conservation the moment the surface is steep.
Diffractor is built the other way round — geometry, media, interface response
and transport are separate things, each one measured against something that
cannot be argued with:

```
    exact analytic     →      groundtruth (BOR-BEM)      →      diffractor
   scalar ball, planar        Müller second-kind solver,        exact propagators
   interface, R+T = 1         O(h²) convergence                 + interface operators
```

Nothing enters the physical core without a reference to check it against, and
when the reference turns out to be wrong the claim is retracted rather than
kept (see [Status](#status-the-validation-ladder) — that has already happened
once, and the retraction is in the repository).

![A stigmatic Cartesian ovoid against a sphere of the same vertex curvature: the ovoid focuses to one wavelength-sized point, the sphere spreads the same light over 130 wavelengths of axis.](docs/images/stigmatic_vs_spherical.png)

*The whole point of the library in one figure: two surfaces of the **same
vertex curvature**, so the same paraxial focus, differing only in how they bend
the outer rays. The Cartesian ovoid is stigmatic by construction — its measured
wavefront error is 10⁻¹² waves at any aperture — while the sphere loses 9.4
waves peak-to-valley and smears its focus over 130 λ. From
[`examples/04_stigmatic_vs_spherical.py`](examples/04_stigmatic_vs_spherical.py).*

---

## Contents

- [Install](#install) · [60 seconds](#60-seconds)
- [Quickstarts](#quickstarts) — six short, complete programs
- [Gallery](#gallery) — diffraction, gratings, white light, aberration
- [Status: the validation ladder](#status-the-validation-ladder)
- [What is in the box](#what-is-in-the-box) · [Conventions and traps](#conventions-and-traps)
- [Notebooks](#notebooks) · [Tests](#tests) · [Repository layout](#repository-layout)

---

## Install

Python ≥ 3.10, NumPy and SciPy. From the repository root:

```bash
git clone https://github.com/blancusjh/Diffractor.git
cd Diffractor
pip install -e diffractor -e groundtruth     # two packages, one workspace
pip install pytest matplotlib                # tests, and the example figures
python -m pytest tests/                      # 25 assertions, ~8 s
```

`groundtruth` is only needed to *check* `diffractor`; the physical core never
imports it.

## 60 seconds

```python
import numpy as np
from diffractor.propagation import asm_axisym

lam, n, a, z = 0.5, 1.0, 25.0, 300.0        # µm — any consistent length unit
r = np.linspace(0.0, a, 2001)               # the aperture IS the domain: a hard
U = np.ones_like(r, dtype=complex)          # edge with no sampling error at all
r_out = np.linspace(0.0, 60.0, 400)

Uz, rho, spectrum = asm_axisym(U, r, z, n, lam, r_out, n_rho=20000)

k = 2 * np.pi * n / lam                     # the closed form on the axis
exact = np.exp(1j*k*z) - z/np.hypot(a, z) * np.exp(1j*k*np.hypot(a, z))
print(abs(Uz[0]), abs(exact))               # 0.4937   0.4949
```

That is the whole style of the library: exact operators, and a closed form next
to them whenever one exists.

---

## Quickstarts

Each block runs on its own (after the imports of the one before it).

### 1 · Diffract from a plane, exactly

`propagation.exact` solves Helmholtz in a homogeneous half-space with no
further hypothesis. Two independent operators, one physics:

```python
from diffractor.propagation import asm_axisym, rs1_plane

U_asm, rho, spectrum = asm_axisym(U, r, 300.0, n, lam, r_out, n_rho=20000)
U_rs1 = rs1_plane(U, r, 300.0, n, lam, r_out)     # Rayleigh–Sommerfeld I
# they agree to 7e-3 of the peak here; RS1 reproduces the closed-form axial
# value to six digits, and the Airy pattern in the far field to 3e-5
```

`asm_axisym` transforms with a Hankel pair, multiplies by `exp(i kz z)` and
transforms back, so it is fast and near-field friendly; `rs1_plane` integrates
in real space, so it stays accurate at any distance. Pick by regime, not by
taste — [`examples/01_apertures.py`](examples/01_apertures.py) shows where each
one is comfortable.

### 2 · A stigmatic interface, and its measured stigmatism

An `Interface` is a locus plus two media — nothing else. What light *does*
there lives in `scattering`, `propagation` and `analysis`:

```python
import numpy as np
from diffractor.optics import Medium, stigmatic_interface
from diffractor.analysis import opd_waves

iface = stigmatic_interface(Medium(1.0), Medium(1.5), zo=-200.0, zi=100.0)
g = iface.sample(n_pts=2001, rim_frac=0.98)       # vertex → rim, pure geometry

print(g["r"][-1], np.degrees(g["th2"][-1]))       # 40.5 µm, 38.2° image-side
print(np.abs(g["snell"]).max())                   # 1.6e-15  Snell, as a residual
print(np.abs(opd_waves(iface, g, lam)).max())     # 5.7e-13 waves of wavefront error
```

Both numbers are *measurements*, not assumptions: `opd_waves` rebuilds `d₁` and
`d₂` from the `(r, z)` coordinates and compares the two-leg path with the axial
one. Stigmatism is what the residual says, never an input.

### 3 · The pupil: a special case, and the general law

```python
from diffractor.propagation import ray_tube_amplitude, stigmatic_pupil
from diffractor.scattering import T_s
from diffractor.analysis import energy_through_sphere, predicted_pupil

P     = stigmatic_pupil(iface, g)                  # P = t_s · d₂/d₁
P_gen = ray_tube_amplitude(1.0, 1.5, g["w1"], g["w2"],
                           T_s(1.0, 1.5, g["cos_i1"], g["cos_i2"]))
print(np.abs(P[1:]/P_gen[1:] - 1).max())           # 6.7e-16 — the same thing

A = predicted_pupil(iface, g)                      # with the source amplitude
print(energy_through_sphere(A, g["th2"]))          # 2π∫|A|² sinθ dθ
```

`ray_tube_amplitude` is the general statement — `n₁|U₁|²dΩ₁·T = n₂|U₂|²dΩ₂` —
and holds for *any* surface, aberrated or not, because it never assumes the
outgoing bundle is homocentric. `stigmatic_pupil` is the collapse of that law
when both legs are exact spherical waves. Machine agreement between them is a
standing test of both.

### 4 · A flat interface, with no approximation at all

For a plane boundary the transmission operator is diagonal in the angular
spectrum, so it can be applied exactly:

```python
from diffractor.scattering import transmit_axisym, t_spectral

U2, rho, t = transmit_axisym(U, r, 1.0, 1.5, lam)
print(t[0].real, 2*1.0/(1.0+1.5))                  # 0.8  0.8   (normal incidence)
```

This is rung 1 of the validation ladder, and the primitive every curved-
interface scheme has to reduce to in the flat limit.

### 5 · Paraxial, only where it is legal

The Fresnel propagator is a good tool inside its regime and a wrong answer
outside it, so the paraxial module refuses to be used outside it:

```python
from diffractor.propagation.paraxial import fresnel_plane, fresnel_validity_distance

print(fresnel_validity_distance(a, lam))           # 254.9 µm — 3× Goodman's z
fresnel_plane(U, r, 100.0, n, lam, r_out)          # ValueError: outside validity
fresnel_plane(U, r, 100.0, n, lam, r_out, policy="warn")    # warns and proceeds
U_far = fresnel_plane(U, r, 4e5, n, lam, np.linspace(0, 4e4, 200))   # fine
```

`policy` is `"raise"` (default), `"warn"` or `"force"`.

### 6 · How much of your field is not going to propagate

```python
from diffractor.propagation import spectral_budget

frac, rho, spectrum = spectral_budget(U, r, n, lam)
print(frac)             # 1.1e-3 of the power lies beyond ρ = n/λ
```

The fraction beyond the propagating cone is the energy a band-limited angular
spectrum silently discards — the reason a thin-element boundary condition
leaks. Measure it before trusting a result near a sharp feature.

---

## Gallery

Every figure below is produced by a script in [`examples/`](examples); the
numbers quoted in the captions are printed by those scripts when they run.

### Apertures — and what the answer should be

![Circular and annular apertures: meridional maps of the diffracted field, the on-axis intensity against its closed form, and the far field against the Airy pattern.](docs/images/apertures.png)

A hard circular aperture and an annulus, propagated exactly. The on-axis
intensity swings between 0 and 4× the incident value as the Fresnel number
`a²/λz` runs through the integers, and it lands on the closed form
`|e^{ikz} − (z/R)e^{ikR}|` to 2×10⁻³; the far field lands on `[2J₁(x)/x]²` to
7×10⁻⁴; the two exact propagators agree with each other to 7×10⁻⁴.
The annulus — same area, moved to the rim — narrows the core and pumps the
rings, which is the whole idea behind apodizing masks.
→ [`01_apertures.py`](examples/01_apertures.py)

### Gratings that are rings, and a lens made of them

![Ring gratings diffracting into orders at sinθ = mλ/d, and a Fresnel zone plate focusing at f, f/3 and f/5.](docs/images/gratings_zoneplate.png)

A sinusoidal ring grating carries only `m = 0, ±1`; a binary one carries every
odd order, at the angles the grating equation predicts. Chirp the same binary
screen so every clear zone arrives in phase — `r_m = √(mλf)` — and it becomes a
lens: the Fresnel zone plate, focusing to the Airy pattern of its own aperture
while keeping the focal series `f`, `f/3`, `f/5` of the orders it came from.
The measured peak sits 2 % inside the geometric `f`: the focal shift of a
finite-aperture beam, not an error.
→ [`02_gratings_zoneplate.py`](examples/02_gratings_zoneplate.py)

### White light is just many monochromatic solves

![A circular aperture and a zone plate in white light: coloured Airy rings, the same pattern wavelength by wavelength, and a zone plate spreading its focus across 60 % of its focal length.](docs/images/polychromatic.png)

Wavelength is an argument to a propagator here, not a state of the field, so
broadband is bookkeeping: solve at each λ on a shared output grid, then
integrate against the CIE 1931 colour-matching functions
([`examples/colorimetry.py`](examples/colorimetry.py), the analytic Wyman fit —
no data files). A diffraction pattern scales as λ, so the aperture's rings fan
out into spectra; a zone plate's focal length goes as `1/λ`, so white light is
not focused at all but smeared over 59 % of `f` — diffractive optics has
chromatic aberration of the opposite sign, and a hundred times worse, than a
glass singlet. The measured foci follow `f₀λ₀/λ` to 5 µm across 430–680 nm.
→ [`03_polychromatic.py`](examples/03_polychromatic.py)

### A hexagon in white light — the six-pointed star

![A hexagonal aperture in white light: the near-field hexagon with fringes, the pattern dissolving into rays, and the far-field six-pointed star in monochromatic and true colour, with one spike resolved wavelength by wavelength.](docs/images/hexagon_polychromatic.png)

The one figure the core cannot produce by itself — a hexagon is not
axisymmetric — so it uses the 2-D Fresnel integral in
[`examples/cartesian.py`](examples/cartesian.py), which reproduces the core's
exact Rayleigh–Sommerfeld result for a circular aperture to 10⁻⁵ of the peak.
Near the aperture the pattern is still a hexagon with fringes and almost no
colour: at a Fresnel number of 6 every wavelength does much the same thing.
Far from it the six edges throw six spikes **perpendicular to themselves** —
the measured directions land within 0.2° of the edge normals, and the far field
falls as r⁻² along a normal against r⁻³·⁶ towards a corner, which is *why*
there is a star: a straight edge radiates along its own normal, and away from
the normals only the corners are left. Each spike is a channelled spectrum, its
fringes sitting at radii ∝ λ. This is what a camera iris does to a bright
light.
→ [`05_hexagon_polychromatic.py`](examples/05_hexagon_polychromatic.py)

### Stigmatic against spherical

The figure at the top of this page, in numbers: at an 85 λ aperture radius
(NA 0.34) the sphere holds 29 % of the ovoid's peak intensity, and only if you
move the screen 52 λ; at the paraxial image itself, 5 %. Sweeping the aperture
shows the Strehl ratio tracking Maréchal's `exp[−(2πσ)²]` while the error is
small and leaving it behind once it is not — the classical estimate measured,
not quoted.
→ [`04_stigmatic_vs_spherical.py`](examples/04_stigmatic_vs_spherical.py)

| Script | Runtime | What it computes |
| --- | --- | --- |
| [`01_apertures.py`](examples/01_apertures.py) | ~60 s | circular and annular apertures; ASM vs RS1 vs closed form vs Airy |
| [`02_gratings_zoneplate.py`](examples/02_gratings_zoneplate.py) | ~20 s | ring gratings, the grating equation, and the zone plate's focal series |
| [`03_polychromatic.py`](examples/03_polychromatic.py) | ~90 s | 31 monochromatic solves composited to sRGB; chromatic focal spread |
| [`04_stigmatic_vs_spherical.py`](examples/04_stigmatic_vs_spherical.py) | ~10 s | exact ray trace, ray-tube pupils, Debye focal fields, Strehl sweep |
| [`05_hexagon_polychromatic.py`](examples/05_hexagon_polychromatic.py) | ~15 s | a hexagon from near field to six-pointed star, in white light |

Three support modules sit beside them: `style.py` (the shared plot style),
`colorimetry.py` (CIE 1931 → sRGB) and `cartesian.py` (the 2-D Fresnel integral,
with `python examples/cartesian.py` running its own check against the core).

```bash
python examples/01_apertures.py     # from anywhere; figures land in docs/images/
```

---

## Status: the validation ladder

The interface operator is built one rung at a time; a rung is not climbed until
the one below it passes ([`benchmarks/cases.md`](benchmarks/cases.md)).

| # | Case | Reference | Status |
|---|------|-----------|--------|
| 1 | Planar interface, oblique plane wave | exact spectral `t(k⊥)` | primitive implemented (`scattering/planar.py`), ladder test pending |
| 2 | Planar interface, point source | RS1 + `t(k⊥)` | pending |
| 3 | Shallow spherical cap | BEM + exact scalar ball | pending |
| 4 | Stigmatic ovoid, NA 0.3–0.75 | `P = t_s·d₂/d₁` + BEM | **blocked** — see below |
| 5 | Full sphere, caustic regime | exact scalar ball | pending |

The solver itself is checked and works: Müller BOR-BEM against the exact series
for a homogeneous ball, `O(h²)`, agreeing to ~10⁻³ over the usable range of
`k₂a`.

![Müller BOR-BEM checked against the exact scalar solution for a dielectric ball: interior field maps, axial and transverse cuts, and the error against electrical size.](benchmarks/f4_methods.png)

Rung 4 is blocked, and the reason is worth reading before trusting any
benchmark: closing the ovoid so the boundary-integral problem is well posed
turns it into a **lossless dielectric cavity**. Varying only the length of the
closing shroud — a parameter downstream of the converging cone, which cannot
affect the physics — swings the measured edge factor by ±38 %. So the earlier
measurement that appeared to confirm `P = t_s·d₂/d₁` to 0.25 % is **retracted**:
it was contaminated by a reverberant field. Mesh convergence had been verified
and was excellent; it converged beautifully to the answer for the wrong
geometry. Integrated quantities survive averaging (energy through S₂ is stable
at 1.20 ± 0.01 × prediction), the pointwise edge factor does not. The fix is an
open-surface formulation — which is also the physically right model of a lens
with a rim.

![Meridional sections of the stigmatic ovoid from the boundary-integral solver, lossless and absorbing, against the tangent-plane model on the cap alone.](benchmarks/f1_longitudinal.png)

*The same body solved three ways. Middle: an absorbing interior damps the
cavity while leaving the direct converging beam intact — which is how the
trapped field was diagnosed. Bottom: the tangent-plane model on the cap alone,
the thing `diffractor` actually implements.*

The focal region and the field on the two reference spheres — the measurement
protocol itself — are in [`benchmarks/f2_focal.png`](benchmarks/f2_focal.png)
and [`benchmarks/f3_spheres.png`](benchmarks/f3_spheres.png).
[`PLAN.md`](PLAN.md) has the phase roadmap (in Spanish).

---

## What is in the box

`diffractor` is ordered by *the nature of each thing*, not by convenience:

| Module | Holds | Key names |
| --- | --- | --- |
| `geometry/` | pure shape — no media, no fields | `DescartesOvoid` (exact, closed-form, parametrised by `d₁`) |
| `optics/` | matter and where it ends | `Medium`, `Interface`, `stigmatic_interface` |
| `scattering/` | the interface response | `t_s`, `r_s`, `T_s`, `R_s`, `snell_cos`, `point_source_transmission`, `t_spectral`, `transmit_axisym` |
| `propagation/` | transport through space | `hankel`, `asm_axisym`, `rs1_plane`, `spectral_budget`, `paraxial.fresnel_plane` (gated), `ray_tube_amplitude`, `stigmatic_pupil` |
| `sources/` | what emits | `point_source`, `point_source_normal_derivative` |
| `analysis/` | what is measured | `two_leg_opl`, `invariant_opl`, `opd_waves`, `demodulate`, `predicted_pupil`, `energy_through_sphere` |

`groundtruth` holds the references: `exact/` (closed-form scalar ball),
`bem/` (Müller BOR-BEM, field maps, bodies) and `protocol/` (demodulation on
the reference sphere, energy budget). It is never imported by `diffractor`.

## Conventions and traps

- **Units.** Any consistent length unit — the examples use λ = 1 or
  micrometres. `lam` is always the *vacuum* wavelength and `n` the index of
  the medium, so the medium wavelength is `lam/n` wherever it matters.
- **Everything is axisymmetric.** Every propagator is a Hankel/BOR operator on
  a radial grid `r`. There are no 2-D Cartesian grids in the core yet — which
  is why the gallery uses ring gratings and zone plates rather than line ones.
  The one 2-D figure, the hexagon, is computed by
  [`examples/cartesian.py`](examples/cartesian.py): a separable-matrix-DFT
  Fresnel integral that deliberately lives in `examples/` rather than the core,
  because the core's contract is that nothing enters it without a rung on the
  validation ladder. What it has instead is `check_against_core()`, which puts
  it beside the exact axisymmetric propagator on a circular aperture (10⁻⁵ of
  the peak). Promoting it is the natural first half of the Cartesian FFT
  version `scattering/planar.py` already promises.
- **The aperture is the domain.** Radial grids run `0 → a`, so a hard edge is
  exact and costs no sampling error; there is no mask array to alias.
- **`propagation.fresnel_plane` is the raw kernel.** The *gated* one — the one
  that refuses to run outside the paraxial regime — is
  `propagation.paraxial.fresnel_plane`. Import from `paraxial` unless you
  deliberately want the ungated kernel.
- **`n_rho` is yours to choose, and it is not always enough.** The default ρ
  sampling is set by the aperture; long propagations need more, because
  `exp(i kz z)` oscillates ever faster in ρ as `z` grows. Converge it — double
  `n_rho` and see whether the answer moved. Past that there is a floor: at the
  edge of the propagating cone `kz → 0` and the phase turns infinitely fast in
  ρ, so for a hard-edged aperture (whose spectrum decays only as ρ^−3/2) the
  angular spectrum keeps a slowly growing quadrature error — a few times 10⁻³
  of the peak at several hundred λ, which more `n_rho` does not remove.
  `rs1_plane` integrates in real space and has no such limit: it reproduces the
  closed-form axial value to six digits at any distance. Near field → ASM;
  far field → RS1.
- **`t_s` is the scalar/TE coefficient, and only that.** It is what the scalar
  Helmholtz transmission conditions give. Do not average it with `t_p`: that
  belongs to Maxwell, not to this framework.
- **Amplitudes are absolute.** The point source carries `A₀ = 1/4π` and its
  `1/d`; pupils and energies are never rescaled to fit a reference.

## Notebooks

Four notebooks, committed with their outputs, in the order the physics is built
up — see [`notebooks/README.md`](notebooks/README.md):

1. `01_stigmatic_geometry.ipynb` — the Cartesian oval; stigmatism, Snell and the
   ray tubes verified from the coordinates.
2. `02_solver_validation.ipynb` — Müller BOR-BEM against the exact scalar ball.
3. `03_stigmatic_focus.ipynb` — the field of the ovoid: sections, focal region,
   reference spheres.
4. `04_models_against_the_solver.ipynb` — tangent-plane, Debye–Wolf and
   thin-element models, scored.

```bash
pip install -e diffractor -e groundtruth
jupyter lab notebooks/
```

## Tests

The suite asserts physics, not plumbing: Snell as a residual, the invariant
OPL, `R_s + T_s = 1`, the ray-tube identity against `t_s·d₂/d₁`, RS1 against the
angular spectrum, and the Fresnel gate refusing what it should refuse.

```bash
python -m pytest tests/          # 25 passed
```

## Repository layout

```
diffractor/          the product — geometry, optics, scattering, propagation,
                     sources, analysis (installable package, src/ layout)
groundtruth/         the references — exact/, bem/, protocol/ (never imported
                     by diffractor)
benchmarks/          frozen results in golden/*.npz + the figures that read them
                     (compute_fields.py ~15 min, fig_fields.py seconds)
examples/            the gallery above; each script prints its own checks
notebooks/           the physics, built up in order, with outputs committed
tests/               every claim from the validation session, as assertions
docs/images/         figures produced by examples/
PLAN.md              phases, the validation ladder, and the recorded failures
```
