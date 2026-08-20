"""Claims from the validation session, as executable assertions.

Each test here guards something that took work to get right and could
plausibly break again.  Nothing tests that the language works.
"""

import numpy as np
import pytest

from diffractor import Medium, stigmatic_interface
from diffractor.analysis.opl import opd_waves
from diffractor.analysis.pupil import demodulate, energy_through_sphere, predicted_pupil
from diffractor.geometry import DescartesOvoid
from diffractor import Grid, MonochromaticField
from diffractor.propagation import angular_spectrum, fresnel, rayleigh_sommerfeld
from diffractor.transport import (ray_tube_amplitude,
                                              stigmatic_pupil,
                                              stigmatic_pupil_from_tubes)
from diffractor.scattering.fresnel import R_s, T_s, t_s
from diffractor.scattering.planar import t_spectral, transmit
from diffractor.sources import point_source, point_source_normal_derivative

from groundtruth.exact import ScalarBall

LAM = 530e-9


# ── the ovoid parametrisation ─────────────────────────────────────────────────
class TestDescartesOvoid:
    """The closed-form d1 parametrisation, against what it replaced."""

    ov = DescartesOvoid(1.0, 1.5, -0.050, 0.025)
    g = ov.sample(20001, cluster=2.0)

    def test_snell(self):
        # non-trivial: Snell is a consequence of the parametrisation, and the
        # cross-product form of sin(i) is what keeps it exact at grazing
        assert np.abs(self.g["snell"]).max() < 1e-13

    def test_rim_is_grazing(self):
        # at the true rim the incident ray goes tangent -> i1 = 90, T -> 0.
        # Pins d1_rim = sqrt(c/a).
        assert self.g["cos_i1"][-1] == pytest.approx(0.0, abs=1e-7)

    def test_reaches_full_na(self):
        # the regression that motivated the rewrite: Newton-in-z at fixed r
        # dies at the double root and caps this at 0.37
        assert 1.5 * self.g["sin2"][-1] > 0.94


def test_stigmatism_is_observed_not_assumed():
    """OPD rebuilt from the (r, z) coordinates, independent of the
    equal-path condition used to generate them."""
    si = stigmatic_interface(Medium(1.0), Medium(1.5), zo=-0.050, zi=0.025)
    assert np.abs(opd_waves(si, si.sample(20001), LAM)).max() < 1e-8


# ── scalar interface response ─────────────────────────────────────────────────
def test_fresnel_conserves_energy():
    c1 = np.cos(np.radians(np.linspace(0, 89.9, 200)))
    assert np.abs(R_s(1.0, 1.5, c1) + T_s(1.0, 1.5, c1) - 1).max() < 1e-12


@pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.0, 2.5), (1.5, 1.0),
                                    (1.33, 1.52)])
def test_fresnel_energy_conservation_parametrized(n1, n2):
    """R + T = 1 for a range of refractive-index pairs."""
    c1 = np.cos(np.radians(np.linspace(0, 89.9, 200)))
    residual = np.abs(R_s(n1, n2, c1) + T_s(n1, n2, c1) - 1).max()
    assert residual < 1e-12


def test_surface_and_spectral_coefficients_agree():
    """t_s(i1) and t(k_perp) are the same object reached two ways."""
    k0 = 2 * np.pi / LAM
    i1 = np.radians(37.0)
    assert t_spectral(1.0 * k0 * np.sin(i1), 1.0, 1.5, k0) == pytest.approx(
        t_s(1.0, 1.5, np.cos(i1)), rel=1e-12)


# ── the pupil amplitude ───────────────────────────────────────────────────────
class TestStigmaticPupil:
    si = stigmatic_interface(Medium(1.0), Medium(1.5), zo=-0.050, zi=0.025)
    g = si.sample(20001, i1_max_deg=80.0)

    def test_simple_form_equals_ray_tube_form(self):
        """P = t_s d2/d1  ==  sqrt(T_s (n1/n2) w1/w2).

        Two independently coded expressions; the collapse of the second into
        the first is the result the session actually discovered."""
        P1 = stigmatic_pupil(self.si, self.g)
        P2 = stigmatic_pupil_from_tubes(self.si, self.g)
        m = slice(1, None)
        assert np.abs((P1[m] - P2[m]) / P2[m]).max() < 1e-12

    def test_edge_over_axis_matches_measurement(self):
        # golden value; the wave-equation measurement gave 0.213 +/- 0.028
        P = stigmatic_pupil(self.si, self.g)
        assert P[-1] / P[0] == pytest.approx(0.2071, abs=2e-3)


# ── propagators ───────────────────────────────────────────────────────────────
def test_asm_equals_rs1():
    """Two representations of the same exact operator: the angular spectrum
    (Fourier space) and Rayleigh–Sommerfeld I (real space)."""
    n = 1.5
    k = 2 * np.pi * n / LAM
    r = np.linspace(0, 1e-3, 8001)          # >5 samples per radial fringe
    z = 5e-3
    Rc = np.hypot(r, z)
    field = MonochromaticField(Grid.polar(r), (np.exp(-1j * k * Rc) / Rc)[:, None],
                               LAM, medium=Medium(n))
    out = Grid.polar(np.linspace(0, 20e-6, 60))
    Ia = np.abs(angular_spectrum(field, z, output_grid=out).u[:, 0]) ** 2
    Ir = np.abs(rayleigh_sommerfeld(field, z, output_grid=out).u[:, 0]) ** 2
    assert np.abs(Ia / Ia.max() - Ir / Ir.max()).max() < 5e-3


def test_paraxial_refuses_outside_its_regime():
    r = np.linspace(0, 9e-3, 100)
    field = MonochromaticField(Grid.polar(r), np.ones((100, 1)), LAM,
                               medium=Medium(1.5))
    with pytest.raises(ValueError, match="validity"):
        fresnel(field, 16e-3)


# ── transmit (the exact planar operator, on fields) ──────────────────────────
def test_transmit_reduces_to_t_spectral():
    """A field through a flat interface, measured in the spectral domain,
    must reproduce the diagonal operator t(k⊥) — the multiplicative identity
    F_out(k) = t(k)·F_in(k) on the field's own spectral content."""
    from diffractor import FT2

    k0 = 2 * np.pi / LAM
    n1, n2 = 1.0, 1.5
    r = np.linspace(0, 1.2e-3, 6001)
    grid = Grid.polar(r)
    field = MonochromaticField(grid, np.exp(-(r[:, None] / 3e-4) ** 2), LAM)
    kg = grid.reciprocal(k_max=3.0e4, n_k=800)   # the Gaussian's own band
    out = transmit(field, Medium(n2), kgrid=kg)

    F_in = FT2(field, kgrid=kg)
    F_out = FT2(out.like(out.values, medium=Medium(n1)), kgrid=kg)
    t_ref = t_spectral(kg.axes[0], n1, n2, k0)
    err = np.abs(F_out.u[:, 0] - t_ref * F_in.u[:, 0]).max()
    assert err < 1e-4 * np.abs(F_in.u).max()
    assert out.medium.n == n2


# ── analysis: demodulate and energy ──────────────────────────────────────────
def test_demodulate_recovers_amplitude():
    """For a perfect converging spherical wave, demodulation should recover
    a constant amplitude."""
    R2 = 0.01
    k2 = 2 * np.pi * 1.5 / LAM
    th = np.linspace(0.01, 0.3, 100)
    A0 = 1.0 / (4 * np.pi)
    psi = A0 / R2 * np.exp(-1j * k2 * R2)
    A = demodulate(psi, R2, k2)
    assert np.abs(A - A0).max() < 1e-15


def test_energy_through_sphere_uniform():
    """A uniform amplitude |A|=1 over a hemisphere should give 2*pi."""
    th = np.linspace(0, np.pi / 2, 1000)
    A = np.ones_like(th, dtype=complex)
    E = energy_through_sphere(A, th)
    assert E == pytest.approx(2 * np.pi, rel=1e-4)


def test_predicted_pupil_matches_stigmatic_pupil():
    """predicted_pupil wraps stigmatic_pupil with the 1/(4pi) amplitude."""
    si = stigmatic_interface(Medium(1.0), Medium(1.5), zo=-0.050, zi=0.025)
    g = si.sample(5001, i1_max_deg=80.0)
    P_pred = predicted_pupil(si, g)
    P_raw = stigmatic_pupil(si, g)
    assert np.allclose(P_pred, (1.0 / (4 * np.pi)) * P_raw)


# ── sources ──────────────────────────────────────────────────────────────────
def test_point_source_imports_from_package():
    """Verify sources are accessible through the package __init__."""
    k = 2 * np.pi * 1.5 / LAM
    rho = np.array([0.0, 1e-3])
    z = np.array([5e-3, 5e-3])
    psi = point_source(rho, z, 0.0, k)
    assert psi.shape == (2,)
    assert np.isfinite(psi).all()


def test_point_source_normal_derivative_sign():
    """Normal derivative should be purely outgoing for a point source."""
    k = 2 * np.pi / LAM
    z_src = 0.0
    rho = np.array([0.0])
    z = np.array([5e-3])
    n_rho = np.array([0.0])
    n_z = np.array([1.0])
    v = point_source_normal_derivative(rho, z, n_rho, n_z, z_src, k)
    u = point_source(rho, z, z_src, k)
    # for on-axis propagation with n_z=1, v/u ≈ ik - 1/d
    d = 5e-3
    expected_ratio = (1j * k - 1.0 / d)
    assert v[0] / u[0] == pytest.approx(expected_ratio, rel=1e-10)


# ── axis regularization robustness ───────────────────────────────────────────
def test_ray_tube_axis_with_sparse_sampling():
    """Axis regularization should work even with few, non-uniformly spaced points."""
    si = stigmatic_interface(Medium(1.0), Medium(1.5), zo=-0.050, zi=0.025)
    g = si.sample(51, cluster=3.0)
    T = T_s(si.n1, si.n2, g["cos_i1"], g["cos_i2"])
    P = ray_tube_amplitude(si.n1, si.n2, g["w1"], g["w2"], T)
    assert np.isfinite(P[0])
    assert P[0] > 0


# ── ground truth chain ────────────────────────────────────────────────────────
def test_scalar_ball_is_exact():
    b = ScalarBall(1.0, 1.5, 5.0, 1.0)
    e1, e2 = b.bc_residual()
    assert e1 < 1e-12 and e2 < 1e-12
    assert b.flux_balance() < 1e-14


@pytest.mark.parametrize("n1,n2,a", [(1.0, 1.5, 5.0), (1.0, 2.5, 3.0)])
def test_scalar_ball_parametrized(n1, n2, a):
    """Exact ball solution verified across different index contrasts and sizes."""
    b = ScalarBall(n1, n2, a, 1.0)
    e1, e2 = b.bc_residual()
    assert e1 < 1e-11 and e2 < 1e-11
    assert b.flux_balance() < 1e-13


def test_muller_bem_reproduces_the_exact_ball():
    """The whole chain in one assertion, plus the flat conditioning that the
    second-kind formulation buys (the direct one grows like N)."""
    from groundtruth.bem import solve_muller, sphere_generator

    b = ScalarBall(1.0, 1.5, 1.0, 1.0, lmax_pad=25)
    g = sphere_generator(1.0, 120)
    if (g.n_rho * g.rho + g.n_z * g.z).mean() < 0:
        g.flip_normal()
    ue = b.surface_field(g.z / b.a)
    ui = np.exp(1j * b.k1 * g.z)
    u, _, M = solve_muller(g, b.k1, b.k2, ui, 1j * b.k1 * g.n_z * ui, n_phi=48)
    assert np.abs(u - ue).max() / np.abs(ue).max() < 5e-3
    assert np.linalg.cond(M) < 500
