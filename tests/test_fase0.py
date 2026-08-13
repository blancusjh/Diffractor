"""Claims from the validation session, as executable assertions.

Each test here guards something that took work to get right and could
plausibly break again.  Nothing tests that the language works.
"""

import numpy as np
import pytest

from diffractor import Medium, stigmatic_interface
from diffractor.analysis.opl import opd_waves
from diffractor.geometry import DescartesOvoid
from diffractor.propagation.exact import asm_axisym, rs1_plane
from diffractor.propagation.paraxial import fresnel_plane
from diffractor.propagation.transport import (stigmatic_pupil,
                                              stigmatic_pupil_from_tubes)
from diffractor.scattering.fresnel import R_s, T_s, t_s
from diffractor.scattering.planar import t_spectral

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
    """Two representations of the same exact operator."""
    n = 1.5
    k = 2 * np.pi * n / LAM
    r = np.linspace(0, 1e-3, 8001)          # >5 samples per radial fringe
    z = 5e-3
    Rc = np.hypot(r, z)
    U = np.exp(-1j * k * Rc) / Rc           # converging spherical wave
    rho = np.linspace(0, 20e-6, 60)
    Ia = np.abs(asm_axisym(U, r, z, n, LAM, rho)[0]) ** 2
    Ir = np.abs(rs1_plane(U, r, z, n, LAM, rho)) ** 2
    assert np.abs(Ia / Ia.max() - Ir / Ir.max()).max() < 5e-3


def test_paraxial_refuses_outside_its_regime():
    r = np.linspace(0, 9e-3, 100)
    with pytest.raises(ValueError, match="validity"):
        fresnel_plane(np.ones_like(r, dtype=complex), r, 16e-3, 1.5, LAM,
                      np.array([0.0]))


# ── ground truth chain ────────────────────────────────────────────────────────
def test_scalar_ball_is_exact():
    b = ScalarBall(1.0, 1.5, 5.0, 1.0)
    e1, e2 = b.bc_residual()
    assert e1 < 1e-12 and e2 < 1e-12
    assert b.flux_balance() < 1e-14


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
