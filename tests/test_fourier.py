"""The Fourier operators: one transform, several representations.

The acceptance test of the whole design is cross-basis agreement: the SAME
physical function, sampled on a cartesian grid and on a polar grid, must give
the SAME transform values at the SAME frequency nodes — the basis decides how
the integral is evaluated, never what it is.
"""

import numpy as np
import pytest
from scipy.special import j0

from diffractor.field import Field, MonochromaticField
from diffractor.fourier import FT2, IFT2, PolarPlan
from diffractor.hankel import hankel_transform
from diffractor.space import Grid
from diffractor.spectrum import Spectrum

SIG = 1.0
R_AXIS = np.linspace(0.0, 10.0 * SIG, 1200)
X_AXIS = (np.arange(256) - 128) * 0.08 * SIG


def _cartesian_gaussian():
    g = Grid.cartesian(X_AXIS, X_AXIS)
    X, Y = g.meshes()
    return MonochromaticField(g, np.exp(-(X**2 + Y**2) / (2 * SIG**2)), 0.5)


def _polar_gaussian(n_theta=1):
    g = Grid.polar(R_AXIS, n_theta=n_theta)
    R, _ = g.meshes()
    return MonochromaticField(g, np.exp(-R**2 / (2 * SIG**2)), 0.5)


def _polar_cos2theta():
    """f = r² e^{-r²/2} cos2θ, whose transform is closed form:
    FT2[g(r)e^{±i2θ}] = 2π(−i)² H₂{g} e^{±i2φ} and H₂{r²e^{-r²/2}} = k²e^{-k²/2},
    so FT2[f] = −2π k² e^{-k²/2} cos2φ."""
    g = Grid.polar(R_AXIS, n_theta=16)
    R, TH = g.meshes()
    return MonochromaticField(g, R**2 * np.exp(-R**2 / 2) * np.cos(2 * TH), 0.5)


def test_ft2_cartesian_gaussian_closed_form():
    F = FT2(_cartesian_gaussian())
    KX, KY = F.grid.meshes()
    exact = 2 * np.pi * SIG**2 * np.exp(-(KX**2 + KY**2) * SIG**2 / 2)
    assert np.abs(F.u - exact).max() < 1e-12 * exact.max()


def test_ft2_cartesian_fft_equals_matrix_path():
    f = _cartesian_gaussian()
    F_fft = FT2(f)
    F_mat = FT2(f, kgrid=F_fft.grid)
    assert np.abs(F_fft.values - F_mat.values).max() < 1e-10


def test_ft2_polar_gaussian_closed_form():
    F = FT2(_polar_gaussian())
    k = F.grid.axes[0]
    exact = 2 * np.pi * SIG**2 * np.exp(-(k * SIG) ** 2 / 2)
    assert np.abs(F.u[:, 0] - exact).max() < 1e-5 * exact.max()


def test_ft2_polar_cos2theta_closed_form():
    f = _polar_cos2theta()
    F = FT2(f, kgrid=f.grid.reciprocal(k_max=8.0))
    K, PHI = F.grid.meshes()
    exact = -2 * np.pi * K**2 * np.exp(-K**2 / 2) * np.cos(2 * PHI)
    assert np.abs(F.u - exact).max() < 1e-6 * np.abs(exact).max()


def test_ft2_cross_basis_agreement():
    """THE acceptance test: cartesian samples of the same anisotropic function,
    transformed by the generic node-sum path onto the polar transform's own
    frequency nodes, agree pointwise with the Jacobi-Anger/Hankel path."""
    fp = _polar_cos2theta()
    kg = fp.grid.reciprocal(k_max=4.0)      # inside the cartesian band, small
    Fp = FT2(fp, kgrid=kg)

    g = Grid.cartesian(X_AXIS, X_AXIS)
    X, Y = g.meshes()
    R2 = X**2 + Y**2
    cos2t = np.divide(X**2 - Y**2, R2, out=np.zeros_like(R2), where=R2 > 0)
    fc = MonochromaticField(g, R2 * np.exp(-R2 / 2) * cos2t, 0.5)
    Fc = FT2(fc, kgrid=kg)

    scale = np.abs(Fp.u).max()
    assert np.abs(Fc.u - Fp.u).max() < 1e-6 * scale


def test_round_trip_cartesian_is_machine_exact():
    f = _cartesian_gaussian()
    back = IFT2(FT2(f), grid=f.grid)
    assert np.abs(back.u - f.u).max() < 1e-13


def test_round_trip_polar_content_band():
    """Band-limited to the field's content, the polar round trip sits at the
    quadrature floor of the k-axis sampling rule (a few 1e-4 at β = 5)."""
    f = _polar_gaussian()
    kg = f.grid.reciprocal(k_max=8.0 / SIG)
    back = IFT2(FT2(f, kgrid=kg), grid=f.grid)
    assert np.abs(back.u - f.u).max() < 1e-3


def test_round_trip_polar_default_band_collects_noise():
    """Over the full default band the k·dk measure collects the quadrature
    noise of the empty high-k region — documented behaviour, bounded here."""
    f = _polar_gaussian()
    back = IFT2(FT2(f), grid=f.grid)
    err = np.abs(back.u - f.u).max()
    assert 1e-4 < err < 5e-2


def test_polar_round_trip_error_scaling():
    """Halving Δk (doubling β) cuts the round-trip error by ≥ 4×."""
    f = _polar_gaussian()
    errs = []
    for beta in (2.5, 5.0, 10.0):
        n_k = int(np.ceil(beta * 8.0 * R_AXIS[-1] / (2 * np.pi))) + 1
        kg = f.grid.reciprocal(k_max=8.0, n_k=n_k)
        back = IFT2(FT2(f, kgrid=kg), grid=f.grid)
        errs.append(np.abs(back.u - f.u).max())
    assert errs[1] < errs[0] / 4 and errs[2] < errs[1] / 4


def test_parseval_both_bases():
    fc = _cartesian_gaussian()
    Fc = FT2(fc)
    lhs = (fc.spectral_intensity[..., 0] * fc.grid.weights()).sum()
    rhs = (Fc.spectral_intensity[..., 0] * Fc.grid.weights()).sum() / (2 * np.pi) ** 2
    assert np.isclose(lhs, rhs, rtol=1e-10)
    assert np.isclose(lhs, np.pi * SIG**2, rtol=1e-9)   # ∬e^{-r²/σ²} = πσ²

    fp = _polar_gaussian(n_theta=4)
    Fp = FT2(fp, kgrid=fp.grid.reciprocal(k_max=8.0, n_k=400))
    lhs = (fp.spectral_intensity[..., 0] * fp.grid.weights()).sum()
    rhs = (Fp.spectral_intensity[..., 0] * Fp.grid.weights()).sum() / (2 * np.pi) ** 2
    assert np.isclose(lhs, rhs, rtol=1e-4)


def test_nyquist_mode_even_ntheta_round_trip():
    """cos((n_θ/2)θ) lands entirely on the even-n_θ Nyquist mode; both signs of
    m share one operator, so the round trip is as exact as any other mode's."""
    n_theta = 8
    g = Grid.polar(R_AXIS, n_theta=n_theta)
    R, TH = g.meshes()
    f = MonochromaticField(g, R**4 * np.exp(-R**2 / 2) * np.cos(4 * TH), 0.5)
    kg = g.reciprocal(k_max=8.0)
    back = IFT2(FT2(f, kgrid=kg), grid=g)
    assert np.abs(back.u - f.u).max() < 1e-3 * np.abs(f.u).max()


def test_polar_ntheta1_equals_legacy_cycles_hankel():
    """The n_θ = 1 path is the old axisymmetric operator: the legacy
    cycles-convention hankel (J0(2π x y), plain trapezoid) evaluated at
    ρ = k/2π equals FT2/2π up to the quadrature upgrade (h² endpoint term)."""
    f = _polar_gaussian()
    r = R_AXIS
    kg = f.grid.reciprocal(k_max=8.0)
    F = FT2(f, kgrid=kg)
    rho = kg.axes[0] / (2 * np.pi)
    legacy = 2 * np.pi * np.trapezoid(
        j0(2 * np.pi * np.outer(rho, r)) * (f.u[:, 0] * r)[None, :], r, axis=1)
    assert np.abs(F.u[:, 0] - legacy).max() < 1e-4


def test_plan_reproduces_unplanned_transform():
    f = _polar_cos2theta()
    kg = f.grid.reciprocal(k_max=8.0)
    plan = PolarPlan.build(f.grid, kg)
    assert plan is not None
    F0 = FT2(f, kgrid=kg)
    F1 = FT2(f, kgrid=kg, plan=plan)
    assert np.array_equal(F0.values, F1.values)
    b0 = IFT2(F0, grid=f.grid)
    b1 = IFT2(F1, grid=f.grid, plan=plan)
    assert np.array_equal(b0.values, b1.values)


def test_domain_flip_and_refusals():
    f = _polar_gaussian()
    F = FT2(f)
    assert f.domain == "space" and F.domain == "frequency"
    with pytest.raises(ValueError, match="space"):
        FT2(F)
    with pytest.raises(ValueError, match="frequency"):
        IFT2(f, grid=f.grid)
    with pytest.raises(ValueError, match="grid="):
        IFT2(F)
    with pytest.raises(ValueError, match="theta axis"):
        FT2(f, kgrid=Grid.polar(np.linspace(0, 1, 8), n_theta=4))


def test_spectral_axis_rides_through_both_bases():
    s = Spectrum([0.4, 0.5, 0.6])
    g = Grid.polar(R_AXIS, n_theta=4)
    R, _ = g.meshes()
    vals = np.stack([np.exp(-R**2 / 2), 2 * np.exp(-R**2 / 2),
                     3 * np.exp(-R**2)], axis=-1)
    f = Field(g, vals, s)
    F = FT2(f, kgrid=g.reciprocal(k_max=8.0))
    assert F.values.shape[-1] == 3
    for l in range(3):
        mono = MonochromaticField(g, vals[..., l], s.wavelengths[l])
        Fm = FT2(mono, kgrid=F.grid)
        assert np.allclose(F.values[..., l], Fm.values[..., 0],
                           rtol=0, atol=1e-12 * np.abs(Fm.values).max())
