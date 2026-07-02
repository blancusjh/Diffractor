import numpy as np
import pytest

from diffraction import FFT2, FT2, IFFT2, IFT2, Field, Grid, frequency_grid, make_grid

try:
    import scipy.signal  # noqa: F401

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _random_field(n=32, L=2e-3, seed=0):
    grid = make_grid(n, L)
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return Field(grid, values)


def test_ft2_native_matches_fft2():
    field = _random_field()
    out = FT2(field)  # engine="auto" -> fft on native grid
    np.testing.assert_allclose(out.values, FFT2(field.values), atol=1e-12)
    assert out.domain == "frequency"
    # native k-grid is frequency_grid(input grid)
    kgrid = frequency_grid(field.grid)
    np.testing.assert_array_equal(out.grid.x, kgrid.x)


def test_matrix_engine_reproduces_fft2_on_native_grid():
    field = _random_field()
    kgrid = frequency_grid(field.grid)
    out = FT2(field, kgrid=kgrid, engine="matrix")
    np.testing.assert_allclose(out.values, FFT2(field.values), atol=1e-10)


def test_inverse_matrix_reproduces_ifft2():
    field = _random_field()
    spec = FT2(field)
    back = IFT2(spec, grid=field.grid, engine="matrix")
    np.testing.assert_allclose(back.values, IFFT2(spec.values), atol=1e-10)


def test_roundtrip_native():
    field = _random_field()
    back = IFT2(FT2(field))
    np.testing.assert_allclose(back.values, field.values, atol=1e-10)
    np.testing.assert_allclose(back.grid.x, field.grid.x, atol=1e-15)
    assert back.domain == "space"


def test_roundtrip_matrix():
    field = _random_field(n=24)
    spec = FT2(field, engine="matrix")
    back = IFT2(spec, engine="matrix")
    np.testing.assert_allclose(back.values, field.values, atol=1e-10)


def test_arbitrary_zoom_kgrid_matches_matrix_reference():
    # A user-chosen zoomed frequency window, non-native and non-square.
    field = _random_field(n=32)
    fx = np.linspace(-3e4, 3e4, 20)
    fy = np.linspace(-1e4, 1e4, 12)
    kx, ky = np.meshgrid(fx, fy)
    kgrid = Grid(kx, ky)
    out = FT2(field, kgrid=kgrid, engine="matrix")
    # direct DFT reference
    x1 = field.grid.x[0, :]
    y1 = field.grid.y[:, 0]
    Ex = np.exp(-2j * np.pi * np.outer(fx, x1))
    Ey = np.exp(-2j * np.pi * np.outer(fy, y1))
    ref = Ey @ field.values @ Ex.T
    np.testing.assert_allclose(out.values, ref, atol=1e-9)
    assert out.shape == (12, 20)


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not installed")
def test_czt_agrees_with_matrix_on_uniform_grid():
    field = _random_field(n=32)
    fx = np.linspace(-3e4, 3e4, 24)
    fy = np.linspace(-2e4, 2e4, 16)
    kx, ky = np.meshgrid(fx, fy)
    kgrid = Grid(kx, ky)
    m = FT2(field, kgrid=kgrid, engine="matrix")
    c = FT2(field, kgrid=kgrid, engine="czt")
    np.testing.assert_allclose(c.values, m.values, atol=1e-9)


def test_engine_validation():
    field = _random_field()
    kgrid = frequency_grid(field.grid)
    with pytest.raises(ValueError):
        FT2(field, engine="bogus")
    with pytest.raises(ValueError):
        FT2(field, kgrid=kgrid, engine="fft")  # fft only for native
