import numpy as np
import pytest

from diffraction import Field, make_grid


def _field(n=16, L=1.0, domain="space"):
    grid = make_grid(n, L)
    values = np.ones((n, n), dtype=complex)
    return Field(grid, values, domain=domain), grid


def test_construction_and_accessors():
    field, grid = _field()
    assert field.grid is grid
    assert field.shape == (16, 16)
    assert field.domain == "space"
    # coordinate access lives on the grid, not the field
    assert not hasattr(field, "x")
    assert not hasattr(field, "spacing")


def test_intensity():
    grid = make_grid(8, 1.0)
    field = Field(grid, 2.0 * np.ones((8, 8), dtype=complex))
    np.testing.assert_allclose(field.intensity, 4.0)


def test_mul_by_scalar_and_array():
    field, _ = _field()
    scaled = field * 3.0
    np.testing.assert_allclose(scaled.values, 3.0)
    assert scaled.grid is field.grid
    mask = np.zeros((16, 16), dtype=complex)
    masked = field * mask
    np.testing.assert_allclose(masked.values, 0.0)
    # commutes
    np.testing.assert_allclose((3.0 * field).values, 3.0)


def test_mul_by_field_same_grid():
    grid = make_grid(8, 1.0)
    a = Field(grid, 2.0 * np.ones((8, 8), dtype=complex))
    b = Field(grid, 3.0 * np.ones((8, 8), dtype=complex))
    np.testing.assert_allclose((a * b).values, 6.0)


def test_mul_by_field_different_grid_raises():
    a, _ = _field(n=8, L=1.0)
    b, _ = _field(n=8, L=2.0)  # same shape, different coordinates
    with pytest.raises(ValueError):
        _ = a * b


def test_like_preserves_grid_and_domain():
    field, grid = _field(domain="frequency")
    new = field.like(np.zeros((16, 16), dtype=complex))
    assert new.grid is grid
    assert new.domain == "frequency"
    np.testing.assert_allclose(new.values, 0.0)


def test_shape_mismatch_rejected():
    grid = make_grid(8, 1.0)
    with pytest.raises(ValueError):
        Field(grid, np.ones((4, 4), dtype=complex))


def test_bad_domain_rejected():
    grid = make_grid(8, 1.0)
    with pytest.raises(ValueError):
        Field(grid, np.ones((8, 8), dtype=complex), domain="momentum")


def test_frozen():
    field, _ = _field()
    with pytest.raises(Exception):
        field.values = np.zeros((16, 16), dtype=complex)


def test_xp_is_numpy_on_cpu():
    field, _ = _field()
    assert field.xp is np
