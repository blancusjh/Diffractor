import numpy as np
import pytest

from diffractor import Grid, grid_spacing, make_grid


def test_make_grid_returns_grid():
    grid = make_grid(32, 2e-3)
    assert isinstance(grid, Grid)
    assert grid.shape == (32, 32)


def test_unpacking_back_compat():
    grid = make_grid(16, 1.0)
    x, y = grid  # the historical idiom must keep working
    assert x.shape == (16, 16)
    assert y.shape == (16, 16)
    assert x is grid[0]
    assert y is grid[1]


def test_spacing_property():
    n, L = 64, 3e-3
    grid = make_grid(n, L)
    dx, dy = grid.spacing
    np.testing.assert_allclose(dx, L / n)
    np.testing.assert_allclose(dy, L / n)


def test_grid_spacing_free_function_delegates():
    grid = make_grid(64, 3e-3)
    assert grid_spacing(grid) == grid.spacing


def test_spacing_rejects_decreasing_order():
    grid = make_grid(8, 1.0)
    x, y = grid
    flipped = Grid(x[:, ::-1], y)
    with pytest.raises(ValueError):
        _ = flipped.spacing


def test_grid_is_frozen():
    grid = make_grid(8, 1.0)
    with pytest.raises(Exception):
        grid.x = np.zeros((8, 8))


def test_grid_holds_given_arrays():
    x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    grid = Grid(x, y)
    assert grid.x is x
    assert grid.y is y
