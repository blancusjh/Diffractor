import numpy as np
import pytest

from diffractor import Field, Grid, MonochromaticField, make_grid, sample_field, sample_intensity


def test_complex_bilinear_field_is_exact_on_nonuniform_grid():
    x, y = np.meshgrid([-2e-3, -0.5e-3, 1e-3], [-1e-3, 0, 2e-3])
    field = Field(Grid(x, y), 2 + 3j * x + 4 * y + 5j * x * y)
    qx = np.array([-2e-3, -0.7e-3, 1e-3])
    qy = np.array([[-1e-3], [0.3e-3], [2e-3]])
    np.testing.assert_allclose(sample_field(field, qx, qy),
                               2 + 3j * qx + 4 * qy + 5j * qx * qy, atol=1e-15)


def test_intensity_is_sampled_without_interfering_adjacent_pixels():
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    field = MonochromaticField(Grid(x, y), np.array([[1, -1], [1, -1]]))
    assert sample_field(field, 0.5, 0.5) == 0
    assert sample_intensity(field, 0.5, 0.5) == 1


@pytest.mark.parametrize("x,y", [(1, 0), (0, -1), (np.nan, 0), (0, np.inf)])
def test_probes_reject_invalid_locations(x, y):
    field = MonochromaticField(make_grid(8, 1e-3))
    with pytest.raises(ValueError):
        sample_field(field, x, y)


def test_probes_reject_curvilinear_and_frequency_grids():
    grid = make_grid(8, 1e-3)
    field = Field(grid, np.ones(grid.shape), domain="frequency")
    with pytest.raises(ValueError, match="spatial domain"):
        sample_intensity(field, 0, 0)
    warped = Grid(grid.x + grid.y * 0.1, grid.y)
    with pytest.raises(ValueError, match="rectilinear"):
        sample_field(Field(warped, np.ones(grid.shape)), 0, 0)
