import numpy as np
import pytest

from diffractor import (
    annular_aperture,
    circular_aperture,
    elliptical_aperture,
    lattice_aperture,
    lattice_sites,
    make_grid,
    nslit_aperture,
    polygon_aperture,
    rectangular_aperture,
    slit_aperture,
    square_aperture,
)

# `antialiased` is the internal area-coverage helper behind
# MonochromaticField.add_aperture(..., antialiased=True); unit-tested directly.
from diffractor.physics.apertures import antialiased

GRID = make_grid(256, 2.0)
X, Y = GRID
DX = 2.0 / 256


def test_circular_aperture_area():
    R = 0.5
    mask = circular_aperture(X, Y, R)
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, np.pi * R**2, rtol=1e-2)


def test_circular_aperture_center_offset():
    mask = circular_aperture(X, Y, 0.2, center=(0.5, -0.3))
    ys, xs = np.nonzero(mask)
    np.testing.assert_allclose(X[0, xs].mean(), 0.5, atol=DX)
    np.testing.assert_allclose(Y[ys, 0].mean(), -0.3, atol=DX)


def test_rectangular_and_square_aperture():
    mask = rectangular_aperture(X, Y, 1.0, 0.5)
    area = mask.sum() * DX * DX
    # The inclusive <= boundary adds up to one sample row/column per edge.
    np.testing.assert_allclose(area, 0.5, rtol=3e-2)
    np.testing.assert_array_equal(
        square_aperture(X, Y, 0.8), rectangular_aperture(X, Y, 0.8, 0.8)
    )


def test_annular_aperture():
    mask = annular_aperture(X, Y, 0.3, 0.6)
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, np.pi * (0.6**2 - 0.3**2), rtol=2e-2)


def test_elliptical_aperture():
    mask = elliptical_aperture(X, Y, 0.8, 0.4)
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, np.pi * 0.8 * 0.4, rtol=2e-2)


def test_antialiased_area_is_more_accurate():
    R = 0.5
    hard = circular_aperture(X, Y, R).sum() * DX * DX
    soft = antialiased(circular_aperture, GRID, R, factor=8).values.real.sum() * DX * DX

    exact = np.pi * R**2
    assert abs(soft - exact) < abs(hard - exact)
    np.testing.assert_allclose(soft, exact, rtol=1e-4)


def test_antialiased_returns_field_and_kwargs():
    from diffractor import Field

    field = antialiased(circular_aperture, GRID, 0.5, center=(0.1, 0.0))
    assert isinstance(field, Field)
    assert field.grid is GRID
    t = field.values.real
    assert ((0.0 <= t) & (t <= 1.0)).all()
    # Fully interior and exterior pixels stay binary.
    assert t[128, 128] == 1.0
    assert t[0, 0] == 0.0

    with pytest.raises(ValueError):
        antialiased(circular_aperture, GRID, 0.5, factor=0)


def test_slit_aperture():
    horizontal = slit_aperture(X, Y, 0.2, orientation="x")
    assert horizontal[np.abs(Y) <= 0.1].all()
    assert not horizontal[np.abs(Y) > 0.11].any()

    vertical = slit_aperture(X, Y, 0.2, orientation="y")
    np.testing.assert_array_equal(vertical, horizontal.T)

    with pytest.raises(ValueError):
        slit_aperture(X, Y, 0.2, orientation="diagonal")


def test_lattice_sites_count_and_centering():
    sq = lattice_sites(0.1, lattice="square", size=(3, 4))
    assert sq.shape == (12, 2)
    np.testing.assert_allclose(sq.mean(axis=0), [0.0, 0.0], atol=1e-12)
    hx = lattice_sites(0.1, lattice="hexagonal", size=(4, 4))
    assert hx.shape == (16, 2)
    # hex row height is a√3/2
    ys = np.unique(hx[:, 1].round(9))
    np.testing.assert_allclose(np.diff(ys), 0.1 * np.sqrt(3) / 2, atol=1e-9)


def test_lattice_sites_validation():
    with pytest.raises(ValueError):
        lattice_sites(-1.0)
    with pytest.raises(ValueError):
        lattice_sites(0.1, lattice="triangular")
    with pytest.raises(ValueError):
        lattice_sites(0.1, size=(0, 3))


def test_lattice_aperture_area_and_centering():
    R = 0.03
    mask = lattice_aperture(X, Y, circular_aperture, 0.3, lattice="square", size=(3, 3), R=R)
    # nine well-separated holes -> total area ~ 9 π R²
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, 9 * np.pi * R**2, rtol=5e-2)
    # the pattern is centered on the origin (centroid within a pixel)
    total = mask.sum()
    np.testing.assert_allclose((X * mask).sum() / total, 0.0, atol=DX)
    np.testing.assert_allclose((Y * mask).sum() / total, 0.0, atol=DX)


def test_lattice_aperture_composes_with_antialiased():
    from diffractor import Field

    field = antialiased(lattice_aperture, GRID, circular_aperture, 0.4, size=(2, 2), R=0.05)
    assert isinstance(field, Field)
    assert np.all((field.values.real >= 0.0) & (field.values.real <= 1.0))


def test_polygon_aperture_hexagon_area():
    R = 0.4
    mask = polygon_aperture(X, Y, 6, R)
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, 1.5 * np.sqrt(3) * R**2, rtol=5e-3)  # (3√3/2)R²


def test_polygon_aperture_square_and_origin_inside():
    # n_sides=4 is a square with apothem R·cos(45°); side = 2·apothem.
    R = 0.5
    mask = polygon_aperture(X, Y, 4, R)
    side = 2 * R * np.cos(np.pi / 4)
    np.testing.assert_allclose(mask.sum() * DX * DX, side**2, rtol=2e-2)
    # the origin is always inside a centered polygon
    assert bool(polygon_aperture(np.array(0.0), np.array(0.0), 6, R))


def test_polygon_aperture_rotation_shifts_vertex():
    # With rotation=0 an edge normal points along +x, so +x reaches only the
    # apothem; rotating by π/6 puts a vertex on +x, reaching to R. A point
    # between apothem and R on +x is therefore outside at rotation=0 and
    # inside at rotation=π/6.
    R = 0.5
    apo = R * np.cos(np.pi / 6)
    p = 0.5 * (apo + R)  # between apothem and circumradius, on +x
    x = np.array(p)
    y = np.array(0.0)
    assert not bool(polygon_aperture(x, y, 6, R, rotation=0.0))  # past an edge
    assert bool(polygon_aperture(x, y, 6, R, rotation=np.pi / 6))  # inside near a vertex


def test_polygon_aperture_validation():
    with pytest.raises(ValueError):
        polygon_aperture(X, Y, 2, 0.5)
    with pytest.raises(ValueError):
        polygon_aperture(X, Y, 6, -0.1)


def test_nslit_aperture_geometry():
    slits = nslit_aperture(X, Y, 3, 0.05, 0.4)
    # three slits, each of width ~0.05 along x, infinite in y
    row = slits[128]  # a horizontal cut
    # transmitting fraction of the row ~ n_slits · width / domain
    frac = row.mean()
    np.testing.assert_allclose(frac, 3 * 0.05 / 2.0, atol=1e-2)
    # centered: symmetric slit centers about 0
    centers = X[128][row]
    np.testing.assert_allclose(centers.mean(), 0.0, atol=DX)


def test_nslit_aperture_double_slit_fringe_spacing():
    from diffractor import Field, fresnel_zoom_propagator

    L, N, wavelength, z, d, w = 6e-3, 1024, 550e-9, 1.0, 0.2e-3, 0.03e-3
    grid = make_grid(N, L)
    x, y = grid
    mask = nslit_aperture(x, y, 2, w, d) * square_aperture(x, y, 1.5e-3)
    U0 = Field(grid, mask.astype(complex))
    fringe = wavelength * z / d
    out = fresnel_zoom_propagator(
        U0, z=z, wavelength=wavelength, output_half_width=5 * fringe, output_samples=2048
    )
    row = np.abs(out.values[1024]) ** 2
    xo = out.grid.x[0, :]
    # peak nearest the first fringe should sit at ≈ λ z / d
    near = np.abs(xo - fringe) < 0.4 * fringe
    peak = xo[near][np.argmax(row[near])]
    np.testing.assert_allclose(peak, fringe, rtol=5e-2)


def test_nslit_aperture_validation():
    with pytest.raises(ValueError):
        nslit_aperture(X, Y, 0, 0.05, 0.4)
    with pytest.raises(ValueError):
        nslit_aperture(X, Y, 2, -0.05, 0.4)
    with pytest.raises(ValueError):
        nslit_aperture(X, Y, 2, 0.05, 0.4, orientation="diagonal")
