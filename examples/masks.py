"""Hard masks on cartesian grids, evaluated with grey (area-coverage) edges.

A hard-edged mask sampled point-wise has pixelated edges whose spurious
spectral content shows up as streaks; supersampling each cell and averaging
gives the area-coverage value instead.  These helpers exist for the examples —
the package itself has no aperture module, because an aperture is data
(a mask array), not physics.
"""
import numpy as np

__all__ = ["regular_polygon_mask", "disc_mask"]


def _supersampled(grid, inside, supersample):
    x, y = grid.axes
    dx, dy = x[1] - x[0], y[1] - y[0]
    off = (np.arange(supersample) + 0.5) / supersample - 0.5
    acc = np.zeros(grid.shape)
    for oy in off:
        for ox in off:
            X, Y = np.meshgrid(x + ox * dx, y + oy * dy, indexing="ij")
            acc += inside(X, Y)
    return acc / supersample**2


def regular_polygon_mask(grid, n_sides, radius, *, orientation=np.pi / 2,
                         supersample=4):
    """Regular polygon of circumradius `radius` on a cartesian Grid, as the
    intersection of n half-planes.  Returns (mask, edge-normal angles)."""
    normals = (orientation + np.pi / n_sides
               + 2 * np.pi * np.arange(n_sides) / n_sides)
    apothem = radius * np.cos(np.pi / n_sides)

    def inside(X, Y):
        keep = np.ones(X.shape, bool)
        for th in normals:
            keep &= (X * np.cos(th) + Y * np.sin(th)) <= apothem
        return keep

    return _supersampled(grid, inside, supersample), normals


def disc_mask(grid, radius, *, supersample=4):
    return _supersampled(grid, lambda X, Y: X**2 + Y**2 <= radius**2,
                         supersample)
