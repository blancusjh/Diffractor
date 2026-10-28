"""Real scalar waves in spatially varying, stationary optical media.

The model is ``u_tt + 2 sigma u_t = (c0 / n)^2 Laplacian(u) + s``.
For an out-of-plane electric field Ez, constant permeability and a stationary,
isotropic dielectric, the lossless equation is the two-dimensional scalar
Maxwell reduction. It permits refraction, diffraction and reflection at index
interfaces. It does not represent a three-dimensional vector field.

Coordinates are meters, time is seconds, ``sigma`` is inverse seconds, and
forcing ``s`` has units of field amplitude per second squared. Damping is a
phenomenological loss term. A graded damping layer is a sponge, not a PML.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..mathutils.grids import Grid

VACUUM_LIGHT_SPEED = 299_792_458.0

__all__ = ["WaveEquation2D"]

Source = np.ndarray | float


def _real_array(
    value, shape: tuple[int, int], name: str, *, copy: bool = True
) -> np.ndarray:
    """Validate a finite real scalar or an array matching the simulation grid."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.broadcast_to(array, shape)
    elif array.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(array, dtype=float, copy=copy)


def _grid_spacing(grid: Grid) -> tuple[float, float]:
    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    if x.ndim != 2 or x.shape != y.shape or min(x.shape) < 3:
        raise ValueError(
            "grid must contain matching 2-D coordinate arrays "
            "with at least 3 samples per axis."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("grid coordinates must be finite.")
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    if dx <= 0 or dy <= 0:
        raise ValueError("grid coordinates must increase along x and y.")
    # Tolerances scale with a cell rather than with an arbitrary SI offset.
    expected_x = x[0, 0] + np.arange(x.shape[1]) * dx
    expected_y = y[0, 0] + np.arange(y.shape[0]) * dy
    if not np.allclose(x, expected_x[None, :], rtol=0, atol=dx * 1e-8):
        raise ValueError("grid x coordinates must form a regular Cartesian lattice.")
    if not np.allclose(y, expected_y[:, None], rtol=0, atol=dy * 1e-8):
        raise ValueError("grid y coordinates must form a regular Cartesian lattice.")
    return dx, dy


class WaveEquation2D:
    """A centered, second-order finite-difference scalar wave solver.

    ``courant`` is a fraction of the conservative two-dimensional stability
    limit ``1 / (max(c0/n) * sqrt(dx^-2 + dy^-2))``. An explicit ``dt`` must
    not exceed that limit. The five-point Laplacian supports unequal dx/dy.

    The outer ring and ``solid_mask`` cells have fixed zero field (Dirichlet
    reflectors). Supply a smoothly graded nonnegative ``damping`` array to
    attenuate waves before they reach the outer ring. Refractive index,
    damping and the solid mask are copied at construction and remain static.

    ``u`` is the current real field, ``u_previous`` the previous time level.
    Buffers rotate during stepping; read ``solver.u`` after each call instead
    of retaining a reference to its initial array. ``mean_square`` averages
    the computed ``u**2`` samples since reset. It is a field statistic, not
    electromagnetic energy flux or a calibrated optical intensity.
    """

    def __init__(
        self,
        grid: Grid,
        refractive_index=1.0,
        damping=0.0,
        *,
        solid_mask=None,
        dt: float | None = None,
        c0: float = VACUUM_LIGHT_SPEED,
        courant: float = 0.65,
    ) -> None:
        self.dx, self.dy = _grid_spacing(grid)
        if not np.isfinite(c0) or c0 <= 0:
            raise ValueError("c0 must be finite and positive.")
        if not np.isfinite(courant) or not 0 < courant < 1:
            raise ValueError(
                "courant must be finite and lie strictly between zero and one."
            )

        shape = grid.shape
        index = _real_array(refractive_index, shape, "refractive_index")
        sigma = _real_array(damping, shape, "damping")
        if np.any(index <= 0):
            raise ValueError("refractive_index must be positive everywhere.")
        if np.any(sigma < 0):
            raise ValueError("damping must be nonnegative everywhere.")
        index.setflags(write=False)
        sigma.setflags(write=False)

        self.grid = grid
        self.c0 = float(c0)
        self.refractive_index = index
        self.damping = sigma
        speed = self.c0 / index
        self.cfl_limit = float(1 / (speed.max() * np.hypot(1 / self.dx, 1 / self.dy)))
        selected_dt = courant * self.cfl_limit if dt is None else dt
        if not np.isfinite(selected_dt) or selected_dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if selected_dt > self.cfl_limit:
            raise ValueError(
                f"dt exceeds the two-dimensional CFL limit {self.cfl_limit:g} s."
            )
        self.dt = float(selected_dt)

        fixed = np.zeros(shape, dtype=bool)
        if solid_mask is not None:
            mask = np.asarray(solid_mask)
            if mask.shape != shape or mask.dtype != bool:
                raise ValueError(
                    "solid_mask must be a boolean array matching the grid."
                )
            fixed |= mask
        fixed[[0, -1], :] = True
        fixed[:, [0, -1]] = True
        self._fixed = fixed

        # (1 + sigma*dt) u_next = 2 u + (sigma*dt - 1) u_previous
        #                              + dt^2 (c^2 Laplacian(u) + forcing).
        inverse_damping = 1 / (1 + sigma * self.dt)
        self._current_factor = 2 * inverse_damping
        self._previous_factor = (sigma * self.dt - 1) * inverse_damping
        self._laplace_factor = self.dt**2 * speed**2 * inverse_damping
        self._forcing_factor = self.dt**2 * inverse_damping
        self._speed_squared = speed**2
        self._inverse_speed_squared = 1 / speed**2
        self._inverse_dx_squared = 1 / self.dx**2
        self._inverse_dy_squared = 1 / self.dy**2

        self.u = np.zeros(shape)
        self.u_previous = np.zeros(shape)
        self._next = np.zeros(shape)
        self._laplacian = np.zeros(shape)
        self._work = np.zeros(shape)
        self.mean_square = np.zeros(shape)
        self.time = 0.0
        self.step_count = 0

    def _differentiate(self, field: np.ndarray) -> np.ndarray:
        """Five-point Laplacian, reusing work arrays to avoid per-step copies."""
        center = self._laplacian[1:-1, 1:-1]
        work = self._work[1:-1, 1:-1]
        np.multiply(
            field[1:-1, 1:-1],
            -2 * (self._inverse_dx_squared + self._inverse_dy_squared),
            out=center,
        )
        neighbors = (
            (field[1:-1, 2:], self._inverse_dx_squared),
            (field[1:-1, :-2], self._inverse_dx_squared),
            (field[2:, 1:-1], self._inverse_dy_squared),
            (field[:-2, 1:-1], self._inverse_dy_squared),
        )
        for neighbor, factor in neighbors:
            np.multiply(neighbor, factor, out=work)
            center += work
        return self._laplacian

    def reset(self, initial_field=None, initial_velocity=None) -> None:
        """Restart at t=0 with field and field/second initial conditions.

        The previous level is reconstructed to second order using the initial
        acceleration ``c² Laplacian(u0) - 2 sigma v0``. If a source is supplied
        on the first :meth:`step` call, its t=0 acceleration is included in the
        startup then. Fixed boundary/solid cells are zeroed in both states.
        """
        self.u[:] = _real_array(
            0.0 if initial_field is None else initial_field,
            self.u.shape,
            "initial_field",
        )
        velocity = _real_array(
            0.0 if initial_velocity is None else initial_velocity,
            self.u.shape,
            "initial_velocity",
        )
        self.u[self._fixed] = 0
        velocity[self._fixed] = 0
        acceleration = (
            self._speed_squared * self._differentiate(self.u)
            - 2 * self.damping * velocity
        )
        self.u_previous[:] = (
            self.u - self.dt * velocity + 0.5 * self.dt**2 * acceleration
        )
        self.u_previous[self._fixed] = 0
        self._next.fill(0)
        self.mean_square.fill(0)
        self.time = 0.0
        self.step_count = 0

    def step(
        self,
        steps: int = 1,
        forcing: Source | Callable[[float], Source] | None = None,
    ) -> np.ndarray:
        """Advance ``steps`` intervals and return the current real field.

        ``forcing`` is a finite real array, scalar, or callable ``s(time)`` in
        field/second². A callable is evaluated at each current time level before
        the centered update. No field is prescribed directly by a source.
        """
        if (
            isinstance(steps, (bool, np.bool_))
            or not isinstance(steps, (int, np.integer))
            or steps < 0
        ):
            raise ValueError("steps must be a nonnegative integer.")
        constant_source = None
        if forcing is not None and not callable(forcing):
            constant_source = _real_array(forcing, self.u.shape, "forcing", copy=False)

        for _ in range(steps):
            source = constant_source
            if callable(forcing):
                source = _real_array(
                    forcing(self.time), self.u.shape, "forcing", copy=False
                )
            if self.step_count == 0 and source is not None:
                self.u_previous += 0.5 * self.dt**2 * source
                self.u_previous[self._fixed] = 0
            laplacian = self._differentiate(self.u)
            np.multiply(self.u, self._current_factor, out=self._next)
            np.multiply(self.u_previous, self._previous_factor, out=self._work)
            self._next += self._work
            np.multiply(laplacian, self._laplace_factor, out=self._work)
            self._next += self._work
            if source is not None:
                np.multiply(source, self._forcing_factor, out=self._work)
                self._next += self._work
            self._next[self._fixed] = 0

            self.u_previous, self.u, self._next = self.u, self._next, self.u_previous
            self.step_count += 1
            self.time = self.step_count * self.dt
            np.multiply(self.u, self.u, out=self._work)
            self._work /= self.step_count
            self.mean_square *= (self.step_count - 1) / self.step_count
            self.mean_square += self._work
        return self.u

    def energy(self) -> float:
        """Return the unforced scheme's staggered discrete energy.

        ``0.5 integral [(u-u_previous)^2/(c² dt²)
        + grad(u) dot grad(u_previous)] dx dy`` is conserved without damping
        or sources and decreases under damping. It is a stability diagnostic
        in relative field units, not a calibrated electromagnetic energy.
        """
        velocity = (self.u - self.u_previous) / self.dt
        kinetic = np.sum(self._inverse_speed_squared * velocity**2)
        gradient_x = np.diff(self.u, axis=1) / self.dx
        previous_gradient_x = np.diff(self.u_previous, axis=1) / self.dx
        gradient_y = np.diff(self.u, axis=0) / self.dy
        previous_gradient_y = np.diff(self.u_previous, axis=0) / self.dy
        potential = np.sum(gradient_x * previous_gradient_x) + np.sum(
            gradient_y * previous_gradient_y
        )
        return float(0.5 * self.dx * self.dy * (kinetic + potential))
