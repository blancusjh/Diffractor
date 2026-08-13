"""Recompute the convergence envelope in method_check.npz.

Surface error of the Müller BOR-BEM against the exact ball, as a function
of the interior electrical size k₂a, for two refractive indices.  The mesh
is kept at a fixed 40 panels per interior wavelength; the azimuthal
quadrature scales automatically with ka (see assemble_muller).

The env_* keys of method_check.npz are replaced; the single-run keys of the
original file are carried over unchanged.
"""

import time
from pathlib import Path

import numpy as np

from groundtruth.bem import sphere_generator, solve_muller
from groundtruth.exact import ScalarBall

LAM = 1.0
OUT = str(Path(__file__).resolve().parent / "golden") + "/"
PPL2 = 40.0                     # panels per interior wavelength


def surface_error(a, n2, N, n1=1.0):
    b = ScalarBall(n1, n2, a, LAM)
    g = sphere_generator(a, int(N))
    if (g.n_rho * g.rho + g.n_z * g.z).mean() < 0:
        g.flip_normal()
    ui = np.exp(1j * b.k1 * g.z)
    vi = 1j * b.k1 * g.n_z * ui
    u, v, M = solve_muller(g, b.k1, b.k2, ui, vi)
    ue = b.surface_field(g.z / a)
    return np.abs(u - ue).max() / np.abs(ue).max()


def envelope():
    k2a_targets = 2 * np.pi * np.array([1.5, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0])
    rows = []
    for n2 in (1.5, 2.5):
        for k2a in k2a_targets:
            a = k2a / (2 * np.pi * n2)
            # meridian length is pi*a = (k2a/2) interior wavelengths
            N = int(round(PPL2 * k2a / 2))
            t0 = time.time()
            err = surface_error(a, n2, N)
            rows.append((k2a, n2, err))
            print(f"  n2={n2}  k2a={k2a:6.1f}  N={N:5d}  err={err:.3e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    rows = np.array(rows)
    old = dict(np.load(OUT + "method_check.npz"))
    old.pop("env_cond", None)          # never plotted; the SVD dominated the run
    old.update(env_k2a=rows[:, 0], env_n2=rows[:, 1], env_err=rows[:, 2])
    np.savez(OUT + "method_check.npz", **old)
    print("  -> method_check.npz")


if __name__ == "__main__":
    print(f"convergence envelope at {PPL2:.0f} panels per interior wavelength")
    envelope()
