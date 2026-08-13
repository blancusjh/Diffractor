"""
A CLOSED body of revolution whose front face is the stigmatic ovoid.

BEM needs a closed surface, but the optical ovoid is only a cap.  Closing it
naively puts a back face inside the converging cone, which reintroduces exactly
the cavity contamination that ruined the ball experiment.  The design here
keeps every reflecting surface out of the cone:

      cap (the Cartesian oval)          cylinder            end cone
      vertex ........ rim  ------------------------------>  \
        |                                                     \
      axis ------------------- A' (focus) ----------------------- axis

  * every ray from the source converges to A' BEFORE reaching anything else,
    so the focal region sees only the refracted wave;
  * the cylinder sits at the rim radius, strictly outside the cone;
  * the end is a cone rather than a disc, so whatever does reach it is
    reflected radially outward instead of straight back through the focus.
"""

from __future__ import annotations

import numpy as np

from ._ovoid_geometry import Ovoid
from .direct import Generator


def ovoid_body(n1, n2, zo, zi, lam, i1_max_deg=80.0,
               tail=1.6, cone_frac=0.55, ppl=40.0, n_cap=None):
    """Closed generator for the ovoid + shroud.

    Parameters
    ----------
    zo, zi, lam : float
        Object distance, image distance and wavelength in the SAME units.
        Scaling all three together holds NA fixed and changes only kR.
    tail : float
        Cylinder end placed at z = zi + tail*zi, i.e. how far past the focus
        the body is closed.
    cone_frac : float
        Axial length of the closing cone, as a fraction of the rim radius.
    ppl : float
        Target panels per wavelength IN THE DENSER MEDIUM.
    """
    ov = Ovoid(n1, n2, zo, zi)
    g0 = ov.sample(60001, i1_max_deg=i1_max_deg, cluster=2.0)
    r_rim, z_rim = g0["r"][-1], g0["z"][-1]

    lam2 = lam / n2
    hmax = lam2 / ppl

    # -- 1. the ovoid cap, resampled to uniform arclength ---------------------
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(g0["r"]),
                                                  np.diff(g0["z"])))])
    n_cap = n_cap or max(8, int(np.ceil(s[-1] / hmax)))
    sc = np.linspace(0.0, s[-1], n_cap + 1)
    r_cap = np.interp(sc, s, g0["r"])
    z_cap = np.interp(sc, s, g0["z"])

    # -- 2. cylinder from the rim forward -------------------------------------
    z_end = zi + tail * zi
    n_cyl = max(4, int(np.ceil((z_end - z_rim) / hmax)))
    z_cyl = np.linspace(z_rim, z_end, n_cyl + 1)[1:]
    r_cyl = np.full_like(z_cyl, r_rim)

    # -- 3. closing cone, apex on the axis ------------------------------------
    z_apex = z_end + cone_frac * r_rim
    L = np.hypot(r_rim, z_apex - z_end)
    n_cone = max(4, int(np.ceil(L / hmax)))
    t = np.linspace(0.0, 1.0, n_cone + 1)[1:]
    r_cone = r_rim * (1 - t)
    z_cone = z_end + (z_apex - z_end) * t

    rn = np.concatenate([r_cap, r_cyl, r_cone])
    zn = np.concatenate([z_cap, z_cyl, z_cone])
    g = Generator(rn, zn)

    # outward normal must point OUT of the glass; at the vertex that is -z
    if g.n_z[0] > 0:
        g.flip_normal()

    g.meta = dict(ov=ov, r_rim=r_rim, z_rim=z_rim, z_end=z_end, z_apex=z_apex,
                  NA=n2 * g0["sin2"][-1], n_cap=n_cap, n_cyl=n_cyl,
                  n_cone=n_cone, cap_len=s[-1], lam=lam, zi=zi, zo=zo,
                  n1=n1, n2=n2, g0=g0)
    return g


def point_source_cauchy(g, k1, zo):
    """Incident Cauchy data on the surface for an on-axis point source at zo."""
    d = np.hypot(g.rho, g.z - zo)
    u = np.exp(1j * k1 * d) / (4 * np.pi * d)
    # r^ . n  with r = x - A
    rn = (g.rho * g.n_rho + (g.z - zo) * g.n_z) / d
    v = (1j * k1 - 1.0 / d) * u * rn
    return u, v


if __name__ == "__main__":
    LAM = 1.0
    g = ovoid_body(1.0, 1.5, -16.0, 8.0, LAM, ppl=40.0)
    m = g.meta
    print("CLOSED OVOID BODY")
    print(f"  NA                = {m['NA']:.4f}")
    print(f"  rim               = (r {m['r_rim']:.3f}, z {m['z_rim']:.3f}) lam")
    print(f"  focus A'          = z {m['zi']:.3f} lam")
    print(f"  body closes at    = z {m['z_end']:.3f} .. apex {m['z_apex']:.3f} lam")
    print(f"  panels: cap {m['n_cap']}, cylinder {m['n_cyl']}, cone {m['n_cone']}"
          f"  -> N = {g.N}")
    print(f"  generator length  = {g.h.sum():.2f} lam = "
          f"{g.h.sum()*m['n2']/LAM:.1f} interior wavelengths")
    print(f"  outward normal at the vertex: (nr {g.n_rho[0]:+.3f}, nz {g.n_z[0]:+.3f})"
          "   [should be ~ (0,-1)]")

    # does the converging cone clear the shroud?
    g0 = m["g0"]
    zq = np.linspace(m["z_rim"], m["zi"], 40)
    # cone radius at z, from the marginal ray
    rr = m["r_rim"] * (m["zi"] - zq) / (m["zi"] - m["z_rim"])
    print(f"  max cone radius between rim and focus = {rr.max():.3f} lam "
          f"vs cylinder at {m['r_rim']:.3f} lam  -> clearance OK: "
          f"{bool(rr.max() <= m['r_rim'] + 1e-9)}")
