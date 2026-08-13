"""
Exact, singularity-free parametrisation of the Cartesian oval (ovoid of Descartes).

Why not use CartesianSurface.sag()
----------------------------------
`diffractor`'s sag solver runs Newton on the equal-optical-path residual as a
function of z at fixed r.  Near the rim of the oval the two solution sheets
merge into a double root, so the Jacobian collapses and Newton bails out long
before the physical edge of the surface.  That is what limits the achievable
NA to ~0.37 in practice; the same oval actually supports NA ~ 0.95.

The exact way
-------------
Impose the stigmatic condition  n1 d1 + n2 d2 = C  with
    d2^2 = d1^2 - 2 L d1 cos(th1) + L^2 ,   L = zi - zo ,  C = n1|zo| + n2 zi
Eliminating d2 gives a QUADRATIC in d1,

    a d1^2 + 2 (n2^2 L cos th1 - C n1) d1 + c = 0 ,
    a = n1^2 - n2^2 ,   c = C^2 - n2^2 L^2

which can be inverted for cos(th1) *linearly*:

    cos th1(d1) = ( -a d1 + 2 C n1 - c/d1 ) / (2 n2^2 L)                    (*)

So parametrising the surface by the object-side ray length d1 makes every
quantity a closed-form, smooth function -- no iteration, no branch point.
The physical sheet runs

    d1 : |zo|  ->  d1_rim = sqrt(c/a)

(d(cos th1)/d d1 = (-a + c/d1^2)/(2 n2^2 L) vanishes exactly at d1 = sqrt(c/a),
which is the rim: there the incident ray is tangent to the surface.)

Consequence worth noting: at the rim the ray grazes the surface, so the angle
of incidence reaches 90 deg and the Fresnel transmission falls to zero.  The
ovoid apodizes its own pupil to zero at its geometric edge.
"""

from __future__ import annotations

import numpy as np


class Ovoid:
    """Stigmatic Cartesian-oval diopter imaging A(zo) -> A'(zi), exactly."""

    def __init__(self, n1: float, n2: float, zo: float, zi: float):
        if not (zo < 0 < zi):
            raise ValueError("need a real object and a real image: zo < 0 < zi")
        if n1 == n2:
            raise ValueError("n1 and n2 must differ")
        self.n1, self.n2, self.zo, self.zi = n1, n2, zo, zi
        self.L = zi - zo
        self.C = n1 * abs(zo) + n2 * zi           # the invariant optical path
        self.a = n1**2 - n2**2
        self.c = self.C**2 - n2**2 * self.L**2
        self.d1_vertex = abs(zo)
        self.d1_rim = np.sqrt(self.c / self.a)    # double root of the quadratic

    # -- the parametrisation ---------------------------------------------------
    def _cos_th1(self, d1):
        """Equation (*)."""
        return (-self.a * d1 + 2 * self.C * self.n1 - self.c / d1) / (2 * self.n2**2 * self.L)

    def _dcos_th1(self, d1):
        """d(cos th1)/d d1 -- vanishes exactly at the rim."""
        return (-self.a + self.c / d1**2) / (2 * self.n2**2 * self.L)

    def sample(self, n_pts=20001, *, i1_max_deg=None, rim_frac=1.0, cluster=2.0):
        """Sample the oval along d1, optionally truncated by angle of incidence.

        Parameters
        ----------
        n_pts : int
            Number of surface samples.
        i1_max_deg : float, optional
            Truncate the aperture where the angle of incidence exceeds this.
            The Fresnel transmission dies at grazing incidence, so the natural
            stop is somewhere in the 65-80 deg range.  ``None`` = full oval.
        rim_frac : float
            Use only this fraction of the way from vertex to rim (in d1).
        cluster : float
            Exponent clustering samples toward the rim, where everything varies
            fastest.  1.0 = uniform in d1.
        """
        t = np.linspace(0.0, 1.0, n_pts) ** cluster
        d1 = self.d1_vertex + t * rim_frac * (self.d1_rim - self.d1_vertex)
        g = self.geometry(d1)

        if i1_max_deg is not None:
            keep = np.degrees(np.arccos(np.clip(g["cos_i1"], -1, 1))) <= i1_max_deg
            if not keep.all():
                # re-sample smoothly up to the truncation point
                d1_cut = d1[keep][-1]
                t = np.linspace(0.0, 1.0, n_pts) ** cluster
                d1 = self.d1_vertex + t * (d1_cut - self.d1_vertex)
                g = self.geometry(d1)
        return g

    def geometry(self, d1):
        """Full ray geometry and its exact d/d(d1) derivatives."""
        n1, n2, zo, zi, L, C = self.n1, self.n2, self.zo, self.zi, self.L, self.C

        cos1 = np.clip(self._cos_th1(d1), -1.0, 1.0)
        sin1 = np.sqrt(np.maximum(1.0 - cos1**2, 0.0))
        th1 = np.arctan2(sin1, cos1)

        r = d1 * sin1
        z = zo + d1 * cos1
        d2 = (C - n1 * d1) / n2

        sin2 = r / d2
        cos2 = (zi - z) / d2
        th2 = np.arctan2(sin2, cos2)

        # -- exact derivatives with respect to the parameter d1 ---------------
        # dth1 = -dcos1/sin1 blows up on axis, but every quantity the physics
        # needs carries a compensating sin1.  Keep the regularised combinations
        # instead of dth1 itself, so r = 0 is exact rather than extrapolated.
        dcos1 = self._dcos_th1(d1)              # < 0 on the physical sheet

        Dz = cos1 + d1 * dcos1                  # dz/d(d1)          -- finite
        S = sin1**2 - d1 * cos1 * dcos1         # sin1 * dr/d(d1)   -- finite, > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            Dr = S / sin1                       # dr/d(d1)  (-> inf on axis)
            dth1 = -dcos1 / sin1

        zsp = Dz * sin1 / S                     # dz/dr -- finite, exactly 0 on axis

        # Two more regularised groups:
        G = (zi - z) * S + d1 * sin1**2 * Dz
        H = (zi - z) * Dz - d1 * S

        # ray-tube solid-angle densities, singularity-free:
        #   w1 = sin(th1) dth1/d(d1) = -dcos1
        #   w2 = sin(th2) dth2/d(d1) = d1 G / d2^3
        w1 = -dcos1
        w2 = d1 * G / d2**3
        with np.errstate(divide="ignore", invalid="ignore"):
            dth2 = w2 / sin2

        # -- local normal, angles of incidence / refraction --------------------
        # Writing  u1^.n^  and  u1^ x n^  out in full, the surface-slope terms
        # cancel analytically against each other:
        #     Q = sqrt(sin1^2 + (d1 dcos1)^2)
        #     cos i1 = -d1 dcos1 / Q ,   sin i1 = sin1 / Q
        # This removes the catastrophic cancellation that the naive
        # (-sin1*zsp + cos1)/sqrt(1+zsp^2) form suffers at grazing incidence,
        # and makes i1 -> 90 deg exact at the rim (where dcos1 = 0).
        Q = np.sqrt(sin1**2 + (d1 * dcos1) ** 2)
        nrm = np.sqrt(1.0 + zsp**2)
        cos_i1 = -d1 * dcos1 / Q
        sin_i1 = sin1 / Q
        cos_i2 = G / (d2 * Q)
        sin_i2 = sin1 * np.abs(H) / (d2 * Q)

        # -- Fresnel intensity transmittance (scalar = s-polarisation only) -----
        A1, A2 = n1 * cos_i1, n2 * cos_i2
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = 2 * A1 / (A1 + A2)
            Ts = np.where(A1 > 0, (A2 / A1) * ts**2, 0.0)

        return dict(
            d1=d1, d2=d2, r=r, z=z,
            sin1=sin1, cos1=cos1, th1=th1,
            sin2=sin2, cos2=cos2, th2=th2,
            Dr=Dr, Dz=Dz, S=S, G=G, H=H, Q=Q, w1=w1, w2=w2,
            dth1=dth1, dth2=dth2, zsp=zsp,
            cos_i1=cos_i1, cos_i2=cos_i2, sin_i1=sin_i1, sin_i2=sin_i2,
            Ts=Ts, nrm=nrm,
            # invariants for verification
            OPL=n1 * d1 + n2 * d2,
            snell=n1 * sin_i1 - n2 * sin_i2,
        )

    # -- the accurate exit-pupil apodization ----------------------------------
    def apodization(self, g, *, fresnel=True):
        """Amplitude on the exit reference sphere, from ray-tube energy balance.

            n1 |U1|^2 R1^2 sin(th1) dth1 * T  =  n2 |U2|^2 R2^2 sin(th2) dth2

        Both solid-angle densities are written as sin(th) * dth/d(d1), so the
        on-axis 0/0 never appears.
        """
        w1, w2 = g["w1"], g["w2"]
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.sqrt((self.n1 / self.n2) * w1 / w2)
        if fresnel:
            P = P * np.sqrt(g["Ts"])
        return P, w1, w2


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    N1, N2, ZO, ZI = 1.0, 1.5, -0.050, 0.025
    ov = Ovoid(N1, N2, ZO, ZI)

    print("=" * 74)
    print("EXACT CARTESIAN OVAL  (closed-form parametrisation in d1)")
    print("=" * 74)
    print(f"  n1={N1}  n2={N2}   zo={ZO*1e3:.1f} mm   zi={ZI*1e3:.1f} mm")
    print(f"  invariant OPL C   = {ov.C*1e3:.6f} mm")
    print(f"  d1 vertex -> rim  = {ov.d1_vertex*1e3:.4f} -> {ov.d1_rim*1e3:.4f} mm")

    # full oval, right up to the rim
    g = ov.sample(40001, cluster=2.0)
    print("\n-- full oval (to the geometric rim) --")
    print(f"  r_max      = {g['r'][-1]*1e3:.4f} mm")
    print(f"  sag_max    = {g['z'][-1]*1e3:.4f} mm")
    print(f"  th1_max    = {np.degrees(g['th1'][-1]):.3f} deg")
    print(f"  th2_max    = {np.degrees(g['th2'][-1]):.3f} deg")
    print(f"  NA         = {N2*np.sin(g['th2'][-1]):.4f}")
    print(f"  slope dz/dr at rim = {g['zsp'][-1]:.4f}   (cot th1 = {1/np.tan(g['th1'][-1]):.4f})")
    print(f"  angle of incidence at rim = {np.degrees(np.arccos(g['cos_i1'][-1])):.3f} deg")
    print(f"  Fresnel Ts at rim         = {g['Ts'][-1]:.6f}")

    print("\n-- exactness checks (full oval) --")
    print(f"  max |OPL - C| / C            = {np.abs(g['OPL']-ov.C).max()/ov.C:.3e}")
    print(f"  max |n1 sin i1 - n2 sin i2|  = {np.abs(g['snell']).max():.3e}   <- Snell")

    # verify dz/dr against a numerical derivative
    num = np.gradient(g["z"], g["r"])
    m = slice(10, -10)
    print(f"  max rel err  dz/dr analytic vs numeric = "
          f"{np.abs((g['zsp'][m]-num[m])/g['zsp'][m]).max():.3e}")

    print("\n-- transmission roll-off toward the rim --")
    print(f"  {'i1 [deg]':>9} {'th2 [deg]':>10} {'NA':>7} {'Ts':>9} {'r [mm]':>9}")
    for frac in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
        k = min(int(frac * (len(g["r"]) - 1)), len(g["r"]) - 1)
        print(f"  {np.degrees(np.arccos(g['cos_i1'][k])):9.2f} "
              f"{np.degrees(g['th2'][k]):10.2f} {N2*g['sin2'][k]:7.4f} "
              f"{g['Ts'][k]:9.5f} {g['r'][k]*1e3:9.4f}")

    for cut in (85.0, 80.0, 75.0, 70.0, 65.0):
        gc = ov.sample(20001, i1_max_deg=cut, cluster=2.0)
        P, _, _ = ov.apodization(gc)
        print(f"  cut i1<={cut:4.0f} deg -> NA = {N2*gc['sin2'][-1]:.4f}, "
              f"r={gc['r'][-1]*1e3:6.3f} mm, sag={gc['z'][-1]*1e3:6.3f} mm, "
              f"T_edge={gc['T'][-1]:.4f}, P_edge/P_0={P[-1]/P[0]:.4f}")
