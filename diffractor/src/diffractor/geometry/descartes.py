"""Ovals of Descartes — the meridian curve, as pure geometry.

The oval is the classical curve  n1 d1 + n2 d2 = C  with d1, d2 the distances
to two fixed foci A = (0, zo) and A' = (0, zi).  Here it is parametrised by the
focal ray length d1, which makes every quantity closed-form and smooth (the
sag-solver alternative — Newton in z at fixed r — has a double root at the rim
and dies long before the true edge of the curve).

This module knows NOTHING about media, Fresnel factors or fields.  n1, n2
enter only as the weights that define the curve family; the optical reading
of the same numbers lives in :mod:`diffractor.optics`.

Everything returned by :meth:`DescartesOvoid.geometry` is exact:

* points (r, z), slope dz/dr, principal-angle data,
* the local frame angles cos/sin i1, i2 against the curve normal, computed in
  the cancellation-free cross-product form (exact at grazing),
* solid-angle densities w1 = sin(th1) dth1/d(d1), w2 = sin(th2) dth2/d(d1)
  written without any 0/0 on the axis,
* the invariant OPL = n1 d1 + n2 d2 (constant by construction — a residual
  check, not an assumption).
"""

from __future__ import annotations

import numpy as np

__all__ = ["DescartesOvoid"]


class DescartesOvoid:
    """The oval  n1|XA| + n2|XA'| = n1|zo| + n2 zi,  parametrised by d1."""

    def __init__(self, n1: float, n2: float, zo: float, zi: float):
        if not (zo < 0 < zi):
            raise ValueError("foci must straddle the origin: zo < 0 < zi")
        if n1 == n2:
            raise ValueError("n1 and n2 must differ")
        self.n1, self.n2, self.zo, self.zi = n1, n2, zo, zi
        self.L = zi - zo
        self.C = n1 * abs(zo) + n2 * zi
        self.a = n1**2 - n2**2
        self.c = self.C**2 - n2**2 * self.L**2
        self.d1_vertex = abs(zo)
        self.d1_rim = np.sqrt(self.c / self.a)     # double root: the true rim

    # -- parametrisation -------------------------------------------------------
    def _cos_th1(self, d1):
        return (-self.a * d1 + 2 * self.C * self.n1 - self.c / d1) \
            / (2 * self.n2**2 * self.L)

    def _dcos_th1(self, d1):
        return (-self.a + self.c / d1**2) / (2 * self.n2**2 * self.L)

    def sample(self, n_pts=20001, *, i1_max_deg=None, rim_frac=1.0, cluster=2.0):
        """Sample vertex→rim, optionally truncated at an incidence angle."""
        t = np.linspace(0.0, 1.0, n_pts) ** cluster
        d1 = self.d1_vertex + t * rim_frac * (self.d1_rim - self.d1_vertex)
        g = self.geometry(d1)
        if i1_max_deg is not None:
            keep = np.degrees(np.arccos(np.clip(g["cos_i1"], -1, 1))) <= i1_max_deg
            if not keep.all():
                d1_cut = d1[keep][-1]
                t = np.linspace(0.0, 1.0, n_pts) ** cluster
                d1 = self.d1_vertex + t * (d1_cut - self.d1_vertex)
                g = self.geometry(d1)
        return g

    def geometry(self, d1):
        n1, n2, zo, zi, C = self.n1, self.n2, self.zo, self.zi, self.C

        cos1 = np.clip(self._cos_th1(d1), -1.0, 1.0)
        sin1 = np.sqrt(np.maximum(1.0 - cos1**2, 0.0))
        th1 = np.arctan2(sin1, cos1)

        r = d1 * sin1
        z = zo + d1 * cos1
        d2 = (C - n1 * d1) / n2
        sin2 = r / d2
        cos2 = (zi - z) / d2
        th2 = np.arctan2(sin2, cos2)

        # exact derivatives w.r.t. the parameter d1, regularised on axis
        dcos1 = self._dcos_th1(d1)
        Dz = cos1 + d1 * dcos1
        S = sin1**2 - d1 * cos1 * dcos1
        with np.errstate(divide="ignore", invalid="ignore"):
            Dr = S / sin1
            dth1 = -dcos1 / sin1
        zsp = Dz * sin1 / S

        G = (zi - z) * S + d1 * sin1**2 * Dz
        H = (zi - z) * Dz - d1 * S
        w1 = -dcos1
        w2 = d1 * G / d2**3
        with np.errstate(divide="ignore", invalid="ignore"):
            dth2 = w2 / sin2

        # local frame angles, cancellation-free (exact at grazing incidence)
        Q = np.sqrt(sin1**2 + (d1 * dcos1) ** 2)
        nrm = np.sqrt(1.0 + zsp**2)
        cos_i1 = -d1 * dcos1 / Q
        sin_i1 = sin1 / Q
        cos_i2 = G / (d2 * Q)
        sin_i2 = sin1 * np.abs(H) / (d2 * Q)

        return dict(
            d1=d1, d2=d2, r=r, z=z,
            sin1=sin1, cos1=cos1, th1=th1, sin2=sin2, cos2=cos2, th2=th2,
            Dr=Dr, Dz=Dz, S=S, G=G, H=H, Q=Q, w1=w1, w2=w2,
            dth1=dth1, dth2=dth2, zsp=zsp, nrm=nrm,
            cos_i1=cos_i1, cos_i2=cos_i2, sin_i1=sin_i1, sin_i2=sin_i2,
            OPL=n1 * d1 + n2 * d2,
            snell=n1 * sin_i1 - n2 * sin_i2,
        )
