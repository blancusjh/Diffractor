"""Spectrum → sRGB, for rendering broadband fields in true colour.

Small and self-contained: the CIE 1931 colour-matching functions are the
analytic multi-lobe Gaussian fit of Wyman, Sloan & Shirley, *Simple Analytic
Approximations to the CIE XYZ Color Matching Functions* (JCGT 2013), so no
data file is needed, and the XYZ → sRGB step is the standard D65 matrix with
the sRGB transfer curve.

This is a rendering utility for the examples — the physics lives in
`diffractor`, which is monochromatic by construction: a broadband field is
just many monochromatic solves sharing one output grid.
"""
import numpy as np

__all__ = ["cie_xyz_bar", "planck", "spectrum_to_srgb", "wavelength_to_rgb"]

_XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])


def _g(x, mu, s1, s2):
    t = (x - mu) * np.where(x < mu, 1.0 / s1, 1.0 / s2)
    return np.exp(-0.5 * t * t)


def cie_xyz_bar(nm):
    """x̄, ȳ, z̄ at wavelengths `nm` (nanometres)."""
    nm = np.asarray(nm, float)
    x = (1.056 * _g(nm, 599.8, 37.9, 31.0) + 0.362 * _g(nm, 442.0, 16.0, 26.7)
         - 0.065 * _g(nm, 501.1, 20.4, 26.2))
    y = 0.821 * _g(nm, 568.8, 46.9, 40.5) + 0.286 * _g(nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(nm, 437.0, 11.8, 36.0) + 0.681 * _g(nm, 459.0, 26.0, 13.8)
    return x, y, z


def planck(nm, T=6500.0):
    """Blackbody spectral radiance at temperature T, normalised to its peak."""
    lam = np.asarray(nm, float) * 1e-9
    h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    L = (2 * h * c**2 / lam**5) / np.expm1(h * c / (lam * kB * T))
    return L / L.max()


def _encode(rgb, saturation=1.0, gamma=True):
    """Linear RGB (…,3) → displayable sRGB, with add-white gamut handling."""
    rgb = np.asarray(rgb, float)
    lum = rgb.mean(-1, keepdims=True)
    rgb = lum + saturation * (rgb - lum)
    low = rgb.min(-1, keepdims=True)
    rgb = np.where(low < 0, rgb - low, rgb)            # desaturate into gamut
    rgb = np.clip(rgb, 0.0, 1.0)
    if gamma:
        rgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                       1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0)


def spectrum_to_srgb(nm, spectral_intensity, *, illuminant=None,
                     saturation=1.0, brightness=1.0, stretch=1.0, floor=0.0,
                     gamma=True):
    """Composite per-wavelength intensities to sRGB.

    `spectral_intensity` has shape (..., n_lambda).  The result is normalised so
    that the *undiffracted* illuminant renders as white — i.e. colour here means
    "how this pattern differs, spectrally, from the light that made it".

    `stretch` < 1 applies I**stretch before compositing, which lifts the dim
    outer structure into view without touching hue; `floor` then subtracts the
    stretched value of that intensity, so the background goes back to black
    instead of grey.  Both are display curves, applied identically at every
    wavelength — they change how much you can see, not what colour it is.
    """
    nm = np.asarray(nm, float)
    S = np.asarray(spectral_intensity, float)
    w = np.ones_like(nm) if illuminant is None else np.asarray(illuminant, float)
    xb, yb, zb = cie_xyz_bar(nm)

    if stretch != 1.0:
        S = np.power(np.maximum(S, 0.0), stretch)
    if floor > 0.0:
        f = floor ** stretch
        S = np.clip((S - f) / (1.0 - f), 0.0, None)

    def integrate(vals):
        XYZ = np.stack([np.trapezoid(vals * b * w, nm, axis=-1)
                        for b in (xb, yb, zb)], axis=-1)
        return XYZ @ _XYZ_TO_RGB.T

    rgb = integrate(S)
    white = integrate(np.ones_like(nm))
    rgb = rgb / white.max()
    return _encode(brightness * rgb, saturation=saturation, gamma=gamma)


def wavelength_to_rgb(nm, brightness=1.0, saturation=1.0):
    """The colour of a single spectral line (for legends and axis ticks)."""
    nm = np.atleast_1d(np.asarray(nm, float))
    xb, yb, zb = cie_xyz_bar(nm)
    rgb = np.stack([xb, yb, zb], -1) @ _XYZ_TO_RGB.T
    rgb = rgb / max(rgb.max(), 1e-12)
    return _encode(brightness * rgb, saturation=saturation)
