"""Color science: turning spectra into sRGB, for polychromatic rendering.

Wavelengths become colors through the CIE 1931 standard observer. The
color-matching functions are the analytic multi-lobe Gaussian fit of Wyman,
Sloan & Shirley (*JCGT* 2013) — accurate to a small fraction of a percent and
requiring no data file. A spectrum is integrated against them to CIE XYZ
tristimulus, then mapped to sRGB (IEC 61966-2-1) with a gamut step and the
standard gamma.

Two gamut modes for out-of-sRGB-gamut spectral colors:

- ``"desaturate"`` (default): add the least white that makes every channel
  non-negative, preserving hue while pulling saturated colors toward white —
  colorimetrically faithful (this is diffractsim's ``CLIP_ADD_WHITE``).
- ``"clip"``: clamp negative channels to zero — more saturated, less faithful.

All RGB outputs are float arrays in ``[0, 1]`` with a trailing size-3 axis.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .grids import Array

__all__ = [
    "cie_xyz",
    "wavelength_to_rgb",
    "xyz_to_srgb",
    "spectrum_to_srgb",
    "d65_weights",
    "blackbody_weights",
]

# Linear-sRGB (D65) from CIE XYZ — IEC 61966-2-1.
_XYZ_TO_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)

# CIE D65 relative spectral power distribution, 380–780 nm at 10 nm.
_D65_WAVELENGTHS = np.arange(380.0, 781.0, 10.0)
_D65_SPD = np.array(
    [
        49.98, 54.65, 82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81,
        114.86, 115.92, 108.81, 109.35, 107.80, 104.79, 107.69, 104.41,
        104.05, 100.00, 96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29,
        83.70, 80.03, 80.21, 82.28, 78.28, 69.72, 71.61, 74.35, 61.60,
        69.89, 75.09, 63.59, 46.42, 66.81, 63.38,
    ]
)


def _gaussian(x: Array, mu: float, sigma_lo: float, sigma_hi: float) -> Array:
    """Piecewise (asymmetric) Gaussian used by the Wyman CMF fit."""
    sigma = np.where(x < mu, sigma_lo, sigma_hi)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def cie_xyz(wavelength_nm: Array) -> Tuple[Array, Array, Array]:
    """CIE 1931 color-matching functions ``x̄, ȳ, z̄`` at ``wavelength_nm``.

    Analytic multi-lobe Gaussian fit (Wyman, Sloan & Shirley, JCGT 2013),
    array-aware; wavelengths in nanometers.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    x = (
        1.056 * _gaussian(lam, 599.8, 37.9, 31.0)
        + 0.362 * _gaussian(lam, 442.0, 16.0, 26.7)
        - 0.065 * _gaussian(lam, 501.1, 20.4, 26.2)
    )
    y = (
        0.821 * _gaussian(lam, 568.8, 46.9, 40.5)
        + 0.286 * _gaussian(lam, 530.9, 16.3, 31.1)
    )
    z = (
        1.217 * _gaussian(lam, 437.0, 11.8, 36.0)
        + 0.681 * _gaussian(lam, 459.0, 26.0, 13.8)
    )
    return x, y, z


def _srgb_gamma(c: Array) -> Array:
    """Piecewise sRGB companding (linear → display), inputs assumed ≥ 0."""
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def xyz_to_srgb(
    XYZ: Array,
    *,
    gamut: str = "desaturate",
    gamma: bool = True,
    normalize: bool = True,
    saturation: float = 1.0,
    stretch: float = 1.0,
    brightness: float = 1.0,
) -> Array:
    """Convert CIE XYZ to display sRGB in ``[0, 1]``.

    Parameters
    ----------
    XYZ : array (..., 3)
        Tristimulus values (last axis is X, Y, Z).
    gamut : {"desaturate", "clip"}
        Out-of-gamut handling (see module docstring).
    gamma : bool
        Apply the sRGB gamma (default True).
    normalize : bool
        Scale the whole array so its brightest linear channel is 1 before the
        display transforms. Default True.
    saturation : float
        Chroma multiplier applied about each pixel's channel mean (1.0 = none;
        >1 more vivid). Default 1.0 (colorimetrically faithful).
    stretch : float
        Display-gamma exponent applied to the normalized linear intensity
        (``linear ** stretch``) before the sRGB gamma. ``<1`` compresses the
        dynamic range to reveal faint structure (spikes, outer fringes) without
        blowing out bright cores. Default 1.0.
    brightness : float
        Exposure multiplier applied after normalization; ``>1`` brightens and
        lets bright regions clip. Default 1.0.

    Notes
    -----
    The three display controls all default to the identity, so the default
    output is the physically faithful render; raise ``saturation`` / lower
    ``stretch`` / raise ``brightness`` purely for punchier visualization.
    """
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.shape[-1] != 3:
        raise ValueError("XYZ must have a trailing axis of length 3.")

    linear = XYZ @ _XYZ_TO_SRGB.T

    if gamut == "desaturate":
        # Add the least white (constant across channels) to clear negatives.
        deficit = np.clip(-linear.min(axis=-1, keepdims=True), 0.0, None)
        linear = linear + deficit
    elif gamut == "clip":
        linear = np.clip(linear, 0.0, None)
    else:
        raise ValueError("gamut must be 'desaturate' or 'clip'.")

    if normalize:
        peak = linear.max()
        if peak > 0:
            linear = linear / peak

    if brightness != 1.0:
        linear = np.clip(linear * brightness, 0.0, 1.0)
    if stretch != 1.0:
        linear = np.clip(linear, 0.0, None) ** stretch
    if saturation != 1.0:
        grey = linear.mean(axis=-1, keepdims=True)
        linear = np.clip(grey + (linear - grey) * saturation, 0.0, None)

    linear = np.clip(linear, 0.0, 1.0)
    rgb = _srgb_gamma(linear) if gamma else linear
    return np.clip(rgb, 0.0, 1.0)


def wavelength_to_rgb(wavelength_nm: Array, *, gamma: bool = True, gamut: str = "desaturate") -> Array:
    """The sRGB color of a single (monochromatic) wavelength [nm]."""
    x, y, z = cie_xyz(wavelength_nm)
    XYZ = np.stack([x, y, z], axis=-1)
    return xyz_to_srgb(XYZ, gamut=gamut, gamma=gamma, normalize=True)


def spectrum_to_srgb(
    wavelengths: Array,
    intensity_stack: Array,
    *,
    weights: Optional[Array] = None,
    gamut: str = "desaturate",
    gamma: bool = True,
    saturation: float = 1.0,
    stretch: float = 1.0,
    brightness: float = 1.0,
) -> Array:
    """Composite per-wavelength intensity maps into an sRGB image.

    ``XYZ(x, y) = Σ_k weights_k · intensity_stack[k] · cmf(λ_k) · Δλ_k``.

    Parameters
    ----------
    wavelengths : array (K,)
        Wavelengths [nm] of each slice.
    intensity_stack : array (K, H, W)
        Per-wavelength intensity maps (e.g. ``|field_λ|²`` on a shared grid).
    weights : array (K,), optional
        Spectral power of the source at each wavelength (e.g.
        :func:`d65_weights`). Defaults to flat (equal energy).
    gamut, gamma, saturation, stretch, brightness :
        Passed to :func:`xyz_to_srgb` (see there for the display controls).

    Returns
    -------
    array (H, W, 3) in ``[0, 1]``.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    stack = np.asarray(intensity_stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] != wavelengths.shape[0]:
        raise ValueError("intensity_stack must have shape (K, H, W) matching wavelengths.")

    if weights is None:
        weights = np.ones_like(wavelengths)
    else:
        weights = np.asarray(weights, dtype=float)

    # Per-slice integration weight (spectral power × bandwidth).
    if wavelengths.size > 1:
        dlam = np.gradient(wavelengths)
    else:
        dlam = np.ones_like(wavelengths)
    w = weights * dlam

    x, y, z = cie_xyz(wavelengths)  # each (K,)
    # XYZ image (H, W, 3) = Σ_k (w_k cmf_k) I_k
    cmf = np.stack([x, y, z], axis=-1) * w[:, None]  # (K, 3)
    XYZ = np.tensordot(stack, cmf, axes=([0], [0]))  # (H, W, 3)
    return xyz_to_srgb(
        XYZ, gamut=gamut, gamma=gamma, normalize=True,
        saturation=saturation, stretch=stretch, brightness=brightness,
    )


def d65_weights(wavelengths: Array) -> Array:
    """CIE D65 (daylight) relative spectral power at ``wavelengths`` [nm]."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    return np.interp(wavelengths, _D65_WAVELENGTHS, _D65_SPD, left=0.0, right=0.0)


def blackbody_weights(wavelengths: Array, temperature: float) -> Array:
    """Planck blackbody spectral radiance at ``wavelengths`` [nm], peak-normalized.

    Parameters
    ----------
    wavelengths : array
        Wavelengths [nm].
    temperature : float
        Blackbody temperature [K] (e.g. 6500 for daylight-like, 3000 for warm).
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    lam = np.asarray(wavelengths, dtype=float) * 1e-9  # m
    h = 6.62607015e-34
    c = 2.99792458e8
    kB = 1.380649e-23
    spectral = (2.0 * h * c**2) / (lam**5 * (np.exp(h * c / (lam * kB * temperature)) - 1.0))
    peak = spectral.max()
    return spectral / peak if peak > 0 else spectral
