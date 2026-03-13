import matplotlib.pyplot as plt
import numpy as np

from diffraction.apertures import circular_aperture
from diffraction.propagation import asm_propagator


def cartesian_surface_coeffs(n1: float, n2: float, zo: float, zi: float):
    G = (n2**2 - n1**2) ** 2 / (n2 * n1 * (n2 * zi - n1 * zo) * (n2 * zo - n1 * zi))
    O = (n2 * zo - n1 * zi) / (zi * zo * (n2 - n1))
    T = (n2 - n1) * (n1 + n2) ** 2 / (4 * n2 * n1 * zi * zo * (n2 * zi - n1 * zo))
    S = (n2 + n1) * (n2**2 * zi - n1**2 * zo) / (2 * n2 * n1 * zi * zo * (n2 * zi - n1 * zo))
    return G, O, T, S


def cartesian_surface_sag(X: np.ndarray, Y: np.ndarray, n1: float, n2: float, zo: float, zi: float):
    rho2 = X**2 + Y**2
    G, O, T, S = cartesian_surface_coeffs(n1, n2, zo, zi)
    radicand = 1.0 + (2.0 * S - (O**2) * G) * rho2
    radicand = np.maximum(radicand, 0.0)
    return ((O + T * rho2) * rho2) / (1.0 + S * rho2 + np.sqrt(radicand))


N = 1024
L = 6e-3
wavelength = 530e-9
z_prop = 0.20

# Cartesian surface parameters requested by user
n1 = 1.0
n2 = 1.5
zo = -100000.0
zi = 1.0

x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, y)

R0 = 0.8e-5
U_amp = circular_aperture(X, Y, R0).astype(np.complex128)

sag = cartesian_surface_sag(X, Y, n1=n1, n2=n2, zo=zo, zi=zi)
k0 = 2.0 * np.pi / wavelength
phase_mask = np.exp(1.0j * k0 * (n2 - n1) * sag)

U_after_surface = U_amp * phase_mask
Uz = asm_propagator(U_after_surface, (X, Y), z=z_prop, λ=wavelength, n=n2, include_evanescent=False)

I_in = np.abs(U_after_surface) ** 2
I_out = np.abs(Uz) ** 2
I_in_log = np.log10(I_in + 1e-16)
I_out_log = np.log10(I_out + 1e-16)

fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

im0 = ax[0].imshow(
    sag,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="viridis",
)
ax[0].set_title("Cartesian surface sag z(x,y) [m]")
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

ax[1].imshow(
    I_in_log,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="hot",
    vmin=-12,
    vmax=0,
)
ax[1].set_title("Intensity after phase mask")
ax[1].set_xlabel("x [m]")
ax[1].set_ylabel("y [m]")

ax[2].imshow(
    I_out_log,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="hot",
    vmin=-12,
    vmax=0,
)
ax[2].set_title(f"ASM propagated intensity (z={z_prop} m)")
ax[2].set_xlabel("x [m]")
ax[2].set_ylabel("y [m]")

plt.suptitle(f"ASM with Cartesian-surface phase (n1={n1}, n2={n2}, zo={zo}, zi={zi})")
plt.show()
