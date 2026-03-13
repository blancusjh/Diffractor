import matplotlib.pyplot as plt
import numpy as np

from diffraction.apertures import circular_aperture
from diffraction.propagation import fresnel_output_grid, fresnel_propagator


N = 4024
L = 6e-3
λ = 530e-9
n = 1.0
z = 1.15

x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, y)

R0 = 0.5e-4  # Radius of the circular aperture (meters).
U0 = circular_aperture(X, Y, R0).astype(np.complex128)

x_, y_ = fresnel_output_grid((X, Y), z=z, λ=λ, n=n)
Uz = fresnel_propagator(U0, (X, Y), z=z, λ=λ, n=n)

I0 = np.abs(U0) ** 2
Iz = np.abs(Uz) ** 2
eps = 1e-16
I0_log = np.log10(I0 + eps)
Iz_log = np.log10(Iz + eps)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
ax[0].imshow(
    I0_log,
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="hot",
    origin="lower",
    vmin=-12.0,
    vmax=0.0,
)
ax[0].set_title("Input intensity")
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")

ax[1].imshow(
    Iz_log,
    extent=[x_.min(), x_.max(), y_.min(), y_.max()],
    cmap="hot",
    origin="lower",
    vmin=-12.0,
    vmax=0.0,
)
ax[1].set_title(f"Propagated intensity (z={z} m)")
ax[1].set_xlabel("x [m]")
ax[1].set_ylabel("y [m]")

plt.show()
