import matplotlib.pyplot as plt
import numpy as np

from diffraction.apertures import circular_aperture
from diffraction.propagation import fresnel_output_grid, fresnel_propagator
from diffraction.surfaces import ParabolicSurface


N = 1024
L = 6e-3
wavelength = 530e-9
z_prop = 0.20

n1 = 1.0
n2 = 1.5
f = 0.12  # Parabola focal length [m]
z_vertex = 0.0

x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, y)

R0 = 0.8e-3
U_amp = circular_aperture(X, Y, R0).astype(np.complex128)

surface = ParabolicSurface(focal_length=f, z0=z_vertex)
sag = surface.sag(X, Y)
k0 = 2.0 * np.pi / wavelength
phase_mask = np.exp(1.0j * k0 * (n2 - n1) * sag)

U_after_surface = U_amp * phase_mask
x_out, y_out = fresnel_output_grid((X, Y), z=z_prop, λ=wavelength, n=n2)
Uz = fresnel_propagator(U_after_surface, (X, Y), z=z_prop, λ=wavelength, n=n2)

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
ax[0].set_title("Parabolic surface sag z(x,y) [m]")
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
    extent=[x_out.min(), x_out.max(), y_out.min(), y_out.max()],
    origin="lower",
    cmap="hot",
    vmin=-12,
    vmax=0,
)
ax[2].set_title(f"Fresnel propagated intensity (z={z_prop} m)")
ax[2].set_xlabel("x [m]")
ax[2].set_ylabel("y [m]")

plt.suptitle(f"Fresnel with parabolic-surface phase (n1={n1}, n2={n2}, f={f} m)")
plt.show()
