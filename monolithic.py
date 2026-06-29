import numpy as np
import matplotlib.pyplot as plt

exp = np.exp
pi = np.pi
π = np.pi


def FFT2(g):
    """Centered 2D FFT."""
    g_ = np.fft.ifftshift(np.fft.ifftshift(g)) 
    G_ = np.fft.fft2(g_) 
    G = np.fft.fftshift(G_)

    return G 

def IFFT2(G):
    """Centered inverse 2D FFT."""

    G_ = np.fft.ifftshift(G) 
    g_ = np.fft.ifft2(G_)
    g = np.fft.fftshift(g_) 

    return g 

def frequency_grid(grid):
    """Return centered spatial-frequency grids (fx, fy) from (x, y) sampling grids."""
    x, y = grid
    if x.shape != y.shape:
        raise ValueError("x and y grids must have the same shape.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")

    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    ny, nx = x.shape

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    fx, fy = np.meshgrid(fx, fy)
    return fx, fy


def fresnel_output_grid(grid, z, λ, n=1):
    """Return propagated-plane coordinates (x_, y_) for the single-FFT Fresnel method."""
    if z == 0:
        raise ValueError("z must be non-zero for propagation.")
    if λ <= 0:
        raise ValueError("λ must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")

    λm = λ / n
    fx, fy = frequency_grid(grid)
    x_ = λm * z * fx
    y_ = λm * z * fy
    return x_, y_


def fresnel_prop(U, grid, z, λ, n=1):
    """
    Fresnel propagation of a scalar field U over distance z.

    Parameters
    ----------
    U : 2D complex array
        Input field on the spatial grid.
    grid : tuple (x, y)
        Spatial coordinate grids from np.meshgrid.
    z : float
        Propagation distance.
    λ : float
        Wavelength in vacuum.
    n : float
        Refractive index of medium.
    """
    x, y = grid

    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    λm = λ / n
    k = 2 * π / λm
    x_, y_ = fresnel_output_grid(grid, z, λ, n=n)

    T0 = exp(1.0j * k * z) / (1.0j * λm * z)
    T1 = exp(1.0j * k * (x_**2 + y_**2) / (2.0 * z))
    T2 = exp(1.0j * k * (x**2 + y**2) / (2.0 * z))

    # dx*dy scales FFT2 into a Riemann-sum approximation of the continuous FT integral.
    U_ = T0 * T1 * (FFT2(T2 * U) * dx * dy)
    return U_


def circular_aperture(grid, R): 

    x, y = grid 

    mask = x**2 + y** 2 <= R**2

    return mask.astype(np.complex128)



if __name__ == "__main__":
    N = 2512
    L = 6e-3
    λ = 532e-9
    z = 1.15

    x1d = np.linspace(-L / 2, L / 2, N, endpoint=False)
    y1d = np.linspace(-L / 2, L / 2, N, endpoint=False)
    x, y = np.meshgrid(x1d, y1d)

    w0 = 0.4e-3
    #U0 = np.exp(-(x**2 + y**2) / (w0**2)).astype(np.complex128)
    U0 = circular_aperture((x, y), 0.3e-3) 
    Uz = fresnel_prop(U0, (x, y), z=z, λ=λ, n=1.0)

    xz, yz = fresnel_output_grid((x, y), z=z, λ=λ, n=1.0)
    xz1d = xz[0, :]
    yz1d = yz[:, 0]

    I0 = np.abs(U0) ** 2
    Iz = np.abs(Uz) ** 2
    I0 = I0 / I0.max() if I0.max() > 0 else I0
    Iz = Iz / Iz.max() if Iz.max() > 0 else Iz
    eps = 1e-12
    I0_log = np.log10(I0 + eps)
    Iz_log = np.log10(Iz + eps)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].imshow(
        I0_log,
        extent=[x1d[0], x1d[-1], y1d[0], y1d[-1]],
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
        extent=[xz1d[0], xz1d[-1], yz1d[0], yz1d[-1]],
        cmap="hot",
        origin="lower",
        vmin=-12.0,
        vmax=0.0,
    )
    ax[1].set_title(f"Propagated intensity (z={z} m)")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("y [m]")
    plt.show()


    

