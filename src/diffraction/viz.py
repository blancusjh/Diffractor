"""Interactive scalar-field visualization with VisPy (optional).

GPU-rendered, pan/zoom viewers for large diffraction maps, where matplotlib's
raster path becomes sluggish:

- :func:`plot_scalar_field` — 2D image with centered metric axes, grid,
  colorbar and an optional origin crosshair.
- :func:`plot_scalar_field_3d` — the same field as a shaded 3D surface mesh.
- :func:`animate` — a z-sweep movie from a stack of intensity frames, with
  play/pause and frame-stepping.

VisPy (and a GUI backend such as PyQt6) is an optional dependency; install the
``viz`` extra. The functions import VisPy lazily and raise a clear error if it
is missing, so importing :mod:`diffraction` never requires it.

Each viewer takes a real 2D array (an intensity map or any scalar field). Pass
``log_scale=True`` to display ``log10`` of the data, or pre-transform it with
:func:`diffraction.plotting.intensity`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["plot_scalar_field", "plot_scalar_field_3d", "animate"]

_Extent = Tuple[float, float, float, float]


def _require_vispy():
    try:
        from vispy import app, scene  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "This viewer needs VisPy and a GUI backend. Install the 'viz' "
            "extra, e.g. `pip install \"diffraction[viz]\"` or "
            "`pip install vispy PyQt6`."
        ) from exc
    from vispy import app, scene
    from vispy.color import get_colormap
    from vispy.visuals.transforms import STTransform

    return app, scene, get_colormap, STTransform


def _to_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _maybe_log(a: np.ndarray, log_scale: bool) -> np.ndarray:
    return np.log10(a + 1e-6) if log_scale else a


def _norm01(a: np.ndarray) -> np.ndarray:
    m, M = float(np.min(a)), float(np.max(a))
    if M == m:
        return np.zeros_like(a, dtype=np.float32)
    return (a - m) / (M - m)


def _default_extent(shape: Tuple[int, int]) -> _Extent:
    """Symmetric, centered pixel extent when no physical extent is given."""
    ny, nx = shape
    return (-nx / 2.0, nx / 2.0, -ny / 2.0, ny / 2.0)


def plot_scalar_field(
    field: np.ndarray,
    title: Optional[str] = None,
    cmap: str = "inferno",
    *,
    log_scale: bool = False,
    xlabel: str = "x",
    ylabel: str = "y",
    extent: Optional[Sequence[float]] = None,
    show_colorbar: bool = True,
    grid_color: str = "white",
    show_center_cross: bool = False,
) -> None:
    """Interactive 2D viewer for a scalar field.

    Parameters
    ----------
    field : 2D array
        Real scalar field (e.g. an intensity map).
    title : str, optional
        Title drawn above the axes.
    cmap : str
        VisPy colormap name.
    log_scale : bool
        Display ``log10(field + 1e-6)`` instead of the raw values.
    extent : [xmin, xmax, ymin, ymax], optional
        Physical extent; if omitted, a symmetric pixel extent centers the
        origin at the image center.
    show_colorbar, show_center_cross : bool
        Toggle the right-hand colorbar and an origin crosshair.

    Opens a window and blocks until it is closed.
    """
    app, scene, _get_colormap, STTransform = _require_vispy()

    field = _to_float32(_maybe_log(field, log_scale))
    clim = (float(field.min()), float(field.max()))
    cbar_label = "log10(Intensity)" if log_scale else "Intensity"

    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
    grid = canvas.central_widget.add_grid(margin=18)

    if title:
        title_lbl = scene.Label(title, color="white")
        title_lbl.height_max = 40
        grid.add_widget(title_lbl, row=0, col=0, col_span=3)

    view = grid.add_view(row=1, col=1)
    view.camera = scene.cameras.PanZoomCamera(aspect=1)

    img = scene.visuals.Image(field, cmap=cmap, parent=view.scene, clim=clim)

    if extent is None:
        xmin, xmax, ymin, ymax = _default_extent(field.shape)
    else:
        xmin, xmax, ymin, ymax = map(float, extent)

    ny, nx = field.shape
    sx = (xmax - xmin) / float(nx)
    sy = (ymax - ymin) / float(ny)
    img.transform = STTransform(scale=(sx, sy), translate=(xmin, ymin))

    view.camera.set_range(x=(xmin, xmax), y=(ymin, ymax))
    view.camera.center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 0.0)

    scene.visuals.GridLines(color=grid_color, parent=view.scene)

    if show_center_cross and (xmin < 0 < xmax) and (ymin < 0 < ymax):
        scene.visuals.Line(
            np.array([[xmin, 0.0], [xmax, 0.0]], dtype=np.float32),
            color=(1, 1, 1, 0.6),
            parent=view.scene,
        )
        scene.visuals.Line(
            np.array([[0.0, ymin], [0.0, ymax]], dtype=np.float32),
            color=(1, 1, 1, 0.6),
            parent=view.scene,
        )

    yaxis = scene.AxisWidget(orientation="left")
    yaxis.stretch = (0.1, 1)
    xaxis = scene.AxisWidget(orientation="bottom")
    xaxis.stretch = (1, 0.1)
    grid.add_widget(yaxis, row=1, col=0)
    yaxis.link_view(view)
    grid.add_widget(xaxis, row=2, col=1)
    xaxis.link_view(view)

    ylab = scene.Label(ylabel, color="white", rotation=-90)
    ylab.width_max = 40
    xlab = scene.Label(xlabel, color="white")
    xlab.height_max = 40
    grid.add_widget(ylab, row=1, col=0)
    grid.add_widget(xlab, row=3, col=1)

    if show_colorbar:
        cbar = scene.ColorBarWidget(
            cmap=cmap,
            clim=clim,
            orientation="right",
            label=cbar_label,
            label_color="white",
            border_width=42,
        )
        cbar.stretch = (0.1, 1)
        grid.add_widget(cbar, row=1, col=2)
        cbar.label.rotation = -90

    app.run()


def plot_scalar_field_3d(
    field: np.ndarray,
    title: Optional[str] = None,
    cmap: str = "inferno",
    *,
    log_scale: bool = False,
    extent: Optional[Sequence[float]] = None,
    show_colorbar: bool = True,
    subsample: Optional[int] = None,
    elevation: float = 25.0,
    azimuth: float = -60.0,
    shading: str = "smooth",
) -> None:
    """Interactive 3D surface view of a scalar field.

    The field is drawn as a shaded triangle mesh (height = value). Large maps
    are automatically decimated to keep the mesh interactive; override with
    ``subsample`` (stride in pixels). Opens a window and blocks until closed.
    """
    app, scene, get_colormap, _STTransform = _require_vispy()

    data = _to_float32(_maybe_log(field, log_scale))
    clim = (float(data.min()), float(data.max()))
    cbar_label = "log10(Intensity)" if log_scale else "Intensity"

    ny, nx = data.shape
    if subsample is None:
        target = 400
        subsample = max(1, int(np.ceil(max(nx, ny) / target)))
    data = data[::subsample, ::subsample]
    ny, nx = data.shape

    if extent is None:
        xmin, xmax, ymin, ymax = _default_extent((ny, nx))
    else:
        xmin, xmax, ymin, ymax = map(float, extent)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    Z = data.astype(np.float32)

    verts = np.c_[X.ravel(), Y.ravel(), Z.ravel()].astype(np.float32)
    # Two triangles per grid cell, vectorized over the (ny-1)×(nx-1) cells.
    r = np.arange(ny - 1)[:, None] * nx
    c = np.arange(nx - 1)[None, :]
    a = (r + c).ravel()
    b = a + 1
    cc = a + nx
    d = cc + 1
    faces = np.empty((a.size * 2, 3), dtype=np.uint32)
    faces[0::2] = np.column_stack((a, b, d))
    faces[1::2] = np.column_stack((a, d, cc))

    colors = get_colormap(cmap).map(_norm01(Z).ravel()).astype(np.float32)

    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
    grid = canvas.central_widget.add_grid(margin=18)

    if title:
        title_lbl = scene.Label(title, color="white")
        title_lbl.height_max = 40
        grid.add_widget(title_lbl, row=0, col=0, col_span=(3 if show_colorbar else 1))

    if show_colorbar:
        left_pad = scene.Widget()
        left_pad.width_min = left_pad.width_max = 60
        left_pad.stretch = (1, 1)
        grid.add_widget(left_pad, row=1, col=0)

        view = grid.add_view(row=1, col=1)
        view.stretch = (1, 1)

        cbar = scene.ColorBarWidget(
            cmap=cmap,
            clim=clim,
            orientation="right",
            label=cbar_label,
            label_color="white",
            border_width=6,
        )
        cbar.width_min = cbar.width_max = 60
        cbar.stretch = (1, 1)
        grid.add_widget(cbar, row=1, col=2)
        cbar.label.rotation = -90
    else:
        view = grid.add_view(row=1, col=0)
        view.stretch = (1, 1)

    view.camera = scene.cameras.TurntableCamera(
        elevation=elevation, azimuth=azimuth, up="+z", fov=45
    )

    scene.visuals.Mesh(
        vertices=verts,
        faces=faces,
        vertex_colors=colors,
        shading=shading,
        parent=view.scene,
    )
    scene.visuals.XYZAxis(parent=view.scene)

    zmin, zmax = float(Z.min()), float(Z.max())
    view.camera.set_range(x=(xmin, xmax), y=(ymin, ymax), z=(zmin, zmax))
    view.camera.center = (
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0,
    )

    app.run()


def animate(
    frames: Sequence[np.ndarray],
    z_distances: Sequence[float],
    title: str = "Diffraction z-sweep",
    *,
    extent: Optional[Sequence[float]] = None,
    xlabel: str = "x [mm]",
    ylabel: str = "y [mm]",
    cmap: str = "hot",
    log_scale: bool = True,
    fps: int = 30,
    loop: bool = True,
    show_colorbar: bool = True,
    show_center_cross: bool = True,
    grid_color: str = "white",
    annotation: str = "",
) -> None:
    """Animate a stack of intensity frames over propagation distance.

    Parameters
    ----------
    frames : sequence of 2D arrays
        Intensity (or log-intensity) maps, one per plane.
    z_distances : sequence of float
        Propagation distance [m] of each frame; the current value is shown in
        the title. Must match ``frames`` in length.
    log_scale : bool
        Whether the frames are already log-scaled (affects the colorbar label
        only; the frames are displayed as given).
    fps, loop : int, bool
        Playback rate and whether to restart at the end.
    annotation : str
        Extra tag appended to the title (e.g. ``"[GPU]"``).

    Controls: SPACE play/pause, R reset, LEFT/RIGHT step, mouse pan/zoom.
    Opens a window and blocks until closed.
    """
    app, scene, _get_colormap, STTransform = _require_vispy()

    n_frames = len(frames)
    if len(z_distances) != n_frames:
        raise ValueError(
            f"frames and z_distances must have the same length "
            f"({n_frames} != {len(z_distances)})."
        )
    if n_frames == 0:
        raise ValueError("frames is empty.")

    frames = [f.astype(np.float32) for f in frames]
    all_data = np.concatenate([f.ravel() for f in frames])
    clim = (float(all_data.min()), float(all_data.max()))

    if extent is None:
        ny, nx = frames[0].shape
        extent = [-nx / 2, nx / 2, -ny / 2, ny / 2]
    xmin, xmax, ymin, ymax = map(float, extent)

    def _title(idx: int) -> str:
        text = f"{title} | z = {z_distances[idx] * 1e3:.2f} mm"
        return f"{text} {annotation}" if annotation else text

    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black", title=title)
    grid = canvas.central_widget.add_grid(margin=18)

    title_lbl = scene.Label(_title(0), color="white", font_size=14)
    title_lbl.height_max = 40
    grid.add_widget(title_lbl, row=0, col=0, col_span=3)

    view = grid.add_view(row=1, col=1)
    view.camera = scene.cameras.PanZoomCamera(aspect=1)

    img = scene.visuals.Image(frames[0], cmap=cmap, parent=view.scene, clim=clim)
    ny, nx = frames[0].shape
    sx = (xmax - xmin) / float(nx)
    sy = (ymax - ymin) / float(ny)
    img.transform = STTransform(scale=(sx, sy), translate=(xmin, ymin))

    view.camera.set_range(x=(xmin, xmax), y=(ymin, ymax))
    view.camera.center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 0.0)

    scene.visuals.GridLines(color=grid_color, parent=view.scene)

    if show_center_cross and (xmin < 0 < xmax) and (ymin < 0 < ymax):
        scene.visuals.Line(
            np.array([[xmin, 0.0], [xmax, 0.0]], dtype=np.float32),
            color=(1, 1, 1, 0.6),
            parent=view.scene,
        )
        scene.visuals.Line(
            np.array([[0.0, ymin], [0.0, ymax]], dtype=np.float32),
            color=(1, 1, 1, 0.6),
            parent=view.scene,
        )

    yaxis = scene.AxisWidget(orientation="left")
    yaxis.stretch = (0.1, 1)
    xaxis = scene.AxisWidget(orientation="bottom")
    xaxis.stretch = (1, 0.1)
    grid.add_widget(yaxis, row=1, col=0)
    yaxis.link_view(view)
    grid.add_widget(xaxis, row=2, col=1)
    xaxis.link_view(view)

    ylab = scene.Label(ylabel, color="white", rotation=-90)
    ylab.width_max = 40
    xlab = scene.Label(xlabel, color="white")
    xlab.height_max = 40
    grid.add_widget(ylab, row=1, col=0)
    grid.add_widget(xlab, row=3, col=1)

    if show_colorbar:
        cbar = scene.ColorBarWidget(
            cmap=cmap,
            clim=clim,
            orientation="right",
            label="log10(Intensity)" if log_scale else "Intensity",
            label_color="white",
            border_width=42,
        )
        cbar.stretch = (0.1, 1)
        grid.add_widget(cbar, row=1, col=2)
        cbar.label.rotation = -90

    state = {"idx": 0, "playing": True, "forward": True}

    def show(idx: int) -> None:
        img.set_data(frames[idx])
        title_lbl.text = _title(idx)
        canvas.update()

    def on_timer(event):  # noqa: ARG001
        if not state["playing"]:
            return
        if state["forward"]:
            state["idx"] += 1
            if state["idx"] >= n_frames:
                if loop:
                    state["idx"] = 0
                else:
                    state["idx"] = n_frames - 1
                    state["forward"] = False
        else:
            state["idx"] -= 1
            if state["idx"] < 0:
                state["idx"] = 0
                state["forward"] = True
        show(state["idx"])

    @canvas.events.key_press.connect
    def on_key_press(event):
        if event.key == " ":
            state["playing"] = not state["playing"]
        elif event.key == "r":
            state["idx"] = 0
            state["forward"] = True
            show(0)
        elif event.key == "Right":
            state["idx"] = min(state["idx"] + 1, n_frames - 1)
            show(state["idx"])
        elif event.key == "Left":
            state["idx"] = max(state["idx"] - 1, 0)
            show(state["idx"])

    timer = app.Timer(interval=1.0 / fps, connect=on_timer, start=True)
    _keepalive.append(timer)  # keep the timer from being garbage-collected

    app.run()


# app.Timer is only weakly held by VisPy; keep a strong reference alive.
_keepalive: List[object] = []
