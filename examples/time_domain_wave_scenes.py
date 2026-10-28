"""Run functional 2-D scalar wave scenes, interactively or as real PDE movies.

    python examples/time_domain_wave_scenes.py --scene lens --interactive
    python examples/time_domain_wave_scenes.py --scene all --output wave-results

The quick preset resolves the shortest lens wavelength with 12 points. The
refined preset uses 16 points there. Exported frames advance the numerical
solver; they are not phase-rotated snapshots of a monochromatic solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

from diffractor.physics.wave_scene import double_slit_scene, lens_scene


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=("lens", "double-slit", "all"), default="all"
    )
    parser.add_argument("--quality", choices=("quick", "refined"), default="quick")
    parser.add_argument(
        "--source", choices=("continuous", "pulse"), default="continuous"
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("wave-results"))
    parser.add_argument(
        "--mode", choices=("luminous", "signed", "mean square"), default="luminous"
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--steps-per-frame", type=int, default=2)
    parser.add_argument(
        "--warmup-periods",
        type=float,
        default=None,
        help="Numerical drive periods advanced before the first exported frame.",
    )
    parser.add_argument("--no-mp4", action="store_true")
    args = parser.parse_args()
    if args.interactive and args.scene == "all":
        parser.error("Choose one scene for an interactive window.")
    if args.frames < 2 or args.fps < 1 or args.steps_per_frame < 1:
        parser.error(
            "Use at least two frames, a positive fps and positive steps per frame."
        )
    if args.warmup_periods is None:
        args.warmup_periods = 44.0 if args.source == "pulse" else 50.0
    if args.warmup_periods < 0:
        parser.error("warmup-periods must be nonnegative.")
    return args


def source_revision():
    project_root = Path(__file__).resolve().parents[1]
    code_paths = (
        "src/diffractor/physics/time_domain.py",
        "src/diffractor/physics/wave_scene.py",
        "src/diffractor/viz/wave_simulation.py",
        "examples/time_domain_wave_scenes.py",
    )
    hashes = {
        path: hashlib.sha256((project_root / path).read_bytes()).hexdigest()
        for path in code_paths
    }
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=project_root,
        ).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            text=True,
            cwd=project_root,
        )
        dirty = bool(changes.strip())
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
        dirty = None
    return {
        "git_head_sha": revision,
        "working_tree_dirty": dirty,
        "code_sha256": hashes,
        "note": (
            "Code hashes identify the executed working-tree sources; "
            "a dirty HEAD is not an exact source revision."
        ),
    }


def selected_scenes(args):
    sampling = 18 if args.quality == "quick" else 24
    factories = {"lens": lens_scene, "double-slit": double_slit_scene}
    names = factories if args.scene == "all" else (args.scene,)
    return [
        factories[name](points_per_wavelength=sampling, waveform=args.source)
        for name in names
    ]


def main():
    args = parse_arguments()
    provenance = source_revision()
    if not args.interactive:
        matplotlib.use("Agg")

    from diffractor.viz.wave_simulation import WaveViewer, export_wave_scene

    scenes = selected_scenes(args)
    if args.interactive:
        viewer = WaveViewer(
            scenes[0], steps_per_frame=args.steps_per_frame, mode=args.mode
        )
        viewer.show()
        return

    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for scene in scenes:
        print(
            f"Computing {scene.name}: {scene.grid.shape}, {args.quality} sampling.",
            flush=True,
        )
        results[scene.name] = export_wave_scene(
            scene,
            args.output / (scene.name + "_time_domain"),
            frames=args.frames,
            fps=args.fps,
            steps_per_frame=args.steps_per_frame,
            warmup_periods=args.warmup_periods,
            mode=args.mode,
            mp4=not args.no_mp4,
        )
        dx, _ = scene.grid.spacing
        results[scene.name]["minimum_points_per_local_wavelength"] = (
            scene.wavelength / (dx * scene.refractive_index.max())
        )
        print(json.dumps(results[scene.name], indent=2), flush=True)

    metadata = {
        "code_provenance": provenance,
        "quality": args.quality,
        "source": args.source,
        "display_mode": args.mode,
        "wavelength_m": scenes[0].wavelength,
        "equation": "u_tt + 2 sigma u_t = (c0/n)^2 Laplacian(u) + forcing(t)",
        "method": (
            "second-order centered finite differences; "
            "actual numerical stepping between frames"
        ),
        "boundary": (
            "fixed-zero exterior behind a 4-wavelength graded damping sponge, not a PML"
        ),
        "mean_square": (
            "online mean of u squared over all elapsed steps since reset; "
            "not EM intensity"
        ),
        "normalization": (
            "fixed display gain and reference levels; no frame-by-frame renormalization"
        ),
        "limitations": [
            (
                "Two-dimensional scalar field with stationary isotropic "
                "refractive index, not a 3-D vector optical field."
            ),
            (
                "Finite-difference dispersion and pixelated interfaces require "
                "sampling convergence checks for precision results."
            ),
            (
                "The slit wall is a reflecting fixed-zero boundary, "
                "not a perfectly absorbing aperture plate."
            ),
            (
                "The graded absorber has residual reflection; "
                "it is not a perfectly matched layer."
            ),
            (
                "GIF playback restarts its recorded clip; "
                "the last and first PDE states are not assumed equal."
            ),
            (
                "Optical femtosecond evolution is slowed for playback; "
                "compute speed depends on grid and hardware."
            ),
        ],
        "scenes": results,
    }
    (args.output / "time_domain_results.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
