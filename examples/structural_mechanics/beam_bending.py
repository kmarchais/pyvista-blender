# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fedoo I-beam bending rendered as a cropped transparent VP9 movie.

Run from the repository root:

    uv run examples/structural_mechanics/beam_bending.py

The script solves a 3-point bending problem with fedoo, builds a PyVista
scene with supports and a load arrow, renders transparent Cycles frames
through pyvista-blender, crops away empty transparent pixels, and encodes
the result as VP9 WebM with alpha.
"""

# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "fedoo[plot]>=0.8.3",
#   "imageio>=2.37",
#   "imageio-ffmpeg>=0.6.0",
#   "numpy>=2",
#   "pillow>=11",
#   "pyvista>=0.48",
#   "pyvista-blender",
#   "rich>=13.9",
# ]
#
# [tool.uv.sources]
# pyvista-blender = { path = "../..", editable = true }
# ///

from __future__ import annotations

import argparse
import math
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="WARNING: no fast direct sparse solver has been found.*",
    category=UserWarning,
    module="fedoo.core.base",
)

import fedoo as fd
import imageio.v2 as imageio
import numpy as np
import pyvista as pv
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "assets" / "examples" / "structural_mechanics"
FPS = 60
FRAMES = 90
RESOLUTION = (1280, 720)
CONSOLE = Console()


def progress_bar() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=CONSOLE,
        auto_refresh=False,
    )


def solve_beam() -> tuple[
    pv.UnstructuredGrid, np.ndarray, np.ndarray, np.ndarray, dict[str, float]
]:
    profile = fd.mesh.structured_mesh.I_shape_mesh(
        height=10,
        width=10,
        web_thickness=2,
        flange_thickness=2,
        size_elm=0.5,
        elm_type="tri3",
    )
    mesh = fd.mesh.extrude(profile, 100, 41)
    mesh.nodes = mesh.nodes[:, [2, 1, 0]]

    fd.ModelingSpace("3D")
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)),
        mesh,
    )
    problem = fd.problem.Linear(assembly)
    bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)
    top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
    left = np.intersect1d(mesh.find_nodes("X", mesh.bounding_box.xmin), bottom)
    right = np.intersect1d(mesh.find_nodes("X", mesh.bounding_box.xmax), bottom)
    load = np.intersect1d(mesh.find_nodes("X", mesh.bounding_box.center[0]), top)
    problem.bc.add("Dirichlet", left, "Disp", 0)
    problem.bc.add("Dirichlet", right, "DispY", 0)
    problem.bc.add("Dirichlet", load, "DispY", -10)
    problem.solve()

    results = problem.get_results(assembly, ["Stress", "Disp"], "Node")
    stress, _ = results.get_data("Stress", "vm", "Node", True)
    displacement = results.node_data["Disp"].T
    base = mesh.to_pyvista()
    base["von_mises_mpa"] = np.zeros_like(stress)

    deformed = mesh.nodes + displacement
    stats = {
        "p99_stress": float(np.percentile(stress, 99)),
        "xmid": float((deformed[:, 0].min() + deformed[:, 0].max()) * 0.5),
        "zmid": float((mesh.bounding_box.zmin + mesh.bounding_box.zmax) * 0.5),
        "z_width": float(mesh.bounding_box.zmax - mesh.bounding_box.zmin + 7),
        "load_x0": float(mesh.nodes[load, 0].mean()),
        "load_y0": float(mesh.nodes[load, 1].max()),
        "load_z0": float(mesh.nodes[load, 2].mean()),
        "load_x1": float(deformed[load, 0].mean()),
        "load_y1": float(deformed[load, 1].max()),
        "load_z1": float(deformed[load, 2].mean()),
        "left_x0": float(mesh.nodes[left, 0].mean()),
        "left_y0": float(mesh.nodes[left, 1].min()),
        "left_x1": float(deformed[left, 0].mean()),
        "left_y1": float(deformed[left, 1].min()),
        "right_x0": float(mesh.nodes[right, 0].mean()),
        "right_y0": float(mesh.nodes[right, 1].min()),
        "right_x1": float(deformed[right, 0].mean()),
        "right_y1": float(deformed[right, 1].min()),
    }
    return base, mesh.nodes.copy(), displacement, stress, stats


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def force_factor(frame: int, frame_count: int) -> float:
    phase = frame / max(frame_count - 1, 1)
    return smoothstep(0.5 - 0.5 * math.cos(2.0 * math.pi * phase))


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def frame_mesh(
    base: pv.UnstructuredGrid,
    points: np.ndarray,
    displacement: np.ndarray,
    stress: np.ndarray,
    t: float,
) -> pv.UnstructuredGrid:
    mesh = base.copy(deep=True)
    mesh.points = points + t * displacement
    mesh["von_mises_mpa"] = t * stress
    return mesh.extract_surface(algorithm="dataset_surface").triangulate()


def support(x: float, y: float, z: float, width: float) -> list[pv.PolyData]:
    return [
        pv.Cylinder(
            center=(x, y - 1.05, z),
            direction=(0, 0, 1),
            radius=1.05,
            height=width,
            resolution=64,
        ),
        pv.Cube(
            center=(x, y - 2.32, z), x_length=11.5, y_length=0.72, z_length=width + 2.5
        ),
        pv.Cube(
            center=(x, y - 3.55, z), x_length=15.5, y_length=1.55, z_length=width + 4.5
        ),
    ]


def load_arrow(x: float, y: float, z: float) -> list[pv.PolyData]:
    return [
        pv.Cylinder(
            center=(x, y + 6.1, z),
            direction=(0, 1, 0),
            radius=0.28,
            height=8.2,
            resolution=48,
        ),
        pv.Cone(
            center=(x, y + 1.75, z),
            direction=(0, -1, 0),
            height=3.4,
            radius=1.45,
            resolution=64,
        ),
    ]


def make_plotter(mesh: pv.PolyData, stats: dict[str, float], t: float) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=RESOLUTION, lighting="none")
    plotter.add_mesh(
        mesh,
        scalars="von_mises_mpa",
        cmap="turbo",
        clim=(0, stats["p99_stress"]),
        pbr=True,
        metallic=0.03,
        roughness=0.34,
        show_scalar_bar=False,
    )
    for prefix in ("left", "right"):
        x = lerp(stats[f"{prefix}_x0"], stats[f"{prefix}_x1"], t)
        y = lerp(stats[f"{prefix}_y0"], stats[f"{prefix}_y1"], t)
        for i, part in enumerate(support(x, y, stats["zmid"], stats["z_width"])):
            plotter.add_mesh(
                part,
                name=f"{prefix}_support_{i}",
                color="#9aa3b2",
                pbr=True,
                metallic=0.2 if i == 0 else 0.0,
                roughness=0.38 if i == 0 else 0.62,
                show_scalar_bar=False,
            )
    for i, part in enumerate(
        load_arrow(
            lerp(stats["load_x0"], stats["load_x1"], t),
            lerp(stats["load_y0"], stats["load_y1"], t),
            lerp(stats["load_z0"], stats["load_z1"], t),
        )
    ):
        plotter.add_mesh(
            part,
            name=f"load_arrow_{i}",
            color="#e24d63",
            pbr=True,
            roughness=0.24,
            show_scalar_bar=False,
        )

    plotter.add_light(
        pv.Light(
            position=(34, 66, 58), focal_point=(stats["xmid"], -2.8, 0), intensity=1.4
        )
    )
    plotter.add_light(
        pv.Light(
            position=(126, 46, 42),
            focal_point=(stats["xmid"], -2.8, 0),
            color="#a8ccff",
            intensity=0.7,
        )
    )
    plotter.camera_position = [
        (stats["xmid"] + 68, 52, 58),
        (stats["xmid"], -2.8, 0),
        (0, 1, 0),
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 32
    return plotter


def alpha_bounds(paths: list[Path]) -> tuple[int, int, int, int]:
    x0 = y0 = x1 = y1 = None
    for path in paths:
        alpha = np.asarray(Image.open(path).convert("RGBA").getchannel("A"))
        ys, xs = np.where(alpha > 0)
        if xs.size == 0:
            continue
        x0 = int(xs.min()) if x0 is None else min(x0, int(xs.min()))
        y0 = int(ys.min()) if y0 is None else min(y0, int(ys.min()))
        x1 = int(xs.max()) + 1 if x1 is None else max(x1, int(xs.max()) + 1)
        y1 = int(ys.max()) + 1 if y1 is None else max(y1, int(ys.max()) + 1)
    if x0 is None or y0 is None or x1 is None or y1 is None:
        raise RuntimeError("all frames are fully transparent")
    return x0, y0, x1, y1


def crop_and_encode(
    frame_dir: Path, output: Path, fps: int, crf: int
) -> tuple[int, int]:
    frames = sorted(frame_dir.glob("frame_*.png"))
    x0, y0, x1, y1 = alpha_bounds(frames)
    first = Image.open(frames[0])
    if (x1 - x0) % 2:
        x1 = min(x1 + 1, first.width)
    if (y1 - y0) % 2:
        y0 = max(y0 - 1, 0)
    with (
        imageio.get_writer(
            output,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libvpx-vp9",
            bitrate="0",
            pixelformat="yuva420p",
            macro_block_size=1,
            ffmpeg_log_level="quiet",
            output_params=[
                "-crf",
                str(crf),
                "-auto-alt-ref",
                "0",
                "-metadata:s:v:0",
                "alpha_mode=1",
            ],
        ) as writer,
        progress_bar() as progress,
    ):
        task = progress.add_task("Cropping and encoding frames", total=len(frames))
        progress.refresh()
        for path in frames:
            rgba = Image.open(path).convert("RGBA").crop((x0, y0, x1, y1))
            writer.append_data(np.asarray(rgba))
            progress.advance(task)
            progress.refresh()
    return x1 - x0, y1 - y0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--crf", type=int, default=32)
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "beam_bending.webm")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    CONSOLE.rule("Fedoo I-beam bending")
    CONSOLE.print(
        f"[bold]Solving model[/]  engine={args.engine} frames={args.frames} fps={args.fps}"
    )
    base, points, displacement, stress, stats = solve_beam()
    with tempfile.TemporaryDirectory(prefix="beam_frames_") as tmp:
        frame_dir = Path(tmp)
        with progress_bar() as progress:
            task = progress.add_task("Rendering transparent frames", total=args.frames)
            progress.refresh()
            for frame in range(args.frames):
                t = force_factor(frame, args.frames)
                mesh = frame_mesh(base, points, displacement, stress, t)
                plotter = make_plotter(mesh, stats, t)
                path = frame_dir / f"frame_{frame:04d}.png"
                plotter.blender.render(
                    str(path),
                    engine=args.engine,
                    samples=args.samples,
                    transparent_bg=True,
                )
                plotter.close()
                progress.advance(task)
                progress.refresh()
        width, height = crop_and_encode(frame_dir, args.output, args.fps, args.crf)
    CONSOLE.print(f"[green]Video[/] {args.output}")
    CONSOLE.print(f"[green]Crop[/] {width}x{height}")
    CONSOLE.print(f"[green]Size[/] {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
