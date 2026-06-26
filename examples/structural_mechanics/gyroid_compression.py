# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fedoo gyroid compression rendered with a material-to-stress overlay.

Run from the repository root:

    uv run examples/structural_mechanics/gyroid_compression.py

The remeshed gyroid is stored as an example asset until the microgen
generation path is stable enough to keep directly in this script.
"""

# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "fedoo[plot]>=0.8.3",
#   "imageio>=2.37",
#   "imageio-ffmpeg>=0.6.0",
#   "matplotlib>=3.10",
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
warnings.filterwarnings(
    "ignore",
    message="'Material.use_nodes' is expected to be removed in Blender 6.0",
    category=DeprecationWarning,
)

import fedoo as fd
import imageio.v2 as imageio
import matplotlib as mpl
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
MESH_PATH = Path(__file__).with_name("assets") / "gyroid_microgen_remeshed_tet.vtu"
FPS = 60
FRAMES = 90
RESOLUTION = (1920, 1080)
ALPHA_CMAP = "turbo_stress_alpha"
PLATEN_THICKNESS = 0.07
AUTO_SMOOTH_ANGLE_DEGREES = 60
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


def solve_compression() -> tuple[
    pv.UnstructuredGrid, np.ndarray, np.ndarray, np.ndarray, dict[str, float]
]:
    if not MESH_PATH.exists():
        raise FileNotFoundError(f"missing gyroid mesh asset: {MESH_PATH}")
    mesh_pv = pv.read(MESH_PATH).connectivity(extraction_mode="largest")
    mesh_pv.clear_data()
    mesh = fd.Mesh.from_pyvista(mesh_pv)

    fd.ModelingSpace("3D")
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1200.0, 0.34)),
        mesh,
    )
    problem = fd.problem.Linear(assembly)
    height = mesh.bounding_box.zmax - mesh.bounding_box.zmin
    imposed_z = -0.08 * height
    problem.bc.add("Dirichlet", mesh.find_nodes("Z", mesh.bounding_box.zmin), "Disp", 0)
    problem.bc.add(
        "Dirichlet", mesh.find_nodes("Z", mesh.bounding_box.zmax), "DispZ", imposed_z
    )
    problem.solve()

    results = problem.get_results(assembly, ["Stress", "Disp"], "Node")
    stress, _ = results.get_data("Stress", "vm", "Node", True)
    displacement = results.node_data["Disp"].T
    base = mesh.to_pyvista()
    base["stress_mpa"] = np.zeros_like(stress)
    stats = {
        "p99_stress": float(np.percentile(stress, 99)),
        "imposed_z": float(imposed_z),
        "zmin": float(mesh.bounding_box.zmin),
        "zmax": float(mesh.bounding_box.zmax),
    }
    return base, mesh.nodes.copy(), displacement, stress, stats


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def force_factor(frame: int, frame_count: int) -> float:
    phase = frame / max(frame_count - 1, 1)
    return smoothstep(0.5 - 0.5 * math.cos(2.0 * math.pi * phase))


def register_alpha_colormap() -> None:
    if ALPHA_CMAP in mpl.colormaps:
        return
    values = np.linspace(0, 1, 256)
    colors = np.asarray(mpl.colormaps["turbo"](values), dtype=np.float32)
    colors[:, 3] = values**0.72
    mpl.colormaps.register(mpl.colors.ListedColormap(colors, name=ALPHA_CMAP))


def surface(grid: pv.UnstructuredGrid, offset: float = 0.0) -> pv.PolyData:
    surf = grid.extract_surface(algorithm="dataset_surface").triangulate()
    if offset == 0:
        return surf
    surf = surf.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
    )
    surf.points = surf.points + offset * np.asarray(surf.point_data["Normals"])
    return surf


def frame_grid(
    base: pv.UnstructuredGrid,
    points: np.ndarray,
    displacement: np.ndarray,
    stress: np.ndarray,
    t: float,
) -> pv.UnstructuredGrid:
    grid = base.copy(deep=True)
    grid.points = points + t * displacement
    grid["stress_mpa"] = t * stress
    return grid


def platen(z: float) -> pv.PolyData:
    return pv.Cube(
        center=(0, 0, z), x_length=1.18, y_length=1.18, z_length=PLATEN_THICKNESS
    ).triangulate()


def make_plotter(
    grid: pv.UnstructuredGrid, stats: dict[str, float], t: float
) -> pv.Plotter:
    register_alpha_colormap()
    plotter = pv.Plotter(off_screen=True, window_size=RESOLUTION, lighting="none")
    plotter.add_mesh(
        surface(grid),
        name="gyroid_real_material",
        color="#a8a59d",
        pbr=True,
        metallic=0.72,
        roughness=0.36,
    )
    plotter.add_mesh(
        surface(grid, 0.0025),
        name="gyroid_stress_overlay",
        scalars="stress_mpa",
        cmap=ALPHA_CMAP,
        clim=(0, stats["p99_stress"]),
        lighting=False,
        show_scalar_bar=False,
    )
    for name, z in (
        ("bottom_platen", stats["zmin"] - PLATEN_THICKNESS * 0.5),
        (
            "top_platen",
            stats["zmax"] + PLATEN_THICKNESS * 0.5 + t * stats["imposed_z"],
        ),
    ):
        plotter.add_mesh(
            platen(z),
            name=name,
            color="#7f8791",
            pbr=True,
            metallic=0.62,
            roughness=0.38,
        )
    plotter.camera_position = [(2.12, -2.46, 1.42), (0, 0, 0.01), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 3.85
    return plotter


def set_socket(node: object, names: tuple[str, ...], value: object) -> None:
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return


def render_blender_frame(path: Path, samples: int, engine: str) -> None:
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    import bpy  # noqa: PLC0415
    from pyvista_blender.translate.camera import look_at_matrix  # noqa: PLC0415

    scene = bpy.context.scene
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 8
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = samples
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 1.0

    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    for name, location, energy, size in (
        ("softbox_key", (2.4, -3.1, 3.0), 650, 2.4),
        ("cool_rim", (-2.5, 2.4, 1.9), 320, 1.5),
        ("front_fill", (0.2, -1.8, 2.7), 110, 3.0),
        ("metal_strip", (0.4, -0.7, 3.6), 160, 0.55),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.size = size

    scene.camera.matrix_world = look_at_matrix(
        (2.12, -2.46, 1.42), (0, 0, 0.01), (0, 0, 1)
    )
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 3.85

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        lname = obj.name.lower()
        if "gyroid_stress_overlay" in lname:
            material = bpy.data.materials.new("stress_alpha_overlay")
            material.use_nodes = True
            nodes = material.node_tree.nodes
            for node in list(nodes):
                if node.bl_idname != "ShaderNodeOutputMaterial":
                    nodes.remove(node)
            output = nodes.get("Material Output")
            attr = nodes.new("ShaderNodeAttribute")
            attr.attribute_name = "scalars"
            transparent = nodes.new("ShaderNodeBsdfTransparent")
            emission = nodes.new("ShaderNodeEmission")
            mix = nodes.new("ShaderNodeMixShader")
            links = material.node_tree.links
            links.new(attr.outputs["Color"], emission.inputs["Color"])
            links.new(
                attr.outputs.get("Alpha") or attr.outputs.get("Fac"), mix.inputs["Fac"]
            )
            links.new(transparent.outputs["BSDF"], mix.inputs[1])
            links.new(emission.outputs["Emission"], mix.inputs[2])
            links.new(mix.outputs["Shader"], output.inputs["Surface"])
            material.surface_render_method = "BLENDED"
            obj.data.materials.clear()
            obj.data.materials.append(material)
        elif "gyroid_real_material" in lname or "platen" in lname:
            material = obj.active_material
            if material is not None:
                material.use_nodes = True
                bsdf = material.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None and "platen" in lname:
                    bsdf.inputs["Base Color"].default_value = (0.34, 0.36, 0.39, 1)
                    bsdf.inputs["Roughness"].default_value = 0.40
                    set_socket(bsdf, ("Metallic",), 0.72)
        if "gyroid_real_material" in lname or "platen" in lname:
            bpy.ops.object.select_all(action="DESELECT")
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.shade_auto_smooth(
                angle=math.radians(AUTO_SMOOTH_ANGLE_DEGREES)
            )
            obj.select_set(False)

    scene.render.filepath = str(path)
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            bpy.ops.render.render(write_still=True)
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def alpha_bounds(paths: list[Path]) -> tuple[int, int, int, int]:
    x0 = y0 = x1 = y1 = None
    for path in paths:
        alpha = np.asarray(Image.open(path).convert("RGBA").getchannel("A"))
        ys, xs = np.where(alpha > 0)
        if xs.size:
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
    if (x1 - x0) % 2:
        x1 += 1
    if (y1 - y0) % 2:
        y1 += 1
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
    parser.add_argument(
        "--output", type=Path, default=OUT_DIR / "gyroid_compression.webm"
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    CONSOLE.rule("Fedoo gyroid compression")
    CONSOLE.print(
        f"[bold]Solving model[/]  engine={args.engine} frames={args.frames} fps={args.fps}"
    )
    CONSOLE.print(f"[dim]Mesh[/] {MESH_PATH}")
    base, points, displacement, stress, stats = solve_compression()
    with tempfile.TemporaryDirectory(prefix="gyroid_frames_") as tmp:
        frame_dir = Path(tmp)
        scene_path = frame_dir / "scene.blend"
        with progress_bar() as progress:
            task = progress.add_task("Rendering transparent frames", total=args.frames)
            progress.refresh()
            for frame in range(args.frames):
                t = force_factor(frame, args.frames)
                plotter = make_plotter(
                    frame_grid(base, points, displacement, stress, t), stats, t
                )
                path = frame_dir / f"frame_{frame:04d}.png"
                plotter.blender.export_blend(str(scene_path))
                render_blender_frame(path, args.samples, args.engine)
                plotter.close()
                progress.advance(task)
                progress.refresh()
        width, height = crop_and_encode(frame_dir, args.output, args.fps, args.crf)
    CONSOLE.print(f"[green]Video[/] {args.output}")
    CONSOLE.print(f"[green]Crop[/] {width}x{height}")
    CONSOLE.print(f"[green]Size[/] {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
