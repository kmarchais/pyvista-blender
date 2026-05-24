# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a 360° turntable orbit of a static scene.

Uses :meth:`pl.blender.orbit_camera` to build the per-frame updater:
only the camera moves, the mesh and material data blocks are reused
across every frame via the Level 1 identity cache. Frame-to-frame
cost is dominated by Cycles rendering itself, not by Python / bpy
bookkeeping.

The subject is PyVista's airplane mesh — asymmetric enough that each
quadrant of the orbit reads as a distinct view (the front-of-bracket
default would look essentially identical at 0° / 90° / 180° / 270°
because the bracket has 4-fold symmetry).

Run with ``uv run python examples/orbit_animation.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv

N_FRAMES = 48


def main() -> None:
    """Orbit a metallic airplane and write the gif."""
    airplane = pv.examples.load_airplane()
    center = tuple(float(c) for c in airplane.center)

    plotter = pv.Plotter(off_screen=True, window_size=[480, 360])
    plotter.add_mesh(
        airplane,
        color="#9ec5e8",
        pbr=True,
        metallic=0.7,
        roughness=0.3,
    )
    plotter.set_background("#101820")
    plotter.camera_position = [
        (center[0] + 2500.0, center[1] - 2500.0, center[2] + 1200.0),
        center,
        (0.0, 0.0, 1.0),
    ]

    orbit = plotter.blender.orbit_camera(focal_point=center, n_frames=N_FRAMES)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.blender.animate(
        str(out_dir / "orbit_airplane_blender.gif"),
        updater=orbit,
        frames=range(N_FRAMES),
        fps=24,
        samples=32,
    )
    plotter.close()


if __name__ == "__main__":
    main()
