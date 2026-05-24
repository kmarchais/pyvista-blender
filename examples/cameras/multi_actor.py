# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render two PyVista actors in the same scene.

Demonstrates the scene orchestrator's loop over ``renderer.actors``: each
visible :class:`pyvista.Actor` becomes a separate Blender mesh + material,
so a sphere with shiny chrome-like settings and a matte blue cube coexist
faithfully in one render.

Run with ``uv run python examples/multi_actor.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render a sphere and a cube together via PyVista and Cycles."""
    sphere = pv.Sphere(radius=1.0, center=(-1.2, 0.0, 0.0))
    cube = pv.Cube(center=(1.2, 0.0, 0.0))

    plotter = pv.Plotter(off_screen=True, window_size=[1280, 960])
    plotter.add_mesh(sphere, color="#e63946", pbr=True, metallic=0.8, roughness=0.2)
    plotter.add_mesh(cube, color="#457b9d", pbr=True, metallic=0.05, roughness=0.6)
    plotter.set_background("#f1faee")
    plotter.camera_position = [(4.0, -3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_dir / "multi_actor_pyvista.png"))
    plotter.blender.render(str(out_dir / "multi_actor_blender.png"), samples=64)
    plotter.close()


if __name__ == "__main__":
    main()
