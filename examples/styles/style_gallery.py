# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render three actors with the style routing flags.

* **Surface + show_edges** (left) — the default filled shading with an
  edge overlay; the edge color and line width come from the property.
* **Wireframe only** (centre) — ``style="wireframe"`` hides the fill
  surface and renders the wireframe overlay on its own.
* **Plain surface** (right) — the default, kept for comparison.

The wireframe overlay is a duplicate bpy object whose faces are
dissolved when coplanar and then run through the WIREFRAME modifier, so
triangulation diagonals on a cube don't bleed into the visible edges.

Run with ``uv run python examples/style_gallery.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render the three-style comparison."""
    plotter = pv.Plotter(off_screen=True, window_size=[960, 540])
    plotter.add_mesh(
        pv.Sphere(radius=0.9, center=(-2.5, 0.0, 0.0)),
        color="#7faedb",
        show_edges=True,
        edge_color="#1f1f1f",
        line_width=1.8,
        pbr=True,
        metallic=0.05,
        roughness=0.5,
    )
    plotter.add_mesh(
        pv.Cube(center=(0.0, 0.0, 0.0)),
        color="#e07b3b",
        style="wireframe",
        edge_color="#1f1f1f",
        line_width=3.0,
    )
    plotter.add_mesh(
        pv.Sphere(radius=0.9, center=(2.5, 0.0, 0.0)),
        color="#7faedb",
        pbr=True,
        metallic=0.05,
        roughness=0.5,
    )
    plotter.set_background("#f3eed8")
    plotter.camera_position = [(0.0, -6.5, 2.6), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_dir / "style_gallery_pyvista.png"))
    plotter.blender.render(str(out_dir / "style_gallery_blender.png"), samples=96)
    plotter.close()


if __name__ == "__main__":
    main()
