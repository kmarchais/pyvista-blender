# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Demonstrate every HUD overlay on a single render.

* **Scalar bar** — matplotlib colorbar matching PyVista's layout.
* **Text + title** — PIL-rendered at corner-annotation positions.
* **Corner axes triad** — XYZ arrows projected through the camera.
* **Bounds box** — wireframe bounding cube drawn from world-space
  corners projected through the perspective camera.

The mesh is PyVista's random hills with viridis scalars; the resulting
PNG is a complete scivis frame: 3D content + screen-space annotations.

Run with ``uv run python examples/hud_demo.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render a fully-annotated scivis frame."""
    hills = pv.examples.load_random_hills()

    plotter = pv.Plotter(off_screen=True, window_size=[960, 540])
    plotter.add_mesh(
        hills,
        cmap="viridis",
        show_scalar_bar=True,
        scalar_bar_args={"title": "Elevation (m)"},
    )
    plotter.add_title("PyVista to Blender HUD", font_size=18, color="white")
    plotter.add_text(
        "Random Hills",
        position="upper_left",
        font_size=14,
        color="white",
    )
    plotter.show_axes()
    plotter.show_bounds(color="#bbbbbb")
    plotter.set_background("#101820")
    plotter.camera_position = "iso"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_dir / "hud_demo_pyvista.png"))
    plotter.blender.render(str(out_dir / "hud_demo_blender.png"), samples=64)
    plotter.close()


if __name__ == "__main__":
    main()
