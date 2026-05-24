# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the shading branches side by side.

Three actors, one shading mode each:

* **Glossy Phong** (left) — ``specular=0.85, specular_power=80`` driving
  the BSDF's roughness via the Phong→GGX fit. Sharp white highlights.
* **Unlit** (centre) — ``lighting=False`` swaps the BSDF for an
  ``Emission`` shader; the surface reads at full base colour regardless
  of scene lights or shadows.
* **Two-sided** (right) — a hemisphere with a green ``backface_params``
  property; the convex outer surface stays red but the concave interior
  picks up the back-face material via ``Geometry.Backfacing``.

Run with ``uv run python examples/material_modes.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render the three shading modes."""
    plotter = pv.Plotter(off_screen=True, window_size=[960, 540])
    plotter.add_mesh(
        pv.Sphere(radius=0.9, center=(-2.5, 0.0, 0.0)),
        color="#d23030",
        specular=0.85,
        specular_power=80.0,
    )
    plotter.add_mesh(
        pv.Cube(center=(0.0, 0.0, 0.0)).scale(1.2, inplace=False),
        color="#f0c33b",
        lighting=False,
    )
    backface = pv.Property(color="#3ad26a", specular=0.3, specular_power=20.0)
    plotter.add_mesh(
        pv.Sphere(radius=0.9, center=(2.5, 0.0, 0.0), end_phi=90),
        color="#3a6ad2",
        backface_params=backface,
        specular=0.4,
        specular_power=30.0,
    )
    plotter.set_background("#1a1a20")
    # Camera below the row so the open bottom of the hemisphere is in view
    # and the green back-face property reads through the opening.
    plotter.camera_position = [(0.0, -5.0, -1.5), (0.0, 0.0, 0.2), (0.0, 0.0, 1.0)]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_dir / "material_modes_pyvista.png"))
    plotter.blender.render(str(out_dir / "material_modes_blender.png"), samples=96)
    plotter.close()


if __name__ == "__main__":
    main()
