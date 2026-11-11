# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Studio lighting: the default rig, against the translated VTK light kit.

A heap of spheres on a plane, where contact shadows matter. Three images:

* the PyVista preview lit by ``pl.blender.add_studio_lights()``, with
  screen-space ambient occlusion for its contact shadows;
* the Blender render with ``config.lighting = "vtk"`` (the VTK light kit
  as shadowless sun lights);
* the Blender render with the default ``config.lighting = "studio"``.

Run with ``uv run python examples/lights/studio_lighting.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

import pyvista_blender as pvb

_COLORS = ("#2a78d6", "#eb6834", "#5f9f99", "#e3cc96", "#8f7fbf")


def _scene() -> pv.Plotter:
    """Build the heap of spheres on a plane, seen from above at an angle.

    Returns
    -------
    pv.Plotter
        An off-screen plotter with the default light kit.

    """
    rng = np.random.default_rng(3)
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.set_background("white")
    plotter.add_mesh(pv.Plane(i_size=9, j_size=7), color="#d9d9d9")
    # Spheres on a jittered hexagonal patch, a second layer resting in the
    # hollows of the first.
    radius = 0.45
    for layer, z in enumerate((radius, radius * (1 + np.sqrt(8 / 3)))):
        for i in range(-3, 4 - layer):
            for j in range(-2, 3 - layer):
                x = 2 * radius * (i + 0.5 * (j % 2) + 0.5 * layer)
                y = np.sqrt(3) * radius * (j + layer / 3)
                if x * x + y * y > (3.2 - layer) ** 2:
                    continue
                centre = (x + rng.uniform(-0.04, 0.04), y + rng.uniform(-0.04, 0.04), z)
                plotter.add_mesh(
                    pv.Sphere(radius=radius, center=centre, theta_resolution=48),
                    color=_COLORS[rng.integers(len(_COLORS))],
                    smooth_shading=True,
                )
    plotter.camera_position = [(-6.5, -8.0, 6.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)]
    return plotter


def main() -> None:
    """Render the preview and both Blender lightings into the docs assets."""
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = _scene()
    pvb.config.lighting = "vtk"
    plotter.blender.render(str(out_dir / "vtk_lighting_blender.png"), samples=128)
    pvb.config.lighting = "studio"
    plotter.blender.render(str(out_dir / "studio_lighting_blender.png"), samples=128)
    plotter.blender.add_studio_lights()
    plotter.enable_ssao(radius=0.3, kernel_size=128)
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(str(out_dir / "studio_lighting_pyvista.png"))
    plotter.close()


if __name__ == "__main__":
    main()
