# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the same scene under three world-shader configurations.

The bridge's background path has three branches; we hit each one
from a single plotter (cache reuse confirmed by the three renders
sharing the same mesh + material data blocks):

1. **Solid colour** — the default.
2. **Vertical gradient** — ``set_background(bottom, top=...)`` drives a
   Camera-Y ColorRamp tuned to the camera's FOV so the gradient stops
   sit at the framed-view edges.
3. **Environment texture** — ``set_environment_texture(tex)`` loads a
   ``vtkTexture`` as an EXR-equivalent in-memory bpy image, used both
   for the camera-visible background *and* as IBL on the glossy sphere
   (the world is no longer hidden from glossy rays in this branch).

Run with ``uv run python examples/environment.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render the metallic sphere under three world configurations."""
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(
        pv.Sphere(theta_resolution=64, phi_resolution=64),
        color="#cccccc",
        pbr=True,
        metallic=0.95,
        roughness=0.18,
    )
    plotter.camera_position = [(3.0, -3.0, 1.6), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    plotter.set_background("#1d2236")
    plotter.blender.render(str(out_dir / "environment_solid_blender.png"), samples=96)

    plotter.set_background("#0a0a30", top="#ffaa44")
    plotter.blender.render(
        str(out_dir / "environment_gradient_blender.png"), samples=96
    )

    plotter.set_environment_texture(pv.examples.load_globe_texture())
    plotter.blender.render(str(out_dir / "environment_hdri_blender.png"), samples=128)

    plotter.close()


if __name__ == "__main__":
    main()
