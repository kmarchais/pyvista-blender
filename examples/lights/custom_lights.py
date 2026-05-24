# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replace the default light kit with a custom SUN + POINT + SPOT rig.

Exercises every branch of ``translate/light.py``:

* **SUN**  — directional, ``positional=False``. Acts as a soft fill from
  above.
* **POINT** — positional with ``cone_angle = 90``, falls back to omni
  emission. Provides a warm rim light from the side.
* **SPOT**  — positional with ``cone_angle = 25``, gets translated as
  ``bpy.types.Light(type='SPOT', spot_size = 2 * cone_angle)``. Casts a
  tight focused beam onto the model.

Run with ``uv run python examples/custom_lights.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render a glossy torus under a custom 3-light stage rig."""
    torus = pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.35)

    plotter = pv.Plotter(off_screen=True, window_size=[800, 600], lighting="none")
    plotter.add_mesh(
        torus,
        color="#e0c2a6",
        pbr=True,
        metallic=0.2,
        roughness=0.35,
    )
    plotter.set_background("#0c0f14")
    plotter.camera_position = [(3.2, -3.2, 2.6), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    # Soft directional fill from above-back-left. SUN in Blender.
    sun = pv.Light(
        position=(-2.0, -2.5, 4.0),
        focal_point=(0.0, 0.0, 0.0),
        color="#a0c8ff",
        intensity=0.45,
        light_type="scene light",
    )
    sun.positional = False
    plotter.add_light(sun)

    # Warm omnidirectional key from the right. POINT in Blender.
    point = pv.Light(
        position=(3.0, 0.5, 1.5),
        focal_point=(0.0, 0.0, 0.0),
        color="#ffb070",
        intensity=0.9,
        light_type="scene light",
    )
    point.positional = True
    point.cone_angle = 90.0
    plotter.add_light(point)

    # Tight focused beam from below-front, picking out the inner ring.
    # cone_angle < 90 maps to bpy SPOT with spot_size = 2 * cone_angle.
    spot = pv.Light(
        position=(0.0, -3.2, -1.5),
        focal_point=(0.0, 0.0, 0.4),
        color="#ffffff",
        intensity=1.6,
        light_type="scene light",
    )
    spot.positional = True
    spot.cone_angle = 18.0
    plotter.add_light(spot)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_dir / "custom_lights_pyvista.png"))
    plotter.blender.render(str(out_dir / "custom_lights_blender.png"), samples=128)
    plotter.close()


if __name__ == "__main__":
    main()
