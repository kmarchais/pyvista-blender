# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render three orthographic views of the same model from one plotter.

Demonstrates two features at once:

* **Orthographic projection.** Setting ``plotter.enable_parallel_projection()``
  feeds PyVista's ``parallel_scale`` into Blender's ``ortho_scale`` via
  ``translate_camera``. Lines stay parallel, depth is preserved without
  perspective foreshortening — the look engineers expect for technical
  drawings.
* **Level 1 identity-keyed caching.** Three consecutive ``pl.blender.render``
  calls reuse the cached bpy mesh and material data blocks; only the
  camera and lights are rebuilt between calls.

Run with ``uv run python examples/ortho_multi_view.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv

Vec3 = tuple[float, float, float]
View = tuple[str, Vec3, Vec3]


def _views(center: Vec3, offset: float) -> tuple[View, ...]:
    """Build front / side / top camera positions around ``center``.

    Returns
    -------
    tuple of (name, position, up)
        Three views, each a ``(name, eye, up)`` triple. The focal point is
        always ``center``; the eye sits ``offset`` units away along ±x / ±y / +z.

    """
    cx, cy, cz = center
    return (
        ("front", (cx, cy - offset, cz), (0.0, 0.0, 1.0)),
        ("side", (cx + offset, cy, cz), (0.0, 0.0, 1.0)),
        ("top", (cx, cy, cz + offset), (0.0, 1.0, 0.0)),
    )


def main() -> None:
    """Render PyVista's airplane in three orthographic views."""
    airplane = pv.examples.load_airplane()
    center = tuple(float(c) for c in airplane.center)

    plotter = pv.Plotter(off_screen=True, window_size=[640, 480])
    plotter.add_mesh(
        airplane,
        color="#9ec5e8",
        pbr=True,
        metallic=0.85,
        roughness=0.3,
    )
    plotter.set_background("#f4efe6")
    plotter.enable_parallel_projection()

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for view_name, eye, up in _views(center, offset=2500.0):
        plotter.camera_position = [eye, center, up]
        plotter.camera.parallel_scale = 800.0
        plotter.blender.render(
            str(out_dir / f"ortho_{view_name}_blender.png"), samples=64
        )

    plotter.close()


if __name__ == "__main__":
    main()
