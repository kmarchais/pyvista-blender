# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the Stanford Bunny with a metallic PBR material.

Demonstrates the material translator's PBR path: when ``pbr=True`` is
passed to ``add_mesh``, the bridge maps ``metallic`` and ``roughness``
onto the Blender Principled BSDF's matching inputs. Cycles then gives
true Fresnel reflection and view-dependent shading that the VTK Phong
shader cannot reproduce.

Run with ``uv run python examples/bunny_pbr.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render the Stanford Bunny PBR-style via PyVista and Cycles."""
    bunny = pv.examples.download_bunny()

    plotter = pv.Plotter(off_screen=True, window_size=[1280, 960])
    plotter.add_mesh(
        bunny,
        color="#dec27c",
        pbr=True,
        metallic=0.3,
        roughness=0.35,
    )
    plotter.set_background("#1a1d22")
    plotter.camera_position = [(0.2, 0.18, 0.25), (-0.02, 0.1, 0.0), (0, 1, 0)]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_dir / "bunny_pyvista.png"))
    plotter.blender.render(str(out_dir / "bunny_blender.png"), samples=64)
    plotter.close()


if __name__ == "__main__":
    main()
