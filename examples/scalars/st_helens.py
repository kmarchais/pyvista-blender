# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render Mount St. Helens elevation as a Cycles scene.

Demonstrates:

* ``ImageData`` → triangulated surface (the DEM is a 2D height field).
* Point scalars (``"Elevation"``) → POINT FLOAT_COLOR attribute → shader.
* ``warp_by_scalar`` to lift the 2D field into a 3D mesh before rendering.
* ``terrain`` colormap, plumbed end-to-end (matplotlib in the bridge →
  vertex colors → Cycles shader).

Run with ``uv run python examples/st_helens.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Lift the St. Helens elevation field and render the terrain."""
    dem = pv.examples.download_st_helens().warp_by_scalar()

    plotter = pv.Plotter(off_screen=True, window_size=[1280, 960])
    plotter.add_mesh(dem, cmap="terrain", show_scalar_bar=False)
    plotter.set_background("white")
    plotter.camera_position = "iso"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_dir / "st_helens_pyvista.png"))
    plotter.blender.render(str(out_dir / "st_helens_blender.png"), samples=64)
    plotter.close()


if __name__ == "__main__":
    main()
