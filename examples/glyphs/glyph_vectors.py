# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render an arrow-glyph vector field via Geometry-Nodes instancing.

Uses :meth:`pl.blender.add_glyph` to host one copy of the arrow mesh
and instance it at every grid point with per-point orientation +
magnitude. For ``N = 11**3 = 1331`` points and an arrow mesh of ~50
vertices that's ~66 K vertices uploaded once, against the ~66 K *per
arrow* the baked ``mesh.glyph(...)`` route would push through Cycles
(11**3 * 50 = ~66 K total scales the same here only because there are
few points; the win grows linearly with point count).

Run with ``uv run python examples/glyph_vectors.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def main() -> None:
    """Render a swirling vector field as instanced arrows."""
    n = 11
    coords = np.linspace(-1.0, 1.0, n)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    positions = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    points = pv.PolyData(positions)

    # Swirl: rotation around Z plus a small inward component.
    points["vec"] = np.column_stack([
        -points.points[:, 1] - 0.2 * points.points[:, 0],
        points.points[:, 0] - 0.2 * points.points[:, 1],
        0.4 * np.cos(np.linalg.norm(points.points[:, :2], axis=1) * 2.5),
    ])
    points["mag"] = np.linalg.norm(points["vec"], axis=1)

    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.blender.add_glyph(
        points,
        geom=pv.Arrow(),
        orient="vec",
        scale="mag",
        factor=0.25,
    )
    plotter.set_background("#0c0f14")
    plotter.camera_position = [(3.0, -3.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.blender.render(str(out_dir / "glyph_vectors_blender.png"), samples=64)
    plotter.close()


if __name__ == "__main__":
    main()
