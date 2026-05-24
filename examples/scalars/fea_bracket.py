# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render PyVista's FEA bracket example through Blender's Cycles.

Run with ``uv run python examples/fea_bracket.py``. Produces two PNGs side
by side:

* ``fea_bracket_pyvista.png`` - VTK preview via ``pl.screenshot``.
* ``fea_bracket_blender.png`` - Cycles render via ``pl.blender.render``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Build the FEA bracket plotter scene and render it both ways."""
    bracket = pv.examples.download_fea_bracket()

    plotter = pv.Plotter(off_screen=True, window_size=[1280, 960])
    plotter.add_mesh(
        bracket,
        scalars="Equivalent (von-Mises) Stress (psi)",
        cmap="viridis",
        show_scalar_bar=False,
    )
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
    pyvista_png = out_dir / "fea_bracket_pyvista.png"
    blender_png = out_dir / "fea_bracket_blender.png"

    plotter.screenshot(str(pyvista_png))
    plotter.blender.render(str(blender_png), samples=32)
    plotter.close()


if __name__ == "__main__":
    main()
