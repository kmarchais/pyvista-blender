# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render PyVista's random hills with translucency.

Demonstrates the material translator's opacity path: ``add_mesh(opacity=0.7)``
sets ``pv.Property.opacity``, which the bridge wires into the Principled
BSDF's ``Alpha`` input. Cycles' path-tracer handles transparency natively;
no shader graph tweaking is needed at the user level.

Run with ``uv run python examples/random_hills.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render PyVista's random hills, translucent."""
    hills = pv.examples.load_random_hills()

    plotter = pv.Plotter(off_screen=True, window_size=[1280, 960])
    plotter.add_mesh(
        hills,
        cmap="plasma",
        opacity=0.7,
        show_scalar_bar=False,
    )
    plotter.set_background("#0d0d0d")
    plotter.camera_position = "iso"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_dir / "random_hills_pyvista.png"))
    plotter.blender.render(str(out_dir / "random_hills_blender.png"), samples=128)
    plotter.close()


if __name__ == "__main__":
    main()
