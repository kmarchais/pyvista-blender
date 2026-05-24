# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render cell-level scalars in the per-face FEA / CFD look.

When the active scalar lives on ``cell_data`` (not ``point_data``), the
mesh translator writes a CORNER-domain ``FLOAT_COLOR`` attribute so
each face shows a single flat colour — the look engineers expect for
finite-element results where each element is its own measurement, not
something to smooth across.

The scene is a hexahedral beam (PyVista's ``load_hexbeam``) with one
scalar value per cell. Compare against the smooth point-data rendering
in ``examples/fea_bracket.py`` to see the visual difference.

Also demonstrates **MultiBlock** routing: two beams stacked into a
``pv.MultiBlock`` render as a single merged surface with the scalar
array spliced across blocks.

Run with ``uv run python examples/cell_scalars.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def main() -> None:
    """Render two hex beams with per-cell scalars via MultiBlock."""
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    beam_a = pv.examples.load_hexbeam()
    beam_b = pv.examples.load_hexbeam().translate((2.0, 0.0, 0.0), inplace=False)
    beam_a.cell_data["cell_id"] = np.arange(beam_a.n_cells, dtype=np.float32)
    beam_b.cell_data["cell_id"] = (
        np.arange(beam_b.n_cells, dtype=np.float32) + beam_a.n_cells
    )

    multi = pv.MultiBlock([beam_a, beam_b])

    plotter = pv.Plotter(off_screen=True, window_size=[960, 540])
    plotter.add_mesh(
        multi,
        scalars="cell_id",
        cmap="plasma",
        show_scalar_bar=False,
    )
    plotter.set_background("#161b22")
    plotter.camera_position = [(3.5, -5.5, 4.5), (1.0, 0.5, 0.5), (0.0, 0.0, 1.0)]

    plotter.screenshot(str(out_dir / "cell_scalars_pyvista.png"))
    plotter.blender.render(str(out_dir / "cell_scalars_blender.png"), samples=64)
    plotter.close()


if __name__ == "__main__":
    main()
