# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Animate a sine-wave deformation on a structured grid.

Mirrors PyVista's canonical gif example
(https://docs.pyvista.org/examples/02-plot/gif.html): a 2D grid whose
z-coordinate is a phase-shifted ``sin(r)``, rendered as both a viridis
height field and a deforming surface. The same ``update`` callable
drives both the PyVista gif and the Blender mp4 — vertices and the
``height`` scalar field are mutated in place, the bridge's Level 1
cache picks up the change and refreshes the cached bpy mesh per frame
(no re-allocation).

Run with ``uv run python examples/wave_animation.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

N_FRAMES = 30


def _radial_grid() -> tuple[pv.StructuredGrid, np.ndarray]:
    """Build the static (x, y) lattice and return it with the radial field.

    Returns
    -------
    grid
        Initial structured grid at phase 0.
    radius
        Radial distance per node, used by the updater to recompute z.

    """
    xs = np.arange(-10.0, 10.0, 0.25)
    ys = np.arange(-10.0, 10.0, 0.25)
    x, y = np.meshgrid(xs, ys)
    radius = np.sqrt(x**2 + y**2)
    z = np.sin(radius)
    return pv.StructuredGrid(x, y, z), radius


def main() -> None:
    """Generate the PyVista gif and attempt the Blender movie."""
    grid, radius = _radial_grid()
    grid["height"] = grid.points[:, 2].copy()
    phases = np.linspace(0.0, 2.0 * np.pi, N_FRAMES, endpoint=False)

    plotter = pv.Plotter(off_screen=True, window_size=[640, 480])
    plotter.add_mesh(
        grid,
        scalars="height",
        cmap="viridis",
        clim=(-1.0, 1.0),
        show_scalar_bar=False,
    )
    plotter.set_background("#101820")
    plotter.camera_position = [(20.0, -20.0, 18.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pv_path = out_dir / "wave_pyvista.gif"
    blender_path = out_dir / "wave_blender.mp4"

    def update(frame: int) -> None:
        z = np.sin(radius + phases[frame]).ravel()
        grid.points[:, 2] = z
        grid["height"] = z

    plotter.open_gif(str(pv_path))
    for i in range(N_FRAMES):
        update(i)
        plotter.write_frame()

    plotter.blender.animate(
        str(blender_path),
        updater=update,
        frames=range(N_FRAMES),
        fps=30,
        samples=32,
    )

    plotter.close()


if __name__ == "__main__":
    main()
