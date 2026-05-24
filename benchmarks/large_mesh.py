# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-flight: time the full bridge pipeline on a ~1M-vertex synthetic mesh.

Builds a procedural 1000 × 1000 plane with a wavy z-displacement and a
matching scalar field, then walks the bridge stage-by-stage:

1. PyVista scene construction (numpy → ``pv.StructuredGrid`` → actor).
2. ``build_scene_from_plotter`` cold start (mesh + material translation,
   suspected hot spot is the colour-attribute ``foreach_set``).
3. ``build_scene_from_plotter`` warm hit (everything cached, identity
   reuses survive).
4. Full ``pl.blender.render(...)`` to a PNG at low samples (so the
   timing isolates the bridge cost from Cycles itself).

Run: ``uv run python benchmarks/large_mesh.py``.

The script is intentionally bpy-touching, so it lives outside the test
suite — there's nothing to assert, just numbers to inspect.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

from pyvista_blender.translate.scene import build_scene_from_plotter

if TYPE_CHECKING:
    from collections.abc import Iterator

_RES = 1000  # → 1_000_000 vertices, 998_001 quads


@contextmanager
def _stage(label: str) -> Iterator[None]:
    """Print ``label`` with the elapsed wall time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {label:<40} {elapsed * 1000:>10.1f} ms")


def main() -> None:
    """Walk the bridge pipeline on a 1M-vertex plane and print stage timings."""
    print(f"Pre-flight: {_RES}x{_RES} plane ({_RES * _RES:,} vertices)")
    print()

    print("[1] PyVista scene construction")
    with _stage("build StructuredGrid + scalars"):
        u = np.linspace(-2.0, 2.0, _RES, dtype=np.float32)
        v = np.linspace(-2.0, 2.0, _RES, dtype=np.float32)
        x, y = np.meshgrid(u, v)
        z = np.sin(3.0 * x) * np.cos(3.0 * y) * 0.3
        grid = pv.StructuredGrid(x, y, z)
        grid["wave"] = z.flatten()

    with _stage("attach to plotter"):
        pl = pv.Plotter(off_screen=True, window_size=[640, 480])
        pl.add_mesh(grid, scalars="wave", cmap="viridis")
        pl.renderer.set_background("#1a1d22")
        pl.camera_position = "iso"

    print()
    print("[2] build_scene_from_plotter — cold start")
    cache = None
    with _stage("first call (mesh + material upload)"):
        cache = build_scene_from_plotter(pl, cache=cache)

    print()
    print("[3] build_scene_from_plotter — warm hit")
    with _stage("second call (cached, no rebuild)"):
        cache = build_scene_from_plotter(pl, cache=cache)

    print()
    print("[4] pl.blender.render (samples=4 — bridge cost, not Cycles)")
    out = Path("/tmp/perf_large_mesh.png")
    with _stage("full render to PNG"):
        pl.blender.render(str(out), samples=4)

    pl.close()
    print()
    print(f"Output: {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
