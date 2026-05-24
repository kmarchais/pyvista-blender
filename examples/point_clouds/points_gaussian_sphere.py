# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gaussian point splats sized + coloured by a radial scalar field.

Mirrors the PyVista ``PointGaussianMapper.scale_array`` recipe — see
https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotting.mapper.pointgaussianmapper.scale_array
for the canonical pattern (``add_mesh(..., style="points_gaussian",
emissive=False)`` + a per-point ``scale_array``).

* ``emissive=False`` — splats render as translucent scene-lit
  surfaces (the PyVista default). The bridge maps this to a
  Principled BSDF foreground mixed with a Transparent BSDF via the
  camera-facing falloff, so the splats shade like tiny lit spheres.
* The docs example also passes ``render_points_as_spheres=True``;
  VTK's offscreen ``screenshot()`` path renders an empty frame when
  that's set (probably a GL/depth-buffer quirk that the interactive
  ``pl.show()`` path masks), so this example omits it. The bridge
  treats it as a no-op for the gaussian path anyway — Cycles'
  PointCloud primitive always renders points as spheres natively.

The same plotter renders twice:

* ``screenshot()`` — PyVista's native ``vtkPointGaussianMapper`` GL
  output (camera-facing sprites with alpha blending).
* ``pl.blender.render()`` — the bridge's PointCloud path: per-point
  radii on a :class:`bpy.types.PointCloud`, lit spheres with a
  camera-facing alpha falloff in Cycles.

Run with ``uv run python examples/points_gaussian_sphere.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def _sample_points_in_unit_sphere(n: int, seed: int) -> np.ndarray:
    """Return ``n`` points sampled uniformly in the unit-radius ball.

    Uses a rejection sampler — a few percent of rejections is cheaper
    than a Marsaglia transform here and stays exact.

    Returns
    -------
    np.ndarray
        ``(n, 3)`` float32 array of points with ``norm <= 1``.

    """
    rng = np.random.default_rng(seed)
    accepted = np.empty((0, 3), dtype=np.float32)
    while accepted.shape[0] < n:
        batch = rng.uniform(-1.0, 1.0, size=(2 * n, 3)).astype(np.float32)
        norms = np.linalg.norm(batch, axis=1)
        accepted = np.concatenate([accepted, batch[norms <= 1.0]], axis=0)
    return accepted[:n]


def main() -> None:
    """Build the cloud, drive size + colour from radius, render twice."""
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    n_points = 250

    points = _sample_points_in_unit_sphere(n_points, seed=42)
    cloud = pv.PolyData(points)
    # The radial scalar drives BOTH the colormap and the splat size.
    cloud["radius"] = np.linalg.norm(points, axis=1).astype(np.float32)

    plotter = pv.Plotter(off_screen=True, window_size=[960, 720])
    actor = plotter.add_mesh(
        cloud,
        style="points_gaussian",
        scalars="radius",
        cmap="inferno",
        point_size=12,
        emissive=False,
        show_scalar_bar=False,
        render_points_as_spheres=True,
    )
    # ``scale_array`` is the per-point sizing field. Assigning it
    # resets ``scale_factor`` to 1.0 (world-unit scaling), so pin a
    # smaller scale_factor immediately after so VTK's preview doesn't
    # saturate the frame. The bridge normalises ``scale_array`` values
    # by mean and applies them on top of its own
    # POINT_SIZE_TO_WORLD_RADIUS calibration, so it ignores
    # scale_factor; both renders end up at a comparable visual scale.
    actor.mapper.scale_array = "radius"
    actor.mapper.scale_factor = 0.12

    plotter.set_background("#05060a")
    plotter.camera_position = [(2.6, 2.6, 1.7), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    plotter.screenshot(str(out_dir / "points_gaussian_sphere_pyvista.png"))
    plotter.blender.render(
        str(out_dir / "points_gaussian_sphere_blender.png"), samples=64
    )
    plotter.close()


if __name__ == "__main__":
    main()
