# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render with a transparent film and toggle the OIDN denoiser.

Two consecutive renders of the same scene:

1. ``transparent_bg=True`` — Cycles' ``scene.render.film_transparent`` is
   enabled and the PNG carries an alpha channel. Useful for compositing
   the rendered mesh onto a different background in post.
2. ``denoise=False, samples=256`` — same scene with the denoiser
   disabled and a higher sample count, so the output is honest path-
   tracing noise rather than the OIDN-smoothed default. Visible only at
   high zoom; the comparison is the point.

Both calls reuse the cached mesh and material data blocks (Level 1
identity cache) so the second render only rebuilds camera, lights, and
output settings.

Run with ``uv run python examples/transparent_render.py``.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


def main() -> None:
    """Render a glossy sphere twice: once transparent, once raw path-traced."""
    sphere = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64)

    plotter = pv.Plotter(off_screen=True, window_size=[640, 480])
    plotter.add_mesh(
        sphere,
        color="#bb446c",
        pbr=True,
        metallic=0.7,
        roughness=0.25,
    )
    plotter.set_background("#1a1d22")
    plotter.camera_position = [(2.5, -2.5, 1.8), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "examples"
        / Path(__file__).parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Transparent film. RGBA PNG; the dark background colour is dropped.
    plotter.blender.render(
        str(out_dir / "transparent_alpha_blender.png"),
        samples=128,
        transparent_bg=True,
    )

    # 2) Same scene, opaque, denoiser off, more samples — the second call
    # hits the identity cache (mesh + material reused), only render config
    # changes.
    plotter.blender.render(
        str(out_dir / "transparent_raw_blender.png"),
        samples=256,
        denoise=False,
    )

    plotter.close()


if __name__ == "__main__":
    main()
