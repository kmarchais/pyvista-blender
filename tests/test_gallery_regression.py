# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery regression: compare canonical bpy renders against committed baselines.

Each test below builds a small synthetic scene (no internet downloads,
no large datasets) that exercises one branch of the bridge — PBR,
HUD compositing, glyph instancing, vanilla render — then compares the
output PNG against a baseline pinned in ``tests/image_cache/blender/``.

The comparison is a mean-absolute per-channel diff with a generous
threshold (``_MEAN_DIFF_THRESHOLD``); Cycles is mostly deterministic on
identical inputs but tiny per-pixel deltas creep in from libm / libtbb
across machines, so the threshold tolerates that drift while still
catching genuine bridge regressions (shifted shaders, missing overlays,
camera-projection bugs).

Adding a new baseline
---------------------
1. Add a new ``test_*`` that renders to ``tmp_path``.
2. Call ``_compare_images(rendered, _BASELINE_DIR / "<name>.png")``.
3. Run the test once — it will fail with a "baseline missing" message
   that quotes the copy command.
4. Copy the rendered PNG into ``tests/image_cache/blender/<name>.png``
   and review it before committing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from PIL import Image

_BASELINE_DIR = Path(__file__).parent / "image_cache" / "blender"

# Mean absolute per-channel diff (0-255 scale) above which a baseline
# comparison is treated as a regression. Empirically calibrated against
# repeat-rendering the same scene at samples=32 on a Linux + Cycles CPU
# host: typical re-render mean diff is < 5, so 20 leaves headroom for
# cross-machine libm / libtbb drift without masking real shifts.
_MEAN_DIFF_THRESHOLD: float = 20.0


def _compare_images(rendered: Path, baseline: Path) -> None:
    """Compare ``rendered`` against ``baseline`` with a perceptual threshold.

    Parameters
    ----------
    rendered
        Path to the freshly-rendered PNG.
    baseline
        Path to the committed baseline PNG. If it doesn't exist, the
        test fails with a clear copy-command hint so the contributor
        knows how to bless the new baseline.

    """
    if not baseline.exists():
        pytest.fail(
            f"baseline image missing: {baseline}\n"
            f"After reviewing the rendered output, commit it as the new "
            f"baseline:\n"
            f"    cp {rendered} {baseline}"
        )
    with Image.open(rendered) as img:
        rendered_arr = np.asarray(img.convert("RGB"), dtype=np.int32)
    with Image.open(baseline) as img:
        baseline_arr = np.asarray(img.convert("RGB"), dtype=np.int32)
    if rendered_arr.shape != baseline_arr.shape:
        pytest.fail(
            f"shape mismatch: rendered {rendered_arr.shape}, "
            f"baseline {baseline_arr.shape}"
        )
    mean_diff = float(np.abs(rendered_arr - baseline_arr).mean())
    if mean_diff > _MEAN_DIFF_THRESHOLD:
        pytest.fail(
            f"perceptual diff vs {baseline.name} exceeds "
            f"{_MEAN_DIFF_THRESHOLD}: mean={mean_diff:.2f}. "
            f"Inspect the rendered output at {rendered}; if the new "
            f"behaviour is correct, refresh the baseline."
        )


@pytest.mark.bpy
def test_gallery_vanilla_sphere(tmp_path: Path) -> None:
    """Baseline: a default-coloured sphere on a neutral background."""
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    pl.add_mesh(pv.Sphere(), color="#9ec5e8")
    pl.renderer.set_background("#1a1d22")
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    rendered = tmp_path / "vanilla_sphere.png"
    pl.blender.render(str(rendered), samples=32)
    pl.close()

    _compare_images(rendered, _BASELINE_DIR / "vanilla_sphere.png")


@pytest.mark.bpy
def test_gallery_pbr_sphere(tmp_path: Path) -> None:
    """A metallic PBR sphere — exercises the Principled BSDF path."""
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    pl.add_mesh(
        pv.Sphere(),
        color="#dec27c",
        pbr=True,
        metallic=0.8,
        roughness=0.25,
    )
    pl.renderer.set_background("#1a1d22")
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    rendered = tmp_path / "pbr_sphere.png"
    pl.blender.render(str(rendered), samples=32)
    pl.close()

    _compare_images(rendered, _BASELINE_DIR / "pbr_sphere.png")


@pytest.mark.bpy
def test_gallery_scalar_bar_overlay(tmp_path: Path) -> None:
    """HUD overlay: a sphere with scalars + a visible scalar bar."""
    sphere = pv.Sphere()
    sphere["z"] = sphere.points[:, 2]

    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    pl.add_mesh(sphere, scalars="z", cmap="viridis", show_scalar_bar=True)
    pl.renderer.set_background("#1a1d22")
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    rendered = tmp_path / "scalar_bar_overlay.png"
    pl.blender.render(str(rendered), samples=32)
    pl.close()

    _compare_images(rendered, _BASELINE_DIR / "scalar_bar_overlay.png")


@pytest.mark.bpy
def test_gallery_glyph_instances(tmp_path: Path) -> None:
    """Glyph instancing: three oriented cones via Geometry Nodes."""
    points = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    points["vec"] = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    pl.blender.add_glyph(points, geom=pv.Cone(), orient="vec", factor=0.4)
    pl.renderer.set_background("#1a1d22")
    pl.camera_position = [(2.5, 2.5, 1.8), (0.3, 0.3, 0.0), (0.0, 0.0, 1.0)]

    rendered = tmp_path / "glyph_instances.png"
    pl.blender.render(str(rendered), samples=32)
    pl.close()

    _compare_images(rendered, _BASELINE_DIR / "glyph_instances.png")
