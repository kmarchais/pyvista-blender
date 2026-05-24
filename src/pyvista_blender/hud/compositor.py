# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-render HUD compositor.

Each HUD producer (scalar bar, text, axes, bounds) yields an RGBA numpy
array sized to the render frame. After Cycles writes the path-traced
PNG, this module opens it via PIL, alpha-composites every overlay on
top, and writes back to the same path.

We considered driving Blender's built-in compositor (``AlphaOver`` node
chain) but bpy 5.x reworked the compositor into a node group where
``Image`` input doesn't auto-receive the render pass without a complex
pass / view-layer setup that drifts from bpy 4.x. A 2-3 ms PIL pass per
frame is simpler, cross-version, and runs after the view transform so
overlay colours land in the same display space the rendered image is
already in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from pyvista_blender.hud import axes, bounds, scalar_bar, text

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["composite_hud_into_array", "composite_hud_overlays"]


def composite_hud_overlays(
    plotter: pv.BasePlotter,
    output_path: str,
    width: int,
    height: int,
    *,
    renderer_index: int | None = None,
) -> None:
    """Alpha-composite every HUD overlay on top of the freshly-rendered PNG.

    Parameters
    ----------
    plotter
        Source plotter; walked by each HUD producer for its data
        (scalar bars, text annotations, …).
    output_path
        File the renderer wrote. Read, composited, written back in place.
    width, height
        Render resolution; overlays are produced at this size so the
        AlphaComposite is pixel-aligned without scaling.
    renderer_index
        Forwarded to per-renderer-aware producers (currently scalar
        bars) so subplot tiles only draw the bars they own. ``None``
        keeps the single-renderer behaviour: every visible bar is
        drawn.

    """
    overlays = list(
        _collect_overlays(plotter, width, height, renderer_index=renderer_index)
    )
    if not overlays:
        return

    with Image.open(output_path) as opened:
        had_alpha = opened.mode == "RGBA"
        base = opened.convert("RGBA")

    base = _apply_overlays(base, overlays)

    # PyVista writes RGB PNGs by default; preserve that unless the
    # rendered image already carries alpha (transparent_bg=True).
    if had_alpha:
        base.save(output_path)
    else:
        base.convert("RGB").save(output_path)


def composite_hud_into_array(
    plotter: pv.BasePlotter, base_rgba: np.ndarray
) -> np.ndarray:
    """Return ``base_rgba`` with every HUD overlay alpha-composited on top.

    In-memory companion to :func:`composite_hud_overlays`: takes an RGBA
    buffer (e.g. the pixel buffer Cycles wrote into ``bpy.data.images
    ["Render Result"]``), composites every visible HUD overlay over it,
    and returns the result as a new uint8 RGBA array. Useful when the
    output isn't headed to disk — the interactive viewport in
    :mod:`pyvista_blender.interactive.overlay` ships pixels straight
    into a ``vtkImageData`` and never round-trips through a PNG.

    Parameters
    ----------
    plotter
        Source plotter; walked by each HUD producer for its data.
    base_rgba
        Shape ``(height, width, 4)`` uint8 RGBA. Read-only; the return
        value is a freshly-allocated array.

    Returns
    -------
    np.ndarray
        Shape ``(height, width, 4)`` uint8 RGBA with overlays composited
        on top. Returns the input unchanged (still a fresh copy) when no
        overlays are active.

    """
    height, width = base_rgba.shape[:2]
    base = Image.fromarray(base_rgba, mode="RGBA")
    overlays = list(_collect_overlays(plotter, width, height))
    if overlays:
        base = _apply_overlays(base, overlays)
    return np.asarray(base, dtype=np.uint8)


def _apply_overlays(
    base: Image.Image, overlays: list[tuple[str, np.ndarray]]
) -> Image.Image:
    """Alpha-composite ``overlays`` onto ``base`` in order.

    Returns
    -------
    Image.Image
        A new PIL image with every overlay alpha-composited over
        ``base``. PIL's ``alpha_composite`` allocates a fresh image
        per call so the input is left untouched.

    """
    for _name, rgba in overlays:
        overlay_uint8 = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
        overlay_img = Image.fromarray(overlay_uint8, mode="RGBA")
        base = Image.alpha_composite(base, overlay_img)
    return base


def _collect_overlays(
    plotter: pv.BasePlotter,
    width: int,
    height: int,
    *,
    renderer_index: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Walk every HUD producer and return the non-empty RGBA overlays.

    ``renderer_index`` is forwarded to producers that honour per-tile
    filtering (currently the scalar-bar producer). Other producers
    already read from the plotter's active renderer, so for them the
    caller switches ``plotter.subplot(row, col)`` before invoking the
    compositor.

    Returns
    -------
    list of (str, np.ndarray)
        ``(name, rgba)`` pairs. ``rgba`` is shape ``(height, width, 4)``
        in ``float32`` ``[0, 1]`` ready for PIL alpha composite.

    """
    out: list[tuple[str, np.ndarray]] = []
    bounds_rgba = bounds.render_bounds_overlay(plotter, width, height)
    if bounds_rgba is not None:
        out.append(("PVBoundsBox", bounds_rgba))
    bars_rgba = scalar_bar.render_scalar_bars(
        plotter, width, height, renderer_index=renderer_index
    )
    if bars_rgba is not None:
        out.append(("PVScalarBars", bars_rgba))
    text_rgba = text.render_text_overlay(plotter, width, height)
    if text_rgba is not None:
        out.append(("PVTextOverlay", text_rgba))
    axes_rgba = axes.render_axes_overlay(plotter, width, height)
    if axes_rgba is not None:
        out.append(("PVAxesTriad", axes_rgba))
    return out
