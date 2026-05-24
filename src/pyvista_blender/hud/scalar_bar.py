# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render PyVista scalar bars as an RGBA overlay.

VTK's :class:`vtkScalarBarActor` lives in viewport-fraction space — its
``Position`` is ``(x0, y0)`` and ``Position2`` is ``(width, height)``,
both in ``[0, 1]``. Matplotlib's :meth:`Figure.add_axes` uses the same
convention, so a colorbar drawn at full frame size lines up with the
PyVista output without any pixel-space arithmetic.

Returns a single RGBA image carrying every scalar bar on the plotter
(stacked into one overlay rather than one per bar — the compositor sees
fewer nodes and the cost is identical for the linear node count).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["render_scalar_bars"]

_DPI = 100


def render_scalar_bars(
    plotter: pv.BasePlotter,
    width: int,
    height: int,
    *,
    renderer_index: int | None = None,
) -> np.ndarray | None:
    """Render every visible scalar bar on the plotter as one RGBA overlay.

    Parameters
    ----------
    plotter
        Source plotter.
    width, height
        Output resolution.
    renderer_index
        When not ``None``, restrict the overlay to bars that belong to
        the renderer at this index (the bar is in
        ``plotter.renderers[renderer_index].GetActors2D()``). Used by
        the subplot tile path so each tile only draws its own bars.
        ``None`` (the default) keeps the "draw every bar" behaviour
        for single-renderer plotters.

    Returns
    -------
    np.ndarray or None
        Shape ``(height, width, 4)`` float32 RGBA in ``[0, 1]`` ready for
        compositor upload. ``None`` when the plotter has no scalar bars
        or matplotlib isn't importable.

    """
    bars = list(_iter_scalar_bars(plotter, renderer_index=renderer_index))
    if not bars:
        return None

    # Pin the headless Agg backend lazily, only when an overlay is
    # actually drawn. ``force=False`` is a no-op once the backend is set,
    # so the cost is one cheap check per render.
    mpl.use("Agg", force=False)

    fig = plt.figure(
        figsize=(width / _DPI, height / _DPI),
        dpi=_DPI,
        facecolor="none",
    )
    try:
        for cmap_name, params in bars:
            ax = fig.add_axes((params["x0"], params["y0"], params["w"], params["h"]))
            ax.patch.set_alpha(0.0)
            cb = ColorbarBase(
                ax,
                cmap=plt.get_cmap(cmap_name),
                norm=Normalize(vmin=params["vmin"], vmax=params["vmax"]),
                orientation=params["orientation"],
            )
            if params["title"]:
                # ``cb.set_label`` puts the title at the long-edge end
                # of the bar (below for horizontal, right for vertical),
                # which gets clipped when the bar sits flush against an
                # image edge — VTK's ``vtkScalarBarActor`` puts the
                # title at the *top*, where there's usually clearance.
                # ``ax.set_title`` mirrors that convention.
                ax.set_title(
                    params["title"],
                    color=params["text_color"],
                    fontsize=params["title_fontsize"],
                )
            cb.ax.tick_params(
                colors=params["text_color"],
                labelsize=params["label_fontsize"],
            )
            for spine in cb.ax.spines.values():
                spine.set_color(params["text_color"])

        fig.canvas.draw()
        # We pinned the Agg backend just above this block, so
        # ``fig.canvas`` is concretely ``FigureCanvasAgg`` and exposes
        # ``buffer_rgba``; matplotlib's stubs only type the base
        # ``FigureCanvasBase``, which doesn't.
        canvas = cast("Any", fig.canvas)
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    finally:
        plt.close(fig)

    # Matplotlib's Agg canvas is fully transparent where nothing was drawn,
    # which is exactly what the compositor needs. Convert to float [0, 1]
    # and return as-is (top-to-bottom origin; the compositor flips for us).
    return buf.astype(np.float32) / 255.0


def _renderer_actors2d(
    plotter: pv.BasePlotter, renderer_index: int | None
) -> set[object] | None:
    """Return the set of 2D actors owned by ``plotter.renderers[renderer_index]``.

    Walks VTK's :meth:`vtkRenderer.GetActors2D` collection which is the
    source of truth for per-renderer 2D-actor ownership (pyvista's
    plotter-global ``scalar_bars`` dict doesn't carry this association
    explicitly).

    Returns
    -------
    set or None
        ``None`` when ``renderer_index`` is ``None`` (callers use that
        as a "no filtering, draw every bar" sentinel); otherwise the
        set of every 2D actor in the renderer's collection. Returns an
        empty set when ``renderer_index`` is out of range.

    """
    if renderer_index is None:
        return None
    if not 0 <= renderer_index < len(plotter.renderers):
        return set()
    collection = plotter.renderers[renderer_index].GetActors2D()
    owned: set[object] = set()
    collection.InitTraversal()
    while True:
        actor = collection.GetNextActor2D()
        if actor is None:
            break
        owned.add(actor)
    return owned


def _iter_scalar_bars(
    plotter: pv.BasePlotter, *, renderer_index: int | None = None
) -> list[tuple[str, dict]]:
    """Pull layout + style parameters for every scalar bar on the plotter.

    Parameters
    ----------
    plotter
        Source plotter.
    renderer_index
        When not ``None``, only include bars whose VTK actor sits in
        ``plotter.renderers[renderer_index].GetActors2D()``. VTK
        registers each :class:`vtkScalarBarActor` with the renderer
        that owned it at ``add_mesh`` time, so this filter is exact
        per-renderer (no heuristics needed).

    Returns
    -------
    list of (str, dict)
        ``(cmap_name, params)`` per bar; ``params`` carries position,
        orientation, range, title, fonts, text colour.

    """
    owned_actors = _renderer_actors2d(plotter, renderer_index)
    out: list[tuple[str, dict]] = []
    for actor in plotter.scalar_bars.values():
        if owned_actors is not None and actor not in owned_actors:
            continue
        lut = actor.GetLookupTable()
        if lut is None:
            continue
        cmap_name = getattr(lut, "cmap", None) or "viridis"
        vmin, vmax = lut.GetRange()
        x0, y0 = actor.GetPosition()
        w, h = actor.GetPosition2()
        title_prop = actor.GetTitleTextProperty()
        label_prop = actor.GetLabelTextProperty()
        out.append((
            cmap_name,
            {
                "x0": float(x0),
                "y0": float(y0),
                "w": float(w),
                "h": float(h),
                "vmin": float(vmin),
                "vmax": float(vmax),
                "title": str(actor.GetTitle() or ""),
                "orientation": (
                    "horizontal" if actor.GetOrientation() == 0 else "vertical"
                ),
                "title_fontsize": int(title_prop.GetFontSize()),
                "label_fontsize": int(label_prop.GetFontSize()),
                # VTK colours are 0..1 RGB; matplotlib accepts that tuple.
                "text_color": tuple(title_prop.GetColor()),
            },
        ))
    return out
