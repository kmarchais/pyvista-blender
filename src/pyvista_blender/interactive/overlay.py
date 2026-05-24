# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Layer-1 ``vtkActor2D`` overlay + Cycles → VTK pixel-buffer round-trip.

The interactive rendered viewport shipped by ``pl.blender.show()``
keeps VTK in charge of the window and input devices, then displays the
Cycles output as a fullscreen RGBA texture on a second VTK renderer
stacked above the normal 3D scene. This module owns the plumbing:

* :func:`install_overlay` adds the layer-1 renderer + ``vtkImageData``
  + ``vtkImageMapper`` + ``vtkActor2D`` chain to the plotter's render
  window.
* :func:`render_and_blit` runs one Cycles render, reads the float32
  RGBA buffer from ``bpy.data.images["Render Result"]``, applies the
  sRGB OETF to land in display space, alpha-composites every HUD
  overlay (scalar bars, axes, etc.) on top, and writes the result into
  the overlay's ``vtkImageData``.

Observer wiring, throttling, and sample-tier dispatch are handled by
``_component.py``; this module owns the install / blit half of the
pipeline.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import bpy
import numpy as np
import vtk
from PIL import Image
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import vtkImageActor, vtkRenderer

from pyvista_blender._bpy_silence import silence_bpy_stderr
from pyvista_blender.hud.compositor import composite_hud_into_array
from pyvista_blender.render.engine import configure_engine
from pyvista_blender.translate.scene import build_scene_from_plotter

if TYPE_CHECKING:
    import pyvista as pv

    from pyvista_blender._options import _EngineParams, _PlotterSources
    from pyvista_blender.translate.scene import SceneCache

__all__ = ["OverlayHandles", "active_ren_win", "install_overlay", "render_and_blit"]


@dataclass
class OverlayHandles:
    """References to the VTK objects installed for the Cycles overlay.

    The plotter retains these so :meth:`BlenderComponent.__plotter_close__`
    can tear them back down (remove the renderer, drop the actor) when
    the user closes the viewport.

    Attributes
    ----------
    image_data
        The ``vtkImageData`` whose scalar buffer receives Cycles pixels
        on every render-and-blit. :meth:`Modified` is called after
        writes to trigger VTK to repaint.
    actor
        Fullscreen ``vtkImageActor`` displaying the RGBA image data.
        Used over ``vtkActor2D`` + ``vtkImageMapper`` because the latter
        treats the buffer as scalar (one channel through
        ColorWindow/Level) and shows a grayscale render; ``vtkImageActor``
        handles multi-component RGBA natively.
    renderer
        Layer-1 ``vtkRenderer`` added to the plotter's render window.
    width, height
        Overlay resolution at install time. ``_component.py``
        reinstalls on ``ModifiedEvent`` to track window resize.

    """

    image_data: vtkImageData
    actor: vtkImageActor
    renderer: vtkRenderer
    width: int
    height: int


def active_ren_win(plotter: pv.BasePlotter) -> vtk.vtkRenderWindow:
    """Return ``plotter.ren_win`` narrowed to a non-``None`` value.

    PyVista's stub types ``BasePlotter.ren_win`` as
    ``vtkRenderWindow | None``; in practice it is always present on a
    constructed plotter. Routing every access through this helper means
    callers don't repeat the narrowing assertion.

    Returns
    -------
    vtk.vtkRenderWindow
        The render window, narrowed to the non-``None`` branch.

    Raises
    ------
    RuntimeError
        When the plotter has no render window — only reachable on a
        torn-down plotter, which can't host the overlay anyway.

    """
    ren_win = plotter.ren_win
    if ren_win is None:
        msg = "plotter.ren_win is None — plotter has no render window"
        raise RuntimeError(msg)
    return ren_win


def install_overlay(plotter: pv.BasePlotter, width: int, height: int) -> OverlayHandles:
    """Add a layer-1 fullscreen RGBA overlay renderer to ``plotter.ren_win``.

    The render window's layer count is bumped to ``2`` (idempotently);
    layer 0 keeps PyVista's VTK 3D scene, layer 1 hosts the Cycles
    output. The overlay renderer's interactive flag is off so mouse
    events fall through to layer 0 and the trackball keeps working
    without us touching the interactor style.

    Parameters
    ----------
    plotter
        Target plotter; its ``ren_win`` is mutated in place.
    width, height
        Overlay resolution. Should match the render window's current
        size; mismatches stretch in the mapper.

    Returns
    -------
    OverlayHandles
        Handles to the installed VTK objects so the caller can write
        pixels into them and tear them down on close.

    """
    image_data = vtkImageData()
    image_data.SetDimensions(width, height, 1)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)

    actor = vtkImageActor()
    actor.SetInputData(image_data)

    overlay = vtkRenderer()
    overlay.InteractiveOff()
    overlay.SetBackgroundAlpha(0.0)
    overlay.AddViewProp(actor)

    # PyVista's ``Plotter.__init__`` already adds a ``shadow_renderer``
    # at layer 1 (it captures interactions per subplot viewport but
    # paints nothing). Stacking the Cycles overlay there would put our
    # renderer in the same layer group as the shadow — order between
    # the two is unspecified and on the user's machine the shadow ends
    # up on top, hiding the Cycles output. Always allocate a new layer
    # above whatever exists and put ourselves there.
    ren_win = active_ren_win(plotter)
    new_layer = ren_win.GetNumberOfLayers()
    ren_win.SetNumberOfLayers(new_layer + 1)
    overlay.SetLayer(new_layer)
    ren_win.AddRenderer(overlay)

    # Frame the image actor in the overlay viewport. ``vtkImageActor``
    # is a 3D actor (it lives in world space); we set up a parallel
    # camera so its rectangular bounds fill the viewport edge-to-edge.
    # Note we do NOT use ``ResetCamera``: that routine sizes the
    # ``parallel_scale`` to the actor's bounding-SPHERE radius, leaving
    # ~20-30% letterbox margins around any rectangular image. We size
    # it to the bounding-RECTANGLE (half the image height) instead.
    _frame_camera_on_image(overlay, width, height)

    return OverlayHandles(
        image_data=image_data,
        actor=actor,
        renderer=overlay,
        width=width,
        height=height,
    )


def render_and_blit(
    plotter: pv.BasePlotter,
    handles: OverlayHandles,
    *,
    engine_params: _EngineParams,
    cache: SceneCache | None,
    sources: _PlotterSources,
    hud: bool = True,
) -> SceneCache:
    """Run one Cycles render and blit the pixels into the overlay.

    Equivalent to one :func:`pyvista_blender._render_impl.do_render`
    call, except the output never touches disk — pixels go straight from
    ``bpy.data.images["Render Result"]`` into the overlay's
    ``vtkImageData``.

    Parameters
    ----------
    plotter
        Source plotter; the bpy scene is reconciled against it.
    handles
        Overlay handles from :func:`install_overlay`. The
        ``image_data`` scalar buffer is overwritten and ``Modified()``
        is called.
    engine_params
        Bundled engine kwargs forwarded to :func:`configure_engine`.
        ``device`` and ``denoise`` are Cycles-only; Eevee Next ignores
        them.
    cache
        Identity-keyed scene cache to carry across renders. ``None``
        means a cold start.
    sources
        Per-call registries from the component (``glyphs`` and
        ``volume_sources``); the cache reconciliation uses both.
    hud
        Whether to alpha-composite scalar bars / text / axes / bounds
        over the Cycles output before blitting (matches the offline
        ``render()`` behaviour). Set ``False`` to skip the overlay pass.

    Returns
    -------
    SceneCache
        The updated cache; the caller should retain it for the next
        render.

    Raises
    ------
    RuntimeError
        When ``bpy.context.scene`` is ``None`` (Blender did not
        initialise a scene) or ``bpy.data.images["Render Result"]`` is
        missing after the render call returned.

    """
    cache = build_scene_from_plotter(
        plotter, cache, sources.glyphs, sources.volume_sources
    )
    with silence_bpy_stderr():
        configure_engine(
            engine=engine_params.engine,
            device=engine_params.device,
            samples=engine_params.samples,
            denoise=engine_params.denoise,
            transparent_bg=engine_params.transparent_bg,
        )

    scene = bpy.context.scene
    if scene is None:
        msg = "bpy.context.scene is None — Blender did not initialise a scene"
        raise RuntimeError(msg)

    # Re-pin the overlay to the current ren_win size before rendering.
    # The window manager (or pl.show()) often realizes the window at a
    # different size than the configured ``window_size``, and the user
    # may resize it during interaction. Re-querying ``GetSize`` every
    # frame keeps the overlay full-bleed; the alternative is an external
    # ``ModifiedEvent`` observer racing the render path and producing
    # letterboxed frames when the timing is unlucky.
    ren_win = active_ren_win(plotter)
    actual_w, actual_h = (int(d) for d in ren_win.GetSize())
    size_changed = (actual_w, actual_h) != (handles.width, handles.height)
    if size_changed and actual_w > 0 and actual_h > 0:
        handles.image_data.SetDimensions(actual_w, actual_h, 1)
        handles.image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
        handles.width = actual_w
        handles.height = actual_h
        # Re-frame the overlay camera around the new image bounds.
        _frame_camera_on_image(handles.renderer, actual_w, actual_h)
    scene.render.resolution_x = handles.width
    scene.render.resolution_y = handles.height

    # Headless bpy doesn't populate ``bpy.data.images["Render Result"]``
    # — that buffer only exists when Blender's UI is showing the
    # Image Editor. Same disk round-trip as ``_render_impl.do_render``:
    # render to a temp PNG, then PIL-read it back. Adds ~5 ms per
    # frame, negligible next to the Cycles render itself, and gives
    # us sRGB-encoded pixels for free (no manual OETF math).
    with tempfile.TemporaryDirectory(prefix="pvblender_show_") as tmp:
        frame_path = Path(tmp) / "frame.png"
        scene.render.filepath = str(frame_path)
        with silence_bpy_stderr():
            bpy.ops.render.render(write_still=True)
        with Image.open(frame_path) as img:
            rgba_uint8 = np.asarray(img.convert("RGBA"), dtype=np.uint8)

    # PNG is top-down; VTK's vtkImageMapper expects bottom-up.
    rgba_uint8 = np.flipud(rgba_uint8).copy()

    if hud:
        rgba_uint8 = composite_hud_into_array(plotter, rgba_uint8)
    _write_into_image_data(handles.image_data, rgba_uint8)
    return cache


def _frame_camera_on_image(renderer: vtkRenderer, width: int, height: int) -> None:
    """Set up a parallel-projection camera framing a ``width`` by ``height`` image.

    Avoids ``ResetCamera``'s default behaviour of sizing the parallel
    scale to the actor's bounding-sphere radius — for a rectangular
    image that leaves substantial letterbox margins on the long axis.
    Sizing to half the image height makes the image fill the viewport
    edge-to-edge, with the layer 0 / VTK renderer cropped at the same
    rectangle (or invisible when the overlay is opaque, the default).
    """
    cam = renderer.GetActiveCamera()
    cam.ParallelProjectionOn()
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    # Camera distance is arbitrary for a parallel projection but must
    # be large enough that the near/far planes bracket the image plane
    # at z=0.
    cam_distance = max(width, height)
    cam.SetPosition(cx, cy, cam_distance)
    cam.SetFocalPoint(cx, cy, 0.0)
    cam.SetViewUp(0.0, 1.0, 0.0)
    cam.SetParallelScale(height / 2.0)
    cam.SetClippingRange(cam_distance - 1.0, cam_distance + 1.0)


def _write_into_image_data(image_data: vtkImageData, rgba_uint8: np.ndarray) -> None:
    """Copy ``rgba_uint8`` into ``image_data``'s scalar buffer and mark dirty.

    The buffer is mutated in place via ``numpy_support.vtk_to_numpy``
    so the existing ``vtkImageMapper`` keeps its allocation; no need
    to re-bind the actor.
    """
    height, width = rgba_uint8.shape[:2]
    scalars = image_data.GetPointData().GetScalars()
    flat = cast("Any", numpy_support.vtk_to_numpy(scalars))
    np.copyto(flat.reshape((height, width, 4)), rgba_uint8)
    image_data.Modified()
