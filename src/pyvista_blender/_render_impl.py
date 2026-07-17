# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime render + animate implementation.

Both bpy-driven entry points live here so the rest of the package can
be imported (the accessor smoke test, type hints, config) without
loading ``bpy``. ``_component.py`` lazy-imports this module from inside
``render()`` / ``animate()``, those are the only call sites that need
the renderer to be live.

``do_render`` builds the bpy scene from the plotter and runs Cycles
once. ``do_animate`` drives the per-frame loop for animation: the
PyVista scene's ``updater(frame_index)`` mutates dataset positions /
scalars in place, scene reconciliation refreshes the cached bpy mesh,
Cycles renders the frame, and the resulting PNGs are muxed into the
requested output container.

Mux backend is picked from the output extension:

* ``.gif`` → :mod:`imageio` built-in writer (loops indefinitely).
* ``.mp4``, ``.webm``, ``.mov``, ``.mkv`` → :mod:`imageio-ffmpeg`.
"""

from __future__ import annotations

import math
import re
import struct
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import bpy
import imageio.v3 as iio
import numpy as np
import pyvista as pv
from mathutils import Matrix
from PIL import Image

from pyvista_blender._bpy_silence import silence_bpy_stderr
from pyvista_blender._options import (
    _EMPTY_SOURCES,
    _ActorSampleBuckets,
    _AnimationSamples,
    _BakeChannels,
    _EngineParams,
    _LightSnapshot,
    _MaterialSnapshot,
    _PlotterSources,
    _SubplotTileContext,
)
from pyvista_blender.hud import composite_hud_overlays
from pyvista_blender.render.engine import configure_engine
from pyvista_blender.translate import background, camera, light
from pyvista_blender.translate.camera import look_at_matrix
from pyvista_blender.translate.scene import (
    SceneCache,
    build_scene_from_plotter,
    vtk_identity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pyvista_blender._glyph import GlyphSpec

__all__ = [
    "do_animate",
    "do_export_animation_blend",
    "do_export_blend",
    "do_render",
]


#: Concrete ``bpy.types.Light`` subclasses that expose ``energy: float``
#: (the abstract base does not). Used by the light-keyframe baker's
#: cast so attribute access type-checks cleanly across all light kinds.
_AnyLight = (
    bpy.types.AreaLight
    | bpy.types.PointLight
    | bpy.types.SpotLight
    | bpy.types.SunLight
)


_FFMPEG_CODECS = {
    ".mp4": "libx264",
    ".mov": "libx264",
    ".mkv": "libx264",
    ".webm": "libvpx-vp9",
}
_FFMPEG_EXTENSIONS = frozenset(_FFMPEG_CODECS)
_GIF_EXTENSIONS = frozenset({".gif"})


def _active_scene() -> bpy.types.Scene:
    """Return ``bpy.context.scene``, raising if Blender hasn't initialised one.

    fake-bpy-module-5.0 types :data:`bpy.context.scene` as
    ``Scene | None``; on a running Blender the singleton scene is
    always present. Routing every access through this helper means
    ty's narrowing handles the ``None`` branch once instead of at
    every callsite.

    Returns
    -------
    bpy.types.Scene
        The active scene, narrowed to a concrete (non-``None``) value.

    Raises
    ------
    RuntimeError
        When :data:`bpy.context.scene` is ``None`` — only reachable on
        a bpy build that hasn't initialised its default scene, which
        the bridge can't recover from anyway.

    """
    scene = bpy.context.scene
    if scene is None:
        msg = "bpy.context.scene is None — Blender did not initialise a scene"
        raise RuntimeError(msg)
    return scene


def do_render(
    plotter: pv.BasePlotter,
    output: str,
    *,
    engine_params: _EngineParams,
    cache: SceneCache | None,
    sources: _PlotterSources = _EMPTY_SOURCES,
) -> SceneCache:
    """Build the bpy scene from ``plotter`` and render to ``output``.

    Parameters
    ----------
    plotter
        The :class:`pyvista.BasePlotter` to translate.
    output
        Destination PNG path.
    engine_params
        Bundled engine kwargs forwarded to
        :func:`pyvista_blender.render.engine.configure_engine`. Carries
        ``engine``, ``device``, ``samples``, ``denoise``, and
        ``transparent_bg``.
    cache
        Identity-keyed scene cache from a previous render, or ``None`` for
        a cold start.
    sources
        Per-call registries from the component: ``glyphs`` (specs
        from ``pl.blender.add_glyph``) and ``volume_sources`` (live
        datasets from ``pl.blender.add_volume`` so per-frame mutations
        on the user's grid propagate without ``actor.mapper.dataset``
        indirection). Defaults to an empty bundle.

    Returns
    -------
    SceneCache
        The updated cache; the caller should retain it for the next render
        on this plotter.

    """
    cache = build_scene_from_plotter(
        plotter, cache, sources.glyphs, sources.volume_sources
    )

    # Multi-renderer (subplot) layouts go through the tile path so each
    # viewport gets its own camera / lights / background. Single-
    # renderer plotters keep the fast path: one Cycles call, HUD
    # composite over the full frame.
    if len(plotter.renderers) > 1:
        _render_subplot_tiles(plotter, output, engine_params=engine_params, cache=cache)
        return cache

    with silence_bpy_stderr():
        configure_engine(
            engine=engine_params.engine,
            device=engine_params.device,
            samples=engine_params.samples,
            denoise=engine_params.denoise,
            transparent_bg=engine_params.transparent_bg,
        )
    scene = _active_scene()
    scene.render.filepath = output
    with silence_bpy_stderr():
        bpy.ops.render.render(write_still=True)
    composite_hud_overlays(
        plotter, output, int(scene.render.resolution_x), int(scene.render.resolution_y)
    )
    return cache


def _render_subplot_tiles(
    plotter: pv.BasePlotter,
    output: str,
    *,
    engine_params: _EngineParams,
    cache: SceneCache,
) -> None:
    """Render each renderer in ``plotter.renderers`` and composite into ``output``.

    Each tile uses its own camera, light kit, and world shader; layer-0
    actors that belong to *other* renderers are hidden via
    ``hide_render`` for the duration of the tile, then restored.

    HUD overlays are composited **per tile**: before pasting each tile
    into the final image, the bridge switches the plotter's active
    renderer to that tile and re-runs :func:`composite_hud_overlays`
    at the tile's resolution. The HUD producers in
    :mod:`pyvista_blender.hud` read from ``plotter.renderer`` (the
    active renderer) so axes, text, and bounds get the right per-tile
    camera basis. Scalar bars are filtered to each tile's owning
    renderer.
    """
    width, height = int(plotter.window_size[0]), int(plotter.window_size[1])
    n_cols = _resolve_subplot_columns(plotter)
    composite = Image.new(
        "RGBA" if engine_params.transparent_bg else "RGB",
        (width, height),
        (0, 0, 0, 0) if engine_params.transparent_bg else (0, 0, 0),
    )

    # Map every cached actor object back to its owning renderer index so
    # we can flip ``hide_render`` per tile.
    actor_to_renderer: dict[str, int] = {}
    for ri, renderer in enumerate(plotter.renderers):
        for actor in renderer.actors.values():
            if isinstance(actor, pv.Actor):
                actor_to_renderer[vtk_identity(actor)] = ri

    with tempfile.TemporaryDirectory(prefix="pvblender_subplot_") as tmp:
        ctx = _SubplotTileContext(
            tmp_dir=Path(tmp),
            composite=composite,
            cache=cache,
            actor_to_renderer=actor_to_renderer,
            engine_params=engine_params,
            width=width,
            height=height,
            n_cols=n_cols,
        )
        for ri, renderer in enumerate(plotter.renderers):
            _render_one_subplot_tile(plotter, cast("pv.Renderer", renderer), ri, ctx)
        _toggle_other_renderers_hidden(cache, actor_to_renderer, -1, hide=False)

    composite.save(output)


def _render_one_subplot_tile(
    plotter: pv.BasePlotter,
    renderer: pv.Renderer,
    ri: int,
    ctx: _SubplotTileContext,
) -> None:
    """Render one subplot tile, composite its HUD, and paste it into ``ctx.composite``.

    Extracted from :func:`_render_subplot_tiles` so the outer function
    stays under ruff's local-variable threshold. ``ctx`` carries the
    per-call state (canvas, cache, engine params, tile layout) that's
    constant across the loop, so the caller varies only ``renderer``
    and ``ri``.
    """
    vp = renderer.GetViewport()
    x0 = round(vp[0] * ctx.width)
    x1 = round(vp[2] * ctx.width)
    y1 = round(vp[3] * ctx.height)
    sub_w = max(x1 - x0, 1)
    sub_h = max(y1 - round(vp[1] * ctx.height), 1)

    _toggle_other_renderers_hidden(ctx.cache, ctx.actor_to_renderer, ri, hide=True)
    # Match pyvista's screenshot behaviour: auto-fit the camera when the
    # user hasn't set one explicitly. Otherwise default-constructed
    # subplot cameras leave actors out of frame.
    if not getattr(renderer, "camera_set", True):
        renderer.reset_camera()
    # PyVista's ``Plotter.camera`` property lazily resets the active
    # renderer's camera to the iso default when ``camera.is_set`` is
    # False. The HUD producers read ``plotter.camera`` and would
    # silently clobber the pose we just configured for the Cycles
    # render. Pin ``is_set`` so the getter is a no-op and the rendered
    # output and the HUD overlay see the same pose.
    renderer.camera.is_set = True
    camera.translate_camera(renderer.camera, (sub_w, sub_h))
    light.translate_lights(renderer)
    background.translate_background(renderer)

    with silence_bpy_stderr():
        configure_engine(
            engine=ctx.engine_params.engine,
            device=ctx.engine_params.device,
            samples=ctx.engine_params.samples,
            denoise=ctx.engine_params.denoise,
            transparent_bg=ctx.engine_params.transparent_bg,
        )
    scene = _active_scene()
    scene.render.resolution_x = sub_w
    scene.render.resolution_y = sub_h
    tile_path = ctx.tmp_dir / f"tile_{ri:02d}.png"
    scene.render.filepath = str(tile_path)
    with silence_bpy_stderr():
        bpy.ops.render.render(write_still=True)

    # Per-tile HUD: switch the plotter's active renderer so the HUD
    # producers read its camera, text actors, bounds; composite at the
    # tile's resolution so overlays land inside the viewport rect.
    _composite_tile_hud(
        plotter, ri, ctx.n_cols, str(tile_path), sub_w=sub_w, sub_h=sub_h
    )

    with Image.open(tile_path) as tile_img:
        # pyvista's viewport origin is bottom-left; PIL is top-left.
        # Flip the Y coordinate before pasting.
        ctx.composite.paste(tile_img, (x0, ctx.height - y1))


def _resolve_subplot_columns(plotter: pv.BasePlotter) -> int:
    """Return the column count for a ``shape=(R, C)`` plotter.

    Falls back to ``len(plotter.renderers)`` for irregular layouts
    where ``plotter.shape`` isn't a clean ``(rows, cols)`` tuple — the
    subplot tile path still produces a rendered image, it just can't
    map renderer indices to ``(row, col)`` for ``pl.subplot()``-style
    HUD switching.

    Returns
    -------
    int
        The number of columns; never zero (``max(1, ...)`` ensures
        downstream ``divmod`` calls don't blow up).

    """
    rectangular_shape_dims = 2
    grid_shape = getattr(plotter, "shape", None)
    if isinstance(grid_shape, tuple) and len(grid_shape) >= rectangular_shape_dims:
        return max(int(grid_shape[1]), 1)
    return max(len(plotter.renderers), 1)


def _composite_tile_hud(
    plotter: pv.BasePlotter,
    renderer_index: int,
    n_cols: int,
    tile_path: str,
    *,
    sub_w: int,
    sub_h: int,
) -> None:
    """Composite HUD overlays onto one subplot tile.

    PyVista's ``Plotter.subplot(row, col)`` shifts
    ``plotter.renderer`` (and the related accessors the HUD producers
    read from) to the addressed tile. We map the linear renderer
    index ``ri`` back to ``(row, col)`` for the grid's column count
    so the helper works for any ``shape=(R, C)`` layout. Failures
    inside ``subplot()`` (e.g. irregular layouts where ``ri`` doesn't
    correspond to a clean grid coordinate) are caught so the tile
    render still lands without HUD — the alternative would be losing
    the whole subplot output to one decorator misfit.
    """
    row, col = divmod(renderer_index, max(n_cols, 1))
    try:
        plotter.subplot(row, col)
    except (IndexError, ValueError):
        return
    composite_hud_overlays(
        plotter, tile_path, sub_w, sub_h, renderer_index=renderer_index
    )


def _toggle_other_renderers_hidden(
    cache: SceneCache,
    actor_to_renderer: dict[str, int],
    active_renderer: int,
    *,
    hide: bool,
) -> None:
    """Flip ``hide_render`` on cached objects to isolate the active renderer.

    When ``hide=True`` and ``active_renderer >= 0``, every cached
    object whose source actor lives in a *different* renderer is
    hidden from Cycles. When ``hide=False`` (the restore call after
    every tile is rendered), all cached objects come back.
    """
    for actor_key, obj_name in cache.objects.items():
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        if not hide:
            obj.hide_render = False
            continue
        owning_renderer = actor_to_renderer.get(actor_key, active_renderer)
        obj.hide_render = owning_renderer != active_renderer


def do_animate(
    plotter: pv.BasePlotter,
    output: str,
    updater: Callable[[int], None],
    frames: Iterable[int],
    *,
    fps: int,
    engine_params: _EngineParams,
    cache: SceneCache | None,
    sources: _PlotterSources = _EMPTY_SOURCES,
) -> tuple[str, SceneCache | None]:
    """Render the frame sequence and mux it to ``output``.

    Parameters
    ----------
    plotter
        The :class:`pyvista.BasePlotter` to translate, refreshed per frame.
    output
        Destination movie path (``.gif`` / ``.mp4`` / ``.webm`` / ``.mov``).
    updater
        Per-frame mutation callback. Called as ``updater(frame_index)``
        before each render; should mutate the PyVista scene in place.
    frames
        Frame indices passed to ``updater``. Materialised eagerly via
        ``list(frames)`` so the iterable can be a generator.
    fps
        Output frame rate (gif: 1 / frame_delay; ffmpeg: container fps).
    engine_params
        Bundled engine kwargs (``engine``, ``device``, ``samples``,
        ``denoise``, ``transparent_bg``) forwarded to
        :func:`configure_engine` for every per-frame render.
    cache
        Identity-keyed scene cache to carry across frames so meshes are
        refreshed rather than rebuilt. ``None`` for a cold start.
    sources
        Per-call registries from the component (``glyphs`` and
        ``volume_sources``); see :func:`do_render` for the details.

    Returns
    -------
    tuple of (str, SceneCache or None)
        The output path and the (mutated) cache. The cache is ``None``
        only when ``frames`` is empty (a no-op call).

    Raises
    ------
    ValueError
        When ``output``'s extension isn't one of the supported gif / mp4 /
        webm / mov / mkv containers.

    """
    frame_list = list(frames)
    suffix = Path(output).suffix.lower()
    if suffix not in _FFMPEG_EXTENSIONS and suffix not in _GIF_EXTENSIONS:
        msg = (
            f"unsupported animation output extension {suffix!r}; "
            f"use one of {sorted(_GIF_EXTENSIONS | _FFMPEG_EXTENSIONS)}"
        )
        raise ValueError(msg)

    with tempfile.TemporaryDirectory(prefix="pvblender_anim_") as tmp:
        tmp_dir = Path(tmp)
        frame_paths: list[Path] = []
        for ordinal, frame_index in enumerate(frame_list):
            updater(frame_index)
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
            frame_path = tmp_dir / f"frame_{ordinal:06d}.png"
            scene = _active_scene()
            scene.render.filepath = str(frame_path)
            with silence_bpy_stderr():
                bpy.ops.render.render(write_still=True)
            composite_hud_overlays(
                plotter,
                str(frame_path),
                int(scene.render.resolution_x),
                int(scene.render.resolution_y),
            )
            frame_paths.append(frame_path)

        _encode_animation_frames(frame_paths, output, suffix=suffix, fps=fps)

    return output, cache


def _encode_animation_frames(
    frame_paths: list[Path],
    output: str,
    *,
    suffix: str,
    fps: int,
) -> None:
    """Mux rendered PNG frames into the destination container."""
    # Read every frame back as a numpy array so the imageio writer
    # gets a single in-memory stack — avoids any "is this a glob?"
    # ambiguity with the writer plugins.
    stack = [iio.imread(p) for p in frame_paths]

    if suffix in _GIF_EXTENSIONS:
        # imageio's gif plugin takes ``duration`` per frame in seconds.
        # disposal=2 clears each frame to background before drawing the
        # next; without it transparent pixels composite over the previous
        # frame and opaque content ghosts across the animation.
        iio.imwrite(output, stack, duration=1.0 / float(fps), loop=0, disposal=2)
    else:
        # Pick the codec from the container: webm uses VP9, the rest
        # ride libx264. imageio-ffmpeg reads ``fps`` directly.
        iio.imwrite(output, stack, fps=int(fps), codec=_FFMPEG_CODECS[suffix])


def do_export_blend(
    plotter: pv.BasePlotter,
    path: str,
    *,
    cache: SceneCache | None,
    sources: _PlotterSources = _EMPTY_SOURCES,
) -> tuple[str, SceneCache]:
    """Translate the live PyVista scene and save the bpy state as a ``.blend``.

    Builds the bpy scene from the plotter exactly like :func:`do_render`
    (so everything that would render is in the saved file), then writes
    a ``.blend`` archive via ``bpy.ops.wm.save_as_mainfile``. The user
    can open the result in Blender's UI for manual tweaks — adjusting
    materials, lighting, adding props, baking animations — without
    having to recreate the scene by hand.

    Parameters
    ----------
    plotter
        The :class:`pyvista.BasePlotter` to translate.
    path
        Destination ``.blend`` path. Overwrites any existing file.
    cache
        Identity-keyed scene cache from a previous render, or ``None``
        for a cold start.
    sources
        Per-call registries from the component (``glyphs`` and
        ``volume_sources``); see :func:`do_render` for the details.

    Returns
    -------
    tuple of (str, SceneCache)
        ``(path, cache)``. The cache is returned for parity with the
        other ``do_*`` entry points; callers typically retain it on
        the component for the next render call.

    """
    cache = build_scene_from_plotter(
        plotter, cache, sources.glyphs, sources.volume_sources
    )
    with silence_bpy_stderr():
        # ``copy=True`` writes the .blend without redirecting the
        # in-memory session to the new file — keeps subsequent renders
        # working against the same scene the user has been building.
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).resolve()), copy=True)
    return path, cache


_DeformationMode = Literal["none", "mdd", "shape_keys"]
_DEFORMATION_MODES: frozenset[str] = frozenset({"none", "mdd", "shape_keys"})


def _resolve_deformation_mode(*, bake_deformation: bool | str) -> _DeformationMode:
    """Normalise the public ``bake_deformation`` kwarg to an internal mode.

    ``True`` defaults to ``"mdd"`` (smaller files, no auto-exec). ``False``
    disables. Explicit strings ``"mdd"`` / ``"shape_keys"`` route to their
    respective backends. Anything else raises so typos surface early.

    Returns
    -------
    str
        One of ``"none"``, ``"mdd"``, ``"shape_keys"``.

    Raises
    ------
    ValueError
        When ``bake_deformation`` is a string that isn't a known backend.

    """
    if bake_deformation is True:
        return "mdd"
    if bake_deformation is False:
        return "none"
    if bake_deformation in _DEFORMATION_MODES - {"none"}:
        return cast("_DeformationMode", bake_deformation)
    msg = (
        f"bake_deformation={bake_deformation!r} is invalid; "
        f"expected True, False, 'mdd', or 'shape_keys'"
    )
    raise ValueError(msg)


def do_export_animation_blend(
    plotter: pv.BasePlotter,
    path: str,
    updater: Callable[[int], None],
    frames: Iterable[int],
    *,
    fps: int,
    bake: _BakeChannels,
    cache: SceneCache | None,
    sources: _PlotterSources = _EMPTY_SOURCES,
) -> tuple[str, SceneCache]:
    """Save a ``.blend`` whose timeline plays the per-frame ``updater``.

    Per-channel selective baking: each :class:`_BakeChannels` field
    gates one feature (camera fcurves, mesh deformation via MDD or
    Shape Keys, scalar-field PNG atlas, light keyframes, actor
    transforms, BSDF material inputs, volume atlas, glyph atlas). See
    the public :meth:`BlenderComponent.export_animation_blend`
    docstring for the per-channel mechanism descriptions.

    Parameters
    ----------
    plotter
        The :class:`pyvista.BasePlotter` to translate. ``updater`` is
        applied to it in place before each per-frame snapshot.
    path
        Destination ``.blend`` path. Overwrites any existing file.
    updater
        Per-frame mutation callback, invoked as ``updater(frame_index)``.
    frames
        Frame indices passed to ``updater``. Materialised eagerly so the
        iterable can be a generator. Becomes the saved scene's frame
        range.
    fps
        Output frame rate written to ``scene.render.fps``.
    bake
        Per-channel selection (:class:`_BakeChannels`). Each field
        gates one independent bake: ``camera`` (default ``True``),
        ``deformation`` (``False`` / ``True`` / ``"mdd"`` /
        ``"shape_keys"``), ``scalars``, ``lights``, ``transforms``,
        ``materials``, ``volume``, ``glyphs`` (all default ``False``).
        See the public :meth:`BlenderComponent.export_animation_blend`
        docstring for the per-channel mechanism.
    cache
        Identity-keyed scene cache from a previous render, or ``None``
        for a cold start.
    sources
        Per-call registries from the component (``glyphs`` and
        ``volume_sources``); see :func:`do_render` for the details.

    Returns
    -------
    tuple of (str, SceneCache)
        ``(path, cache)``. The cache is returned for parity with the
        other ``do_*`` entry points.

    Raises
    ------
    ValueError
        When ``frames`` is empty, or ``bake_deformation`` is an
        unknown string. :func:`_bake_camera_animation` additionally
        raises :class:`RuntimeError` when no scene camera resolves
        and ``bake_camera`` is on; that's unreachable on a well-formed
        plotter.

    """
    frame_list = [int(f) for f in frames]
    if not frame_list:
        msg = "frames is empty — nothing to animate"
        raise ValueError(msg)

    deformation_mode = _resolve_deformation_mode(bake_deformation=bake.deformation)

    samples = _sample_animation(
        plotter,
        updater,
        frame_list,
        bake=bake,
        deformation_active=deformation_mode != "none",
        sources=sources,
    )

    # Translate the (now last-frame) scene; this is the static state
    # in the saved .blend.
    cache = build_scene_from_plotter(
        plotter, cache, sources.glyphs, sources.volume_sources
    )

    scene = _active_scene()
    if bake.camera:
        _bake_camera_animation(scene, samples.cam)

    _dispatch_deformation_bake(
        samples.vertex, cache, path, deformation_mode=deformation_mode, fps=fps
    )

    scalar_pngs_to_cleanup: list[Path] = []
    if bake.scalars and samples.scalar:
        scalar_pngs_to_cleanup = _bake_scalar_animation(
            samples.scalar,
            samples.scalar_domains,
            cache,
            plotter,
            path,
            frame_start=frame_list[0],
        )

    if bake.lights and samples.light:
        _bake_light_animation(samples.light)

    if bake.transforms and samples.transform:
        _bake_transform_animation(samples.transform, cache)

    if bake.materials and samples.material:
        _bake_material_animation(samples.material, cache)

    if bake.volume and samples.volume:
        _bake_volume_animation(samples.volume, cache, frame_start=frame_list[0])

    if bake.glyphs and samples.glyph:
        _bake_glyph_animation(samples.glyph, cache, frame_start=frame_list[0])

    scene.frame_start = frame_list[0]
    scene.frame_end = frame_list[-1]
    scene.render.fps = int(fps)
    scene.frame_current = frame_list[0]

    with silence_bpy_stderr():
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).resolve()), copy=True)

    # The scalar PNGs are packed into the .blend by ``_load_or_replace_image``
    # — the external sidecars are no longer needed. Cleaning them up here
    # (after save) keeps the .blend the single source of truth so users
    # who copy or share just the ``.blend`` don't lose the animation.
    for png in scalar_pngs_to_cleanup:
        png.unlink(missing_ok=True)
    return path, cache


def _dispatch_deformation_bake(
    vertex_samples: dict[str, list[tuple[int, np.ndarray]]],
    cache: SceneCache,
    blend_path: str,
    *,
    deformation_mode: _DeformationMode,
    fps: int,
) -> None:
    """Route per-frame vertex samples through the chosen deformation backend.

    Extracted from :func:`do_export_animation_blend` so the entry point
    stays under ruff's branch / complexity thresholds. ``deformation_mode``
    has already been normalised by :func:`_resolve_deformation_mode`.
    """
    if not vertex_samples:
        return
    if deformation_mode == "shape_keys":
        _bake_shape_key_animation(vertex_samples, cache)
    elif deformation_mode == "mdd":
        _bake_mdd_deformation(vertex_samples, cache, blend_path, fps=fps)


def _bake_camera_animation(
    scene: bpy.types.Scene,
    cam_samples: list[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ],
) -> None:
    """Keyframe ``scene.camera`` from per-frame pose snapshots.

    Quaternion rotation mode is set so Blender's playback avoids the
    Euler-interpolation surprises (gimbal, wrong-side flips) that show
    up on long camera moves. Raises when the scene has no camera —
    unreachable on a well-formed plotter but kept explicit because the
    static path may legitimately omit it.

    Raises
    ------
    RuntimeError
        When ``scene.camera`` is ``None`` after ``build_scene_from_plotter``.

    """
    cam_obj = scene.camera
    if cam_obj is None:
        msg = "scene.camera is None after build — cannot bake animation"
        raise RuntimeError(msg)
    cam_obj.rotation_mode = "QUATERNION"
    for frame_index, position, focal_point, up in cam_samples:
        cam_obj.matrix_world = look_at_matrix(position, focal_point, up)
        cam_obj.keyframe_insert("location", frame=frame_index)
        cam_obj.keyframe_insert("rotation_quaternion", frame=frame_index)


def _sample_animation(
    plotter: pv.BasePlotter,
    updater: Callable[[int], None],
    frame_list: list[int],
    *,
    bake: _BakeChannels,
    deformation_active: bool,
    sources: _PlotterSources = _EMPTY_SOURCES,
) -> _AnimationSamples:
    """Walk the timeline once, returning per-frame samples for each channel.

    Camera samples are always collected; the channels enabled on
    ``bake`` drive which of the other per-actor / per-light / per-
    volume / per-glyph samples get captured. Actors whose point count
    changes between frames are dropped with a :class:`UserWarning`.

    Parameters
    ----------
    plotter
        Live :class:`pyvista.BasePlotter` whose state ``updater``
        mutates per frame.
    updater
        Per-frame mutation callback, invoked as ``updater(frame_index)``.
    frame_list
        Materialised list of frame indices to walk.
    bake
        Bake-channel selection. The per-channel booleans gate which
        samples are captured.
    deformation_active
        Pre-resolved "should we sample vertex positions" boolean,
        accounting for ``bake.deformation`` being either ``False``,
        ``True``, ``"mdd"``, or ``"shape_keys"``.
    sources
        Per-call registries (``glyphs`` and ``volume_sources``); the
        volume sampler reads from ``sources.volume_sources`` when
        present, the glyph sampler reads from ``sources.glyphs`` when
        ``bake.glyphs`` is on.

    Returns
    -------
    _AnimationSamples
        Per-channel samples, one attribute per channel.

    """
    cam_samples: list[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    buckets = _ActorSampleBuckets(vertex={}, scalar={}, transform={}, material={})
    light_samples: dict[int, list[tuple[int, _LightSnapshot]]] = {}
    volume_samples: dict[str, list[tuple[int, np.ndarray]]] = {}
    glyph_samples: dict[int, dict[str, list[tuple[int, np.ndarray]]]] = {}
    state = _ActorSampleState()

    sample_actors = (
        deformation_active or bake.scalars or bake.transforms or bake.materials
    )

    for frame_index in frame_list:
        updater(frame_index)
        pv_cam = plotter.camera
        cam_samples.append((
            frame_index,
            tuple(pv_cam.position),
            tuple(pv_cam.focal_point),
            tuple(pv_cam.up),
        ))
        if bake.lights:
            _sample_lights(plotter, frame_index, light_samples)
        if sample_actors:
            for renderer in plotter.renderers:
                for actor in renderer.actors.values():
                    _sample_one_actor(
                        actor,
                        frame_index,
                        state,
                        buckets,
                        bake=bake,
                        deformation_active=deformation_active,
                    )
        if bake.volume:
            _sample_volumes(
                plotter, frame_index, volume_samples, sources.volume_sources
            )
        if bake.glyphs and sources.glyphs:
            _sample_glyphs(sources.glyphs, frame_index, glyph_samples)
    return _AnimationSamples(
        cam=cam_samples,
        vertex=buckets.vertex,
        scalar=buckets.scalar,
        light=light_samples,
        scalar_domains=dict(state.scalar_domain),
        transform=buckets.transform,
        material=buckets.material,
        volume=volume_samples,
        glyph=glyph_samples,
    )


def _sample_glyphs(
    specs: list[GlyphSpec],
    frame_index: int,
    glyph_samples: dict[int, dict[str, list[tuple[int, np.ndarray]]]],
) -> None:
    """Snapshot every registered glyph spec's source-dataset state at one frame.

    Each spec contributes up to three per-frame channels — positions,
    ``orient`` vectors, ``scale`` scalars — keyed by the spec's ordinal
    in the plotter's glyph registry. Only the channels the spec
    actually declares (via ``add_glyph(orient=..., scale=...)``) are
    captured. The captured arrays drive the per-channel atlas images
    that :func:`_bake_glyph_animation` packs into the saved .blend.
    """
    for ordinal, spec in enumerate(specs):
        per_spec = glyph_samples.setdefault(ordinal, {})
        positions = np.asarray(spec.source.points, dtype=np.float32).copy()
        per_spec.setdefault("positions", []).append((frame_index, positions))

        if spec.orient and spec.orient in spec.source.point_data:
            vectors = np.asarray(
                spec.source.point_data[spec.orient], dtype=np.float32
            ).copy()
            per_spec.setdefault("orient", []).append((frame_index, vectors))

        if spec.scale and spec.scale in spec.source.point_data:
            scalars = np.asarray(
                spec.source.point_data[spec.scale], dtype=np.float32
            ).copy()
            per_spec.setdefault("scale", []).append((frame_index, scalars))


def _sample_volumes(
    plotter: pv.BasePlotter,
    frame_index: int,
    volume_samples: dict[str, list[tuple[int, np.ndarray]]],
    volume_sources: dict[str, pv.DataSet] | None,
) -> None:
    """Snapshot every visible volume's scalar field at one frame.

    Walks ``plotter.renderers`` for :class:`pv.Volume` actors. Each
    volume's underlying dataset is resampled to ImageData (when not
    already regular) and its active scalar field is reshaped to
    ``(nz, ny, nx)``. The captured arrays drive the multi-frame atlas
    that :func:`_bake_volume_animation` packs into the .blend.

    When a volume's actor key has a registered live dataset in
    ``volume_sources`` (set up by ``pl.blender.add_volume``), that
    dataset is sampled instead of ``actor.mapper.dataset`` so the
    user's per-frame mutations of the original grid propagate.
    """
    from pyvista_blender.translate.volume import (  # noqa: PLC0415
        resolve_array_name,
        resolve_image_data,
        resolve_scalar_array,
    )

    for renderer in plotter.renderers:
        for actor in renderer.actors.values():
            if not isinstance(actor, pv.Volume):
                continue
            actor_key = vtk_identity(actor)
            array_name = resolve_array_name(actor)
            source = (
                volume_sources.get(actor_key) if volume_sources else None
            ) or actor.mapper.dataset
            image_data = resolve_image_data(source, array_name)
            values = resolve_scalar_array(image_data, array_name).copy()
            volume_samples.setdefault(actor_key, []).append((frame_index, values))


def _sample_lights(
    plotter: pv.BasePlotter,
    frame_index: int,
    light_samples: dict[int, list[tuple[int, _LightSnapshot]]],
) -> None:
    """Snapshot every visible light's pose / intensity / colour at one frame.

    Walks ``plotter.renderer.lights`` to match :func:`translate_lights`'s
    iteration order; the index in that list is the light's stable key
    (and matches the ``PVLight_{i}`` naming used by the translator).
    Lights with ``on=False`` are skipped just like the static path does.
    """
    lights = list(getattr(plotter.renderer, "lights", []))
    for index, pv_light in enumerate(lights):
        if not getattr(pv_light, "on", True):
            continue
        color = pv_light.diffuse_color.float_rgb
        snapshot: _LightSnapshot = (
            tuple(pv_light.world_position),
            tuple(pv_light.world_focal_point),
            float(getattr(pv_light, "intensity", 1.0)),
            (float(color[0]), float(color[1]), float(color[2])),
        )
        light_samples.setdefault(index, []).append((frame_index, snapshot))


class _ActorSampleState:
    """Per-walk topology bookkeeping for :func:`_sample_animation`.

    Tracks the first-seen point count for each actor, the set of
    actors whose topology has gone unstable (and are therefore
    permanently dropped from later frames), and the set of actors
    we've already warned about for cell-data scalars (we warn once
    per actor, not once per frame).
    """

    __slots__ = ("initial_n_points", "scalar_domain", "unstable")

    def __init__(self) -> None:
        """Initialise empty bookkeeping maps."""
        self.initial_n_points: dict[str, int] = {}
        self.unstable: set[str] = set()
        # Per-actor scalar domain (``"POINT"`` or ``"FACE"``), captured on
        # first sample. Subsequent frames must match — a mid-animation
        # switch between point-data and cell-data scalars is unsupported.
        self.scalar_domain: dict[str, str] = {}


def _sample_one_actor(
    actor: object,
    frame_index: int,
    state: _ActorSampleState,
    buckets: _ActorSampleBuckets,
    *,
    bake: _BakeChannels,
    deformation_active: bool,
) -> None:
    """Capture vertex / scalar / transform / material samples for one actor.

    Detects topology drift the first time it occurs (warns and drops
    the actor from subsequent samples). Centralising this logic lets
    :func:`_sample_animation` stay under ruff's branch / complexity
    thresholds. ``deformation_active`` is the pre-resolved
    "vertex-sample this frame" bool (accounts for
    ``bake.deformation`` being ``False`` / ``True`` / ``"mdd"`` /
    ``"shape_keys"``).
    """
    if not isinstance(actor, pv.Actor) or not actor.visibility:
        return
    actor_key = vtk_identity(actor)
    if actor_key in state.unstable:
        return
    dataset = actor.mapper.dataset
    if dataset is None:
        return
    if bake.transforms:
        _capture_actor_transform(actor, actor_key, frame_index, buckets.transform)
    if bake.materials:
        _capture_actor_material(
            actor, vtk_identity(actor.prop), frame_index, buckets.material
        )
    points = np.asarray(dataset.points, dtype=np.float32)
    n = int(points.shape[0])
    if not _check_topology(
        actor_key,
        n,
        frame_index,
        state=state,
        vertex_samples=buckets.vertex,
        scalar_samples=buckets.scalar,
    ):
        return
    if deformation_active:
        buckets.vertex.setdefault(actor_key, []).append((frame_index, points.copy()))
    if bake.scalars:
        _capture_actor_scalars(
            actor, actor_key, frame_index, n, state=state, scalar_samples=buckets.scalar
        )


def _capture_actor_scalars(
    actor: pv.Actor,
    actor_key: str,
    frame_index: int,
    n_points: int,
    *,
    state: _ActorSampleState,
    scalar_samples: dict[str, list[tuple[int, np.ndarray]]],
) -> None:
    """Snapshot one actor's active scalar array, tracking the domain.

    Pulls the active scalars via :func:`_extract_active_scalars` and
    enforces a single domain (POINT or FACE) across frames — switching
    between the two mid-animation would force a re-layout of the
    Geometry Nodes image, which the bridge doesn't try to do.
    """
    result = _extract_active_scalars(actor, n_points)
    if result is None:
        return
    scalars, domain = result
    prior_domain = state.scalar_domain.get(actor_key)
    if prior_domain is None:
        state.scalar_domain[actor_key] = domain
    elif prior_domain != domain:
        return
    scalar_samples.setdefault(actor_key, []).append((frame_index, scalars))


def _capture_actor_transform(
    actor: pv.Actor,
    actor_key: str,
    frame_index: int,
    transform_samples: dict[str, list[tuple[int, np.ndarray]]],
) -> None:
    """Snapshot one actor's ``user_matrix`` at one frame.

    The matrix is captured independently of mesh topology, so actors
    whose mesh becomes unstable mid-animation still get their transforms
    keyframed if the caller asked for them. Non-finite or oddly-shaped
    matrices are silently skipped.
    """
    user_matrix = np.asarray(getattr(actor, "user_matrix", np.eye(4)), dtype=np.float64)
    expected_shape = (4, 4)
    if user_matrix.shape == expected_shape and np.isfinite(user_matrix).all():
        transform_samples.setdefault(actor_key, []).append((
            frame_index,
            user_matrix.copy(),
        ))


def _capture_actor_material(
    actor: pv.Actor,
    prop_key: str,
    frame_index: int,
    material_samples: dict[str, list[tuple[int, _MaterialSnapshot]]],
) -> None:
    """Snapshot the Principled-BSDF-relevant property values at one frame.

    Mirrors the static material translator's logic so the snapshot lines
    up with whatever ``translate_actor_material`` would have written to
    the BSDF — Phong-shaded properties get their ``specular_power``
    converted to GGX roughness, PBR-shaded properties use ``roughness``
    and ``metallic`` directly.
    """
    prop = actor.prop
    color = prop.color.float_rgb
    interpolation = (
        getattr(getattr(prop, "interpolation", None), "name", "")
        or str(getattr(prop, "interpolation", ""))
    ).lower()
    if interpolation == "pbr":
        metallic = float(prop.metallic)
        roughness = float(prop.roughness)
    else:
        metallic = 0.0
        roughness = _phong_specular_power_to_roughness(
            float(getattr(prop, "specular_power", 100.0))
        )
    opacity = float(getattr(prop, "opacity", 1.0))
    snapshot: _MaterialSnapshot = (
        (float(color[0]), float(color[1]), float(color[2])),
        metallic,
        roughness,
        opacity,
    )
    material_samples.setdefault(prop_key, []).append((frame_index, snapshot))


def _phong_specular_power_to_roughness(specular_power: float) -> float:
    """Mirror :func:`translate.material._phong_power_to_roughness`.

    Replicating the formula keeps :mod:`_render_impl` from depending on
    a private helper in the static translator; the math is the
    canonical Walter et al. (2007) GGX fit.

    Returns
    -------
    float
        Roughness in ``[0.02, 1.0]``; the lower clamp matches the
        static path so the animated value won't pop relative to the
        last-frame static output.

    """
    n = max(specular_power, 0.0)
    return max(math.sqrt(2.0 / (n + 2.0)), 0.02)


def _check_topology(
    actor_key: str,
    n: int,
    frame_index: int,
    *,
    state: _ActorSampleState,
    vertex_samples: dict[str, list[tuple[int, np.ndarray]]],
    scalar_samples: dict[str, list[tuple[int, np.ndarray]]],
) -> bool:
    """Verify the actor's point count is stable; warn + drop if it drifted.

    Returns
    -------
    bool
        ``True`` when the actor's topology is still consistent with
        the first frame, ``False`` after a drift has been recorded
        (the caller should skip this actor for this frame).

    """
    initial = state.initial_n_points.get(actor_key)
    if initial is None:
        state.initial_n_points[actor_key] = n
        return True
    if initial != n:
        warnings.warn(
            f"actor {actor_key}: point count changed from {initial} to {n} "
            f"at frame {frame_index}; skipping deformation / scalar bake "
            f"for this mesh (constant topology is required)",
            UserWarning,
            stacklevel=4,
        )
        state.unstable.add(actor_key)
        vertex_samples.pop(actor_key, None)
        scalar_samples.pop(actor_key, None)
        return False
    return True


def _extract_active_scalars(
    actor: pv.Actor,
    n_points: int,
) -> tuple[np.ndarray, str] | None:
    """Return the actor's active scalar array plus its domain.

    Routes between point-data (returns ``(arr, "POINT")``) and cell-data
    (returns ``(arr, "FACE")``); both are supported. Returns ``None``
    when the actor has no visible scalars, the array is missing, or
    the shape doesn't match the expected element count.

    Returns
    -------
    tuple[np.ndarray, str] or None
        ``(scalars, domain)`` where ``domain`` is ``"POINT"`` or
        ``"FACE"``; or ``None`` when there's nothing to capture.

    """
    mapper = actor.mapper
    if not getattr(mapper, "scalar_visibility", False):
        return None
    array_name = getattr(mapper, "array_name", None)
    dataset = mapper.dataset
    if not array_name or dataset is None:
        return None
    return _extract_point_scalars(
        dataset, array_name, n_points
    ) or _extract_cell_scalars(dataset, array_name)


def _extract_point_scalars(
    dataset: pv.DataSet, array_name: str, n_points: int
) -> tuple[np.ndarray, str] | None:
    """Return the active point-data scalar array (POINT domain), if shape matches.

    Returns
    -------
    tuple[np.ndarray, str] or None
        ``(scalars, "POINT")`` when the array is 1D and has
        ``n_points`` entries; ``None`` otherwise.

    """
    point_data = dataset.point_data
    if array_name not in point_data:
        return None
    scalars = np.asarray(point_data[array_name], dtype=np.float32)
    if scalars.ndim != 1 or scalars.shape[0] != n_points:
        return None
    return scalars.copy(), "POINT"


def _extract_cell_scalars(
    dataset: pv.DataSet, array_name: str
) -> tuple[np.ndarray, str] | None:
    """Return the active cell-data scalar array aligned to the bpy mesh.

    The bridge's mesh translator extracts a triangulated surface, so
    the bpy mesh's polygon count differs from ``dataset.n_cells`` in
    general (a quad source becomes two bpy polygons). Mirror that
    pipeline here so per-frame samples align with the bpy mesh post-
    translation: VTK's triangulate filter propagates cell data per
    output cell automatically.

    Returns
    -------
    tuple[np.ndarray, str] or None
        ``(scalars, "FACE")`` keyed on the post-triangulation polygon
        count; ``None`` when the dataset doesn't expose
        ``extract_surface`` or the array can't be propagated.

    """
    if array_name not in dataset.cell_data:
        return None
    if not hasattr(dataset, "extract_surface"):
        return None
    surface = dataset.extract_surface(algorithm="dataset_surface").triangulate()
    if array_name not in surface.cell_data:
        return None
    scalars = np.asarray(surface.cell_data[array_name], dtype=np.float32)
    if scalars.ndim != 1:
        return None
    return scalars.copy(), "FACE"


def _bake_shape_key_animation(
    vertex_samples: dict[str, list[tuple[int, np.ndarray]]],
    cache: SceneCache,
) -> None:
    """Materialise per-frame vertex snapshots as keyframed Shape Keys.

    For each actor with stable topology and at least one frame where
    the vertex positions differ from frame 0: add a Basis shape key,
    then one shape key per sampled frame. Each per-frame key is
    keyframed with ``value=0`` at neighbouring frames and ``value=1``
    at its own frame — adjacent keys cross-fade through 0.5 at
    sub-frame times, giving linear vertex interpolation. All
    keyframe points are switched to ``LINEAR`` so Blender doesn't
    Bezier-overshoot the morph.
    """
    # At least two frames are needed to detect deformation. A single
    # sample can't differ from itself, so skip — single-frame animation
    # is just a static export with a one-frame timeline.
    min_samples_to_compare = 2
    for actor_key, frames_and_points in vertex_samples.items():
        if len(frames_and_points) < min_samples_to_compare:
            continue
        first_pts = frames_and_points[0][1]
        if all(np.array_equal(pts, first_pts) for _, pts in frames_and_points[1:]):
            continue
        obj_name = cache.objects.get(actor_key)
        if obj_name is None:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        mesh_data = obj.data
        if not isinstance(mesh_data, bpy.types.Mesh):
            # Cached actor's bpy object isn't a mesh (curve, volume, ...).
            # Shape keys are mesh-only.
            continue
        n_obj_verts = len(mesh_data.vertices)
        if n_obj_verts != first_pts.shape[0]:
            # The bpy mesh was rebuilt (or wasn't translated) at a
            # different point count than the source dataset. Likely
            # because the mesh translator extracts a triangulated
            # surface that may not be one-to-one with dataset.points
            # for unstructured grids — shape keys can't address those
            # extra vertices, so skip with a clear warning.
            warnings.warn(
                f"mesh {obj.name}: bpy n_verts ({n_obj_verts}) != "
                f"dataset n_verts ({first_pts.shape[0]}); shape-key bake "
                f"requires a 1:1 mapping (typical for PolyData surfaces) — "
                f"skipping this mesh",
                UserWarning,
                stacklevel=3,
            )
            continue

        obj.shape_key_add(name="Basis", from_mix=False)
        for frame_index, pts in frames_and_points:
            key = obj.shape_key_add(name=f"f_{frame_index}", from_mix=False)
            key.data.foreach_set("co", pts.astype(np.float32).ravel())
            key.value = 0.0
            key.keyframe_insert("value", frame=frame_index - 1)
            key.value = 1.0
            key.keyframe_insert("value", frame=frame_index)
            key.value = 0.0
            key.keyframe_insert("value", frame=frame_index + 1)
        _set_shape_key_fcurves_linear(mesh_data.shape_keys)


def _set_shape_key_fcurves_linear(shape_keys: bpy.types.Key | None) -> None:
    """Switch every keyframe point on a Key block's fcurves to LINEAR.

    Bezier (the default) overshoots between value=0 and value=1, which
    on a triangular per-frame envelope produces visible morph artefacts
    at sub-frame times. Linear gives the textbook
    ``key_N * (1-t) + key_{N+1} * t`` blend.
    """
    if shape_keys is None or shape_keys.animation_data is None:
        return
    action = shape_keys.animation_data.action
    if action is None:
        return
    for fc in _iter_action_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _iter_action_fcurves(action: bpy.types.Action) -> Iterable[bpy.types.FCurve]:
    """Yield every fcurve on ``action`` across bpy 4.x and 5.x layouts.

    bpy 4.x exposes ``action.fcurves`` directly; 5.x reorganised actions
    into layers / strips / slots with curves under
    ``action.layers[*].strips[*].channelbag(slot).fcurves``.

    Yields
    ------
    bpy.types.FCurve
        Each fcurve in turn, regardless of bpy major version.

    """
    legacy = getattr(action, "fcurves", None)
    if legacy:
        yield from legacy
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for slot in getattr(action, "slots", []):
                cb = strip.channelbag(slot)
                if cb is not None:
                    yield from cb.fcurves


_MDD_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


def _bake_mdd_deformation(
    vertex_samples: dict[str, list[tuple[int, np.ndarray]]],
    cache: SceneCache,
    blend_path: str,
    *,
    fps: int,
) -> None:
    """Write per-mesh MDD sidecars and attach Mesh Cache modifiers.

    For each actor with stable topology and at least one varying frame:
    stack the per-frame vertex snapshots into a single
    ``(N_frames, N_verts, 3)`` array, write it as an MDD file next to
    the .blend, and add a ``MESH_CACHE`` modifier on the corresponding
    bpy object pointing at it. Blender's built-in modifier replays the
    cache on file open, without requiring Python execution.

    The MDD sidecar paths are derived from ``blend_path``'s stem plus
    the bpy object name, e.g. ``scene__PolyData.mdd``.
    """
    blend = Path(blend_path).resolve()
    blend_dir = blend.parent
    blend_stem = blend.stem
    min_samples_to_compare = 2
    for actor_key, frames_and_points in vertex_samples.items():
        if len(frames_and_points) < min_samples_to_compare:
            continue
        first_pts = frames_and_points[0][1]
        if all(np.array_equal(pts, first_pts) for _, pts in frames_and_points[1:]):
            continue
        obj_name = cache.objects.get(actor_key)
        if obj_name is None:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        mesh_data = obj.data
        if not isinstance(mesh_data, bpy.types.Mesh):
            continue
        if len(mesh_data.vertices) != first_pts.shape[0]:
            warnings.warn(
                f"mesh {obj.name}: bpy n_verts ({len(mesh_data.vertices)}) != "
                f"dataset n_verts ({first_pts.shape[0]}); MDD bake requires a "
                f"1:1 mapping (typical for PolyData surfaces) — skipping",
                UserWarning,
                stacklevel=3,
            )
            continue

        vertices = np.stack([pts.astype(np.float32) for _, pts in frames_and_points])
        frame_indices = [fi for fi, _ in frames_and_points]
        safe_name = _MDD_NAME_SAFE_RE.sub("_", obj.name)
        mdd_path = blend_dir / f"{blend_stem}__{safe_name}.mdd"
        _write_mdd(mdd_path, vertices, fps=fps)

        # ``Object.modifiers.new`` returns the abstract ``Modifier`` type
        # in fake-bpy stubs; cast to the concrete subtype so ty accepts
        # the MeshCache-specific attribute assignments below.
        mod = cast(
            "bpy.types.MeshCacheModifier",
            obj.modifiers.new(name="pvblender_meshcache", type="MESH_CACHE"),
        )
        mod.cache_format = "MDD"
        mod.filepath = str(mdd_path)
        mod.deform_mode = "OVERWRITE"
        mod.time_mode = "FRAME"
        mod.play_mode = "SCENE"
        # ``frame_start`` shifts the cache so its frame 0 aligns with
        # the first scene frame the user actually sampled.
        mod.frame_start = float(frame_indices[0])


def _write_mdd(path: Path, vertices: np.ndarray, *, fps: int) -> None:
    """Write a Lightwave MDD vertex-cache file.

    Format (big-endian throughout):

    * ``uint32`` total frames
    * ``uint32`` total vertices
    * ``float32`` x ``N_frames`` per-frame timestamps (in seconds)
    * ``float32`` x ``N_frames`` x ``N_vertices`` x ``3`` per-vertex positions

    Blender's :class:`MeshCacheModifier` reads this layout natively.
    Vertex positions are stored in object-local space; the modifier
    overwrites the bound mesh's vertex coordinates per frame.

    Parameters
    ----------
    path
        Destination ``.mdd`` file. Overwrites any existing file.
    vertices
        ``(N_frames, N_verts, 3)`` array of per-frame vertex positions.
    fps
        Frame rate used to convert frame indices to seconds for the
        timestamp table.

    """
    n_frames, n_verts, _ = vertices.shape
    header = struct.pack(">II", int(n_frames), int(n_verts))
    times = (np.arange(n_frames, dtype=np.float32) / float(fps)).astype(">f4")
    verts_be = vertices.astype(">f4", copy=False)
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(times.tobytes())
        fh.write(verts_be.tobytes())


def _bake_light_animation(
    light_samples: dict[int, list[tuple[int, _LightSnapshot]]],
) -> None:
    """Keyframe every sampled :class:`pv.Light`'s pose / energy / colour.

    For each light index ``i``, looks up the corresponding ``PVLight_i``
    bpy object created by :func:`translate_lights`, and keyframes only
    the attributes that varied across frames:

    * ``location`` + ``rotation_quaternion`` on the light object
    * ``energy`` on the light data-block (intensity → bpy energy via
      the translator's scaling, sampled directly from the per-frame
      pyvista intensity values)
    * ``color`` on the light data-block

    Skipping static values keeps the saved action small for the
    common scientific-viz case where most lights are fixed.
    """
    for index, frames in light_samples.items():
        if len(frames) < 2:  # noqa: PLR2004
            continue
        obj_name = f"PVLight_{index}"
        light_obj = bpy.data.objects.get(obj_name)
        if light_obj is None or light_obj.type != "LIGHT":
            continue
        light_data = cast("bpy.types.Light", light_obj.data)
        _keyframe_light_transform(light_obj, frames)
        _keyframe_light_energy(light_obj, light_data, frames)
        _keyframe_light_color(light_obj, light_data, frames)


def _keyframe_light_transform(
    light_obj: bpy.types.Object,
    frames: list[tuple[int, _LightSnapshot]],
) -> None:
    """Bake location + rotation_quaternion fcurves when the pose varies."""
    poses = [(pos, focal) for _, (pos, focal, _, _) in frames]
    if all(p == poses[0] for p in poses[1:]):
        return
    light_obj.rotation_mode = "QUATERNION"
    for frame_index, (pos, focal, _, _) in frames:
        light_obj.matrix_world = look_at_matrix(pos, focal, (0.0, 0.0, 1.0))
        light_obj.keyframe_insert("location", frame=frame_index)
        light_obj.keyframe_insert("rotation_quaternion", frame=frame_index)


def _keyframe_light_energy(
    light_obj: bpy.types.Object,
    light_data: bpy.types.Light,
    frames: list[tuple[int, _LightSnapshot]],
) -> None:
    """Bake the data-block ``energy`` fcurve when intensity varies.

    The static :func:`translate_lights` path scales pyvista
    ``intensity`` by a kind-specific multiplier (sun / point / spot
    each get a different factor). To keep the bake faithful to the
    static state, we read the multiplier off the current
    ``light_data.energy`` (which reflects the last-frame intensity)
    and apply it per frame.
    """
    intensities = [intensity for _, (_, _, intensity, _) in frames]
    if all(i == intensities[0] for i in intensities[1:]):
        return
    last_intensity = intensities[-1]
    # ``last_intensity`` came from ``light_data.energy`` ÷ multiplier, so
    # the only way this is meaningfully zero is the caller set
    # ``light.intensity = 0`` at the last frame — in which case there's
    # nothing to scale and the energy curve would be flat anyway.
    if abs(last_intensity) < 1e-12:  # noqa: PLR2004
        return
    # fake-bpy-module types ``bpy.types.Light`` as the abstract base;
    # ``energy`` is declared on each concrete sub-class. The Union
    # cast is structurally honest (any light data-block is one of
    # these four at runtime) and exposes ``energy: float`` for ty.
    light = cast("_AnyLight", light_data)
    current_energy = light.energy
    energy_per_intensity = current_energy / last_intensity
    for frame_index, (_, _, intensity, _) in frames:
        light.energy = intensity * energy_per_intensity
        light.keyframe_insert("energy", frame=frame_index, group=light_obj.name)


def _keyframe_light_color(
    light_obj: bpy.types.Object,
    light_data: bpy.types.Light,
    frames: list[tuple[int, _LightSnapshot]],
) -> None:
    """Bake the data-block ``color`` fcurve when the colour varies."""
    colors = [color for _, (_, _, _, color) in frames]
    if all(c == colors[0] for c in colors[1:]):
        return
    for frame_index, (_, _, _, color) in frames:
        light_data.color = color
        light_data.keyframe_insert("color", frame=frame_index, group=light_obj.name)


def _bake_volume_animation(
    volume_samples: dict[str, list[tuple[int, np.ndarray]]],
    cache: SceneCache,
    *,
    frame_start: int,
) -> None:
    """Bake per-frame volume scalar fields as a multi-frame packed atlas.

    For each volume whose scalar field actually changes across frames:

    1. Stack the per-frame ``(nz, ny, nx)`` arrays into a single
       ``(ny * n_frames, nx * nz)`` byte atlas, packed inside the
       .blend.
    2. Replace the existing static atlas image on the volume's
       material with the stacked one.
    3. Inject a keyframed ``ShaderNodeValue`` into the material's
       atlas-V coordinate so playback scrolls through the frame
       bands without any Python execution.

    Volumes whose scalars are constant across the sampled frames are
    left static, with the same shader graph and atlas image as the
    static-render path produces.
    """
    del frame_start  # ordinal=0 in keyframes already pins frames[0]
    for actor_key, frames in volume_samples.items():
        if not _volume_animation_varies(frames):
            continue
        target = _resolve_volume_bake_target(actor_key, cache)
        if target is None:
            continue
        _apply_volume_animation(target, frames)


def _volume_animation_varies(frames: list[tuple[int, np.ndarray]]) -> bool:
    """Return ``True`` when ``frames`` carries actual per-frame variation.

    Returns
    -------
    bool
        ``False`` for single-sample or constant fields (no bake needed).

    """
    min_samples = 2
    if len(frames) < min_samples:
        return False
    first = frames[0][1]
    return not all(np.array_equal(arr, first, equal_nan=True) for _, arr in frames[1:])


def _resolve_volume_bake_target(
    actor_key: str, cache: SceneCache
) -> tuple[bpy.types.Object, bpy.types.Material, bpy.types.Node] | None:
    """Resolve ``(object, material, image_node)`` for an animated volume.

    Returns
    -------
    tuple or None
        ``None`` when the cache entry is stale or the material was
        not built by the volume translator (missing image node).

    """
    from pyvista_blender.translate.volume import NODE_NAME_IMAGE  # noqa: PLC0415

    obj_name = cache.volumes.get(actor_key)
    if obj_name is None:
        return None
    obj = bpy.data.objects.get(obj_name)
    if obj is None or not isinstance(obj.data, bpy.types.Mesh):
        return None
    material = obj.data.materials[0] if obj.data.materials else None
    node_tree = material.node_tree if material is not None else None
    img_node = node_tree.nodes.get(NODE_NAME_IMAGE) if node_tree is not None else None
    if material is None or img_node is None:
        return None
    return obj, material, img_node


def _apply_volume_animation(
    target: tuple[bpy.types.Object, bpy.types.Material, bpy.types.Node],
    frames: list[tuple[int, np.ndarray]],
) -> None:
    """Swap in a multi-frame atlas + keyframe the frame-offset Value node."""
    from pyvista_blender.translate.volume import (  # noqa: PLC0415
        build_animated_atlas,
        inject_frame_offset,
    )

    obj, material, img_node = target
    per_frame = [arr for _, arr in frames]
    new_image, n_frames = build_animated_atlas(per_frame, obj.name)

    img_node_typed = cast("bpy.types.ShaderNodeTexImage", img_node)
    old_image = img_node_typed.image
    img_node_typed.image = new_image
    if old_image is not None and old_image is not new_image:
        bpy.data.images.remove(old_image)

    value_node = inject_frame_offset(material, n_frames)
    value_socket = cast("bpy.types.NodeSocketFloat", value_node.outputs[0])
    for ordinal, (frame_index, _) in enumerate(frames):
        value_socket.default_value = float(ordinal)
        value_socket.keyframe_insert("default_value", frame=frame_index)


_GlyphChannels = dict[str, list[tuple[int, np.ndarray]]]


def _bake_glyph_animation(
    glyph_samples: dict[int, _GlyphChannels],
    cache: SceneCache,
    *,
    frame_start: int,
) -> None:
    """Bake per-frame glyph state as packed images + GN sub-graph overrides.

    For each registered glyph whose source state varies, the bridge:

    1. Captures the per-frame (positions, orient, scale) arrays.
    2. Packs each varying channel into a single float-precision image
       inside the .blend: rows = frames, columns = point indices,
       channels = RGBA carrying the per-point vector / scalar values.
    3. Injects a Geometry Nodes sub-graph upstream of the existing
       instancer that samples the image at
       ``((index + 0.5) / N_points, (frame - frame_start + 0.5) / N_frames)``
       and either overrides the point positions (positions channel)
       or writes the value to the ``pv_orient`` / ``pv_scale`` named
       attribute the existing instancer reads.

    Constant channels are detected and skipped so the saved action
    only carries variation that actually happens. No Python in the
    .blend, no auto-execution prompt — playback rides on a Scene Time
    node driving the GN modifier.
    """
    for ordinal, channels in glyph_samples.items():
        target = _resolve_glyph_bake_target(ordinal, cache)
        if target is None:
            continue
        _apply_glyph_animation(target, channels, frame_start=frame_start)


def _resolve_glyph_bake_target(
    ordinal: int, cache: SceneCache
) -> tuple[bpy.types.Object, bpy.types.NodeTree] | None:
    """Resolve ``(points_obj, gn_tree)`` for an animated glyph spec.

    Returns
    -------
    tuple or None
        ``None`` when the cache entry is stale, the points object was
        removed, or the GN node group is missing.

    """
    entry = cache.glyphs.get(ordinal)
    if entry is None:
        return None
    points_obj_name, _geom_obj_name, gn_name = entry
    points_obj = bpy.data.objects.get(points_obj_name)
    if points_obj is None or not isinstance(points_obj.data, bpy.types.Mesh):
        return None
    tree = bpy.data.node_groups.get(gn_name)
    if tree is None:
        return None
    return points_obj, tree


def _apply_glyph_animation(
    target: tuple[bpy.types.Object, bpy.types.NodeTree],
    channels: _GlyphChannels,
    *,
    frame_start: int,
) -> None:
    """Bake images + splice override sub-graphs into the glyph's GN tree."""
    from pyvista_blender.translate.glyph import (  # noqa: PLC0415
        build_glyph_channel_image,
        inject_glyph_channel_override,
    )

    points_obj, tree = target
    for channel_name in ("positions", "orient", "scale"):
        frames = channels.get(channel_name)
        if not frames or not _glyph_channel_varies(frames):
            continue
        image = build_glyph_channel_image(
            channel_name, frames, base_name=points_obj.name
        )
        inject_glyph_channel_override(
            tree,
            channel_name,
            image,
            n_frames=len(frames),
            frame_start=frame_start,
        )


def _glyph_channel_varies(frames: list[tuple[int, np.ndarray]]) -> bool:
    """Return ``True`` when a glyph channel actually changes across frames.

    Returns
    -------
    bool
        ``False`` for single-sample or constant fields (no bake needed).

    """
    min_samples = 2
    if len(frames) < min_samples:
        return False
    first = frames[0][1]
    return not all(np.array_equal(arr, first) for _, arr in frames[1:])


def _bake_transform_animation(
    transform_samples: dict[str, list[tuple[int, np.ndarray]]],
    cache: SceneCache,
) -> None:
    """Keyframe each actor's :attr:`pv.Actor.user_matrix` per frame.

    For each actor whose user_matrix actually varies across the sampled
    frames: look up the cached bpy object via ``cache.objects``, switch
    the rotation mode to ``"QUATERNION"`` (avoiding Euler-interpolation
    surprises), decompose each per-frame 4x4 into translation /
    rotation / scale, and keyframe ``location`` / ``rotation_quaternion``
    / ``scale`` per frame. Actors whose matrix is identity-stable (the
    common static case) are skipped so the saved action only carries
    channels that actually animate.
    """
    min_samples_to_compare = 2
    for actor_key, frames in transform_samples.items():
        if len(frames) < min_samples_to_compare:
            continue
        first = frames[0][1]
        if all(np.allclose(m, first, atol=1e-9) for _, m in frames[1:]):
            continue
        obj_name = cache.objects.get(actor_key)
        if obj_name is None:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        obj.rotation_mode = "QUATERNION"
        for frame_index, matrix in frames:
            obj.matrix_world = Matrix(matrix.tolist())
            obj.keyframe_insert("location", frame=frame_index)
            obj.keyframe_insert("rotation_quaternion", frame=frame_index)
            obj.keyframe_insert("scale", frame=frame_index)


def _bake_material_animation(
    material_samples: dict[str, list[tuple[int, _MaterialSnapshot]]],
    cache: SceneCache,
) -> None:
    """Keyframe each cached material's Principled BSDF inputs per frame.

    For each material with varying property snapshots: look up the bpy
    :class:`bpy.types.Material` via ``cache.materials``, find every
    Principled BSDF node in its node tree (a double-sided material has
    one per side), and keyframe Base Color / Metallic / Roughness /
    Alpha — only the channels that actually vary across frames. Base
    Color is skipped when its socket is driven by a Color Attribute
    (the scalar-bake path owns that wiring).
    """
    min_samples_to_compare = 2
    for prop_key, frames in material_samples.items():
        if len(frames) < min_samples_to_compare:
            continue
        if all(snap == frames[0][1] for _, snap in frames[1:]):
            continue
        mat_name = cache.materials.get(prop_key)
        if mat_name is None:
            continue
        mat = bpy.data.materials.get(mat_name)
        if mat is None or mat.node_tree is None:
            continue
        bsdfs = [
            n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"
        ]
        if not bsdfs:
            continue
        _keyframe_bsdf_inputs(bsdfs, frames)


def _keyframe_bsdf_inputs(
    bsdfs: list[bpy.types.Node],
    frames: list[tuple[int, _MaterialSnapshot]],
) -> None:
    """Write the per-frame snapshots to every Principled BSDF in ``bsdfs``.

    Splitting this out keeps :func:`_bake_material_animation` under
    ruff's branch threshold; each BSDF can be processed independently
    because Blender's Animation API keys by socket identity.
    """
    colors = [snap[0] for _, snap in frames]
    metallics = [snap[1] for _, snap in frames]
    roughnesses = [snap[2] for _, snap in frames]
    alphas = [snap[3] for _, snap in frames]
    color_varies = any(c != colors[0] for c in colors[1:])
    metallic_varies = any(m != metallics[0] for m in metallics[1:])
    roughness_varies = any(r != roughnesses[0] for r in roughnesses[1:])
    alpha_varies = any(a != alphas[0] for a in alphas[1:])
    for bsdf in bsdfs:
        base_color_socket = bsdf.inputs.get("Base Color")
        # Skip Base Color when it's already wired up — the scalar bake
        # owns that socket via a Color Attribute / Image Texture
        # downstream of a ShaderNodeAttribute("scalars").
        if (
            color_varies
            and base_color_socket is not None
            and not base_color_socket.is_linked
        ):
            for frame_index, (color, *_) in frames:
                _set_socket_default(base_color_socket, (*color, 1.0))
                base_color_socket.keyframe_insert("default_value", frame=frame_index)
        if metallic_varies and (metallic_socket := bsdf.inputs.get("Metallic")):
            for frame_index, snap in frames:
                _set_socket_default(metallic_socket, snap[1])
                metallic_socket.keyframe_insert("default_value", frame=frame_index)
        if roughness_varies and (roughness_socket := bsdf.inputs.get("Roughness")):
            for frame_index, snap in frames:
                _set_socket_default(roughness_socket, snap[2])
                roughness_socket.keyframe_insert("default_value", frame=frame_index)
        if alpha_varies and (alpha_socket := bsdf.inputs.get("Alpha")):
            for frame_index, snap in frames:
                _set_socket_default(alpha_socket, snap[3])
                alpha_socket.keyframe_insert("default_value", frame=frame_index)


def _bake_scalar_animation(
    scalar_samples: dict[str, list[tuple[int, np.ndarray]]],
    scalar_domains: dict[str, str],
    cache: SceneCache,
    plotter: pv.BasePlotter,
    blend_path: str,
    *,
    frame_start: int,
) -> list[Path]:
    """Bake per-frame scalar fields as a packed PNG + Geometry Nodes graph.

    Dispatches on each actor's scalar domain: ``"POINT"`` (per-vertex)
    or ``"FACE"`` (per-cell). The image layout, GN graph index source,
    and Store Named Attribute domain all switch accordingly:

    * **POINT**: image shape ``(N_frames, N_verts, 4)``; ``Index`` node
      yields vertex index; attribute stored on POINT domain.
    * **FACE**: image shape ``(N_frames, N_cells, 4)``; ``Index`` node
      yields polygon index when evaluated at FACE domain; attribute
      stored on FACE domain.

    Either way the attribute name stays ``"scalars"`` so the material's
    existing scalar-aware shader graph picks it up.

    Returns
    -------
    list of Path
        The external PNG files that were written; the caller deletes
        them after :func:`bpy.ops.wm.save_as_mainfile` completes so
        the .blend remains self-contained.

    """
    actors_by_key = _collect_actors_by_identity(plotter)
    blend = Path(blend_path).resolve()
    written_pngs: list[Path] = []
    for actor_key, frames_and_scalars in scalar_samples.items():
        actor = actors_by_key.get(actor_key)
        obj = _resolve_bake_target(actor_key, cache)
        if actor is None or obj is None:
            continue
        domain = scalar_domains.get(actor_key, "POINT")
        png = _bake_one_actor_scalars(
            actor,
            obj,
            frames_and_scalars,
            blend,
            frame_start=frame_start,
            domain=domain,
        )
        if png is not None:
            written_pngs.append(png)
    return written_pngs


def _resolve_bake_target(actor_key: str, cache: SceneCache) -> bpy.types.Object | None:
    """Resolve the bpy ``Object`` to bind a scalar bake to, or ``None``.

    Returns
    -------
    bpy.types.Object or None
        The cached bpy mesh object for ``actor_key``, or ``None`` when
        the cache has no entry, the object was removed, or the
        object's data isn't a mesh (shape keys + GN scalar attribute
        only apply to meshes).

    """
    obj_name = cache.objects.get(actor_key)
    if obj_name is None:
        return None
    obj = bpy.data.objects.get(obj_name)
    if obj is None or not isinstance(obj.data, bpy.types.Mesh):
        return None
    return obj


def _bake_one_actor_scalars(
    actor: pv.Actor,
    obj: bpy.types.Object,
    frames_and_scalars: list[tuple[int, np.ndarray]],
    blend: Path,
    *,
    frame_start: int,
    domain: str,
) -> Path | None:
    """Build the PNG + GN modifier for one actor.

    Returns
    -------
    Path or None
        The external PNG path the bridge wrote (so the caller can
        delete it after :func:`bpy.ops.wm.save_as_mainfile`), or
        ``None`` when nothing was baked (single-frame, constant
        scalars, or element-count mismatch with the bpy mesh).

    """
    min_samples_to_compare = 2
    if len(frames_and_scalars) < min_samples_to_compare:
        return None
    first = frames_and_scalars[0][1]
    if all(np.array_equal(s, first) for _, s in frames_and_scalars[1:]):
        return None  # constant scalars; static "scalars" attribute is fine
    mesh_data = cast("bpy.types.Mesh", obj.data)
    n_elements_bpy = (
        len(mesh_data.vertices) if domain == "POINT" else len(mesh_data.polygons)
    )
    domain_label = "n_verts" if domain == "POINT" else "n_polys"
    if n_elements_bpy != first.shape[0]:
        warnings.warn(
            f"mesh {obj.name}: bpy {domain_label} ({n_elements_bpy}) != "
            f"dataset element count ({first.shape[0]}); {domain.lower()} "
            f"scalar bake requires a 1:1 mapping — skipping",
            UserWarning,
            stacklevel=4,
        )
        return None

    rgba_per_frame = _colormap_scalar_frames(actor, frames_and_scalars)
    safe_name = _MDD_NAME_SAFE_RE.sub("_", obj.name)
    png_path = blend.parent / f"{blend.stem}__{safe_name}_scalars.png"
    _write_scalar_png(png_path, rgba_per_frame)
    image = _load_or_replace_image(png_path)
    _attach_scalar_nodes_modifier(
        obj,
        image,
        n_elements=first.shape[0],
        n_frames=len(frames_and_scalars),
        frame_start=frame_start,
        domain=domain,
    )
    return png_path


def _collect_actors_by_identity(plotter: pv.BasePlotter) -> dict[str, pv.Actor]:
    """Map each visible actor's identity key to the actor itself.

    Returns
    -------
    dict
        ``{vtk_identity(actor): actor}`` for every renderer in the
        plotter; used to find the source actor (and its colormap)
        when rebuilding the bake from per-actor cache keys.

    """
    out: dict[str, pv.Actor] = {}
    for renderer in plotter.renderers:
        for actor in renderer.actors.values():
            if isinstance(actor, pv.Actor):
                out[vtk_identity(actor)] = actor
    return out


def _colormap_scalar_frames(
    actor: pv.Actor,
    frames_and_scalars: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    """Apply the actor's colormap to per-frame scalar arrays.

    Resolves the scalar range from the mapper (with a fall-back to the
    observed min/max across frames when the mapper's range is
    degenerate). Uses the matplotlib colormap on
    ``actor.mapper.lookup_table.cmap`` for vectorised lookup.

    Returns
    -------
    np.ndarray
        ``(N_frames, N_verts, 4)`` float32 RGBA in ``[0, 1]``.

    Raises
    ------
    RuntimeError
        When the actor's lookup table has no matplotlib colormap (the
        mesh was added without ``cmap=...``, so the bridge has nothing
        to colormap with).

    """
    scalar_stack = np.stack([s for _, s in frames_and_scalars])
    mapper = actor.mapper
    vmin, vmax = mapper.scalar_range
    if not (vmax > vmin):
        vmin = float(scalar_stack.min())
        vmax = float(scalar_stack.max())
        if not (vmax > vmin):
            vmax = vmin + 1.0  # degenerate constant field; arbitrary spread
    normalized = (scalar_stack - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    cmap = mapper.lookup_table.cmap
    if cmap is None:
        msg = (
            f"actor {actor.name!r} has scalar visibility but no matplotlib "
            f"colormap on its lookup table; add the mesh with `cmap=...` "
            f"so the bridge can colormap the scalar field"
        )
        raise RuntimeError(msg)
    rgba = cmap(normalized)  # shape (N_frames, N_verts, 4) in [0, 1]
    return rgba.astype(np.float32, copy=False)


def _write_scalar_png(path: Path, rgba_per_frame: np.ndarray) -> None:
    """Write the per-frame scalar colour table as a PNG.

    Layout: row ``i`` (top-to-bottom in pixel space) carries the
    colours for frame ``i``; column ``j`` carries vertex ``j``. The
    array is flipped vertically before writing so that the PNG row at
    the bottom is frame 0 — Blender's Image Texture samples ``V=0`` at
    the bottom row, so the GN graph can use the natural mapping
    ``V = (frame_offset + 0.5) / N_frames``.
    """
    flipped = np.flip(rgba_per_frame, axis=0)
    byte_image = np.clip(flipped * 255.0, 0.0, 255.0).astype(np.uint8)
    iio.imwrite(path, byte_image)


def _load_or_replace_image(path: Path) -> bpy.types.Image:
    """Load and **pack** the PNG so the .blend is self-contained.

    Blender's :class:`bpy.types.Image` data-block can hold both an
    external ``filepath`` and an internal ``packed_file``. After
    calling :meth:`bpy.types.Image.pack`, the image bytes live inside
    the ``.blend`` and the external file is no longer required at
    load time — Blender prefers the packed copy. The bridge deletes
    the external PNG after the .blend is saved so the .blend stays
    the single source of truth and users can't accidentally lose
    the animation by shipping the .blend alone.

    Returns
    -------
    bpy.types.Image
        The freshly loaded + packed image. Any previously cached image
        of the same name is removed first so re-runs in the same bpy
        session don't accumulate stale entries.

    """
    abs_path = str(path.resolve())
    name = path.name
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    image = bpy.data.images.load(abs_path)
    image.pack()
    return image


def _attach_scalar_nodes_modifier(
    obj: bpy.types.Object,
    image: bpy.types.Image,
    *,
    n_elements: int,
    n_frames: int,
    frame_start: int,
    domain: str,
) -> None:
    """Build the Geometry Nodes graph that samples ``image`` per frame.

    The graph computes ``(U, V)`` from the per-element index (vertex
    for POINT, polygon for FACE) and the scene frame, samples the bound
    image, and stores the resulting RGBA into a ``FLOAT_COLOR``
    attribute named ``"scalars"`` (the same name the material's shader
    graph already reads via :class:`ShaderNodeAttribute`).
    """
    group = _build_scalar_node_group(
        obj.name,
        image,
        n_elements=n_elements,
        n_frames=n_frames,
        frame_start=frame_start,
        domain=domain,
    )
    mod = cast(
        "bpy.types.NodesModifier",
        obj.modifiers.new(name="pvblender_scalars", type="NODES"),
    )
    mod.node_group = group


def _build_scalar_node_group(
    base_name: str,
    image: bpy.types.Image,
    *,
    n_elements: int,
    n_frames: int,
    frame_start: int,
    domain: str,
) -> bpy.types.GeometryNodeTree:
    """Build the Geometry Nodes group that samples ``image`` per frame.

    Returns
    -------
    bpy.types.GeometryNodeTree
        The new node group, suitable for assigning to a NODES
        modifier's ``node_group`` attribute.

    Raises
    ------
    RuntimeError
        When the freshly-created node group exposes no interface
        (Blender invariant; included for ty narrowing).

    """
    group = cast(
        "bpy.types.GeometryNodeTree",
        bpy.data.node_groups.new(f"pvblender_scalars_{base_name}", "GeometryNodeTree"),
    )
    interface = group.interface
    if interface is None:
        msg = f"node group {group.name!r} has no interface"
        raise RuntimeError(msg)
    interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = group.nodes
    links = group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")

    uv_socket = _build_uv_subgraph(
        nodes,
        links,
        n_elements=n_elements,
        n_frames=n_frames,
        frame_start=frame_start,
    )
    img_color_socket = _build_image_lookup(nodes, links, image, uv_socket)
    _build_store_attribute(
        nodes,
        links,
        group_in,
        group_out,
        value_socket=img_color_socket,
        domain=cast('Literal["POINT", "FACE"]', domain),
    )
    return group


def _build_uv_subgraph(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    *,
    n_elements: int,
    n_frames: int,
    frame_start: int,
) -> bpy.types.NodeSocket:
    """Wire ``Index`` and ``Scene Time`` into a combined ``(U, V, 0)`` vector.

    The Geometry Nodes ``Index`` node is domain-aware — it yields the
    index in whichever domain the consuming node (here, Store Named
    Attribute) evaluates at. So one UV subgraph serves both the POINT
    (vertex-indexed) and FACE (polygon-indexed) paths; the domain
    switch happens at the Store Named Attribute, not here.

    Returns
    -------
    bpy.types.NodeSocket
        The output of the ``Combine XYZ`` node carrying the UV vector.

    """
    index = nodes.new("GeometryNodeInputIndex")
    u_add = _math_node(nodes, "ADD", const=0.5)
    links.new(index.outputs["Index"], u_add.inputs[0])
    u_div = _math_node(nodes, "DIVIDE", const=float(n_elements))
    links.new(u_add.outputs[0], u_div.inputs[0])

    scene_time = nodes.new("GeometryNodeInputSceneTime")
    v_sub = _math_node(nodes, "SUBTRACT", const=float(frame_start))
    links.new(scene_time.outputs["Frame"], v_sub.inputs[0])
    v_add = _math_node(nodes, "ADD", const=0.5)
    links.new(v_sub.outputs[0], v_add.inputs[0])
    v_div = _math_node(nodes, "DIVIDE", const=float(n_frames))
    links.new(v_add.outputs[0], v_div.inputs[0])

    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(u_div.outputs[0], combine.inputs[0])
    links.new(v_div.outputs[0], combine.inputs[1])
    return combine.outputs[0]


def _build_image_lookup(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    image: bpy.types.Image,
    vector_socket: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Create an ``Image Texture`` node bound to ``image`` and ``vector_socket``.

    Returns
    -------
    bpy.types.NodeSocket
        The image node's ``Color`` output.

    """
    img_node = cast(
        "bpy.types.GeometryNodeImageTexture",
        nodes.new("GeometryNodeImageTexture"),
    )
    # Unlike :class:`ShaderNodeTexImage`, the geometry-nodes image
    # texture exposes the image as a socket input rather than a direct
    # attribute. Set the default value on the ``Image`` socket so the
    # graph binds to ``image`` without needing an external link.
    _set_socket_default(img_node.inputs["Image"], image)
    img_node.interpolation = "Closest"
    img_node.extension = "EXTEND"
    links.new(vector_socket, img_node.inputs["Vector"])
    return img_node.outputs["Color"]


def _build_store_attribute(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    group_in: bpy.types.Node,
    group_out: bpy.types.Node,
    *,
    value_socket: bpy.types.NodeSocket,
    domain: Literal["POINT", "FACE"],
) -> None:
    """Add a ``Store Named Attribute('scalars')`` node and wire it into the group.

    The ``domain`` (``"POINT"`` or ``"FACE"``) controls both where the
    attribute lives on the evaluated mesh and which domain the upstream
    ``Index`` node yields values for, since the Store node propagates
    its evaluation domain backwards through its ``Value`` input.
    Typed as the narrow ``Literal`` of the two domains this helper
    actually supports.
    """
    store = cast(
        "bpy.types.GeometryNodeStoreNamedAttribute",
        nodes.new("GeometryNodeStoreNamedAttribute"),
    )
    store.data_type = "FLOAT_COLOR"
    store.domain = domain
    _set_socket_default(store.inputs["Name"], "scalars")
    links.new(group_in.outputs[0], store.inputs["Geometry"])
    links.new(value_socket, store.inputs["Value"])
    links.new(store.outputs["Geometry"], group_out.inputs[0])


def _math_node(
    nodes: bpy.types.Nodes,
    operation: Literal["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"],
    *,
    const: float,
) -> bpy.types.ShaderNodeMath:
    """Create a :class:`ShaderNodeMath` with one constant input baked in.

    fake-bpy-module's stubs type ``Nodes.new`` as returning the abstract
    ``Node``, which lacks ``operation``; the cast unblocks ty without
    affecting runtime. ``operation`` is typed as the narrow ``Literal``
    of the ops this helper's callers actually use, which is a subset
    of ``ShaderNodeMath.operation``'s ``NodeMathItems`` enum, so direct
    assignment type-checks without a stub-internal import.

    Returns
    -------
    bpy.types.ShaderNodeMath
        The configured math node; the second input carries ``const``.

    """
    node = cast("bpy.types.ShaderNodeMath", nodes.new("ShaderNodeMath"))
    node.operation = operation
    _set_socket_default(node.inputs[1], const)
    return node


def _set_socket_default(socket: bpy.types.NodeSocket, value: object) -> None:
    """Set ``socket.default_value``.

    fake-bpy-module declares ``default_value`` only on the concrete
    socket subclasses (``NodeSocketFloat``, ``NodeSocketColor``, ...)
    even though every socket has it at runtime. Centralising the
    typing-bypass here keeps the rest of the node-tree builder calling
    a normal helper instead of scattering suppressions everywhere.
    """
    socket.default_value = value  # ty: ignore[unresolved-attribute]
