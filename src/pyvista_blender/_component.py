# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``blender`` plotter component, registered via PyVista 0.48 entry points.

This module is referenced from ``pyproject.toml``::

    [project.entry-points."pyvista.plotter_components"]
    blender = "pyvista_blender._component"

PyVista discovers the entry point lazily on first ``plotter.blender`` access,
so users see ``pl.blender.render(...)`` without ever importing
``pyvista_blender`` explicitly.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Literal, TypedDict, cast, overload

import pyvista as pv

from pyvista_blender import config
from pyvista_blender._glyph import GlyphSpec
from pyvista_blender._options import (
    _BakeChannels,
    _EngineParams,
    _PlotterSources,
    _RenderKwargs,
)
from pyvista_blender.animate import orbit_camera as _orbit_camera
from pyvista_blender.config import SUPPORTED_DEVICES, SUPPORTED_ENGINES
from pyvista_blender.hud.axes import render_axes_overlay
from pyvista_blender.hud.bounds import render_bounds_overlay
from pyvista_blender.hud.scalar_bar import render_scalar_bars
from pyvista_blender.hud.text import render_text_overlay
from pyvista_blender.interactive.tiers import resolve_tier_samples
from pyvista_blender.interactive.visibility import (
    hide_underlying_actors,
    restore_underlying_actors,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np
    import vtk
    from pyvista.plotting.render_window_interactor import RenderWindowInteractor

    from pyvista_blender.config import Device, Engine
    from pyvista_blender.interactive.overlay import OverlayHandles
    from pyvista_blender.interactive.visibility import VisibilitySnapshot
    from pyvista_blender.translate.scene import SceneCache

__all__ = ["BlenderComponent"]

_HUD_PRODUCERS: dict[str, Callable[[pv.BasePlotter, int, int], np.ndarray | None]] = {
    "scalar_bar": render_scalar_bars,
    "text": render_text_overlay,
    "axes": render_axes_overlay,
    "bounds": render_bounds_overlay,
}


class _AddVolumeKwargs(TypedDict, total=False):
    """Subset of :meth:`pl.add_volume` kwargs the bridge wrapper forwards.

    ``total=False`` so each key is optional: the wrapper only inserts
    entries the user actually passed, letting pyvista's own defaults
    apply for the rest.
    """

    scalars: str
    cmap: str
    opacity: str | float | list[float]
    clim: tuple[float, float]
    opacity_unit_distance: float
    show_scalar_bar: bool


class _Unset:
    """Internal sentinel; instances are truthy and identity-compared.

    Used as a "fall through to the next layer of config" default for
    component attributes so callers can pass ``engine=None`` explicitly
    to mean "use the module default".
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a recognizable sentinel name in tracebacks.

        Returns
        -------
        str
            The literal string ``"<unset>"``.

        """
        return "<unset>"


_UNSET: _Unset = _Unset()


@pv.register_plotter_component("blender")
class BlenderComponent:
    """Namespace attached to every :class:`pyvista.BasePlotter` as ``.blender``.

    Holds the lazily-built Blender scene state and exposes the four entry
    points: :meth:`render`, :meth:`animate`, :meth:`show`, :meth:`export_blend`.

    Resolution order for ``engine`` / ``device`` / ``samples`` / etc.:

    1. Keyword arguments to the specific call.
    2. Attributes set on this component instance (``pl.blender.engine = ...``).
    3. Module-level defaults in :mod:`pyvista_blender.config`.

    The component is constructed on first ``plotter.blender`` access and cached
    on the plotter; untouched plotters pay zero cost.
    """

    def __init__(self, plotter: pv.BasePlotter) -> None:
        """Bind the component to its owning plotter.

        Parameters
        ----------
        plotter
            The :class:`pyvista.BasePlotter` that owns this component.

        """
        self._plotter = plotter
        # Identity-keyed scene cache, lazily populated on first render.
        # Holds bpy data-block names (not references), so it survives across
        # bpy lifecycle events. See :class:`SceneCache` in translate/scene.py.
        self._scene: SceneCache | None = None

        # Glyph specs registered via :meth:`add_glyph`. Reconciled into
        # Geometry-Nodes-instanced bpy objects on each render.
        self._glyphs: list[GlyphSpec] = []

        # Live volume datasets registered via :meth:`add_volume`. Maps the
        # volume actor's ``vtk_identity`` to the user's original
        # :class:`pv.DataSet` so the bridge can read mutations on the
        # original grid directly — pyvista's :meth:`pl.add_volume` copies
        # the input, so without this registry per-frame updates of the
        # original grid would not propagate. Used by both the static
        # translator (``translate_volume``) and the animation bake
        # (``_sample_volumes``).
        self._volume_sources: dict[str, pv.DataSet] = {}

        # Per-component overrides for module defaults. ``_UNSET`` means
        # "fall through to pyvista_blender.config".
        self.engine: Engine | _Unset = _UNSET
        self.device: Device | _Unset = _UNSET
        self.samples: int | _Unset = _UNSET
        self.denoise: bool | _Unset = _UNSET
        self.transparent_bg: bool | _Unset = _UNSET

        # Interactive viewport state, populated when ``show()`` runs.
        # ``_overlay`` holds the layer-1 VTK handles; ``_visibility``
        # snapshots layer-0 actor flags so we can restore them on close;
        # ``_show_kwargs`` retains the per-call render config so the
        # resize observer can re-render at the new resolution.
        # ``_overlay_funcs`` caches the lazy-imported overlay module's
        # entry points so the resize observer doesn't re-import.
        self._overlay: OverlayHandles | None = None
        self._visibility: VisibilitySnapshot | None = None
        self._show_kwargs: dict[str, object] | None = None
        self._overlay_funcs: (
            tuple[
                Callable[[pv.BasePlotter], vtk.vtkRenderWindow],
                Callable[..., OverlayHandles],
                Callable[..., SceneCache],
            ]
            | None
        ) = None
        # Wall time of the most recent end-of-interaction render. Kept
        # for instrumentation / future per-frame deltas; not currently
        # read by any throttle path (the hybrid viewport doesn't render
        # Cycles during drag).
        self._last_drag_render_at: float = 0.0
        # Idle-tier state: ``_iren`` is the cached interactor so
        # observer callbacks can schedule / cancel VTK timers without
        # re-resolving it. ``_idle_timer_id`` is the live one-shot
        # timer id (``None`` when no idle render is pending).
        # ``_idle_delay_ms`` is the delay between EndInteractionEvent
        # and the idle-quality render.
        self._iren: RenderWindowInteractor | None = None
        self._idle_timer_id: int | None = None
        self._idle_delay_ms: float = 1000.0
        # Hybrid-viewport state. The Cycles overlay is hidden
        # during ``InteractionEvent`` so VTK's layer-0 render is what
        # shows on-screen for real-time drag; it comes back when the
        # settling / idle Cycles render lands. Starts visible (the
        # first paint shows the Cycles output).
        self._overlay_visible: bool = True

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def add_glyph(
        self,
        source: pv.DataSet,
        geom: pv.DataSet,
        *,
        orient: str | None = None,
        scale: str | None = None,
        factor: float = 1.0,
        name: str | None = None,
    ) -> None:
        """Register a glyph instancing layer for the next render.

        Unlike ``plotter.add_mesh(source.glyph(geom=geom))`` — which
        bakes one copy of ``geom`` per source point into a single
        polydata — this routes through Blender's
        :class:`GeometryNodeInstanceOnPoints`. The glyph mesh data is
        stored once and instances are generated at render time, so
        memory + render cost scale with ``N + V`` instead of ``N * V``.

        Parameters
        ----------
        source
            Dataset whose points become instance origins.
        geom
            Glyph shape to instance at every source point (typically
            ``pv.Arrow()``, ``pv.Cone()``, etc.).
        orient
            Name of a point-data 3D vector field used to orient each
            instance (instance's +Z aligns to that vector). ``None`` →
            identity rotation.
        scale
            Name of a point-data scalar field used to scale each
            instance. ``None`` → uniform scale.
        factor
            Global scale multiplier, applied on top of ``scale``.
        name
            Optional base name for the resulting bpy data blocks.

        """
        self._glyphs.append(
            GlyphSpec(
                source=source,
                geom=geom,
                orient=orient,
                scale=scale,
                factor=factor,
                name=name,
            )
        )

    def add_volume(
        self,
        dataset: pv.DataSet,
        *,
        scalars: str | None = None,
        cmap: str | None = None,
        opacity: str | float | list[float] | None = None,
        clim: tuple[float, float] | None = None,
        opacity_unit_distance: float | None = None,
        show_scalar_bar: bool | None = None,
    ) -> pv.Volume:
        """Add a volume to the plotter and pin the original dataset to the bridge.

        Thin wrapper around :meth:`pl.add_volume`. PyVista's ``add_volume``
        copies the input dataset, so per-frame updates of the original
        ``dataset`` don't propagate to the actor's mapper. Routing
        through this method registers the *original* dataset on the
        bridge so the static translator and animation baker read from
        it directly — users mutate their own grid (``grid[scalars] =
        ...``) and renders / animations see the new values without any
        ``actor.mapper.dataset`` indirection.

        Only the kwargs the bridge actively uses are exposed. For other
        pyvista volume kwargs (``ambient``, ``diffuse``, ``shade``,
        ``blending``, ...) call :meth:`pl.add_volume` directly — the
        bridge falls back to ``actor.mapper.dataset`` when a volume
        actor isn't registered here, at the cost of losing the
        live-grid update path.

        Parameters
        ----------
        dataset
            The dataset to render volumetrically. Mutations to its
            ``point_data[scalars]`` between renders are visible to the
            bridge after this call.
        scalars, cmap, opacity, clim, opacity_unit_distance, show_scalar_bar
            Forwarded to :meth:`pl.add_volume`. Same semantics, same
            defaults.

        Returns
        -------
        pv.Volume
            The volume actor created by :meth:`pl.add_volume`, for
            chaining or downstream property tweaks.

        """
        # Forward only kwargs the user actually set, so pyvista's own
        # defaults (notably ``opacity="linear"``) apply when we don't
        # pass them. TypedDict carries each value's precise type, so
        # the ``**forwarded`` splat satisfies pyvista's signature
        # without needing ``Any`` anywhere.
        forwarded: _AddVolumeKwargs = {}
        if scalars is not None:
            forwarded["scalars"] = scalars
        if cmap is not None:
            forwarded["cmap"] = cmap
        if opacity is not None:
            forwarded["opacity"] = opacity
        if clim is not None:
            forwarded["clim"] = clim
        if opacity_unit_distance is not None:
            forwarded["opacity_unit_distance"] = opacity_unit_distance
        if show_scalar_bar is not None:
            forwarded["show_scalar_bar"] = show_scalar_bar
        actor = cast("pv.Volume", self._plotter.add_volume(dataset, **forwarded))
        # Avoid a stale entry sticking around if the same actor key is
        # somehow reused; new registration wins. Use the same identity
        # function as the rest of the bridge.
        from pyvista_blender.translate.scene import vtk_identity  # noqa: PLC0415

        self._volume_sources[vtk_identity(actor)] = dataset
        return actor

    @property
    def volume_sources(self) -> dict[str, pv.DataSet]:
        """Live volume datasets registered via :meth:`add_volume`.

        Read-only snapshot of the actor-key → user-grid mapping the
        bridge consults when sampling volume scalars. Mutating the
        returned dict does not mutate the registry (it's a defensive
        copy); call :meth:`add_volume` to add, :meth:`clear_volume_sources`
        to reset.

        Returns
        -------
        dict
            ``{vtk_identity(actor): user_dataset}`` for every volume
            registered via this component's :meth:`add_volume`.

        """
        return dict(self._volume_sources)

    def clear_volume_sources(self) -> None:
        """Drop every live-dataset registration made via :meth:`add_volume`.

        Useful when the user wants the bridge to fall back to reading
        from each volume's ``actor.mapper.dataset`` (pyvista's copy),
        e.g. after the original grids have been replaced or freed.
        """
        self._volume_sources.clear()

    @property
    def registered_glyphs(self) -> tuple[GlyphSpec, ...]:
        """Glyph specs registered via :meth:`add_glyph`, in registration order.

        Returned as a tuple so callers can introspect the queue without
        mutating it; use :meth:`add_glyph` to add and :meth:`clear_glyphs`
        to reset.

        Returns
        -------
        tuple of GlyphSpec
            Snapshot of the current glyph queue.

        """
        return tuple(self._glyphs)

    def orbit_camera(
        self,
        focal_point: tuple[float, float, float] | None = None,
        *,
        n_frames: int,
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        direction: int = 1,
    ) -> Callable[[int], None]:
        """Build an :meth:`animate` updater that orbits this plotter's camera.

        Thin wrapper around :func:`pyvista_blender.animate.orbit_camera`
        that fixes the ``plotter`` argument to the owning plotter, so the
        full animation idiom stays on ``pl.blender``::

            update = pl.blender.orbit_camera(n_frames=48)
            pl.blender.animate("turn.gif", updater=update, frames=range(48))

        Parameters
        ----------
        focal_point
            Point the camera circles around. Defaults to the plotter
            camera's current focal point.
        n_frames
            Number of frames in a full 360° loop.
        axis
            World-space rotation axis. Defaults to ``+Z``.
        direction
            ``+1`` counter-clockwise, ``-1`` clockwise.

        Returns
        -------
        Callable[[int], None]
            Updater suitable for :meth:`animate`.

        """
        return _orbit_camera(
            self._plotter,
            focal_point,
            n_frames=n_frames,
            axis=axis,
            direction=direction,
        )

    def render_hud_overlay(
        self, kind: str, *, width: int, height: int
    ) -> np.ndarray | None:
        """Render a single HUD overlay to an RGBA array without running Cycles.

        Useful for debugging an overlay in isolation: change a scalar
        bar's font size, an axes triad's position, or a bounds box's
        camera projection and inspect the pixels directly instead of
        round-tripping a full render.

        Parameters
        ----------
        kind
            One of ``"scalar_bar"``, ``"text"``, ``"axes"``, ``"bounds"``.
        width, height
            Output resolution in pixels.

        Returns
        -------
        np.ndarray or None
            Shape ``(height, width, 4)`` float32 RGBA in ``[0, 1]``, or
            ``None`` when the producer found nothing to draw (e.g. no
            scalar bars on the plotter).

        Raises
        ------
        ValueError
            When ``kind`` isn't a known overlay name.

        """
        producer = _HUD_PRODUCERS.get(kind)
        if producer is None:
            known = ", ".join(sorted(_HUD_PRODUCERS))
            msg = f"unknown HUD overlay {kind!r}; known overlays: {known}"
            raise ValueError(msg)
        return producer(self._plotter, int(width), int(height))

    def clear_glyphs(self) -> None:
        """Drop every glyph spec registered via :meth:`add_glyph`."""
        self._glyphs.clear()

    def render(
        self,
        output: str,
        *,
        engine: Engine | None = None,
        device: Device | None = None,
        samples: int | None = None,
        denoise: bool | None = None,
        transparent_bg: bool | None = None,
    ) -> str:
        """Translate the PyVista scene to ``bpy`` and render one frame.

        The scene is rebuilt in bpy and the configured engine writes a
        PNG to ``output``. Each kwarg follows the three-tier resolution
        order (per-call > component > module default); ``None`` falls
        through to the next tier.

        Parameters
        ----------
        output
            Output PNG path.
        engine
            Render engine. ``"cycles"`` (path tracer) or ``"eevee"``
            (Eevee Next rasterizer).
        device
            Compute device. See :data:`~pyvista_blender.config.Device`.
        samples
            Cycles samples per pixel.
        denoise
            Whether to enable the OpenImageDenoise post-pass.
        transparent_bg
            Whether the film alpha is preserved (sets ``film_transparent``).

        Returns
        -------
        str
            The ``output`` path, for chaining.

        """
        resolved_engine = self._resolve_engine(engine)
        self._resolve_device(device)

        # ``_render_impl`` imports ``bpy`` at module top, so loading it
        # eagerly would force every ``import pyvista_blender`` (triggered
        # automatically by PyVista's entry-point discovery the moment a
        # user touches ``pl.blender``) to pay bpy's ~200 MB / ~3 s import
        # cost — even when no render is requested. Keeping the import
        # inside ``render()`` means only callers that actually render
        # pay it. PLC0415 silenced because the rule cannot be honoured
        # without breaking this contract.
        from pyvista_blender._render_impl import do_render  # noqa: PLC0415

        self._scene = do_render(
            self._plotter,
            output,
            engine_params=_EngineParams(
                engine=resolved_engine,
                device=self.resolve_config("device", call_value=device),
                samples=self.resolve_config("samples", call_value=samples),
                denoise=self.resolve_config("denoise", call_value=denoise),
                transparent_bg=self.resolve_config(
                    "transparent_bg", call_value=transparent_bg
                ),
            ),
            cache=self._scene,
            sources=self._sources(),
        )
        return output

    def animate(
        self,
        output: str,
        updater: Callable[[int], None],
        frames: Iterable[int],
        *,
        fps: int = 30,
        engine: Engine | None = None,
        device: Device | None = None,
        samples: int | None = None,
        denoise: bool | None = None,
        transparent_bg: bool | None = None,
    ) -> str:
        """Render the scene as an animation.

        Drives a per-frame loop: ``updater(frame_index)`` mutates the
        PyVista scene in place, the bridge reconciles against the
        identity cache (mesh data blocks are *refreshed*, not rebuilt),
        Cycles renders the frame, and the resulting PNGs are muxed into
        a gif / mp4 / webm / mov / mkv selected from the output extension.

        Parameters
        ----------
        output
            Output movie path. Extension picks the muxer.
        updater
            Per-frame mutation callback, invoked as ``updater(frame_index)``.
        frames
            Frame indices to render.
        fps
            Output frame rate.
        engine, device, samples, denoise, transparent_bg
            Same three-tier resolution as :meth:`render`.

        Returns
        -------
        str
            The output path, for chaining.

        """
        resolved_engine = self._resolve_engine(engine)
        self._resolve_device(device)

        # Same lazy-import reasoning as ``render()``: ``_render_impl``
        # owns the bpy import, so we keep it off the path that fires on
        # plain ``import pyvista_blender`` / ``pl.blender`` access.
        from pyvista_blender._render_impl import do_animate  # noqa: PLC0415

        _, self._scene = do_animate(
            self._plotter,
            output,
            updater,
            frames,
            fps=int(fps),
            engine_params=_EngineParams(
                engine=resolved_engine,
                device=self.resolve_config("device", call_value=device),
                samples=self.resolve_config("samples", call_value=samples),
                denoise=self.resolve_config("denoise", call_value=denoise),
                transparent_bg=self.resolve_config(
                    "transparent_bg", call_value=transparent_bg
                ),
            ),
            cache=self._scene,
            sources=self._sources(),
        )
        return output

    def show(  # noqa: PLR0913
        self,
        *,
        backend: str = "desktop",
        engine: Engine | None = None,
        device: Device | None = None,
        samples: int | None = None,
        samples_interacting: int | None = None,
        samples_settling: int | None = None,
        samples_idle: int | None = None,
        idle_delay_ms: float = 1000.0,
        denoise: bool | None = None,
        transparent_bg: bool | None = None,
        hide_underlying: bool = False,
        hud: bool = True,
        port: int = 0,
        host: str = "127.0.0.1",
        open_browser: bool = True,
    ) -> None:
        """Open the plotter window with a fullscreen Cycles overlay.

        **Hybrid viewport**: the Cycles overlay is hidden while the
        user is dragging the mouse, so VTK's 60 fps real-time
        render handles interaction. On mouse release Cycles renders
        the new pose and the overlay reappears with the path-traced
        result. ``samples_interacting`` is therefore unused by drag
        (no Cycles call happens) but kept for future use.

        Sample tiers:

        * **settling** — mouse release (``EndInteractionEvent``).
          Default 32 samples; comes back ~1 s after release.
        * **idle** — fires once ``idle_delay_ms`` after release.
          Default 128 samples; publication-quality settle. Cancelled
          by any new interaction.

        Parameters
        ----------
        backend
            ``"desktop"`` (default) opens the in-process hybrid VTK +
            Cycles viewport described below. ``"web"`` boots a Trame
            server, opens a browser tab, and serves a slim Vuetify
            page with a VtkLocalView widget + Cycles ``<img>``
            overlay. The web backend reuses the same translator /
            cache as the desktop one; only the display surface
            differs.
        engine
            Render engine. ``"cycles"`` (path-traced) is the default.
            ``"eevee"`` is wired up offline but **not supported inside
            ``show()``**: Eevee Next and VTK both want to own the X11
            / OpenGL context for the window, and on shared displays
            they crash with ``X_GLXMakeCurrent`` ``BadAccess``. Use
            Eevee through :meth:`render` / :meth:`animate` (off-screen
            plotters avoid the conflict).
        device
            Cycles compute device. See
            :data:`~pyvista_blender.config.Device`.
        samples
            **Legacy single-tier override.** When set and the per-tier
            kwargs are not, every tier uses this value. Per-tier
            kwargs override it.
        samples_interacting, samples_settling, samples_idle
            Cycles samples for each tier. ``None`` falls through to
            :data:`~pyvista_blender.config.interactive_samples` /
            :data:`~pyvista_blender.config.settled_samples` /
            :data:`~pyvista_blender.config.idle_samples`.
        idle_delay_ms
            Milliseconds after the last ``EndInteractionEvent`` before
            the idle render fires. Default 1000 ms.
        denoise
            Whether to enable the OpenImageDenoise post-pass.
        transparent_bg
            Whether the film alpha is preserved. Off by default —
            an opaque overlay is the usual interactive case.
        hide_underlying
            Hide every layer-0 ``pv.Actor`` to skip rendering the VTK
            3D scene behind the opaque Cycles overlay. Off by default:
            when every layer-0 actor is invisible, pyvista's renderer
            paints only its transparent background and the layered
            compositor on some VTK / driver combinations discards the
            higher layers along with it, leaving the window blank.
            Leaving the VTK render running costs a few ms per frame
            and is invisible behind the opaque overlay anyway. Restored
            when the plotter closes.
        hud
            Alpha-composite scalar bars / text / axes / bounds over
            the Cycles output before blitting. Matches the offline
            ``render()`` behaviour; set ``False`` for a pure Cycles view.
        port
            ``backend="web"`` only. TCP port for the Trame server.
            ``0`` (default) picks an unused port automatically; the
            resolved port appears in the URL written to stdout.
        host
            ``backend="web"`` only. Interface to bind. Default
            ``127.0.0.1`` (local access only). Set to ``"0.0.0.0"``
            for remote access — no auth, be careful.
        open_browser
            ``backend="web"`` only. Auto-open the system browser at
            the served URL. ``False`` is useful when running headless
            or driving the server from external tooling.

        Raises
        ------
        RuntimeError
            When the plotter is off-screen (``pl.off_screen=True``)
            and ``backend="desktop"``; the desktop viewport needs a
            real VTK window for its event loop. The web backend
            doesn't need one.
        ValueError
            When ``backend`` is neither ``"desktop"`` nor ``"web"``.

        """
        if backend == "web":
            self._show_web(
                samples=samples,
                samples_settling=samples_settling,
                samples_idle=samples_idle,
                idle_delay_ms=idle_delay_ms,
                engine=engine,
                device=device,
                denoise=denoise,
                transparent_bg=transparent_bg,
                port=port,
                host=host,
                open_browser=open_browser,
            )
            return
        if backend != "desktop":
            msg = (
                f"unknown show() backend {backend!r}; "
                f"supported: 'desktop' (in-process VTK + Cycles overlay), "
                f"'web' (Trame browser viewport)"
            )
            raise ValueError(msg)

        resolved_engine = self._resolve_engine(engine)
        resolved_device = self._resolve_device(device)

        pl = self._plotter
        if pl.off_screen:
            msg = (
                "pl.blender.show() needs an interactive plotter; "
                "off_screen=True has no VTK event loop. Use "
                "pl.blender.render(...) instead."
            )
            raise RuntimeError(msg)

        # Same lazy-import reasoning as render() / animate(): the
        # overlay module imports bpy transitively, so we keep it off the
        # path that fires on plain ``pl.blender`` access. The functions
        # are cached on the component so the resize observer doesn't
        # need its own lazy import.
        from pyvista_blender.interactive import overlay  # noqa: PLC0415

        self._overlay_funcs = (
            overlay.active_ren_win,
            overlay.install_overlay,
            overlay.render_and_blit,
        )
        active_ren_win, install_overlay, render_and_blit = self._overlay_funcs

        width, height = int(pl.window_size[0]), int(pl.window_size[1])
        tiers = resolve_tier_samples(
            samples=samples,
            samples_interacting=samples_interacting,
            samples_settling=samples_settling,
            samples_idle=samples_idle,
        )
        resolved_denoise = self.resolve_config("denoise", call_value=denoise)
        resolved_transparent = self.resolve_config(
            "transparent_bg", call_value=transparent_bg
        )

        # First paint pays bpy startup + scene translation + Cycles —
        # easily 1-5 s on a cold start. One stderr line so the user
        # knows the frozen-looking window is loading.
        sys.stderr.write("[pyvista-blender] rendering first frame...\n")
        sys.stderr.flush()

        if hide_underlying:
            self._visibility = hide_underlying_actors(pl)

        self._overlay = install_overlay(pl, width, height)
        # Snapshot the render config so the observer callbacks
        # (resize, drag, end, idle) can pick the per-tier sample count
        # without rethreading every render parameter through.
        self._show_kwargs = {
            "engine": resolved_engine,
            "device": resolved_device,
            "samples_interacting": tiers.interacting,
            "samples_settling": tiers.settling,
            "samples_idle": tiers.idle,
            "denoise": resolved_denoise,
            "transparent_bg": resolved_transparent,
            "hud": hud,
        }
        self._idle_delay_ms = float(idle_delay_ms)
        # First paint uses the settling tier: it's the "snapshot at
        # rest" the user expects when the window opens; the idle timer
        # then promotes to high-quality after ``idle_delay_ms``.
        self._scene = render_and_blit(
            pl,
            self._overlay,
            engine_params=_EngineParams(
                engine=resolved_engine,
                device=resolved_device,
                samples=tiers.settling,
                denoise=resolved_denoise,
                transparent_bg=resolved_transparent,
            ),
            cache=self._scene,
            sources=self._sources(),
            hud=hud,
        )

        # Resize observer: ``render_and_blit`` is the actual resizer
        # (it queries ``ren_win.GetSize`` every call and reshapes the
        # vtkImageData in place); this observer just nudges a re-render
        # when the window dimensions change so the user doesn't have to
        # wait for the next interaction. VTK's stub types
        # ``AddObserver``'s event arg as ``int``; the C++ API also
        # accepts the string event name (which is how every Python
        # example writes it).
        active_ren_win(pl).AddObserver(
            cast("int", "ModifiedEvent"), self._on_window_resize
        )

        # Camera tracking + three-tier sampling.
        # PyVista's trackball interactor mutates ``pl.camera`` on drag.
        # We re-render on:
        #   * ``InteractionEvent``    (throttled) → interacting tier
        #   * ``EndInteractionEvent`` (always)    → settling tier, schedule idle
        #   * ``TimerEvent``          (one-shot)  → idle tier
        iren = pl.iren
        if iren is None:
            msg = "plotter.iren is None, no interactor to bind viewport observers to"
            raise RuntimeError(msg)
        self._iren = iren
        iren.add_observer("InteractionEvent", self._on_interaction_drag)
        iren.add_observer("EndInteractionEvent", self._on_interaction_end)
        iren.add_observer("TimerEvent", self._on_idle_timer)
        # Schedule the first idle render: if the user opens the window
        # and just looks, ``idle_delay_ms`` later they get the high-
        # quality settled frame for free.
        self._schedule_idle_render()

        pl.show(auto_close=False)

    def _on_window_resize(self, *_unused: object) -> None:
        """Trigger a settling-tier re-render on ``ren_win`` ``ModifiedEvent``.

        :func:`render_and_blit` queries ``ren_win.GetSize`` and reshapes
        the overlay's ``vtkImageData`` in place each call, so all we do
        here is fire a re-render when the window size has drifted. The
        early-return short-circuits the common case (Modified fires for
        many reasons; we only want to act on actual resize).
        """
        if self._overlay is None or self._overlay_funcs is None:
            return
        active_ren_win, _, _ = self._overlay_funcs
        new_size = active_ren_win(self._plotter).GetSize()
        new_width, new_height = int(new_size[0]), int(new_size[1])
        if (new_width, new_height) == (self._overlay.width, self._overlay.height):
            return
        # Resize is a "settled" event from the user's POV — they
        # finished dragging the window corner. Schedule the idle render
        # to follow.
        self._rerender_overlay(tier="settling")
        self._schedule_idle_render()

    def _rerender_overlay(self, *, tier: str) -> None:
        """Re-run ``render_and_blit`` at ``tier``'s sample count, then redraw.

        Pulls device / denoise / transparent / hud from the snapshot
        ``self._show_kwargs`` took in :meth:`show`; samples come from
        the per-tier entry (``samples_interacting`` /
        ``samples_settling`` / ``samples_idle``). After the blit,
        ``ren_win.Render()`` forces a window repaint.

        No-op when called outside an active ``show()`` (every required
        piece is ``None``).
        """
        if (
            self._overlay is None
            or self._show_kwargs is None
            or self._overlay_funcs is None
        ):
            return
        active_ren_win, _, render_and_blit = self._overlay_funcs
        kwargs = self._show_kwargs
        self._scene = render_and_blit(
            self._plotter,
            self._overlay,
            engine_params=_EngineParams(
                engine=cast("str", kwargs["engine"]),
                device=cast("str", kwargs["device"]),
                samples=cast("int", kwargs[f"samples_{tier}"]),
                denoise=cast("bool", kwargs["denoise"]),
                transparent_bg=cast("bool", kwargs["transparent_bg"]),
            ),
            cache=self._scene,
            sources=self._sources(),
            hud=cast("bool", kwargs["hud"]),
        )
        # Bring the overlay back on top after every Cycles render so
        # the user sees the fresh frame. ``_on_interaction_drag`` hides
        # it during drag; this re-shows it once the settling / idle
        # render lands. Re-shows are no-ops if already visible.
        if not self._overlay_visible:
            self._overlay.actor.SetVisibility(1)
            self._overlay_visible = True
        active_ren_win(self._plotter).Render()

    def _on_interaction_drag(self, *_unused: object) -> None:
        """Hide the Cycles overlay during ``InteractionEvent``; show VTK underneath.

        Cycles inside ``show()`` pays a ~700 ms per-frame fixed cost
        (X11/GL context sharing with VTK, not optimisable from the
        Python side). To keep drag real-time we skip Cycles entirely
        while the mouse is moving: the layer-N image actor is set
        invisible, layer 0's VTK render becomes visible, and the user
        gets 60 fps interaction for free. The settled Cycles frame
        comes back on :meth:`_on_interaction_end`.

        Any pending idle render is cancelled — drag is the antithesis
        of idle.
        """
        if self._overlay is None:
            return
        self._cancel_idle_render()
        if self._overlay_visible:
            self._set_overlay_visibility(visible=False)

    def _on_interaction_end(self, *_unused: object) -> None:
        """Settling-tier re-render fired by ``EndInteractionEvent``, then arm idle.

        Mouse release. Runs Cycles at the settling tier (~1 s on a
        typical scene), which calls :meth:`_set_overlay_visibility`
        to bring the overlay back, then arms the idle timer for the
        high-quality render that follows.
        """
        if self._overlay is None:
            return
        self._last_drag_render_at = time.monotonic()
        self._rerender_overlay(tier="settling")
        self._schedule_idle_render()

    def _set_overlay_visibility(self, *, visible: bool) -> None:
        """Toggle the overlay image actor's visibility and repaint.

        Hiding the overlay lets layer 0's VTK render show through;
        showing it overrides VTK with the latest Cycles output. The
        explicit ``Render()`` call is what actually flips what's on
        screen — without it, the change waits for the next event-loop
        tick.
        """
        if self._overlay is None or self._overlay_funcs is None:
            return
        self._overlay.actor.SetVisibility(1 if visible else 0)
        self._overlay_visible = visible
        active_ren_win, _, _ = self._overlay_funcs
        active_ren_win(self._plotter).Render()

    def _on_idle_timer(self, *_unused: object) -> None:
        """Idle-tier re-render fired by VTK ``TimerEvent``.

        VTK fires ``TimerEvent`` for any timer the interactor manages.
        We check the firing timer id against ours and act only on a
        match; the slot is then cleared so a subsequent
        :meth:`_cancel_idle_render` doesn't try to destroy a fired-and-
        gone timer.
        """
        vtk_iren = self._vtk_interactor()
        if vtk_iren is None or self._idle_timer_id is None:
            return
        if int(vtk_iren.GetTimerEventId()) != self._idle_timer_id:
            return
        self._idle_timer_id = None
        self._rerender_overlay(tier="idle")

    def _schedule_idle_render(self) -> None:
        """Arm a one-shot VTK timer that fires the idle-tier render.

        Cancels any prior pending idle timer first so we never queue
        more than one.
        """
        vtk_iren = self._vtk_interactor()
        if vtk_iren is None:
            return
        self._cancel_idle_render()
        self._idle_timer_id = int(vtk_iren.CreateOneShotTimer(int(self._idle_delay_ms)))

    def _cancel_idle_render(self) -> None:
        """Destroy any pending idle timer.

        Safe to call when no timer is scheduled — the slot is checked
        before touching the interactor.
        """
        vtk_iren = self._vtk_interactor()
        if vtk_iren is None or self._idle_timer_id is None:
            return
        vtk_iren.DestroyTimer(self._idle_timer_id)
        self._idle_timer_id = None

    def _vtk_interactor(self) -> vtk.vtkRenderWindowInteractor | None:
        """Return the underlying VTK interactor narrowed to non-``None``.

        PyVista wraps the VTK ``vtkRenderWindowInteractor`` and types
        ``.interactor`` as optional; in a live ``show()`` it is always
        present. This helper centralises the narrowing so timer
        scheduling / cancellation doesn't repeat the ``None`` check.

        Returns
        -------
        vtk.vtkRenderWindowInteractor or None
            The interactor, or ``None`` outside an active ``show()``.

        """
        if self._iren is None:
            return None
        vtk_iren = self._iren.interactor
        if vtk_iren is None:
            return None
        return vtk_iren

    def export_blend(self, path: str) -> str:
        """Translate the current plotter scene and save it as a ``.blend`` file.

        Everything that ``pl.blender.render()`` would draw lands in the
        archive: meshes, materials, lights, world, camera, registered
        glyphs. Open the resulting file in Blender's UI to tweak
        materials, add props, bake animations — anything the bridge
        doesn't expose directly.

        Parameters
        ----------
        path
            Destination ``.blend`` file path. Overwrites any existing
            file at that path.

        Returns
        -------
        str
            The resolved ``path`` for chaining.

        """
        # Same lazy-import reasoning as ``render()`` / ``animate()`` /
        # ``show()``: ``_render_impl`` pulls in ``bpy``; deferring the
        # import keeps ``pl.blender`` access cheap when no render is
        # requested.
        from pyvista_blender._render_impl import do_export_blend  # noqa: PLC0415

        _, self._scene = do_export_blend(
            self._plotter,
            path,
            cache=self._scene,
            sources=self._sources(),
        )
        return path

    def export_animation_blend(
        self,
        path: str,
        updater: Callable[[int], None],
        frames: Iterable[int],
        *,
        fps: int = 30,
        bake_camera: bool = True,
        bake_deformation: bool | str = False,
        bake_scalars: bool = False,
        bake_lights: bool = False,
        bake_transforms: bool = False,
        bake_materials: bool = False,
        bake_volume: bool = False,
        bake_glyphs: bool = False,
    ) -> str:
        """Save a ``.blend`` whose timeline plays the per-frame ``updater``.

        Each channel is gated by its own kwarg so callers can bake only
        what they want and finish the rest by hand in Blender:

        * **Camera** (``bake_camera=True``, default): keyframe the bpy
          scene camera's ``location`` and ``rotation_quaternion`` per
          frame. Set ``False`` to leave the camera static so the user
          can animate it manually.
        * **Mesh deformation** (``bake_deformation``):

          - ``False`` (default): no deformation bake.
          - ``True`` or ``"mdd"``: write a sidecar ``.mdd`` cache file
            next to the .blend and attach a :class:`MeshCacheModifier`
            to the deforming mesh. Blender plays it back natively, no
            Python script needed (no auto-execution prompt).
          - ``"shape_keys"``: alternative backend that keeps everything
            inside the ``.blend`` as morph targets (one Shape Key per
            frame). Heavier file but self-contained — no sidecar to
            ship alongside.
        * **Scalar field** (``bake_scalars=True``): bake the active
          scalar field across frames as a packed PNG image and attach
          a Geometry Nodes modifier that overrides the mesh's
          ``"scalars"`` Color Attribute per frame. Handles both
          point-data and cell-data scalars; the material's existing
          scalar-aware shader graph picks the result up automatically.
          Like MDD, no Python in the .blend.
        * **Lights** (``bake_lights=True``): keyframe each
          :class:`pv.Light`'s world position, orientation, intensity,
          and colour per frame, mirroring the camera path. Static
          lights are skipped so the action stays small.
        * **Actor transforms** (``bake_transforms=True``): keyframe
          each actor's ``user_matrix`` per frame. The bpy mesh's
          vertex data stays untouched; the per-frame transform rides
          on ``obj.matrix_world``. Static actors are skipped.
          Complements the static-render path that already honours
          ``user_matrix`` for one-frame output.
        * **Material properties** (``bake_materials=True``): keyframe
          each material's Principled BSDF inputs per frame:
          base colour (when no scalar field is driving it), metallic,
          roughness, alpha. Only the inputs that actually vary land
          in the saved action. Phong shading is converted to PBR
          roughness via the Walter fit, matching the static-render
          path.

        ``scene.frame_start``, ``scene.frame_end``, and
        ``scene.render.fps`` are configured so opening the file in
        Blender plays the animation directly (press Spacebar, or
        render the timeline at higher quality through the UI).

        Limitations:

        * **Constant topology required for deformation bake.** Any
          actor whose point count changes mid-animation emits a
          :class:`UserWarning` and is skipped.
        * Light and material animations beyond scalars are not yet
          baked (light pose / intensity keyframing on the roadmap).

        Parameters
        ----------
        path
            Destination ``.blend`` path. Overwrites any existing file.
            MDD sidecars are written alongside as
            ``<stem>__<mesh_name>.mdd``.
        updater
            Per-frame mutation callback, invoked as
            ``updater(frame_index)``.
        frames
            Frame indices to bake.
        fps
            Output frame rate. Default 30.
        bake_camera
            Keyframe the bpy camera per frame. Default ``True``.
        bake_deformation
            Choose the deformation backend. ``True`` (or ``"mdd"``)
            writes a Mesh Cache sidecar; ``"shape_keys"`` uses the
            self-contained Shape-Key path; ``False`` (default) skips
            deformation entirely.
        bake_scalars
            Bake the active per-vertex scalar field per frame as a
            packed PNG image + Geometry Nodes modifier. The actor must
            have been added with ``scalars=...`` + ``cmap=...`` for
            the colormap to resolve. Default ``False``. Both point-
            data and cell-data scalar fields are supported.
        bake_lights
            Keyframe each light's world position, orientation,
            intensity, and colour per frame. Default ``False``.
        bake_transforms
            Keyframe each actor's ``user_matrix`` per frame
            (decomposed into ``location`` + ``rotation_quaternion`` +
            ``scale`` on the bpy object). Default ``False``. Static
            actors are skipped so the saved action only carries the
            channels that actually animate.
        bake_materials
            Keyframe the Principled BSDF inputs on each cached
            material — Base Color (when scalar-driven), Metallic,
            Roughness, Alpha — per frame. Default ``False``.
        bake_volume
            Bake per-frame volume scalar fields into a multi-frame
            atlas image packed inside the .blend. The volume
            material's shader graph gets a frame-indexed Value node
            that scrolls through the atlas via keyframes. Volumes
            whose scalars are constant across the sampled frames are
            left static. Default ``False``.
        bake_glyphs
            Bake per-frame glyph state — point positions plus any
            ``orient`` vector / ``scale`` scalar field declared via
            :meth:`add_glyph` — into per-channel images packed inside
            the .blend. A Geometry Nodes sub-graph upstream of each
            glyph's instancer samples the images at ``(point_index,
            frame)`` and overrides the corresponding live attributes
            so the instances re-place / re-orient / re-scale per
            frame at playback. Constant channels are detected and
            skipped so the saved action stays clean. Default
            ``False``.

        Returns
        -------
        str
            The resolved ``path`` for chaining.

        """
        # Same lazy-import reasoning as the other entry points:
        # ``_render_impl`` pulls in bpy, so deferring the import keeps
        # ``pl.blender`` access cheap when no render is requested.
        from pyvista_blender._render_impl import (  # noqa: PLC0415
            do_export_animation_blend,
        )

        _, self._scene = do_export_animation_blend(
            self._plotter,
            path,
            updater,
            frames,
            fps=int(fps),
            bake=_BakeChannels(
                camera=bool(bake_camera),
                deformation=bake_deformation,
                scalars=bool(bake_scalars),
                lights=bool(bake_lights),
                transforms=bool(bake_transforms),
                materials=bool(bake_materials),
                volume=bool(bake_volume),
                glyphs=bool(bake_glyphs),
            ),
            cache=self._scene,
            sources=self._sources(),
        )
        return path

    def _show_web(
        self,
        *,
        samples: int | None,
        samples_settling: int | None,
        samples_idle: int | None,
        idle_delay_ms: float,
        engine: Engine | None,
        device: Device | None,
        denoise: bool | None,
        transparent_bg: bool | None,
        port: int,
        host: str,
        open_browser: bool,
    ) -> None:
        """Boot the Trame web viewport for the bound plotter.

        Blocks until the Trame server stops (Ctrl-C or all client tabs
        closed). The same translator / cache / Cycles path is used as
        on :meth:`render`, so the rendered output is bit-identical to
        what an offline render would produce at the same sample count.

        Raises
        ------
        ImportError
            When the Trame stack (``trame``, ``trame-vtk``,
            ``trame-vuetify``) is not installed. The bridge reuses
            pyvista's Trame stack, so the error guides the user to
            ``pip install 'pyvista[jupyter]'``.

        """
        try:
            from pyvista_blender.web import (  # noqa: PLC0415
                DEFAULT_IDLE_SAMPLES,
                DEFAULT_SETTLED_SAMPLES,
                serve,
            )
        except ImportError as err:  # pragma: no cover - import guard
            msg = (
                "show(backend='web') needs the Trame stack (trame, "
                "trame-vtk, trame-vuetify). Install pyvista's jupyter "
                "extra: pip install 'pyvista[jupyter]'"
            )
            raise ImportError(msg) from err

        render_kwargs: _RenderKwargs = {}
        if engine is not None:
            render_kwargs["engine"] = self.resolve_config("engine", call_value=engine)
        if device is not None:
            render_kwargs["device"] = self.resolve_config("device", call_value=device)
        if denoise is not None:
            render_kwargs["denoise"] = self.resolve_config(
                "denoise", call_value=denoise
            )
        if transparent_bg is not None:
            render_kwargs["transparent_bg"] = self.resolve_config(
                "transparent_bg", call_value=transparent_bg
            )

        settled = samples_settling if samples_settling is not None else samples
        if settled is None:
            settled = DEFAULT_SETTLED_SAMPLES
        idle = samples_idle if samples_idle is not None else DEFAULT_IDLE_SAMPLES

        serve(
            self._plotter,
            port=int(port),
            host=str(host),
            open_browser=bool(open_browser),
            samples=int(settled),
            samples_idle=int(idle),
            idle_delay_ms=float(idle_delay_ms),
            render_kwargs=render_kwargs,
        )

    def _sources(self) -> _PlotterSources:
        """Bundle the per-call registries forwarded to every ``do_*`` call.

        Snapshots the current glyph and volume-source registrations into
        a :class:`_PlotterSources` instance. Cheap to construct (frozen
        dataclass with slots), so each entry point can call this without
        worrying about overhead.

        Returns
        -------
        _PlotterSources
            Carrier with the live ``glyphs`` list and ``volume_sources``
            mapping read directly from this component.

        """
        return _PlotterSources(glyphs=self._glyphs, volume_sources=self._volume_sources)

    def _resolve_engine(self, engine: Engine | None) -> Engine:
        """Resolve and validate the engine against :data:`SUPPORTED_ENGINES`.

        Parameters
        ----------
        engine
            Per-call override; ``None`` falls through to component then
            module defaults.

        Returns
        -------
        Engine
            The resolved engine name (currently ``"cycles"`` or
            ``"eevee"``).

        Raises
        ------
        ValueError
            When the resolved engine is not in
            :data:`~pyvista_blender.config.SUPPORTED_ENGINES`.

        """
        resolved = self.resolve_config("engine", call_value=engine)
        if resolved not in SUPPORTED_ENGINES:
            msg = (
                f"unknown engine {resolved!r}; "
                f"supported engines: {sorted(SUPPORTED_ENGINES)}"
            )
            raise ValueError(msg)
        return resolved

    def _resolve_device(self, device: Device | None) -> Device:
        """Resolve and validate the device against the public allowlist.

        Parameters
        ----------
        device
            Per-call override; ``None`` falls through to component then
            module defaults.

        Returns
        -------
        Device
            The resolved device name, lowercased.

        Raises
        ------
        ValueError
            When the resolved device is not in
            :data:`~pyvista_blender.config.SUPPORTED_DEVICES`. The
            same check also fires inside
            :func:`pyvista_blender.render._device.select_cycles_device`
            as defense in depth; running it here lets typos surface
            before paying ``bpy``'s import cost.

        """
        resolved = cast(
            "Device", self.resolve_config("device", call_value=device).lower()
        )
        if resolved not in SUPPORTED_DEVICES:
            msg = (
                f"unknown device {resolved!r}; "
                f"supported devices: {sorted(SUPPORTED_DEVICES)}"
            )
            raise ValueError(msg)
        return resolved

    @overload
    def resolve_config(
        self, attr: Literal["engine"], *, call_value: Engine | None
    ) -> Engine: ...
    @overload
    def resolve_config(
        self, attr: Literal["device"], *, call_value: Device | None
    ) -> Device: ...
    @overload
    def resolve_config(
        self, attr: Literal["samples"], *, call_value: int | None
    ) -> int: ...
    @overload
    def resolve_config(
        self, attr: Literal["denoise"], *, call_value: bool | None
    ) -> bool: ...
    @overload
    def resolve_config(
        self, attr: Literal["transparent_bg"], *, call_value: bool | None
    ) -> bool: ...
    def resolve_config(
        self,
        attr: Literal["engine", "device", "samples", "denoise", "transparent_bg"],
        *,
        call_value: Engine | Device | int | bool | None,
    ) -> Engine | Device | int | bool:
        """Apply the three-tier resolution: call > component > module.

        Exposed as a public method (rather than private ``_resolve``) so
        tests and advanced users can introspect which configuration value
        a future render call would actually use. Overloaded so callers
        get the precise per-attr return type (``Engine`` for
        ``"engine"``, ``int`` for ``"samples"``, …) without a ``cast``.

        Parameters
        ----------
        attr
            Name of the config field (``"engine"``, ``"device"``,
            ``"samples"``, ``"denoise"``, ``"transparent_bg"``).
        call_value
            The per-call value; ``None`` falls through to component then
            module.

        Returns
        -------
        Engine or Device or int or bool
            The first non-sentinel value found, walking call kwarg →
            component attribute → module-level default. Concrete type
            per overload matches ``attr``.

        """
        if call_value is not None:
            return call_value
        component_value = getattr(self, attr)
        if component_value is not _UNSET:
            return cast("Engine | Device | int | bool", component_value)
        return cast("Engine | Device | int | bool", getattr(config, attr))

    # ------------------------------------------------------------------ #
    # Lifecycle hooks (called by PyVista's component registry)           #
    # ------------------------------------------------------------------ #

    def __plotter_close__(self) -> None:  # noqa: PLW3201
        # The dunder name is dictated by PyVista's component registry;
        # the rule literally cannot be satisfied without breaking the
        # framework contract.
        """Release ``bpy`` data blocks and tear down the overlay on close.

        Called by PyVista when the plotter is closed. Fires only if
        this component was actually accessed. If :meth:`show` ran, the
        layer-1 overlay renderer is removed and the layer-0 actors that
        :func:`hide_underlying_actors` masked are restored.
        """
        self._cancel_idle_render()
        if self._overlay is not None:
            ren_win = self._plotter.ren_win
            if ren_win is not None:
                ren_win.RemoveRenderer(self._overlay.renderer)
            self._overlay = None
        if self._visibility is not None:
            restore_underlying_actors(self._plotter, self._visibility)
            self._visibility = None
        self._show_kwargs = None
        self._overlay_funcs = None
        self._iren = None
        self._overlay_visible = True
        self._scene = None
