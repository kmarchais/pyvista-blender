# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Trame application class + ``serve()`` entry point for the web viewport.

Mirrors the desktop hybrid viewport's render-on-EndInteraction pattern:

* :class:`trame_vtk.widgets.vtk.VtkLocalView` renders the plotter's
  ``ren_win`` in the browser via VTK.js (60 fps drag).
* An ``<img>`` overlay shows the latest Cycles render produced by
  :meth:`pl.blender.render`. The overlay is **hidden** when the user
  starts dragging (so VTK's real-time draw shows through) and
  **re-shown** after the bridge finishes the settled-quality Cycles
  pass on ``EndInteractionEvent``.

The Trame state surface is intentionally narrow: ``cycles_data_url``
(the inline ``data:image/png;base64,...`` for the overlay),
``cycles_visible`` (bool, drives the overlay's ``v_show``), and
``samples`` (the per-render Cycles sample count, exposed as a small
slider). Future iterations can layer per-tier sampling and a real
toolbar onto this baseline.
"""

from __future__ import annotations

import base64
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import html, vuetify3
from trame.widgets import vtk as vtk_widgets

if TYPE_CHECKING:
    import pyvista as pv

    from pyvista_blender._options import _RenderKwargs

__all__ = [
    "DEFAULT_IDLE_DELAY_MS",
    "DEFAULT_IDLE_SAMPLES",
    "DEFAULT_SETTLED_SAMPLES",
    "BlenderWebApp",
    "serve",
]

#: Default Cycles samples for the on-release re-render. Matches the
#: desktop viewport's ``settled_samples``.
DEFAULT_SETTLED_SAMPLES = 32

#: Default Cycles samples for the idle-tier promotion render. Matches
#: the desktop viewport's ``idle_samples``.
DEFAULT_IDLE_SAMPLES = 128

#: Default delay between the settled render and the idle promotion.
#: Matches the desktop viewport's ``idle_delay_ms``.
DEFAULT_IDLE_DELAY_MS = 1000.0


class BlenderWebApp(TrameApp):
    """Trame app that serves a PyVista plotter through the bridge.

    The app holds a reference to the live :class:`pv.BasePlotter` and
    delegates rendering to ``plotter.blender.render``. UI state lives
    on :attr:`self.state`; the drag-vs-settled handoff is wired
    through the VTK widget's ``StartInteractionEvent`` /
    ``EndInteractionEvent`` callbacks.
    """

    def __init__(
        self,
        plotter: pv.BasePlotter,
        *,
        server: object | None = None,
        samples: int = DEFAULT_SETTLED_SAMPLES,
        samples_idle: int = DEFAULT_IDLE_SAMPLES,
        idle_delay_ms: float = DEFAULT_IDLE_DELAY_MS,
        render_kwargs: _RenderKwargs | None = None,
    ) -> None:
        """Bind the app to ``plotter`` and build the UI.

        Parameters
        ----------
        plotter
            Live plotter whose state should be rendered. The bridge
            translates exactly what :meth:`pl.blender.render` would.
        server
            Optional pre-built Trame server. ``None`` lets the
            framework create one (the common case).
        samples
            Cycles samples for the settled-tier render fired on
            ``EndInteractionEvent``. Surfaced as a slider in the
            toolbar so users can tune quality at runtime.
        samples_idle
            Cycles samples for the idle-tier promotion render fired
            ``idle_delay_ms`` after the last interaction. Defaults
            to 4x the settled tier — publication-quality settle that
            the user only ever sees when the scene is at rest.
        idle_delay_ms
            Milliseconds between the settled render landing and the
            idle promotion firing. Any new interaction cancels the
            pending timer; the next ``EndInteractionEvent`` starts
            a fresh one.
        render_kwargs
            Forwarded to :meth:`pl.blender.render` on every settled
            and idle render. ``samples`` is overridden per tier so
            the toolbar slider (settled) and ``samples_idle``
            (idle) stay authoritative.

        """
        super().__init__(server, client_type="vue3")
        self._plotter = plotter
        self._render_kwargs: _RenderKwargs = {**render_kwargs} if render_kwargs else {}
        #: Cycles samples for the idle-tier render. Read-only after
        #: construction; the toolbar slider drives the settled tier
        #: instead (state.samples).
        self.samples_idle: int = int(samples_idle)
        #: Seconds between the settled render landing and the idle
        #: promotion timer firing. ``0`` disables idle promotion.
        self.idle_delay_s: float = float(idle_delay_ms) / 1000.0
        #: The live one-shot timer driving the next idle promotion,
        #: or ``None`` when no promotion is pending. Exposed for
        #: introspection — most users call :meth:`cancel_idle_promotion`
        #: instead of touching it directly.
        self.idle_timer: threading.Timer | None = None
        self._idle_lock = threading.Lock()
        self.state.samples = int(samples)
        self.state.cycles_data_url = ""
        self.state.cycles_visible = False
        self.state.rendering = False
        self._build_ui()
        # Trigger one render on first browser connection so the user
        # sees something immediately rather than a blank overlay.
        self.ctrl.on_server_ready.add(self._initial_render)

    def _build_ui(self) -> None:
        """Compose the VTK widget + image overlay + slim toolbar."""
        with SinglePageLayout(self.server, full_height=True) as self.ui:
            self.ui.title.set_text("pyvista-blender")
            with self.ui.toolbar:
                vuetify3.VSpacer()
                vuetify3.VSlider(
                    v_model=("samples", DEFAULT_SETTLED_SAMPLES),
                    min=4,
                    max=512,
                    step=4,
                    label="Samples",
                    hide_details=True,
                    style="max-width: 280px;",
                )
                vuetify3.VBtn(
                    icon="mdi-camera-iris",
                    click=self._settled_render,
                    title="Re-render with Cycles",
                )
                vuetify3.VBtn(
                    icon="mdi-crop-free",
                    click=self._reset_camera,
                    title="Reset camera",
                )
            with (
                self.ui.content,
                html.Div(
                    style=(
                        "position: relative; width: 100%; height: 100%;"
                        " background: #0c1018;"
                    ),
                ),
            ):
                vtk_view = vtk_widgets.VtkLocalView(
                    self._plotter.ren_win,
                    interactive_ratio=1.0,
                    ref="vtkView",
                    StartInteractionEvent=self._on_drag_start,
                    EndInteractionEvent=self._on_drag_end,
                )
                self._vtk_view = vtk_view
                self.ctrl.view_update = vtk_view.update
                self.ctrl.view_reset_camera = vtk_view.reset_camera
                html.Img(
                    src=("cycles_data_url",),
                    v_show=("cycles_visible",),
                    style=(
                        "position: absolute; inset: 0;"
                        " width: 100%; height: 100%;"
                        " object-fit: contain;"
                        " pointer-events: none;"
                    ),
                )

    def _initial_render(self, **_kwargs: object) -> None:
        """Kick off the first Cycles render after the browser connects."""
        # Push to a background thread so the server-ready callback
        # returns quickly (Trame's event loop is single-threaded).
        threading.Thread(target=self._settled_render, daemon=True).start()

    def _on_drag_start(self, *_args: object, **_kwargs: object) -> None:
        """Hide the Cycles overlay so the VTK preview can be seen."""
        self.cancel_idle_promotion()
        self.state.cycles_visible = False
        self.state.flush()

    def _on_drag_end(self, *_args: object, **_kwargs: object) -> None:
        """Render Cycles on a worker and bring the overlay back."""
        threading.Thread(target=self._settled_render, daemon=True).start()

    def _settled_render(self) -> None:
        """Run the settled-tier Cycles render and schedule idle promotion."""
        if self._tier_render(self.state.samples):
            self.schedule_idle_promotion()

    def _idle_render(self) -> None:
        """Run the higher-quality idle-tier Cycles render.

        Skipped silently when the user kicked off a new interaction
        between the timer firing and this method acquiring the
        rendering lock (the timer-cancel path on
        :meth:`_on_drag_start` is the primary defence, but the
        debounce-on-rendering guard is the belt-and-suspenders one).
        """
        self._tier_render(self.samples_idle)

    def _tier_render(self, samples: int) -> bool:
        """Run one Cycles render at ``samples`` and update the overlay.

        Returns
        -------
        bool
            ``True`` when the render actually ran and the overlay
            state was updated; ``False`` when a concurrent render was
            already in flight (the caller should not chain follow-up
            actions like idle-timer scheduling on a skipped render).

        """
        if self.state.rendering:
            return False
        self.state.rendering = True
        self.state.flush()
        try:
            png_bytes = self._render_to_png_bytes(int(samples))
        finally:
            self.state.rendering = False
        b64 = base64.b64encode(png_bytes).decode("ascii")
        self.state.cycles_data_url = f"data:image/png;base64,{b64}"
        self.state.cycles_visible = True
        self.state.flush()
        return True

    def _render_to_png_bytes(self, samples: int) -> bytes:
        """Render the plotter through the bridge and return PNG bytes.

        Returns
        -------
        bytes
            Raw PNG payload from ``pl.blender.render``, ready to
            base64-encode into the ``data:`` URL the overlay reads.

        """
        kwargs: _RenderKwargs = {**self._render_kwargs, "samples": int(samples)}
        with tempfile.NamedTemporaryFile(
            prefix="pvblender_web_", suffix=".png", delete=False
        ) as fh:
            tmp = Path(fh.name)
        try:
            self._plotter.blender.render(str(tmp), **kwargs)
            return tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)

    def schedule_idle_promotion(self) -> None:
        """Arm a one-shot timer for the idle-tier render.

        Cancels any in-flight timer first so back-to-back settled
        renders don't stack pending idle renders. Skipped entirely
        when :attr:`idle_delay_s` is non-positive (treats ``<=0`` as
        "disable idle promotion"), making this a no-op when the user
        opted out via ``idle_delay_ms=0``.
        """
        if self.idle_delay_s <= 0.0:
            return
        with self._idle_lock:
            if self.idle_timer is not None:
                self.idle_timer.cancel()
            timer = threading.Timer(self.idle_delay_s, self._fire_idle)
            timer.daemon = True
            self.idle_timer = timer
            timer.start()

    def cancel_idle_promotion(self) -> None:
        """Cancel any pending idle promotion timer (no-op when unset)."""
        with self._idle_lock:
            if self.idle_timer is not None:
                self.idle_timer.cancel()
                self.idle_timer = None

    def _fire_idle(self) -> None:
        """Timer callback: kick the idle render off on a worker thread."""
        with self._idle_lock:
            self.idle_timer = None
        threading.Thread(target=self._idle_render, daemon=True).start()

    def _reset_camera(self, *_args: object, **_kwargs: object) -> None:
        """Reset the camera and trigger a fresh Cycles render."""
        self.cancel_idle_promotion()
        self.ctrl.view_reset_camera()
        threading.Thread(target=self._settled_render, daemon=True).start()


def serve(
    plotter: pv.BasePlotter,
    *,
    port: int = 0,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    samples: int = DEFAULT_SETTLED_SAMPLES,
    samples_idle: int = DEFAULT_IDLE_SAMPLES,
    idle_delay_ms: float = DEFAULT_IDLE_DELAY_MS,
    render_kwargs: _RenderKwargs | None = None,
) -> BlenderWebApp:
    """Boot a Trame server for ``plotter`` and (optionally) open a browser.

    Blocks the calling thread until the server is stopped (Ctrl-C or
    the user closes the only client tab and the server times out).

    Parameters
    ----------
    plotter
        The :class:`pyvista.BasePlotter` to translate. The same
        bridge cache + translator paths are used as on
        :meth:`pl.blender.render`.
    port
        TCP port to bind. ``0`` picks an unused port automatically;
        the resolved port appears in the URL written to stdout.
    host
        Interface to bind. Default ``127.0.0.1`` (local only); set
        to ``0.0.0.0`` for remote access (no auth — be careful).
    open_browser
        Auto-launch the system browser pointing at the served URL.
        ``False`` is useful when running headless or driving the
        server from external tooling.
    samples
        Cycles samples for the settled-tier (on-release) render. The
        user can tune at runtime via the toolbar slider.
    samples_idle
        Cycles samples for the idle-tier promotion render fired
        ``idle_delay_ms`` after the last interaction. Defaults to
        4x the settled tier — publication-quality settle.
    idle_delay_ms
        Delay between the settled render and the idle promotion.
        Any new interaction cancels the pending timer. Set to ``0``
        (or negative) to disable idle promotion entirely.
    render_kwargs
        Extra kwargs forwarded to :meth:`pl.blender.render` on every
        settled / idle pass (e.g. ``engine``, ``device``, ``denoise``).

    Returns
    -------
    BlenderWebApp
        The app instance for introspection after the server stops.

    """
    app = BlenderWebApp(
        plotter,
        samples=samples,
        samples_idle=samples_idle,
        idle_delay_ms=idle_delay_ms,
        render_kwargs=render_kwargs,
    )
    if open_browser:
        threading.Thread(
            target=_open_when_ready,
            args=(app, host, port),
            daemon=True,
        ).start()
    app.server.start(host=host, port=port, open_browser=False)
    return app


def _open_when_ready(app: BlenderWebApp, host: str, port: int) -> None:
    """Poll until the server has bound a port, then open the browser.

    The Trame server resolves the bound port lazily after ``start()``;
    polling avoids races with ``webbrowser.open`` firing before the
    socket is listening.
    """
    deadline = time.time() + 10.0
    while time.time() < deadline:
        resolved_port = int(getattr(app.server, "port", port) or 0)
        if resolved_port > 0:
            webbrowser.open(f"http://{host}:{resolved_port}/")
            return
        time.sleep(0.1)
