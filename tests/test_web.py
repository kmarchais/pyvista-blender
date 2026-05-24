# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Trame web viewport.

The tests construct the :class:`BlenderWebApp` directly so they
exercise state defaults and UI wiring without booting a live HTTP
server (which would block the test). The :meth:`BlenderComponent.show`
dispatch is checked by stubbing :func:`pyvista_blender.web.serve`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import pyvista as pv

from pyvista_blender.web import (
    DEFAULT_IDLE_DELAY_MS,
    DEFAULT_IDLE_SAMPLES,
    DEFAULT_SETTLED_SAMPLES,
    BlenderWebApp,
)


def test_blender_web_app_has_expected_state_defaults() -> None:
    """The app initializes its Trame state with the documented defaults."""
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    try:
        pl.add_mesh(pv.Sphere(), color="orange")
        app = BlenderWebApp(pl, samples=64)
    finally:
        pl.close()

    if app.state.samples != 64:  # noqa: PLR2004
        pytest.fail(f"expected state.samples=64, got {app.state.samples}")
    if app.state.cycles_visible is not False:
        pytest.fail(
            f"expected cycles_visible=False at boot, got {app.state.cycles_visible!r}"
        )
    if app.state.cycles_data_url:
        pytest.fail(
            f"expected empty cycles_data_url at boot, got {app.state.cycles_data_url!r}"
        )


def test_blender_web_app_default_samples_matches_module_const() -> None:
    """Omitting ``samples`` uses the module-level default (32)."""
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    try:
        pl.add_mesh(pv.Cube())
        app = BlenderWebApp(pl)
    finally:
        pl.close()

    if app.state.samples != DEFAULT_SETTLED_SAMPLES:
        pytest.fail(
            f"default samples {app.state.samples} != "
            f"DEFAULT_SETTLED_SAMPLES ({DEFAULT_SETTLED_SAMPLES})"
        )


def test_blender_web_app_carries_vtk_view_handle() -> None:
    """The app exposes the VtkLocalView for camera-control wiring."""
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    try:
        pl.add_mesh(pv.Sphere())
        app = BlenderWebApp(pl)
    finally:
        pl.close()

    if not hasattr(app, "_vtk_view"):
        pytest.fail("BlenderWebApp did not store a reference to the VtkLocalView")


def test_show_backend_web_dispatches_to_serve(offscreen_plotter: pv.Plotter) -> None:
    """``pl.blender.show(backend='web')`` calls :func:`web.serve`.

    Mocks out the serve call so the test doesn't actually boot a Trame
    server. Verifies that the kwargs round-trip end-to-end through the
    component → ``_show_web`` → ``serve`` pipe.
    """
    pl = offscreen_plotter  # off_screen plotter — web backend should not reject
    pl.add_mesh(pv.Sphere())
    with patch("pyvista_blender.web.serve") as mock_serve:
        pl.blender.show(backend="web", samples=64, port=8765, open_browser=False)
    if mock_serve.call_count != 1:
        pytest.fail(f"web.serve was called {mock_serve.call_count} times, expected 1")
    _args, kwargs = mock_serve.call_args
    if kwargs.get("port") != 8765:  # noqa: PLR2004
        pytest.fail(f"port did not propagate; kwargs={kwargs!r}")
    if kwargs.get("samples") != 64:  # noqa: PLR2004
        pytest.fail(f"samples did not propagate; kwargs={kwargs!r}")
    if kwargs.get("open_browser") is not False:
        pytest.fail(f"open_browser did not propagate; kwargs={kwargs!r}")


def test_blender_web_app_idle_defaults_match_module_consts() -> None:
    """Idle-tier kwargs default to the documented module constants."""
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    try:
        pl.add_mesh(pv.Sphere())
        app = BlenderWebApp(pl)
    finally:
        pl.close()

    if app.samples_idle != DEFAULT_IDLE_SAMPLES:
        pytest.fail(
            f"default _samples_idle={app.samples_idle} != "
            f"DEFAULT_IDLE_SAMPLES ({DEFAULT_IDLE_SAMPLES})"
        )
    expected_idle_s = DEFAULT_IDLE_DELAY_MS / 1000.0
    if app.idle_delay_s != pytest.approx(expected_idle_s):
        pytest.fail(f"default idle_delay_s={app.idle_delay_s} != {expected_idle_s}")


def test_schedule_idle_promotion_arms_timer_and_cancel_clears_it() -> None:
    """``_schedule_idle_promotion`` arms a Timer; cancel clears it."""
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    try:
        pl.add_mesh(pv.Sphere())
        # Use a 10 s delay so the timer doesn't fire during the test.
        app = BlenderWebApp(pl, idle_delay_ms=10_000.0)
    finally:
        pl.close()

    if app.idle_timer is not None:
        pytest.fail("fresh app should have no idle timer scheduled")
    app.schedule_idle_promotion()
    if app.idle_timer is None:
        pytest.fail("schedule_idle_promotion did not arm a Timer")
    armed = app.idle_timer
    app.cancel_idle_promotion()
    if app.idle_timer is not None:
        pytest.fail("cancel_idle_promotion did not clear the timer slot")
    # ``Timer.cancel()`` flips the ``finished`` event; the thread may
    # take a moment to exit, but the flag flip is the reliable signal
    # that the timer will never fire.
    if not armed.finished.is_set():
        pytest.fail("cancelled timer's finished flag was not set")


def test_idle_promotion_disabled_when_delay_is_zero() -> None:
    """Setting ``idle_delay_ms=0`` disables idle promotion entirely.

    Useful for tests, demos, or anyone who wants the settled tier to be
    the final state without the higher-quality follow-up.
    """
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    try:
        pl.add_mesh(pv.Sphere())
        app = BlenderWebApp(pl, idle_delay_ms=0.0)
    finally:
        pl.close()

    app.schedule_idle_promotion()
    if app.idle_timer is not None:
        pytest.fail("idle_delay_ms=0 should disable promotion; got a live timer")


def test_show_backend_web_forwards_idle_kwargs(offscreen_plotter: pv.Plotter) -> None:
    """``samples_idle`` / ``idle_delay_ms`` round-trip to ``serve()``."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere())
    with patch("pyvista_blender.web.serve") as mock_serve:
        pl.blender.show(
            backend="web",
            samples_settling=16,
            samples_idle=256,
            idle_delay_ms=500.0,
            open_browser=False,
        )
    _args, kwargs = mock_serve.call_args
    if kwargs.get("samples") != 16:  # noqa: PLR2004
        pytest.fail(f"samples (settled tier) did not propagate; kwargs={kwargs!r}")
    if kwargs.get("samples_idle") != 256:  # noqa: PLR2004
        pytest.fail(f"samples_idle did not propagate; kwargs={kwargs!r}")
    if kwargs.get("idle_delay_ms") != pytest.approx(500.0):
        pytest.fail(f"idle_delay_ms did not propagate; kwargs={kwargs!r}")


def test_show_unknown_backend_raises(offscreen_plotter: pv.Plotter) -> None:
    """An unknown backend string is rejected with a helpful ValueError."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere())
    with pytest.raises(ValueError, match=r"unknown show\(\) backend"):
        pl.blender.show(backend="vulcan")
