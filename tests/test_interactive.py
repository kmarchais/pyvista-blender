# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Off-screen tests for the ``pl.blender.show()`` overlay plumbing.

The full ``show()`` flow enters VTK's event loop, which CI can't host.
The pieces it composes — ``install_overlay``, ``render_and_blit``,
``hide_underlying_actors`` / ``restore_underlying_actors`` — are each
callable on an off-screen plotter, so the tests below exercise the
overlay setup, the pixel-buffer round-trip, and the visibility
bookkeeping individually.

A separate ``test_show_requires_on_screen_plotter`` in
``test_accessor.py`` covers the bpy-free off-screen guard rail.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
from vtkmodules.util import numpy_support

from pyvista_blender import config
from pyvista_blender._options import _EngineParams, _PlotterSources  # noqa: PLC2701
from pyvista_blender.interactive.overlay import (
    active_ren_win,
    install_overlay,
    render_and_blit,
)
from pyvista_blender.interactive.throttle import (
    INTERACTION_THROTTLE_MS,
    should_render_now,
)
from pyvista_blender.interactive.tiers import TierSamples, resolve_tier_samples
from pyvista_blender.interactive.visibility import (
    hide_underlying_actors,
    restore_underlying_actors,
)

_WIDTH = 64
_HEIGHT = 48


@pytest.mark.bpy
def test_install_overlay_adds_layer_one_renderer(
    offscreen_plotter: pv.Plotter,
) -> None:
    """``install_overlay`` adds the overlay renderer above any existing layers."""
    pl = offscreen_plotter
    ren_win = active_ren_win(pl)
    layers_before = ren_win.GetNumberOfLayers()

    handles = install_overlay(pl, _WIDTH, _HEIGHT)

    if ren_win.GetNumberOfLayers() != layers_before + 1:
        pytest.fail(
            f"install_overlay should bump NumberOfLayers from {layers_before} "
            f"to {layers_before + 1}, got {ren_win.GetNumberOfLayers()}"
        )
    if handles.renderer.GetLayer() != layers_before:
        pytest.fail(
            f"overlay renderer landed on layer {handles.renderer.GetLayer()}, "
            f"expected {layers_before} (one above the previous top)"
        )
    if handles.width != _WIDTH or handles.height != _HEIGHT:
        pytest.fail(
            f"handles dimensions {handles.width}x{handles.height} "
            f"don't match install args {_WIDTH}x{_HEIGHT}"
        )


@pytest.mark.bpy
def test_render_and_blit_fills_overlay(offscreen_plotter: pv.Plotter) -> None:
    """One ``render_and_blit`` call lands non-empty pixels in the overlay.

    Note: ``render_and_blit`` queries ``ren_win.GetSize`` and reshapes
    the buffer to match the current window — so the post-call buffer
    dimensions track the plotter's actual size, not the install args.
    """
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red")

    handles = install_overlay(pl, _WIDTH, _HEIGHT)
    cache = render_and_blit(
        pl,
        handles,
        engine_params=_EngineParams(
            engine="cycles",
            device="cpu",
            samples=4,
            denoise=False,
            transparent_bg=False,
        ),
        cache=None,
        sources=_PlotterSources(),
        hud=False,
    )
    if cache is None:
        pytest.fail("render_and_blit returned no scene cache")

    expected_size = handles.width * handles.height * 4
    flat = numpy_support.vtk_to_numpy(handles.image_data.GetPointData().GetScalars())
    if flat.size != expected_size:
        pytest.fail(
            f"overlay buffer size {flat.size} "
            f"doesn't match {expected_size} expected for "
            f"{handles.width}x{handles.height}"
        )
    if int(flat.sum()) == 0:
        pytest.fail("overlay buffer is all zeros after render_and_blit")


@pytest.mark.bpy
def test_render_and_blit_with_hud_composites_overlays(
    offscreen_plotter: pv.Plotter,
) -> None:
    """``hud=True`` makes the rendered output differ from ``hud=False``."""
    sphere = pv.Sphere()
    sphere["z"] = sphere.points[:, 2]

    pl = offscreen_plotter
    pl.add_mesh(sphere, scalars="z", cmap="viridis", show_scalar_bar=True)

    handles_a = install_overlay(pl, _WIDTH, _HEIGHT)
    render_and_blit(
        pl,
        handles_a,
        engine_params=_EngineParams(
            engine="cycles",
            device="cpu",
            samples=4,
            denoise=False,
            transparent_bg=False,
        ),
        cache=None,
        sources=_PlotterSources(),
        hud=False,
    )
    bare = numpy_support.vtk_to_numpy(
        handles_a.image_data.GetPointData().GetScalars()
    ).copy()

    handles_b = install_overlay(pl, _WIDTH, _HEIGHT)
    render_and_blit(
        pl,
        handles_b,
        engine_params=_EngineParams(
            engine="cycles",
            device="cpu",
            samples=4,
            denoise=False,
            transparent_bg=False,
        ),
        cache=None,
        sources=_PlotterSources(),
        hud=True,
    )
    annotated = numpy_support.vtk_to_numpy(
        handles_b.image_data.GetPointData().GetScalars()
    ).copy()

    if np.array_equal(bare, annotated):
        pytest.fail("hud=True produced an identical buffer to hud=False")


def test_hide_and_restore_round_trip(offscreen_plotter: pv.Plotter) -> None:
    """Hiding then restoring leaves every actor's ``visibility`` flag unchanged."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red", name="sphere")
    pl.add_mesh(pv.Cube(), color="blue", name="cube")

    before = {name: actor.visibility for name, actor in pl.renderer.actors.items()}

    snapshot = hide_underlying_actors(pl)
    for actor in pl.renderer.actors.values():
        if actor.visibility:
            pytest.fail("hide_underlying_actors left an actor visible")

    restore_underlying_actors(pl, snapshot)
    after = {name: actor.visibility for name, actor in pl.renderer.actors.items()}
    if after != before:
        pytest.fail(f"restore mismatch: before={before!r}, after={after!r}")


# Camera tracking throttle.
#
# The observer callbacks themselves are private methods on
# ``BlenderComponent`` and can't be exercised in CI without a real
# event loop, but the throttle decision is a pure function in
# :mod:`pyvista_blender.interactive.throttle`. Tests below cover its
# contract; end-to-end interaction is verified by manual smoke runs.


def test_throttle_allows_the_first_call() -> None:
    """``last_render_at=0.0`` is the "never rendered" sentinel; should fire."""
    if not should_render_now(now=0.5, last_render_at=0.0):
        pytest.fail("first call (last_render_at=0.0) should always render")


def test_throttle_blocks_repeat_within_window() -> None:
    """A second call right after the previous one is dropped."""
    last = 10.0
    # 1 ms elapsed — well under 80 ms.
    if should_render_now(now=last + 0.001, last_render_at=last):
        pytest.fail("throttle should block calls inside the interval")


def test_throttle_releases_at_the_threshold() -> None:
    """Exactly at ``INTERACTION_THROTTLE_MS`` elapsed, the throttle releases."""
    last = 10.0
    boundary = last + (INTERACTION_THROTTLE_MS / 1000.0)
    if not should_render_now(now=boundary, last_render_at=last):
        pytest.fail(f"throttle should release at exactly {INTERACTION_THROTTLE_MS} ms")


def test_throttle_releases_after_threshold() -> None:
    """Well past the throttle interval the throttle releases."""
    last = 10.0
    if not should_render_now(now=last + 1.0, last_render_at=last):
        pytest.fail("throttle should release well past the interval")


# Three-tier sample resolution.
#
# ``resolve_tier_samples`` is a pure function; tests below exercise the
# three resolution layers (per-tier kwarg > legacy ``samples`` blanket >
# config default) and the combinations users actually write.


_BLANKET = 64
_INTERACTING = 4
_SETTLING = 16
_IDLE = 256
_IGNORED_BLANKET = 999


def test_tier_resolution_all_defaults_match_config() -> None:
    """No kwargs → every tier picks its matching ``config`` default."""
    result = resolve_tier_samples()
    expected = TierSamples(
        interacting=config.interactive_samples,
        settling=config.settled_samples,
        idle=config.idle_samples,
    )
    if result != expected:
        pytest.fail(f"expected config defaults {expected!r}, got {result!r}")


def test_tier_resolution_legacy_samples_blankets_every_tier() -> None:
    """``samples=N`` alone sets all three tiers to N."""
    result = resolve_tier_samples(samples=_BLANKET)
    expected = TierSamples(interacting=_BLANKET, settling=_BLANKET, idle=_BLANKET)
    if result != expected:
        pytest.fail(f"expected {expected!r}, got {result!r}")


def test_tier_resolution_per_tier_overrides_legacy_blanket() -> None:
    """An explicit per-tier kwarg wins over the legacy ``samples`` blanket."""
    result = resolve_tier_samples(samples=_BLANKET, samples_interacting=_INTERACTING)
    expected = TierSamples(interacting=_INTERACTING, settling=_BLANKET, idle=_BLANKET)
    if result != expected:
        pytest.fail(f"expected {expected!r}, got {result!r}")


def test_tier_resolution_full_per_tier_ignores_legacy() -> None:
    """All three per-tier kwargs set — the legacy ``samples`` is irrelevant."""
    result = resolve_tier_samples(
        samples=_IGNORED_BLANKET,
        samples_interacting=_INTERACTING,
        samples_settling=_SETTLING,
        samples_idle=_IDLE,
    )
    expected = TierSamples(interacting=_INTERACTING, settling=_SETTLING, idle=_IDLE)
    if result != expected:
        pytest.fail(f"expected {expected!r}, got {result!r}")
