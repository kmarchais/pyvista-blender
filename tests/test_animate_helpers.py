# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for :mod:`pyvista_blender.animate` helpers.

The module is pure numpy / math — no bpy — so each helper can be
exercised against a plain off-screen plotter without the @bpy gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvista_blender.animate import orbit_camera

if TYPE_CHECKING:
    import pyvista as pv


def test_orbit_camera_rejects_non_positive_n_frames(
    offscreen_plotter: pv.Plotter,
) -> None:
    """``n_frames <= 0`` is meaningless and surfaces a clear error."""
    with pytest.raises(ValueError, match=r"n_frames must be positive"):
        orbit_camera(offscreen_plotter, n_frames=0)


def test_orbit_camera_frame_zero_reproduces_starting_pose(
    offscreen_plotter: pv.Plotter,
) -> None:
    """Frame 0 of the orbit must place the camera at its initial position."""
    pl = offscreen_plotter
    pl.camera_position = [(3.0, 0.0, 1.5), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    initial = np.asarray(pl.camera.position, dtype=np.float64)

    update = orbit_camera(pl, n_frames=8)
    update(0)

    if not np.allclose(np.asarray(pl.camera.position), initial, atol=1e-9):
        pytest.fail(
            f"frame 0 moved the camera; expected {tuple(initial)}, "
            f"got {tuple(pl.camera.position)}"
        )


def test_orbit_camera_full_loop_returns_to_start(
    offscreen_plotter: pv.Plotter,
) -> None:
    """After ``n_frames`` steps the camera lands back on the starting pose."""
    pl = offscreen_plotter
    pl.camera_position = [(2.0, 1.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    initial = np.asarray(pl.camera.position, dtype=np.float64)

    n_frames = 12
    update = orbit_camera(pl, n_frames=n_frames)
    update(n_frames)

    if not np.allclose(np.asarray(pl.camera.position), initial, atol=1e-9):
        pytest.fail(
            f"full loop did not return to start; expected {tuple(initial)}, "
            f"got {tuple(pl.camera.position)}"
        )


def test_orbit_camera_direction_reverses_sweep(
    offscreen_plotter: pv.Plotter,
) -> None:
    """``direction=-1`` mirrors ``direction=+1`` across the starting pose."""
    pl = offscreen_plotter
    pl.camera_position = [(2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    n_frames = 8
    quarter_frame = n_frames // 4

    update_cw = orbit_camera(pl, n_frames=n_frames, direction=-1)
    update_cw(quarter_frame)
    pos_cw = tuple(pl.camera.position)

    pl.camera_position = [(2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    update_ccw = orbit_camera(pl, n_frames=n_frames, direction=1)
    update_ccw(quarter_frame)
    pos_ccw = tuple(pl.camera.position)

    if pos_cw == pos_ccw:
        pytest.fail("direction=+1 and direction=-1 produced the same pose")
