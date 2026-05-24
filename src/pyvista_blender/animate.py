# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for building per-frame ``updater`` callables.

Animation is driven by a Python loop: the user supplies a callable
``updater(frame_index)`` that mutates the PyVista scene in
place, and the bridge re-renders the cached scene each frame. These
helpers cover the common cases:

* :func:`orbit_camera` — rotate the camera around a focal point.

A custom mutation (deformation, scalar field update, actor visibility
toggle, …) is just a plain Python function — no need to use these
helpers if the bespoke one is shorter.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyvista as pv

__all__ = ["orbit_camera"]


def orbit_camera(
    plotter: pv.BasePlotter,
    focal_point: tuple[float, float, float] | None = None,
    *,
    n_frames: int,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    direction: int = 1,
) -> Callable[[int], None]:
    """Return an updater that orbits the plotter's camera around a focal point.

    The orbit radius and elevation are derived from the plotter's
    *current* camera position: frame 0 reproduces it exactly, and a full
    loop of ``n_frames`` returns to that pose. The camera always looks
    at ``focal_point`` and uses ``axis`` as its world-up vector.

    Parameters
    ----------
    plotter
        Source plotter; its camera is read once to anchor the orbit.
    focal_point
        Point the camera circles around. Defaults to the plotter
        camera's current focal point.
    n_frames
        Number of frames in a full 360° loop.
    axis
        World-space rotation axis. Defaults to ``+Z`` (turntable about
        the vertical axis); ``(0, 1, 0)`` gives a horizontal sweep.
    direction
        ``+1`` for counter-clockwise (right-handed), ``-1`` for clockwise.

    Returns
    -------
    Callable[[int], None]
        An updater suitable for :meth:`BlenderComponent.animate`: invoking
        it with frame index ``i`` rewrites
        ``plotter.camera_position = [eye_i, focal, up]`` in place.

    Raises
    ------
    ValueError
        When ``n_frames`` is not strictly positive.

    """
    if n_frames <= 0:
        msg = f"n_frames must be positive, got {n_frames}"
        raise ValueError(msg)

    cam = plotter.camera
    initial_position = np.asarray(cam.position, dtype=np.float64)
    initial_focal = (
        np.asarray(focal_point, dtype=np.float64)
        if focal_point is not None
        else np.asarray(cam.focal_point, dtype=np.float64)
    )
    axis_vec = np.asarray(axis, dtype=np.float64)
    axis_vec /= np.linalg.norm(axis_vec)

    radial = initial_position - initial_focal
    radial_along = np.dot(radial, axis_vec) * axis_vec
    radial_perp = radial - radial_along

    sign = 1 if direction >= 0 else -1

    def updater(frame_index: int) -> None:
        angle = sign * 2.0 * math.pi * float(frame_index) / float(n_frames)
        rotated_perp = _rotate_about_axis(radial_perp, axis_vec, angle)
        eye = initial_focal + radial_along + rotated_perp
        plotter.camera_position = [
            tuple(eye),
            tuple(initial_focal),
            tuple(axis_vec),
        ]

    return updater


def _rotate_about_axis(
    vector: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """Rodrigues rotation of ``vector`` about a unit ``axis`` by ``angle`` radians.

    Returns
    -------
    np.ndarray
        The rotated 3-vector.

    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        vector * cos_a
        + np.cross(axis, vector) * sin_a
        + axis * np.dot(axis, vector) * (1.0 - cos_a)
    )
