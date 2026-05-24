# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render PyVista's ``show_bounds`` bounding box as an RGBA overlay.

``vtkCubeAxesActor`` is a 3D actor (the box rotates with the camera);
we mirror its **wireframe** part by projecting the eight corners of the
scene's bounding box through a perspective MVP matrix derived from
``pv.Camera`` and drawing the twelve edges with PIL. Tick labels are
not yet implemented — the box outline is the most common scivis ask;
labels land in a later milestone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["render_bounds_overlay"]

#: Vectors whose squared norm falls below this are treated as degenerate.
_DEGENERATE_NORM_EPS = 1e-9

#: View-space depth below which we skip projection (avoids divide-by-near-zero
#: artefacts at the camera's near plane).
_NEAR_PLANE_EPS = 1e-6

_EDGE_INDICES: tuple[tuple[int, int], ...] = (
    # 8 corners ordered (x,y,z) ∈ {min, max}³; bit 0 = x, 1 = y, 2 = z.
    (0b000, 0b001),
    (0b010, 0b011),
    (0b100, 0b101),
    (0b110, 0b111),
    (0b000, 0b010),
    (0b001, 0b011),
    (0b100, 0b110),
    (0b101, 0b111),
    (0b000, 0b100),
    (0b001, 0b101),
    (0b010, 0b110),
    (0b011, 0b111),
)


def render_bounds_overlay(
    plotter: pv.BasePlotter, width: int, height: int
) -> np.ndarray | None:
    """Return an RGBA overlay carrying the bounding-box wireframe, if enabled.

    Returns
    -------
    np.ndarray or None
        Shape ``(height, width, 4)`` float32 RGBA. ``None`` when the
        plotter has no ``vtkCubeAxesActor`` (i.e. ``show_bounds`` wasn't
        called) or the bounds are degenerate.

    """
    if not _bounds_active(plotter):
        return None

    bounds = plotter.renderer.bounds
    xs = (float(bounds[0]), float(bounds[1]))
    ys = (float(bounds[2]), float(bounds[3]))
    zs = (float(bounds[4]), float(bounds[5]))
    if xs[0] >= xs[1] or ys[0] >= ys[1] or zs[0] >= zs[1]:
        return None

    corners = np.array(
        [(xs[i & 1], ys[(i >> 1) & 1], zs[(i >> 2) & 1]) for i in range(8)],
        dtype=np.float64,
    )
    projected = _project_world_to_screen(plotter, corners, width, height)
    if projected is None:
        return None

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    line_width = max(1, width // 480)
    for i_start, i_end in _EDGE_INDICES:
        if projected[i_start, 2] <= 0 or projected[i_end, 2] <= 0:
            # Edge intersects the near plane; skip rather than draw a
            # diverging line through the frame.
            continue
        draw.line(
            (
                (projected[i_start, 0], projected[i_start, 1]),
                (projected[i_end, 0], projected[i_end, 1]),
            ),
            fill=(200, 200, 200, 220),
            width=line_width,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


def _bounds_active(plotter: pv.BasePlotter) -> bool:
    """Return whether ``pl.show_bounds`` is currently enabled.

    Returns
    -------
    bool
        ``True`` when the renderer carries a ``CubeAxesActor``, otherwise
        ``False``. PyVista doesn't expose a public toggle, so actor
        presence is the load-bearing signal.

    """
    return any(
        type(actor).__name__ == "CubeAxesActor"
        for actor in plotter.renderer.actors.values()
    )


def _project_world_to_screen(
    plotter: pv.BasePlotter, points: np.ndarray, width: int, height: int
) -> np.ndarray | None:
    """Project world-space points through ``pv.Camera`` to PIL pixel coords.

    Parameters
    ----------
    plotter
        Source plotter; its camera defines the view + projection.
    points
        ``(n, 3)`` array of world-space points.
    width, height
        Render resolution.

    Returns
    -------
    np.ndarray or None
        ``(n, 3)`` array ``[u_px, v_px, z_view]`` where ``z_view > 0`` for
        points in front of the camera. ``None`` if the camera basis is
        degenerate.

    """
    basis = _camera_view_basis(plotter.camera)
    if basis is None:
        return None
    position, right, view_up, forward = basis

    relative = points - position
    view_x = relative @ right
    view_y = relative @ view_up
    view_z = relative @ forward

    norm_x, norm_y = _ndc_xy(
        plotter.camera, view_x, view_y, view_z, width=width, height=height
    )
    u_px = (norm_x * 0.5 + 0.5) * width
    v_px = (1.0 - (norm_y * 0.5 + 0.5)) * height
    return np.stack([u_px, v_px, view_z], axis=1)


def _camera_view_basis(
    cam: pv.Camera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ``(position, right, view_up, forward)`` from a ``pv.Camera``.

    Returns
    -------
    tuple or None
        Camera origin + orthonormal basis; ``None`` if the camera is
        degenerate (zero forward or up parallel to forward).

    """
    position = np.asarray(cam.position, dtype=np.float64)
    forward = np.asarray(cam.focal_point, dtype=np.float64) - position
    fwd_norm = np.linalg.norm(forward)
    if fwd_norm < _DEGENERATE_NORM_EPS:
        return None
    forward /= fwd_norm
    right = np.cross(forward, np.asarray(cam.up, dtype=np.float64))
    right_norm = np.linalg.norm(right)
    if right_norm < _DEGENERATE_NORM_EPS:
        return None
    right /= right_norm
    view_up = np.cross(right, forward)
    return position, right, view_up, forward


def _ndc_xy(
    cam: pv.Camera,
    view_x: np.ndarray,
    view_y: np.ndarray,
    view_z: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(norm_x, norm_y)`` in ``[-1, 1]`` NDC space.

    Returns
    -------
    tuple of np.ndarray
        Per-point normalised device coords; behind-camera points get
        whatever divide-by-zero guard ``np.where`` produces — the caller
        filters them via the ``z`` channel.

    """
    aspect = width / max(height, 1)
    if getattr(cam, "parallel_projection", False):
        scale = float(getattr(cam, "parallel_scale", 1.0))
        return view_x / (scale * aspect), view_y / scale

    half_tan = math.tan(math.radians(float(cam.view_angle)) / 2.0)
    safe_z = np.where(view_z > _NEAR_PLANE_EPS, view_z, 1.0)
    return view_x / (safe_z * half_tan * aspect), view_y / (safe_z * half_tan)
