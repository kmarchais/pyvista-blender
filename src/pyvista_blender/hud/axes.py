# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the corner XYZ axes triad as an RGBA overlay.

``plotter.show_axes()`` adds a ``vtkOrientationMarkerWidget`` carrying an
XYZ axes actor in a small viewport (default ``(0, 0, 0.2, 0.2)`` — the
lower-left 20%). The widget rotates with the camera so each axis arrow
points in the world direction it represents.

We mirror that here: project the world basis vectors through the
camera, draw three colour-coded arrows on a PIL canvas in the same
viewport region, and hand the result to the post-render compositor.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["render_axes_overlay"]

#: Bottom-left fraction-of-frame viewport that the axes widget occupies
#: (matches VTK's ``vtkOrientationMarkerWidget`` default).
_VIEWPORT = (0.0, 0.0, 0.2, 0.2)
_RgbTuple = tuple[int, int, int]
_AXIS_COLORS: tuple[_RgbTuple, _RgbTuple, _RgbTuple] = (
    (231, 76, 60),  # X — red
    (46, 204, 113),  # Y — green
    (52, 152, 219),  # Z — blue
)
_AXIS_LABELS = ("X", "Y", "Z")

#: Vectors whose squared norm falls below this are treated as degenerate.
_DEGENERATE_NORM_EPS = 1e-9


def render_axes_overlay(
    plotter: pv.BasePlotter, width: int, height: int
) -> np.ndarray | None:
    """Return an RGBA overlay carrying the XYZ axes triad, if enabled.

    Returns
    -------
    np.ndarray or None
        Shape ``(height, width, 4)`` float32 RGBA. ``None`` when the
        plotter has no orientation widget active.

    """
    if not _axes_widget_active(plotter):
        return None

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    basis: _CameraBasis = _camera_basis(plotter)
    cx, cy, radius = _viewport_origin(width, height)
    font = _load_font(int(radius * 0.35))

    for axis_index in range(3):
        _draw_axis(draw, axis_index, (cx, cy), radius, basis=basis, font=font)

    return np.asarray(image, dtype=np.float32) / 255.0


def _viewport_origin(width: int, height: int) -> tuple[float, float, float]:
    """Return the on-screen ``(cx, cy, radius)`` of the axes viewport.

    Returns
    -------
    tuple of (float, float, float)
        ``cx`` / ``cy`` in PIL pixel coords (y-down); ``radius`` is the
        arrow length in pixels.

    """
    vx0, vy0, vw, vh = _VIEWPORT
    cx = (vx0 + vw / 2.0) * width
    cy = (1.0 - (vy0 + vh / 2.0)) * height
    radius = 0.4 * min(vw * width, vh * height)
    return cx, cy, radius


#: ``(view_x, view_y, view_fwd)`` unit vectors describing the camera frame.
_CameraBasis = tuple[np.ndarray, np.ndarray, np.ndarray]


def _draw_axis(
    draw: ImageDraw.ImageDraw,
    axis_index: int,
    origin: tuple[float, float],
    radius: float,
    *,
    basis: _CameraBasis,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw one axis (shaft + arrowhead + label) onto the PIL canvas.

    When an axis projects to ~zero screen-length (parallel to the view
    direction), drawing an arrow ends up at ``atan2(0, 0)``, which both
    misrepresents the axis and produces a tiny "backward teardrop"
    artefact. In that case the renderer falls back to a viewer-friendly
    glyph at the origin: a filled disc when the axis points *toward*
    the camera, a ringed cross when it points *away*. The label sits
    below the glyph.
    """
    view_x, view_y, view_fwd = basis
    world_axis = np.zeros(3, dtype=np.float64)
    world_axis[axis_index] = 1.0
    sx = float(np.dot(world_axis, view_x))
    sy = float(np.dot(world_axis, view_y))
    color = (*_AXIS_COLORS[axis_index], 255)
    label = _AXIS_LABELS[axis_index]
    head_len = max(6.0, radius * 0.18)

    # Axis-along-view threshold: below this projected length we treat
    # the axis as collapsed and draw a glyph at the origin instead.
    collapsed_threshold = 0.08
    if math.hypot(sx, sy) < collapsed_threshold:
        depth = float(np.dot(world_axis, -view_fwd))
        _draw_collapsed_axis(
            draw,
            origin,
            head_len=head_len,
            color=color,
            label=label,
            font=font,
            toward_viewer=depth >= 0.0,
        )
        return

    _draw_axis_arrow(
        draw,
        origin,
        radius,
        head_len=head_len,
        screen_dir=(sx, sy),
        color=color,
        label=label,
        font=font,
    )


def _draw_axis_arrow(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    radius: float,
    *,
    head_len: float,
    screen_dir: tuple[float, float],
    color: tuple[int, int, int, int],
    label: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw the standard shaft + arrowhead + label for a non-collapsed axis.

    ``screen_dir`` is the ``(sx, sy)`` unit vector in screen space
    (PIL coords, y not yet flipped).
    """
    cx, cy = origin
    sx, sy = screen_dir
    end_x = cx + sx * radius
    end_y = cy - sy * radius  # PIL y inverted
    draw.line(
        ((cx, cy), (end_x, end_y)),
        fill=color,
        width=max(2, int(radius * 0.06)),
    )
    angle = math.atan2(end_y - cy, end_x - cx)
    _draw_arrowhead(draw, (end_x, end_y), angle, head_len, color)
    label_x = end_x + 1.4 * head_len * math.cos(angle)
    label_y = end_y + 1.4 * head_len * math.sin(angle)
    draw.text(
        (label_x, label_y),
        label,
        font=font,
        fill=color,
        anchor="mm",
    )


def _draw_collapsed_axis(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    *,
    head_len: float,
    color: tuple[int, int, int, int],
    label: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    toward_viewer: bool,
) -> None:
    """Render a glyph for an axis that points along the view direction.

    * ``toward_viewer=True``: filled disc (looks like an arrow tip
      pointing out of the screen).
    * ``toward_viewer=False``: a hollow ring with a cross drawn over
      it (the "x" tail-of-arrow seen edge-on).
    """
    cx, cy = origin
    r = head_len * 0.55
    bbox = (cx - r, cy - r, cx + r, cy + r)
    if toward_viewer:
        draw.ellipse(bbox, fill=color, outline=color)
    else:
        draw.ellipse(bbox, fill=None, outline=color, width=max(2, int(r * 0.25)))
        offset = r * 0.55
        line_width = max(2, int(r * 0.25))
        draw.line(
            ((cx - offset, cy - offset), (cx + offset, cy + offset)),
            fill=color,
            width=line_width,
        )
        draw.line(
            ((cx - offset, cy + offset), (cx + offset, cy - offset)),
            fill=color,
            width=line_width,
        )
    # Label below the glyph so it never overlaps the origin dot.
    draw.text(
        (cx, cy + r + head_len * 0.7),
        label,
        font=font,
        fill=color,
        anchor="mm",
    )


def _draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    tip: tuple[float, float],
    angle: float,
    head_len: float,
    color: tuple[int, int, int, int],
) -> None:
    """Stamp an arrowhead triangle at ``tip`` pointing along ``angle``."""
    head_w = head_len * 0.7
    base_x = tip[0] - head_len * math.cos(angle)
    base_y = tip[1] - head_len * math.sin(angle)
    left = (
        base_x + head_w * 0.5 * math.cos(angle + math.pi / 2),
        base_y + head_w * 0.5 * math.sin(angle + math.pi / 2),
    )
    right = (
        base_x + head_w * 0.5 * math.cos(angle - math.pi / 2),
        base_y + head_w * 0.5 * math.sin(angle - math.pi / 2),
    )
    draw.polygon((tip, left, right), fill=color)


def _axes_widget_active(plotter: pv.BasePlotter) -> bool:
    """Return whether ``pl.show_axes`` is currently enabled.

    Returns
    -------
    bool
        ``True`` when the renderer carries an orientation marker widget
        that's enabled, otherwise ``False``.

    """
    widget = getattr(plotter.renderer, "axes_widget", None)
    if widget is None:
        return False
    enabled = getattr(widget, "GetEnabled", None)
    return bool(enabled()) if enabled is not None else True


def _camera_basis(
    plotter: pv.BasePlotter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the camera's screen-X, screen-Y, and forward basis vectors.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        ``(view_right, view_up, view_forward)``; each a unit world-space
        vector pointing along the rendered image's +X / +Y / depth
        respectively. ``view_forward`` is the unit ``focal - position``
        direction (a vector from the camera into the scene), so an axis
        with ``dot(axis, -view_forward) > 0`` points toward the viewer.

    """
    cam = plotter.camera
    position = np.asarray(cam.position, dtype=np.float64)
    focal = np.asarray(cam.focal_point, dtype=np.float64)
    up = np.asarray(cam.up, dtype=np.float64)

    forward = focal - position
    forward /= np.linalg.norm(forward) or 1.0
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < _DEGENERATE_NORM_EPS:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_norm
    view_up = np.cross(right, forward)
    return right, view_up, forward


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a sans-serif TrueType font at ``size`` pixels, with PIL fallbacks.

    Returns
    -------
    ImageFont.FreeTypeFont or ImageFont.ImageFont
        A vector PIL font when a system TrueType file is reachable; the
        bundled bitmap fallback otherwise.

    """
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=max(size, 8))
        except OSError:
            continue
    return ImageFont.load_default()
