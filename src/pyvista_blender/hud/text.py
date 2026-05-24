# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render PyVista text / title overlays as an RGBA image.

``plotter.add_text(...)`` and ``plotter.add_title(...)`` both produce a
:class:`vtkCornerAnnotation` actor in the renderer. The annotation
exposes its text via ``GetText(corner_index)`` where the index encodes
position (0 - 3 = the four corners, 4 - 7 = the four edges). We walk
every annotation, locate the populated slot, and draw the string via
PIL at the corresponding fractional viewport position — same anchor
logic as VTK's screen-space placement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["render_text_overlay"]

#: (anchor_x, anchor_y, h_align, v_align) per vtkCornerAnnotation slot.
#: VTK's edge slots place the text along the middle of the named edge.
_CORNER_LAYOUT: tuple[tuple[float, float, str, str], ...] = (
    (0.02, 0.02, "left", "bottom"),  # 0: lower-left
    (0.98, 0.02, "right", "bottom"),  # 1: lower-right
    (0.02, 0.98, "left", "top"),  # 2: upper-left
    (0.98, 0.98, "right", "top"),  # 3: upper-right
    (0.50, 0.02, "center", "bottom"),  # 4: lower-edge (centred bottom)
    (0.98, 0.50, "right", "center"),  # 5: right-edge
    (0.50, 0.98, "center", "top"),  # 6: upper-edge
    (0.02, 0.50, "left", "center"),  # 7: left-edge (used by add_title)
)


def render_text_overlay(
    plotter: pv.BasePlotter, width: int, height: int
) -> np.ndarray | None:
    """Render every text annotation on the plotter as a single RGBA overlay.

    Returns
    -------
    np.ndarray or None
        Shape ``(height, width, 4)`` float32 RGBA. ``None`` when there
        are no text annotations or PIL isn't importable.

    """
    entries = list(_iter_annotations(plotter))
    if not entries:
        return None

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for text, slot, font_size, color in entries:
        font = _load_font(font_size)
        anchor_x_frac, anchor_y_frac, h_align, v_align = _CORNER_LAYOUT[slot]
        bbox = draw.textbbox(
            (0, 0), text, font=font, anchor=_pil_anchor(h_align, v_align)
        )
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        # Convert VTK's "y up" fraction to PIL's "y down" pixel space.
        x_px = anchor_x_frac * width
        y_px = (1.0 - anchor_y_frac) * height
        # Nudge inward so the text doesn't clip against the frame edge.
        margin = 8
        if h_align == "left":
            x_px = max(x_px, margin)
        elif h_align == "right":
            x_px = min(x_px, width - margin)
        if v_align == "top":
            y_px = max(y_px, margin)
        elif v_align == "bottom":
            y_px = min(y_px, height - margin)
        draw.text(
            (x_px, y_px),
            text,
            font=font,
            fill=color,
            anchor=_pil_anchor(h_align, v_align),
        )
        _ = (text_w, text_h)  # bbox sizing reserved for future text frames

    return np.asarray(image, dtype=np.float32) / 255.0


def _iter_annotations(
    plotter: pv.BasePlotter,
) -> list[tuple[str, int, int, tuple[int, int, int, int]]]:
    """Walk renderer actors yielding text annotations with their layout.

    Returns
    -------
    list of (str, int, int, tuple)
        ``(text, slot, font_size, rgba)`` per populated corner slot.

    """
    out: list[tuple[str, int, int, tuple[int, int, int, int]]] = []
    for raw_actor in plotter.renderer.actors.values():
        if type(raw_actor).__name__ != "CornerAnnotation":
            continue
        # Filtered by class name above; ``vtkCornerAnnotation`` exposes
        # ``GetTextProperty`` / ``GetText(slot)`` that the base
        # ``vtkProp`` doesn't surface in the VTK stubs.
        actor = cast("Any", raw_actor)
        prop = actor.GetTextProperty()
        font_size = int(prop.GetFontSize())
        r, g, b = prop.GetColor()
        rgba = (int(r * 255), int(g * 255), int(b * 255), 255)
        for slot in range(8):
            text = actor.GetText(slot)
            if not text:
                continue
            # add_title produces "\nTitle" (leading newline VTK uses for
            # the edge-centred title style). Strip it so the rendered
            # text doesn't show a phantom blank line.
            cleaned = text.strip("\n")
            if not cleaned:
                continue
            out.append((cleaned, slot, font_size, rgba))
    return out


def _pil_anchor(h_align: str, v_align: str) -> str:
    """Map ``(h_align, v_align)`` to PIL ``ImageDraw.text(anchor=...)``.

    Returns
    -------
    str
        Two-letter PIL anchor (e.g. ``"la"`` for left+ascender / top).

    """
    h = {"left": "l", "center": "m", "right": "r"}[h_align]
    v = {"top": "a", "center": "m", "bottom": "d"}[v_align]
    return h + v


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a sans-serif TrueType font at ``size`` pixels, with PIL fallbacks.

    Returns
    -------
    ImageFont.FreeTypeFont or ImageFont.ImageFont
        A vector PIL font when a system TrueType file is reachable; the
        bundled bitmap fallback otherwise (the bitmap font ignores the
        size argument but stays readable enough for HUD use).

    """
    for candidate in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
