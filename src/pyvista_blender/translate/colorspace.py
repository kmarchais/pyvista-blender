# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Colour-space conversion for flat colours handed to Blender.

PyVista / VTK colours (``prop.color``, ``edge_color``, light colours) are
sRGB-encoded; Blender's shader and light colour inputs are scene-linear.
"""

from __future__ import annotations

__all__ = ["linear_rgb"]

_SRGB_KNEE = 0.04045  # sRGB transfer function: linear segment below, power law above


def linear_rgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert an sRGB colour (channels in 0..1) to scene-linear.

    Returns
    -------
    tuple of float
        The linear red, green and blue.

    """
    r, g, b = (
        c / 12.92 if c <= _SRGB_KNEE else ((c + 0.055) / 1.055) ** 2.4
        for c in (float(v) for v in rgb)
    )
    return r, g, b
