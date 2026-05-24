# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-Python ``GlyphSpec`` data class.

Lives outside :mod:`pyvista_blender.translate.glyph` because the
translator imports ``bpy`` at module top, and :mod:`pyvista_blender._component`
needs to type the component's ``_glyphs`` slot without dragging bpy
into the accessor-wiring path. Pure dataclass, no bpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["GlyphSpec"]


@dataclass(frozen=True)
class GlyphSpec:
    """Description of one glyph instancing layer.

    Stored on :class:`BlenderComponent` by ``pl.blender.add_glyph(...)``;
    consumed by :func:`pyvista_blender.translate.glyph.translate_glyph`
    during scene reconciliation.

    Attributes
    ----------
    source
        Dataset whose points become instance origins.
    geom
        Glyph shape to instance at every source point.
    orient
        Name of a point-data 3D vector field used to orient each
        instance (instance's +Z aligns to that vector). ``None`` →
        identity rotation.
    scale
        Name of a point-data scalar field used to scale each instance.
        ``None`` → uniform scale.
    factor
        Global scale multiplier applied to every instance, on top of
        the per-point ``scale`` field if present.
    name
        Optional base name for the resulting bpy data blocks.

    """

    source: pv.DataSet
    geom: pv.DataSet
    orient: str | None = None
    scale: str | None = None
    factor: float = 1.0
    name: str | None = None
