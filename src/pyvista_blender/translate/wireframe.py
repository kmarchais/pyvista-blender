# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wireframe overlay translation.

Maps two PyVista property knobs onto a Blender object hierarchy:

* ``prop.style == "Wireframe"`` — render edges only. The fill surface is
  hidden from Cycles via ``hide_render = True``; a sibling wire object
  carries the visible geometry.
* ``prop.show_edges == True`` — render surface + edges stacked. The fill
  surface stays visible and a wire object is added on top.

The wire object is a duplicate of the surface mesh with a
``WIREFRAME`` modifier (extrudes each edge into a thin tube) and an
emissive shader (no shading: matches GL line rendering). ``line_width``
sets the modifier thickness; ``edge_color`` drives the emissive shader's
colour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bmesh
import bpy

if TYPE_CHECKING:
    import pyvista as pv

__all__ = [
    "actor_needs_wire",
    "make_wire_material",
    "make_wire_object",
    "thickness_for",
]

#: Empirical scaling from VTK's ``line_width`` (screen pixels) to Blender's
#: WIREFRAME modifier thickness (world units). Tuned so the example scenes
#: at 640x480 / unit-cube produce GL-equivalent line thickness.
#: Per-scene tuning can override via ``actor.prop.line_width``.
_LINE_WIDTH_TO_THICKNESS = 0.005


def actor_needs_wire(actor: pv.Actor) -> bool:
    """Return whether an actor's property requests a wireframe overlay.

    Returns
    -------
    bool
        ``True`` if ``prop.style == "Wireframe"`` or ``prop.show_edges``.

    """
    prop = actor.prop
    return _is_wireframe_style(prop) or bool(getattr(prop, "show_edges", False))


def make_wire_object(
    surface_obj: bpy.types.Object, name: str, prop: pv.Property
) -> bpy.types.Object:
    """Create a wireframe overlay object duplicating ``surface_obj``'s geometry.

    The overlay gets its own mesh data block: coplanar edges (the
    diagonals introduced by ``triangulate()`` in ``translate/mesh.py``)
    are dissolved so the wire only marks the conceptual face boundaries
    — a triangulated cube reads as 12 edges, not 18. A ``WIREFRAME``
    modifier then extrudes each remaining edge into a thin bar.

    Parameters
    ----------
    surface_obj
        The fill-surface bpy object whose geometry to mirror.
    name
        Base name for the new mesh + object; ``"_wire"`` is appended.
    prop
        Source PyVista property; controls ``line_width``.

    Returns
    -------
    bpy.types.Object
        The new wireframe overlay object, linked into the active scene.

    """
    wire_mesh = surface_obj.data.copy()
    wire_mesh.name = f"{name}_wire"
    _dissolve_coplanar_edges(wire_mesh)

    wire_obj = bpy.data.objects.new(f"{name}_wire", wire_mesh)
    wire_obj.matrix_world = surface_obj.matrix_world.copy()
    bpy.context.scene.collection.objects.link(wire_obj)

    modifier = wire_obj.modifiers.new(name="PVWireframe", type="WIREFRAME")
    modifier.thickness = thickness_for(prop)
    modifier.use_replace = True  # drop the fill quads, keep only the wire bars
    modifier.use_even_offset = True
    modifier.use_relative_offset = False
    return wire_obj


def _dissolve_coplanar_edges(mesh: bpy.types.Mesh) -> None:
    """Remove edges whose adjacent faces are (near-)coplanar.

    The mesh translator triangulates every surface for predictable bpy
    geometry; that introduces face diagonals on hulls that were
    originally quad-faced (cubes, extrusions, prisms, …). Without this
    pass the WIREFRAME modifier would draw those diagonals as visible
    bars. The 1° threshold matches what VTK's surface rendering treats
    as "same face".
    """
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.dissolve_limit(
            bm,
            angle_limit=0.0175,  # 1° in radians
            verts=bm.verts,
            edges=bm.edges,
            delimit={"NORMAL"},
        )
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()


def make_wire_material(name: str, prop: pv.Property) -> bpy.types.Material:
    """Build an emissive (unlit) material matching ``prop.edge_color``.

    GL line rendering doesn't shade lines — they always read at their full
    colour. The emissive shader gives Cycles the same look: light from
    lamps in the scene doesn't darken or highlight the wire bars.

    Returns
    -------
    bpy.types.Material
        Material with a single ``ShaderNodeEmission`` driving the surface.

    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    emission = tree.nodes.new("ShaderNodeEmission")
    color = getattr(prop, "edge_color", None)
    rgb = color.float_rgb if color is not None else (0.0, 0.0, 0.0)
    emission.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    emission.inputs["Strength"].default_value = 1.0

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def _is_wireframe_style(prop: pv.Property) -> bool:
    """Return whether ``prop.style`` reads as wireframe.

    Returns
    -------
    bool
        ``True`` for the ``"Wireframe"`` style (case-insensitive).

    """
    style = getattr(prop, "style", "")
    return str(style).lower() == "wireframe"


def thickness_for(prop: pv.Property) -> float:
    """Map ``prop.line_width`` (pixels) to a WIREFRAME modifier thickness.

    Returns
    -------
    float
        World-unit thickness; clamped to a small positive value so a
        ``line_width=0`` actor still produces something visible.

    """
    width = float(getattr(prop, "line_width", 1.0) or 1.0)
    return max(width * _LINE_WIDTH_TO_THICKNESS, 1e-4)
