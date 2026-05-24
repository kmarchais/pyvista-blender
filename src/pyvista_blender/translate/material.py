# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate a ``pyvista.Property`` into a ``bpy.types.Material``.

Three orthogonal axes feed into the shader graph:

* **Interpolation**: ``pbr`` honours ``metallic`` / ``roughness`` directly;
  ``flat`` / ``gouraud`` / ``phong`` approximate VTK's Phong model via
  the standard Phong→GGX fit ``r = sqrt(2 / (n + 2))``.
* **Lighting**: ``prop.lighting = False`` swaps the Principled BSDF for
  a ``ShaderNodeEmission`` so the surface reads at its full base colour
  regardless of scene lights (matches VTK's "unlit" path).
* **Backface**: when ``actor.backface_prop`` is set, the front and back
  sides are mixed by ``ShaderNodeMixShader`` driven by the geometry
  ``Backfacing`` output, so thin surfaces (open disks, surface plots,
  glass) read differently from each side.

Per-vertex scalars routed through a ``ShaderNodeAttribute`` override
``Base Color`` on whichever branch is active.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import bpy

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["make_material"]

# Cycles' Principled BSDF renamed several inputs between 4.x and 4.6;
# probe the actual socket names at runtime so the bridge stays portable.
# `Specular IOR Level` is the post-4.6 name; `Specular` was the pre-4.6
# one. Same socket, different label.
_SPECULAR_INPUT_NAMES = ("Specular IOR Level", "Specular")


def make_material(
    actor: pv.Actor, name: str, *, has_scalars: bool
) -> bpy.types.Material:
    """Build a shader-graph material for the given actor.

    Parameters
    ----------
    actor
        Source actor whose :class:`pyvista.Property` (and optional
        ``backface_prop``) drives the shader graph.
    name
        Material data-block name.
    has_scalars
        Whether the target mesh has a ``"scalars"`` FLOAT_COLOR attribute
        that should be wired into base colour in place of the flat colour.

    Returns
    -------
    bpy.types.Material
        The new material, ready to append to ``obj.data.materials``.

    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node_tree = mat.node_tree
    # Start from a clean slate so we control every node — the factory
    # graph (Principled BSDF -> Output) only fits the single-sided lit case.
    for node in list(node_tree.nodes):
        if node.bl_idname != "ShaderNodeOutputMaterial":
            node_tree.nodes.remove(node)
    output = node_tree.nodes.get("Material Output") or node_tree.nodes.new(
        "ShaderNodeOutputMaterial"
    )

    front_socket = _build_surface_shader(
        node_tree, actor.prop, has_scalars=has_scalars, location_x=-400
    )

    backface_prop = getattr(actor, "backface_prop", None)
    if backface_prop is not None:
        back_socket = _build_surface_shader(
            node_tree, backface_prop, has_scalars=has_scalars, location_x=-400
        )
        front_socket = _mix_by_backfacing(node_tree, front_socket, back_socket)

    node_tree.links.new(front_socket, output.inputs["Surface"])

    _apply_alpha_blend(mat, actor.prop, backface_prop)
    return mat


def _build_surface_shader(
    node_tree: bpy.types.NodeTree,
    prop: pv.Property,
    *,
    has_scalars: bool,
    location_x: int,
) -> bpy.types.NodeSocket:
    """Build one side of a material — BSDF for lit, Emission for unlit.

    Returns
    -------
    bpy.types.NodeSocket
        The shader output socket to feed into MixShader or Material Output.

    """
    if not bool(getattr(prop, "lighting", True)):
        return _build_emission(node_tree, prop, has_scalars=has_scalars, x=location_x)
    return _build_principled(node_tree, prop, has_scalars=has_scalars, x=location_x)


def _build_principled(
    node_tree: bpy.types.NodeTree,
    prop: pv.Property,
    *,
    has_scalars: bool,
    x: int,
) -> bpy.types.NodeSocket:
    """Construct a Principled BSDF configured from ``prop``.

    Returns
    -------
    bpy.types.NodeSocket
        The BSDF's ``BSDF`` output socket.

    """
    bsdf = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (x, 0)
    _wire_base_color(node_tree, bsdf, prop, has_scalars=has_scalars)
    _apply_shading_model(bsdf, prop)
    _apply_alpha_input(bsdf, prop)
    return bsdf.outputs["BSDF"]


def _build_emission(
    node_tree: bpy.types.NodeTree,
    prop: pv.Property,
    *,
    has_scalars: bool,
    x: int,
) -> bpy.types.NodeSocket:
    """Construct an Emission shader for an unlit surface.

    Returns
    -------
    bpy.types.NodeSocket
        The Emission node's output socket.

    """
    emission = node_tree.nodes.new("ShaderNodeEmission")
    emission.location = (x, 0)
    emission.inputs["Strength"].default_value = 1.0
    if has_scalars:
        attr = node_tree.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "scalars"
        attr.location = (x - 250, 0)
        node_tree.links.new(attr.outputs["Color"], emission.inputs["Color"])
    else:
        r, g, b = prop.color.float_rgb
        emission.inputs["Color"].default_value = (r, g, b, 1.0)
    return emission.outputs["Emission"]


def _mix_by_backfacing(
    node_tree: bpy.types.NodeTree,
    front_socket: bpy.types.NodeSocket,
    back_socket: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Combine two shader outputs by the geometry ``Backfacing`` factor.

    Returns
    -------
    bpy.types.NodeSocket
        The MixShader's output, ready to wire into Material Output.

    """
    geometry = node_tree.nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-200, -300)

    mix = node_tree.nodes.new("ShaderNodeMixShader")
    mix.location = (-50, 0)
    node_tree.links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
    node_tree.links.new(front_socket, mix.inputs[1])
    node_tree.links.new(back_socket, mix.inputs[2])
    return mix.outputs["Shader"]


def _wire_base_color(
    node_tree: bpy.types.NodeTree,
    bsdf: bpy.types.Node,
    prop: pv.Property,
    *,
    has_scalars: bool,
) -> None:
    """Feed either the scalar attribute or the flat property colour into BSDF."""
    if has_scalars:
        attr_node = node_tree.nodes.new("ShaderNodeAttribute")
        attr_node.attribute_name = "scalars"
        attr_node.location = (bsdf.location.x - 250, bsdf.location.y)
        node_tree.links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        r, g, b = prop.color.float_rgb
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)


def _apply_shading_model(bsdf: bpy.types.Node, prop: pv.Property) -> None:
    """Set metallic / roughness / specular from the property's shading model."""
    interpolation = (
        getattr(getattr(prop, "interpolation", None), "name", "")
        or str(getattr(prop, "interpolation", ""))
    ).lower()

    if interpolation == "pbr":
        bsdf.inputs["Metallic"].default_value = float(prop.metallic)
        bsdf.inputs["Roughness"].default_value = float(prop.roughness)
        _set_specular_input(bsdf, 0.5)  # Cycles' physical default
        return

    # Phong / Gouraud / Flat: VTK's classical shading model.
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = _phong_power_to_roughness(
        float(getattr(prop, "specular_power", 100.0))
    )
    _set_specular_input(bsdf, float(getattr(prop, "specular", 0.0)))


def _apply_alpha_input(bsdf: bpy.types.Node, prop: pv.Property) -> None:
    """Forward ``prop.opacity`` to the BSDF's Alpha socket."""
    opacity = float(getattr(prop, "opacity", 1.0))
    if opacity < 1.0:
        bsdf.inputs["Alpha"].default_value = opacity


def _apply_alpha_blend(
    mat: bpy.types.Material,
    front_prop: pv.Property,
    back_prop: pv.Property | None,
) -> None:
    """Switch the material to alpha-blended rendering if any side is translucent."""
    front_alpha = float(getattr(front_prop, "opacity", 1.0))
    back_alpha = (
        float(getattr(back_prop, "opacity", 1.0)) if back_prop is not None else 1.0
    )
    if front_alpha < 1.0 or back_alpha < 1.0:
        mat.surface_render_method = "BLENDED"


def _phong_power_to_roughness(specular_power: float) -> float:
    """Convert a Phong exponent to a GGX roughness value.

    Uses the standard Walter et al. (2007) fit ``alpha = sqrt(2 / (n + 2))``
    where ``n`` is the Phong exponent. The default ``specular_power = 100``
    maps to ``roughness ≈ 0.14`` (a glossy plastic look).

    Returns
    -------
    float
        Roughness in ``[0.02, 1.0]``; the lower clamp avoids a degenerate
        mirror highlight that path tracing struggles to converge.

    """
    n = max(specular_power, 0.0)
    return max(math.sqrt(2.0 / (n + 2.0)), 0.02)


def _set_specular_input(bsdf: bpy.types.Node, value: float) -> None:
    """Assign ``value`` to whichever specular-weight input this bpy build exposes."""
    for socket_name in _SPECULAR_INPUT_NAMES:
        socket = bsdf.inputs.get(socket_name)
        if socket is not None:
            socket.default_value = value
            return
