# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate ``pyvista.Light`` objects (including the default ``vtkLightKit``).

Dispatch rules:

* ``positional=False`` → Blender ``SUN`` (directional, infinite).
* ``positional=True``, ``cone_angle >= 90`` → ``POINT``.
* ``positional=True``, ``cone_angle < 90``  → ``SPOT`` with
  ``spot_size = 2 * cone_angle``.

Headlight and camera-light lights are parented to the Blender camera so
they follow it (mirroring VTK's ``LightFollowCameraOn``). Scene lights
go in world space.

Energy scaling: PyVista's ``intensity`` is nominally in [0, 1]. Blender's
SUN energy is W/m² (default ≈ 5 = sunlight). Point/Spot use W (typical
~500). The multipliers below produce believable lighting for the default
light kit (total intensity ≈ 1.9) without over-exposing on a Standard
view transform.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import bpy
from mathutils import Euler

from pyvista_blender.translate.camera import look_at_matrix

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["translate_lights"]


_SUN_ENERGY_PER_INTENSITY = 4.0
_POINT_ENERGY_PER_INTENSITY = 500.0
_OMNIDIRECTIONAL_CONE_THRESHOLD = 90.0


def translate_lights(source: pv.BasePlotter | object) -> None:
    """Translate every visible ``pv.Light`` from a plotter or a renderer.

    Parameters
    ----------
    source
        Either a :class:`pyvista.BasePlotter` (walks its active
        renderer's lights) or a single ``pv.Renderer`` (walks that
        renderer's lights, used by the subplot tile path to produce a
        tile-specific light setup).

    """
    # If ``source`` exposes a ``renderer`` attribute we assume it's a
    # plotter and read lights from the active renderer; otherwise
    # assume ``source`` is already a renderer.
    if hasattr(source, "renderer"):
        lights = list(getattr(source.renderer, "lights", []))
    else:
        lights = list(getattr(source, "lights", []))

    if not lights:
        _add_fallback_sun()
        return

    cam_obj = bpy.context.scene.camera

    for index, light in enumerate(lights):
        if not getattr(light, "on", True):
            continue
        _translate_one_light(light, index, cam_obj)


def _translate_one_light(
    light: pv.Light,
    index: int,
    cam_obj: bpy.types.Object | None,
) -> None:
    """Add a single Blender light matching the given ``pv.Light``."""
    light_data = _make_light_data(light, index)
    color = light.diffuse_color.float_rgb
    light_data.color = (color[0], color[1], color[2])

    light_obj = bpy.data.objects.new(f"PVLight_{index}", light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    is_camera_relative = getattr(light, "is_headlight", False) or getattr(
        light, "is_camera_light", False
    )
    if is_camera_relative and cam_obj is not None:
        # `light.position` is in CAMERA-LOCAL coords. Setting matrix_local
        # after `parent =` looked clean but Blender's matrix_parent_inverse
        # silently absorbs the transform and lights end up at world origin.
        # Compose the camera's world matrix with the local light pose
        # directly — no parenting needed for one-shot offline rendering.
        # (If we want lights to follow a moving camera, we'll
        # re-introduce parenting with explicit matrix_parent_inverse
        # handling.)
        local_matrix = look_at_matrix(
            light.position, light.focal_point, (0.0, 0.0, 1.0)
        )
        light_obj.matrix_world = cam_obj.matrix_world @ local_matrix
    else:
        light_obj.matrix_world = look_at_matrix(
            light.world_position, light.world_focal_point, (0.0, 0.0, 1.0)
        )


def _make_light_data(light: pv.Light, index: int) -> bpy.types.Light:
    """Allocate a bpy Light data-block of the right type for ``light``.

    Returns
    -------
    bpy.types.Light
        SUN for directional, POINT for omnidirectional positional, SPOT for
        narrow positional. Energy is scaled from ``light.intensity``.

    """
    intensity = float(getattr(light, "intensity", 1.0))
    name = f"PVLight_{index}"

    if not getattr(light, "positional", False):
        data = bpy.data.lights.new(name, type="SUN")
        data.energy = intensity * _SUN_ENERGY_PER_INTENSITY
        # angle=0 → perfect directional source (zero angular diameter).
        # Default ~5° gives sun-realistic soft shadows and broad specular
        # peaks, which wash out scivis highlights and mute terrain relief.
        data.angle = 0.0
        # PyVista's lights don't cast shadows by default (would require an
        # explicit pl.enable_shadows() call). Matching that means every
        # slope receives the full Lambert from every light direction,
        # giving terrain its natural shaded read and letting every kit
        # light contribute its specular highlight without occlusion.
        data.use_shadow = False
        return data

    cone = float(getattr(light, "cone_angle", _OMNIDIRECTIONAL_CONE_THRESHOLD))
    if cone >= _OMNIDIRECTIONAL_CONE_THRESHOLD:
        data = bpy.data.lights.new(name, type="POINT")
        data.energy = intensity * _POINT_ENERGY_PER_INTENSITY
        return data

    data = bpy.data.lights.new(name, type="SPOT")
    data.energy = intensity * _POINT_ENERGY_PER_INTENSITY
    data.spot_size = math.radians(2.0 * cone)
    return data


def _add_fallback_sun() -> None:
    """Add a single default SUN light when the plotter has no lights at all."""
    light_data = bpy.data.lights.new("FallbackSun", type="SUN")
    light_data.energy = 5.0

    light_obj = bpy.data.objects.new("FallbackSun", light_data)
    light_obj.rotation_euler = Euler(
        (math.radians(55.0), math.radians(15.0), math.radians(35.0)),
        "XYZ",
    )
    bpy.context.scene.collection.objects.link(light_obj)
