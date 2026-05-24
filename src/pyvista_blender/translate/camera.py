# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate ``pyvista.Camera`` state into a ``bpy.types.Camera`` and pose.

VTK's ``view_angle`` is the **vertical** FOV in degrees; Blender's default
camera is horizontal. We set ``sensor_fit='VERTICAL'`` and ``lens_unit='FOV'``
so the same angle produces the same framing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import bpy
from mathutils import Matrix, Vector

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["look_at_matrix", "translate_camera"]


def translate_camera(
    pv_cam: pv.Camera, window_size: tuple[int, int]
) -> bpy.types.Object:
    """Add a ``bpy.types.Camera`` object matching the PyVista camera.

    Parameters
    ----------
    pv_cam
        The :class:`pyvista.Camera` to mirror.
    window_size
        ``(width, height)`` in pixels; mapped to ``scene.render.resolution_*``.

    Returns
    -------
    bpy.types.Object
        The new camera object, linked into the active scene and set as
        ``scene.camera``.

    """
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new("Camera")
    if getattr(pv_cam, "parallel_projection", False):
        cam_data.type = "ORTHO"
        scale = float(getattr(pv_cam, "parallel_scale", 1.0))
        cam_data.ortho_scale = 2.0 * scale * max(1.0, window_size[0] / window_size[1])
    else:
        cam_data.type = "PERSP"
        cam_data.sensor_fit = "VERTICAL"
        cam_data.lens_unit = "FOV"
        cam_data.angle = math.radians(float(pv_cam.view_angle))

    near, far = pv_cam.clipping_range
    cam_data.clip_start = float(near)
    cam_data.clip_end = float(far)

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    cam_obj.matrix_world = look_at_matrix(
        pv_cam.position, pv_cam.focal_point, pv_cam.up
    )
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    width, height = window_size
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100

    return cam_obj


def look_at_matrix(
    position: tuple[float, float, float],
    focal_point: tuple[float, float, float],
    up: tuple[float, float, float],
) -> Matrix:
    """Build a Blender world matrix from a position / focal-point / up triple.

    Blender cameras look down ``-Z``; we construct the basis accordingly.

    Parameters
    ----------
    position
        Camera origin in world space.
    focal_point
        World-space point the camera should look at.
    up
        Approximate up vector; orthogonalised against the view direction.

    Returns
    -------
    mathutils.Matrix
        A 4x4 world matrix suitable for ``cam_obj.matrix_world = ...``.

    """
    pos = Vector(position)
    fwd = (Vector(focal_point) - pos).normalized()
    z_axis = -fwd

    # Guard against an up vector that's (anti-)parallel to the view
    # direction — would produce a zero cross-product and a degenerate
    # basis. Happens in practice for VTK's headlight, which points along
    # the same axis the user's up vector typically lies on (e.g. world Z).
    up_vec = Vector(up).normalized()
    parallel_threshold = 0.999
    if abs(up_vec.dot(z_axis)) > parallel_threshold:
        up_vec = (
            Vector((0.0, 1.0, 0.0))
            if abs(z_axis.y) < parallel_threshold
            else Vector((1.0, 0.0, 0.0))
        )

    x_axis = up_vec.cross(z_axis).normalized()
    y_axis = z_axis.cross(x_axis)
    return Matrix((
        (x_axis.x, y_axis.x, z_axis.x, pos.x),
        (x_axis.y, y_axis.y, z_axis.y, pos.y),
        (x_axis.z, y_axis.z, z_axis.z, pos.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
