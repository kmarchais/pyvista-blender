# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct RGB(A) scalars (``add_mesh(..., rgb=True)``) keep their colours in Blender."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import pyvista as pv

if TYPE_CHECKING:
    import bpy

pytest.importorskip("bpy")

from pyvista_blender.translate import material, mesh

# A float32 colour attribute holds the linearised sRGB value; 1e-6 covers its rounding.
_TOL = 1e-6
_SRGB_KNEE = 0.04045  # sRGB transfer function: linear segment below, power law above


def _linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= _SRGB_KNEE, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _colour(xy: np.ndarray) -> np.ndarray:
    """Colour points by position: red with x, green with y, blue fixed.

    Returns
    -------
    np.ndarray
        ``(n, 3)`` sRGB colours in 0..1.

    """
    return np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), 0.5)])


def _plane() -> pv.PolyData:
    return pv.Plane(
        center=(0.5, 0.5, 0.0), i_size=1.0, j_size=1.0, i_resolution=4, j_resolution=3
    )


def _attribute(obj: bpy.types.Object) -> np.ndarray:
    data = cast("bpy.types.Mesh", obj.data)
    attr = cast("bpy.types.FloatColorAttribute", data.color_attributes["scalars"])
    out = np.empty(len(attr.data) * 4, dtype=np.float32)
    attr.data.foreach_get("color", out)
    return out.reshape(-1, 4)


@pytest.mark.parametrize("dtype", ["uint8", "float"])
def test_point_rgb_scalars_are_used_directly(
    offscreen_plotter: pv.Plotter, dtype: str
) -> None:
    """Point RGB colours land per vertex, linearised, not through a colormap."""
    plane = _plane()
    rgb = _colour(np.asarray(plane.points)[:, :2])
    plane.point_data["rgb"] = (
        np.round(255 * rgb).astype(np.uint8) if dtype == "uint8" else rgb
    )
    actor = offscreen_plotter.add_mesh(plane, scalars="rgb", rgb=True)
    obj = mesh.translate_actor_mesh(actor, "rgb_points")
    data = cast("bpy.types.Mesh", obj.data)
    co = np.array([v.co[:] for v in data.vertices])
    srgb = _colour(co[:, :2])
    if dtype == "uint8":
        srgb = np.round(255 * srgb) / 255
    got = _attribute(obj)
    np.testing.assert_allclose(got[:, :3], _linear(srgb), atol=_TOL)
    np.testing.assert_allclose(got[:, 3], 1.0, atol=_TOL)


def test_cell_rgba_scalars_are_used_directly(offscreen_plotter: pv.Plotter) -> None:
    """Cell RGBA colours land on every corner of the cell, alpha kept."""
    plane = _plane()
    centres = np.asarray(plane.cell_centers().points)[:, :2]
    rgba = np.column_stack([_colour(centres), np.full(len(centres), 0.25)])
    plane.cell_data["rgba"] = np.round(255 * rgba).astype(np.uint8)
    actor = offscreen_plotter.add_mesh(plane, scalars="rgba", rgba=True)
    obj = mesh.translate_actor_mesh(actor, "rgba_cells")
    data = cast("bpy.types.Mesh", obj.data)
    got = _attribute(obj)
    co = np.array([v.co[:] for v in data.vertices])
    loop_vert = np.array([lp.vertex_index for lp in data.loops])
    for poly in data.polygons:
        corners = range(poly.loop_start, poly.loop_start + poly.loop_total)
        # The triangle's cell: the quad whose centre is nearest its centroid.
        tri_centre = co[loop_vert[list(corners)], :2].mean(axis=0)
        cell = int(np.argmin(np.linalg.norm(centres - tri_centre, axis=1)))
        want = np.round(255 * rgba[cell]) / 255
        for k in corners:
            np.testing.assert_allclose(got[k, :3], _linear(want[:3]), atol=_TOL)
            np.testing.assert_allclose(got[k, 3], want[3], atol=_TOL)


def test_solid_colour_is_linearised(offscreen_plotter: pv.Plotter) -> None:
    """A flat ``color=`` lands in Base Color as scene-linear, like scalar colours.

    VTK colours are sRGB; Blender's Base Color is linear. ``#808080`` (sRGB
    0.502) is linear 0.216.
    """
    actor = offscreen_plotter.add_mesh(_plane(), color="#808080")
    mat = material.make_material(actor, "solid", has_scalars=False)
    if mat.node_tree is None:
        pytest.fail("the solid material has no node tree")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    base = np.array(bsdf.inputs["Base Color"].default_value)[:3]  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(base, _linear(np.full(3, 128 / 255)), atol=_TOL)
