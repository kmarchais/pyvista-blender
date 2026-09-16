# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-instance rotation on glyph layers.

``orient`` aligns an instance's +Z to a vector, which fixes two of the
three rotational degrees of freedom and leaves roll about that axis
undefined. That is enough for arrows and cones, whose appearance does
not depend on roll, and wrong for anything that tumbles. ``rotation``
carries the full Euler triple instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import pyvista as pv

if TYPE_CHECKING:
    from pathlib import Path

    import bpy

pytest.importorskip("bpy")

from pyvista_blender.translate.glyph import translate_glyph

ROLLED = math.pi / 4  # radians of roll about the alignment axis


def _wedge() -> pv.PolyData:
    """Build a glyph whose silhouette depends on roll about +Z.

    A cone or a sphere looks the same however it is rolled about its
    own axis, so neither can tell an aligned instance from a rotated
    one. This tetrahedron is deliberately asymmetric in the XY plane.

    Returns
    -------
    pv.PolyData
        A flat asymmetric tetrahedron straddling the XY plane.

    """
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
        dtype=np.float32,
    )
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 1, 3], [3, 0, 2, 3], [3, 1, 2, 3]])
    return pv.PolyData(points, faces)


def _rolled_source(euler: tuple[float, float, float]) -> pv.PolyData:
    """Build a one-point source carrying a rotation field.

    Parameters
    ----------
    euler
        Euler XYZ angles in radians, applied to the single instance.

    Returns
    -------
    pv.PolyData
        A single point with a ``rot`` point-data vector field.

    """
    source = pv.PolyData(np.zeros((1, 3), dtype=np.float32))
    source["rot"] = np.array([euler], dtype=np.float32)
    return source


def test_add_glyph_records_the_rotation_field(offscreen_plotter: pv.Plotter) -> None:
    """The component stores ``rotation`` on the spec it registers."""
    pl = offscreen_plotter
    pl.blender.add_glyph(
        _rolled_source((0.0, 0.0, ROLLED)), geom=_wedge(), rotation="rot"
    )

    registered = pl.blender.registered_glyphs
    if len(registered) != 1:
        pytest.fail(f"expected one glyph spec, got {len(registered)}")
    if registered[0].rotation != "rot":
        pytest.fail(f"expected rotation field 'rot', got {registered[0].rotation!r}")


@pytest.mark.bpy
def test_rotation_drives_the_instancer(offscreen_plotter: pv.Plotter) -> None:
    """A spec with ``rotation`` wires Euler-to-Rotation into the instancer."""
    pl = offscreen_plotter
    pl.blender.add_glyph(
        _rolled_source((0.0, 0.0, ROLLED)), geom=_wedge(), rotation="rot"
    )

    obj = translate_glyph(pl.blender.registered_glyphs[0], 0)
    tree = cast("bpy.types.NodesModifier", obj.modifiers[0]).node_group
    if tree is None:
        pytest.fail("the glyph modifier carries no node group")
    instancer = next(
        n for n in tree.nodes if n.bl_idname == "GeometryNodeInstanceOnPoints"
    )
    links = [
        link for link in tree.links if link.to_socket == instancer.inputs["Rotation"]
    ]
    if len(links) != 1:
        pytest.fail(f"expected the Rotation socket driven once, got {len(links)}")
    driver = links[0].from_node
    if driver is None or driver.bl_idname != "FunctionNodeEulerToRotation":
        pytest.fail(f"Rotation driven by {driver and driver.bl_idname}")


@pytest.mark.bpy
def test_rotation_takes_precedence_over_orient(offscreen_plotter: pv.Plotter) -> None:
    """When both fields are declared, the full rotation wins."""
    pl = offscreen_plotter
    source = _rolled_source((0.0, 0.0, ROLLED))
    source["vec"] = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pl.blender.add_glyph(source, geom=_wedge(), orient="vec", rotation="rot")

    obj = translate_glyph(pl.blender.registered_glyphs[0], 1)
    tree = cast("bpy.types.NodesModifier", obj.modifiers[0]).node_group
    if tree is None:
        pytest.fail("the glyph modifier carries no node group")
    aligners = [
        n for n in tree.nodes if n.bl_idname == "FunctionNodeAlignRotationToVector"
    ]
    if aligners:
        pytest.fail("orient chain was built even though rotation was declared")


@pytest.mark.bpy
def test_roll_about_the_axis_changes_the_render(tmp_path: Path) -> None:
    """Rolling an instance about +Z changes the image, which ``orient`` cannot.

    This is the regression that matters: driven by ``orient`` alone both
    renders would be identical, because aligning +Z to +Z says nothing
    about roll. The repeat render is the control, so a difference can
    never be mistaken for sampling noise.
    """

    def shot(euler: tuple[float, float, float], tag: str) -> np.ndarray:
        pl = pv.Plotter(off_screen=True, window_size=[200, 200])
        pl.blender.add_glyph(_rolled_source(euler), geom=_wedge(), rotation="rot")
        pl.camera_position = [(0.0, 0.0, 4.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        out = tmp_path / f"{tag}.png"
        pl.blender.render(str(out), samples=8)
        pl.close()
        return np.asarray(pv.read_texture(str(out)).to_array(), dtype=np.int16)

    upright = shot((0.0, 0.0, 0.0), "upright")
    repeat = shot((0.0, 0.0, 0.0), "repeat")
    rolled = shot((0.0, 0.0, ROLLED), "rolled")

    noise = float(np.abs(upright - repeat).mean())
    signal = float(np.abs(upright - rolled).mean())
    if signal <= noise:
        pytest.fail(f"roll changed the render by {signal}, noise floor {noise}")
