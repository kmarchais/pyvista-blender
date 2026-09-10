# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Line datasets must create visible geometry and refresh their thickness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
import pyvista as pv

if TYPE_CHECKING:
    import bpy

pytest.importorskip("bpy")

from pyvista_blender.translate import mesh, wireframe


def test_line_width_refresh(offscreen_plotter: pv.Plotter) -> None:
    """A cached polyline follows point movement and line-width changes."""
    line = pv.Line((0, 0, 0), (1, 0, 0))
    actor = offscreen_plotter.add_mesh(line, color="orange", line_width=2)
    obj = mesh.translate_actor_mesh(actor, "line")
    if not cast("bpy.types.Mesh", obj.data).polygons:
        pytest.fail("Line cells produced no renderable faces")
    before = np.array([v.co[:] for v in cast("bpy.types.Mesh", obj.data).vertices])
    line.points += (0, 1, 0)
    actor.prop.line_width *= 2
    if not mesh.refresh_actor_mesh(actor, obj):
        pytest.fail("Moving and widening a line should reuse its topology")
    after = np.array([v.co[:] for v in cast("bpy.types.Mesh", obj.data).vertices])
    np.testing.assert_allclose(after[:, 1].mean() - before[:, 1].mean(), 1)
    np.testing.assert_allclose(np.ptp(after[:, 2]), 2 * np.ptp(before[:, 2]))


@pytest.mark.parametrize("style", ["surface", "wireframe"])
def test_outline_has_visible_faces(
    offscreen_plotter: pv.Plotter, style: Literal["surface", "wireframe"]
) -> None:
    """Disconnected line segments render without a second wireframe overlay."""
    actor = offscreen_plotter.add_mesh(pv.Box().outline(), style=style, show_edges=True)
    obj = mesh.translate_actor_mesh(actor, "outline")
    if not cast("bpy.types.Mesh", obj.data).polygons:
        pytest.fail("Outline cells produced no renderable faces")
    if wireframe.actor_needs_wire(actor):
        pytest.fail("Line geometry must not receive a surface-edge overlay")
