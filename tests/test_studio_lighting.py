# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""``config.lighting = "studio"``: a three-point area-light rig sized to the scene."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import pyvista as pv

pytest.importorskip("bpy")

import bpy

from pyvista_blender import config, studio_lights
from pyvista_blender.translate import light
from pyvista_blender.translate.scene import build_scene_from_plotter

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_N_RIG = 3
_N_STUDIO_BACKGROUNDS = 2  # the background for the camera, a white fill for the rest
# Positions and energies are float32 in bpy: compare to 1e-5 relative.
_RTOL = 1e-5


@pytest.fixture
def studio() -> Iterator[None]:
    """Switch the module default to the studio rig for one test."""
    original = config.lighting
    config.lighting = "studio"
    yield
    config.lighting = original


def _rig(scale: float) -> list[bpy.types.Object]:
    """Translate a box scene (side ``scale``) seen from a fixed camera.

    Returns
    -------
    list of bpy.types.Object
        The light objects of the built scene, sorted by name.

    """
    pl = pv.Plotter(off_screen=True, window_size=[640, 480])
    pl.add_mesh(pv.Box(bounds=(0, 4 * scale, 0, scale, 0, scale)))
    pl.camera_position = [
        (-2 * scale, -5 * scale, 3 * scale),
        (2 * scale, 0.5 * scale, 0.5 * scale),
        (0, 0, 1),
    ]
    build_scene_from_plotter(pl)
    pl.close()
    scene = cast("bpy.types.Scene", bpy.context.scene)
    return sorted((o for o in scene.objects if o.type == "LIGHT"), key=lambda o: o.name)


def _pose(o: bpy.types.Object) -> tuple[np.ndarray, float, float]:
    """Return a rig light's position, power and size.

    Returns
    -------
    tuple of (np.ndarray, float, float)
        World position, energy (W) and side of the area light.

    """
    data = cast("bpy.types.AreaLight", o.data)
    return np.array(o.matrix_world)[:3, 3], float(data.energy), float(data.size)


@pytest.mark.usefixtures("studio")
def test_studio_rig_is_three_shadowed_area_lights() -> None:
    """The rig replaces the VTK lights: three area lights casting shadows."""
    lights = _rig(1.0)
    if len(lights) != _N_RIG:
        pytest.fail(f"expected {_N_RIG} lights, got {[o.name for o in lights]}")
    for o in lights:
        data = cast("bpy.types.AreaLight", o.data)
        if data.type != "AREA" or not data.use_shadow:
            pytest.fail(f"{o.name}: {data.type}, shadow {data.use_shadow}")


@pytest.mark.usefixtures("studio")
def test_studio_rig_aims_at_the_scene() -> None:
    """Each light's -Z axis points at the centre of the scene's bounds."""
    centre = np.array([2.0, 0.5, 0.5])
    for o in _rig(1.0):
        m = np.array(o.matrix_world)
        to_centre = centre - m[:3, 3]
        aim = -m[:3, 2]
        cos = aim @ to_centre / np.linalg.norm(to_centre)
        np.testing.assert_allclose(cos, 1.0, rtol=_RTOL)


@pytest.mark.usefixtures("studio")
def test_studio_rig_scales_with_the_scene() -> None:
    """A scene 10x larger: lights 10x farther and larger, 100x the power.

    The irradiance on the scene (power over distance squared) is then the
    same, so a scene in millimetres looks like the same scene in metres.
    """
    small = [_pose(o) for o in _rig(1.0)]
    large = [_pose(o) for o in _rig(10.0)]
    for (p1, e1, s1), (p10, e10, s10) in zip(small, large, strict=True):
        np.testing.assert_allclose(p10, 10 * p1, rtol=_RTOL)
        np.testing.assert_allclose(e10, 100 * e1, rtol=_RTOL)
        np.testing.assert_allclose(s10, 10 * s1, rtol=_RTOL)


@pytest.mark.parametrize(("lighting", "lit"), [("studio", True), ("vtk", False)])
def test_studio_background_lights_the_scene(lighting: str, *, lit: bool) -> None:
    """With the studio rig the world is a white fill for shading rays.

    VTK draws a solid background behind the scene without lighting it;
    the studio rig keeps that background for the camera and fills the
    shadows with soft white light from all around.
    """
    original = config.lighting
    config.lighting = lighting  # ty: ignore[invalid-assignment]
    try:
        _rig(1.0)
    finally:
        config.lighting = original
    world = cast("bpy.types.Scene", bpy.context.scene).world
    if world is None:
        pytest.fail("no world after the scene build")
    vis = world.cycles_visibility
    if vis is None:
        pytest.fail("the world has no Cycles ray visibility")
    if (vis.diffuse, vis.glossy) != (lit, lit):
        pytest.fail(f"{lighting}: world diffuse/glossy {vis.diffuse}/{vis.glossy}")
    tree = world.node_tree
    if tree is None:
        pytest.fail("the world has no node tree")
    kinds = [n.bl_idname for n in tree.nodes]
    # Studio: camera rays see the background as set, shading rays a white fill.
    split = "ShaderNodeLightPath" in kinds and (
        kinds.count("ShaderNodeBackground") == _N_STUDIO_BACKGROUNDS
    )
    if lit != split:
        pytest.fail(f"{lighting}: world nodes {kinds}")


def test_studio_lighting_is_the_default() -> None:
    """The studio rig is the default; ``"vtk"`` keeps the plotter's lights."""
    if config.lighting != "studio":
        pytest.fail(f"default lighting is {config.lighting!r}")
    original = config.lighting
    config.lighting = "vtk"
    try:
        lights = _rig(1.0)
    finally:
        config.lighting = original
    if any(cast("bpy.types.Light", o.data).type == "AREA" for o in lights):
        pytest.fail("area lights with lighting='vtk'")


def test_preview_rig_matches_the_blender_rig() -> None:
    """``studio_lights()`` puts VTK lights where the Blender rig puts its own."""
    blender = [np.array(o.matrix_world)[:3, 3] for o in _rig(1.0)]
    blender_by_name = dict(zip((o.name for o in _rig(1.0)), blender, strict=True))
    pl = pv.Plotter(off_screen=True, window_size=[640, 480])
    pl.add_mesh(pv.Box(bounds=(0, 4, 0, 1, 0, 1)))
    pl.camera_position = [(-2, -5, 3), (2, 0.5, 0.5), (0, 0, 1)]
    pl.remove_all_lights()
    for lt in studio_lights(pl):
        pl.add_light(lt)
    scene = [lt for lt in pl.renderer.lights if lt.is_scene_light]
    camera = [lt for lt in pl.renderer.lights if lt.is_camera_light]
    pl.close()
    if len(scene) != _N_RIG or len(camera) != 1:
        pytest.fail(f"{len(scene)} scene lights, {len(camera)} camera lights")
    for lt, name in zip(scene, ("key", "fill", "rim"), strict=True):
        np.testing.assert_allclose(
            lt.position, blender_by_name[f"PVStudio_{name}"], rtol=_RTOL, atol=_RTOL
        )


@pytest.mark.usefixtures("studio")
def test_user_placed_lights_are_kept() -> None:
    """A light the user placed in the scene is translated; no rig, no fill."""
    pl = pv.Plotter(off_screen=True, window_size=[640, 480], lighting="none")
    pl.add_mesh(pv.Box())
    pl.add_light(pv.Light(position=(3, -2, 4), light_type="scene light"))
    build_scene_from_plotter(pl)
    pl.close()
    scene = cast("bpy.types.Scene", bpy.context.scene)
    kinds = [
        cast("bpy.types.Light", o.data).type for o in scene.objects if o.type == "LIGHT"
    ]
    if "AREA" in kinds or len(kinds) != 1:
        pytest.fail(f"user light not translated alone: {kinds}")
    world = scene.world
    vis = world.cycles_visibility if world is not None else None
    if vis is None or vis.diffuse:
        pytest.fail("the world lights the scene although the rig is off")


@pytest.mark.usefixtures("studio")
def test_preview_rig_is_recognised() -> None:
    """With ``studio_lights()`` in the plotter the render still uses the rig."""
    pl = pv.Plotter(off_screen=True, window_size=[640, 480])
    pl.add_mesh(pv.Box(bounds=(0, 4, 0, 1, 0, 1)))
    pl.camera_position = [(-2, -5, 3), (2, 0.5, 0.5), (0, 0, 1)]
    pl.remove_all_lights()
    for lt in studio_lights(pl):
        pl.add_light(lt)
    build_scene_from_plotter(pl)
    pl.close()
    scene = cast("bpy.types.Scene", bpy.context.scene)
    kinds = [
        cast("bpy.types.Light", o.data).type for o in scene.objects if o.type == "LIGHT"
    ]
    if kinds != ["AREA"] * _N_RIG:
        pytest.fail(f"preview rig not recognised: {kinds}")


def test_unknown_lighting_is_refused() -> None:
    """An unknown option fails loudly instead of rendering with no lights."""
    original = config.lighting
    config.lighting = "neon"  # ty: ignore[invalid-assignment]
    try:
        with pytest.raises(ValueError, match="neon"):
            light.translate_lights(pv.Plotter(off_screen=True))
    finally:
        config.lighting = original


@pytest.mark.usefixtures("studio")
def test_render_takes_lighting_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``render(lighting=...)`` beats ``pl.blender.lighting``, which beats the module.

    The module default is restored after each render.
    """
    seen: list[str] = []

    def fake_render(*_args: object, **_kwargs: object) -> None:
        seen.append(config.lighting)

    monkeypatch.setattr("pyvista_blender._render_impl.do_render", fake_render)
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Box())
    out = str(tmp_path / "out.png")
    pl.blender.render(out)
    pl.blender.render(out, lighting="vtk")
    pl.blender.lighting = "vtk"
    pl.blender.render(out)
    pl.blender.render(out, lighting="studio")
    pl.close()
    if seen != ["studio", "vtk", "vtk", "studio"] or config.lighting != "studio":
        pytest.fail(f"lighting per render {seen}, module left at {config.lighting!r}")
