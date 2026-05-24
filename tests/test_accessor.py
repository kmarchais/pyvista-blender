# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Acceptance test: ``blender`` namespace reachable via entry-point discovery.

The smoke tests below avoid the bare ``assert`` keyword (``S101`` rule) and
the private-member dot-access pattern (``SLF001``). Failures call
``pytest.fail`` directly, which pytest displays with the same traceback
layout as a bare ``assert``.

See: docs/architecture.md for the entry-point contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import bpy
import numpy as np
import pytest
import pyvista as pv
from PIL import Image

import pyvista_blender as pvb
from pyvista_blender.translate.scene import build_scene_from_plotter

if TYPE_CHECKING:
    from collections.abc import Callable


def test_blender_accessor_resolves_without_explicit_import(
    offscreen_plotter: pv.Plotter,
) -> None:
    """``pl.blender`` exists and is a registered plotter component."""
    pl = offscreen_plotter
    if not hasattr(pl, "blender"):
        pytest.fail("pl.blender missing — entry-point registration failed")


@pytest.mark.bpy
def test_blender_render_writes_a_png(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``render(path)`` returns the path and writes a non-empty PNG."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red")
    out = tmp_path / "render.png"
    result = pl.blender.render(str(out), samples=4)
    if result != str(out):
        pytest.fail(f"render() returned {result!r}, expected {out!s}")
    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("render() did not produce a non-empty PNG")


@pytest.mark.bpy
def test_blender_export_blend_writes_a_file(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``export_blend(path)`` writes a non-empty ``.blend`` with the scene baked in.

    Blender 5.x defaults to zstd-compressed ``.blend`` files; older
    builds wrote uncompressed. We accept either magic to keep the test
    robust across bpy wheels.
    """
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red")
    out = tmp_path / "scene.blend"
    result = pl.blender.export_blend(str(out))
    if result != str(out):
        pytest.fail(f"export_blend() returned {result!r}, expected {out!s}")
    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("export_blend() did not produce a non-empty .blend file")
    blender_magic = b"BLENDER"
    zstd_magic = b"\x28\xb5\x2f\xfd"
    with out.open("rb") as fh:
        head = fh.read(7)
    if not head.startswith((blender_magic, zstd_magic)):
        pytest.fail(f".blend missing BLENDER / zstd magic; got {head!r}")


@pytest.mark.bpy
def test_subplot_layout_renders_per_tile(tmp_path: Path) -> None:
    """A 1-by-2 subplot plotter renders each viewport with its own camera/lights.

    The bridge dispatches to the multi-pass tile path when
    ``len(plotter.renderers) > 1``; each tile gets its own
    ``camera`` / ``light`` / ``background`` translation, then the
    tiles are PIL-composited into the final PNG. The test just
    confirms the dispatch produces a non-empty file at the right
    size — it doesn't pixel-diff against a baseline (Cycles output
    on subplots is sample-dependent).
    """
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[320, 240])
    pl.subplot(0, 0)
    pl.add_mesh(pv.Sphere(), color="red")
    pl.subplot(0, 1)
    pl.add_mesh(pv.Cube(), color="blue")

    out = tmp_path / "subplot.png"
    pl.blender.render(str(out), samples=4)
    pl.close()

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("subplot render produced no output")
    with Image.open(out) as img:
        if img.size != (320, 240):
            pytest.fail(f"subplot composite size {img.size} != (320, 240)")


def test_blender_is_component_per_plotter(offscreen_plotter: pv.Plotter) -> None:
    """Each plotter gets its own component instance (cached on the plotter)."""
    pl1 = offscreen_plotter
    pl2 = pv.Plotter(off_screen=True)
    try:
        if pl1.blender is pl2.blender:
            pytest.fail("two plotters share a single component")
        if pl1.blender is not pl1.blender:
            pytest.fail("the component was not cached on the plotter")
    finally:
        pl2.close()


def test_three_tier_config_resolution(offscreen_plotter: pv.Plotter) -> None:
    """Per-call > component > module defaults."""
    pl = offscreen_plotter

    # 1. Module default
    if pvb.config.engine != "cycles":
        pytest.fail(
            f"module default engine should be 'cycles', got {pvb.config.engine!r}",
        )

    # 2. Component override. Use the public resolver, not the underscore form
    # — keeps tests off SLF001 and matches what a real user would call.
    pl.blender.engine = "eevee"
    resolved = pl.blender.resolve_config("engine", call_value=None)
    if resolved != "eevee":
        pytest.fail(f"component override not picked up; got {resolved!r}")

    # 3. Per-call override wins
    resolved_call = pl.blender.resolve_config("engine", call_value="cycles")
    if resolved_call != "cycles":
        pytest.fail(f"per-call override not picked up; got {resolved_call!r}")


# End-to-end Eevee render (``pl.blender.render(engine="eevee")``) works
# in isolation but contaminates bpy state in a way that hangs the next
# Cycles render in the same process. We don't have a clean way to fully
# reset bpy between tests, so the end-to-end Eevee check is done as a
# manual smoke; the contract that ``engine="eevee"`` resolves through
# ``_resolve_engine`` without raising is covered by
# ``tests/test_validation.py::test_supported_engines_contains_cycles_and_eevee``.


@pytest.mark.bpy
def test_scalar_bar_title_sits_above_bar(offscreen_plotter: pv.Plotter) -> None:
    """scalar-bar title renders above the colorbar (not clipped below).

    Regression for the matplotlib ``cb.set_label`` placement: when the
    bar sits flush against the image edge, ``set_label`` puts the title
    *below* (clipped). The bridge now uses ``ax.set_title`` so the title
    lands above. We render only the scalar-bar overlay (no Cycles needed)
    and check the rows directly above the configured bar rect carry
    non-zero alpha.
    """
    pl = offscreen_plotter
    mesh = pv.Sphere()
    mesh["z"] = mesh.points[:, 2]
    pl.add_mesh(
        mesh,
        scalars="z",
        cmap="viridis",
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Height (m)",
            "color": "white",
            "position_x": 0.35,
            "position_y": 0.05,
            "width": 0.3,
            "height": 0.05,
        },
    )

    width, height = 800, 400
    rgba = pl.blender.render_hud_overlay("scalar_bar", width=width, height=height)
    if rgba is None:
        pytest.fail("scalar_bar overlay returned None despite a visible bar")

    # Bar rect in PIL coords (top-left origin):
    #   bar bottom edge: y_top = (1 - (y0 + h)) * height
    #   bar top edge:    y_top - h * height
    # Title should land in the rows just ABOVE the bar's top edge.
    y0 = 0.05
    h = 0.05
    bar_top_pil = round((1.0 - (y0 + h)) * height)
    title_band = rgba[max(bar_top_pil - 40, 0) : bar_top_pil, :, 3]
    if float(title_band.max()) < 0.5:  # noqa: PLR2004
        pytest.fail(
            f"no opaque pixels in the band above the bar (max alpha "
            f"{float(title_band.max()):.2f}); title may have been clipped below"
        )


def test_hud_overlay_changes_pixels(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A render with a scalar bar produces a PNG with extra non-bg pixels."""
    pl = offscreen_plotter
    sphere = pv.Sphere()
    sphere["z"] = sphere.points[:, 2]

    bare = tmp_path / "bare.png"
    pl.add_mesh(sphere, scalars="z", cmap="viridis", show_scalar_bar=False)
    pl.blender.render(str(bare), samples=4)
    with Image.open(bare) as img:
        bare_pixels = np.asarray(img.convert("RGB")).copy()

    pl.clear()
    pl.add_mesh(sphere, scalars="z", cmap="viridis", show_scalar_bar=True)
    annotated = tmp_path / "annotated.png"
    pl.blender.render(str(annotated), samples=4)
    with Image.open(annotated) as img:
        annotated_pixels = np.asarray(img.convert("RGB")).copy()

    if bare_pixels.shape != annotated_pixels.shape:
        pytest.fail("HUD render changed the output resolution")
    if np.array_equal(bare_pixels, annotated_pixels):
        pytest.fail("show_scalar_bar=True produced an identical render")


@pytest.mark.bpy
def test_add_glyph_renders_instances(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """add_glyph produces a render and stashes a glyph spec on the component."""
    pl = offscreen_plotter
    points = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    points["vec"] = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    pl.blender.add_glyph(points, geom=pv.Cone(), orient="vec", factor=0.3)

    expected_specs = 1
    registered = pl.blender.registered_glyphs
    if len(registered) != expected_specs:
        pytest.fail(
            f"expected {expected_specs} glyph spec stored, got {len(registered)}"
        )

    out = tmp_path / "glyph.png"
    pl.blender.render(str(out), samples=4)
    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("add_glyph render produced no file")


@pytest.mark.bpy
def test_baked_glyph_polydata_emits_hint(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A pre-baked ``mesh.glyph(...)`` polydata triggers the GlyphVector warning."""
    pl = offscreen_plotter
    src = pv.Sphere()
    src["vec"] = np.tile([0.0, 0.0, 1.0], (src.n_points, 1)).astype(np.float32)
    src["mag"] = np.ones(src.n_points, dtype=np.float32)
    baked = src.glyph(orient="vec", scale="mag", geom=pv.Arrow(), factor=0.3)
    pl.add_mesh(baked)

    with pytest.warns(UserWarning, match=r"pre-baked glyph polydata"):
        pl.blender.render(str(tmp_path / "baked.png"), samples=4)


@pytest.mark.bpy
def test_animate_writes_a_gif(offscreen_plotter: pv.Plotter, tmp_path: Path) -> None:
    """End-to-end animate(): two frames of a sphere mutate to produce a gif file."""
    pl = offscreen_plotter
    sphere = pv.Sphere(radius=1.0)
    pl.add_mesh(sphere, color="red")

    def update(frame: int) -> None:
        if frame:
            sphere.points[:, 0] += 0.1

    out = tmp_path / "tiny.gif"
    result = pl.blender.animate(
        str(out), updater=update, frames=range(2), fps=12, samples=4
    )
    if result != str(out):
        pytest.fail(f"animate() returned {result!r}, expected {out!s}")
    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("animate() did not produce a non-empty gif file")


@pytest.mark.bpy
def test_animate_webm_uses_vp9_codec(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A ``.webm`` round-trip produces a non-empty file with a VP9 stream.

    Locks in the codec dispatch: ``libvpx-vp9`` for webm, ``libx264`` for
    the other ffmpeg containers. A hardcoded libx264 would either reject
    the webm container outright or silently mislabel the stream.
    """
    pl = offscreen_plotter
    sphere = pv.Sphere(radius=1.0)
    pl.add_mesh(sphere, color="red")

    def update(frame: int) -> None:
        if frame:
            sphere.points[:, 0] += 0.1

    out = tmp_path / "tiny.webm"
    pl.blender.animate(str(out), updater=update, frames=range(2), fps=12, samples=4)
    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("animate() did not produce a non-empty webm file")

    # WebM files start with the EBML header bytes (0x1A45DFA3) — a sanity
    # check that we wrote a Matroska/WebM container rather than an mp4
    # mislabeled with a .webm extension.
    expected_magic = b"\x1a\x45\xdf\xa3"
    with out.open("rb") as fh:
        header = fh.read(4)
    if header != expected_magic:
        pytest.fail(
            f"webm output missing EBML magic; got {header!r} (codec dispatch broken?)"
        )


def _action_fcurves(action: object) -> list[bpy.types.FCurve]:
    """Flatten an action's fcurves across bpy 4.x and 5.x layouts.

    bpy 4.x exposes ``action.fcurves`` directly. bpy 5.x reorganised
    actions into layers / strips / slots, with curves living under
    ``action.layers[*].strips[*].channelbag(slot).fcurves``. The test
    only cares that *some* fcurves exist for keyframed paths, so we
    flatten across whichever shape this stub-version provides.

    Returns
    -------
    list of bpy.types.FCurve
        Every fcurve attached to ``action``, regardless of bpy major
        version. Empty when no keyframes have been inserted.

    """
    legacy = list(getattr(action, "fcurves", []) or [])
    if legacy:
        return legacy
    flattened: list[bpy.types.FCurve] = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for slot in getattr(action, "slots", []):
                cb = strip.channelbag(slot)
                if cb is not None:
                    flattened.extend(cb.fcurves)
    return flattened


def _bake_orbit_animation(
    pl: pv.Plotter, out_path: Path, *, n_frames: int, fps: int
) -> None:
    """Bake an orbit animation around ``pl``'s camera and reload the saved .blend.

    Centralises the boilerplate for the two animation-export tests so
    each test body stays under ruff's C901 complexity threshold.
    """
    pl.add_mesh(pv.Sphere(), color="red")
    pl.camera_position = [(4.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    updater = pl.blender.orbit_camera(n_frames=n_frames)
    pl.blender.export_animation_blend(
        str(out_path), updater, frames=range(n_frames), fps=fps
    )
    bpy.ops.wm.open_mainfile(filepath=str(out_path))


@pytest.mark.bpy
def test_export_animation_blend_sets_timeline(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``export_animation_blend`` configures frame_start/end/fps."""
    n_frames = 8
    out = tmp_path / "orbit_timeline.blend"
    _bake_orbit_animation(offscreen_plotter, out, n_frames=n_frames, fps=24)
    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("export_animation_blend produced no .blend file")

    scene = bpy.context.scene
    if scene is None:
        pytest.fail("reloaded .blend has no active scene")
    if (scene.frame_start, scene.frame_end) != (0, n_frames - 1):
        pytest.fail(
            f"frame range {scene.frame_start}-{scene.frame_end}, "
            f"expected 0-{n_frames - 1}"
        )
    expected_fps = 24
    if int(scene.render.fps) != expected_fps:
        pytest.fail(f"fps {scene.render.fps}, expected {expected_fps}")


@pytest.mark.bpy
def test_export_animation_blend_keyframes_camera_location(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """Camera ends up with location fcurves that vary across frames."""
    n_frames = 8
    out = tmp_path / "orbit_camera.blend"
    _bake_orbit_animation(offscreen_plotter, out, n_frames=n_frames, fps=24)

    scene = bpy.context.scene
    if scene is None or scene.camera is None:
        pytest.fail("reloaded .blend missing scene or camera")
    anim = scene.camera.animation_data
    if anim is None or anim.action is None:
        pytest.fail("camera has no animation_data.action — keyframes were not baked")

    fcurves = _action_fcurves(anim.action)
    if not fcurves:
        pytest.fail("no fcurves found on camera action — keyframes missing")
    loc_curves = [fc for fc in fcurves if fc.data_path == "location"]
    expected_axes = 3
    if len(loc_curves) != expected_axes:
        pytest.fail(
            f"expected {expected_axes} location fcurves (xyz), got {len(loc_curves)}"
        )
    varies = any(
        len({round(kp.co.y, 6) for kp in fc.keyframe_points}) > 1 for fc in loc_curves
    )
    if not varies:
        pytest.fail("camera location did not vary across orbit frames")


def _bake_deformation_animation(
    pl: pv.Plotter,
    out_path: Path,
    *,
    n_frames: int,
    bake_deformation: bool | str,
    bake_camera: bool = True,
) -> None:
    """Add a sphere, deform its Z coords per frame, export, reload.

    Same complexity-reduction motivation as :func:`_bake_orbit_animation`.
    """
    sphere = pv.Sphere(radius=1.0)
    rest_z = sphere.points[:, 2].copy()
    pl.add_mesh(sphere, color="red")
    pl.camera_position = [(4.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    updater = pl.blender.orbit_camera(n_frames=n_frames)

    def combined(frame: int) -> None:
        updater(frame)
        sphere.points[:, 2] = rest_z + 0.3 * np.sin(frame * 0.5)

    pl.blender.export_animation_blend(
        str(out_path),
        combined,
        frames=range(n_frames),
        fps=24,
        bake_camera=bake_camera,
        bake_deformation=bake_deformation,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out_path))


def _assert_face_domain_scalar_modifier(mesh_obj: bpy.types.Object) -> None:
    """Verify ``mesh_obj`` has a cell-scalars NODES modifier on FACE domain.

    Calls ``pytest.fail`` on any structural mismatch (no NODES mod, no
    Store Named Attribute node, wrong domain). Extracted so the test
    that uses it stays under ruff's C901 complexity threshold.
    """
    nodes_mods = [
        cast("bpy.types.NodesModifier", m)
        for m in mesh_obj.modifiers
        if m.type == "NODES"
    ]
    if not nodes_mods:
        pytest.fail("bake_scalars on cell-data did not add a NODES modifier")
    ng = nodes_mods[0].node_group
    if ng is None:
        pytest.fail("NODES modifier has no node_group attached")
    store = next(
        (
            cast("bpy.types.GeometryNodeStoreNamedAttribute", n)
            for n in ng.nodes
            if n.bl_idname == "GeometryNodeStoreNamedAttribute"
        ),
        None,
    )
    if store is None:
        pytest.fail("NODES modifier has no Store Named Attribute node")
    if store.domain != "FACE":
        pytest.fail(
            f"expected Store domain 'FACE' for cell-data scalars, got {store.domain!r}"
        )


def _sample_face_scalars(
    mesh_obj: bpy.types.Object,
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    *,
    frame: int,
) -> np.ndarray:
    """Set ``scene.frame_current`` and return the FACE ``"scalars"`` array.

    Used by the cell-data test to compare values across frames
    without inflating the test body's branch count.

    Returns
    -------
    np.ndarray
        Flat float32 RGBA buffer (``len == 4 * N_faces``) sampled
        from the evaluated mesh's ``"scalars"`` color attribute.

    """
    scene.frame_set(frame)
    ev_mesh = cast("bpy.types.Mesh", mesh_obj.evaluated_get(depsgraph).data)
    attr = cast(
        "bpy.types.FloatColorAttribute | None",
        ev_mesh.attributes.get("scalars"),
    )
    if attr is None or attr.domain != "FACE":
        pytest.fail(
            f"frame {frame}: 'scalars' missing or wrong domain "
            f"({attr.domain if attr else None})"
        )
    buf = np.zeros(len(attr.data) * 4, dtype=np.float32)
    attr.data.foreach_get("color", buf)
    return buf


def _first_mesh_data() -> bpy.types.Mesh | None:
    """Return the first ``bpy.types.Mesh`` data-block in the loaded scene.

    The reloaded file has cameras + lights besides the deforming
    sphere; we want the single mesh data-block so the bake-deformation
    assertions can inspect its shape keys without hardcoding a name
    (the bridge mangles PyVista actor names with friendly-name rules)
    and without ty complaining that ``obj.data`` is a union over every
    possible data type.

    Returns
    -------
    bpy.types.Mesh or None
        The first mesh data-block found, or ``None`` if the scene has
        no mesh objects.

    """
    for obj in bpy.data.objects:
        data = obj.data
        if isinstance(data, bpy.types.Mesh):
            return data
    return None


@pytest.mark.bpy
def test_export_animation_blend_bakes_shape_keys(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_deformation='shape_keys'`` produces N+1 keys + value fcurves."""
    n_frames = 6
    out = tmp_path / "deform.blend"
    _bake_deformation_animation(
        offscreen_plotter, out, n_frames=n_frames, bake_deformation="shape_keys"
    )

    mesh_data = _first_mesh_data()
    if mesh_data is None:
        pytest.fail("reloaded .blend has no mesh data-block")
    shape_keys = mesh_data.shape_keys
    if shape_keys is None:
        pytest.fail("bake_deformation='shape_keys' did not create shape keys")
    expected_blocks = n_frames + 1  # Basis + one per frame
    if len(shape_keys.key_blocks) != expected_blocks:
        pytest.fail(
            f"expected {expected_blocks} key_blocks (basis + {n_frames}), "
            f"got {len(shape_keys.key_blocks)}: "
            f"{[k.name for k in shape_keys.key_blocks]}"
        )

    anim = shape_keys.animation_data
    if anim is None or anim.action is None:
        pytest.fail("shape keys have no animation_data.action")
    fcurves = _action_fcurves(anim.action)
    value_curves = [fc for fc in fcurves if fc.data_path.endswith("].value")]
    if len(value_curves) != n_frames:
        pytest.fail(
            f"expected {n_frames} shape-key value fcurves, got {len(value_curves)}"
        )


@pytest.mark.bpy
def test_export_animation_blend_default_skips_shape_keys(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """Default ``bake_deformation=False`` leaves meshes shape-key-free."""
    out = tmp_path / "no_deform.blend"
    _bake_deformation_animation(
        offscreen_plotter, out, n_frames=4, bake_deformation=False
    )

    mesh_data = _first_mesh_data()
    if mesh_data is None:
        pytest.fail("reloaded .blend has no mesh data-block")
    if mesh_data.shape_keys is not None:
        pytest.fail(
            "bake_deformation=False should leave shape_keys None; got "
            f"{[k.name for k in mesh_data.shape_keys.key_blocks]}"
        )


@pytest.mark.bpy
def test_export_animation_blend_warns_on_topology_change(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """An actor whose point count changes mid-animation triggers a warning."""
    pl = offscreen_plotter
    # Start with a sphere; mid-animation, swap it for a cube so the
    # actor's underlying dataset.points has a different size.
    sphere = pv.Sphere(radius=1.0)
    cube_points = pv.Cube().points
    pl.add_mesh(sphere, color="red", name="changing")
    pl.camera_position = [(4.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    topology_swap_frame = 2

    def update(frame: int) -> None:
        if frame >= topology_swap_frame:
            # PyVista keeps point-data arrays (Normals, ...) sized to
            # the original vertex count; reassigning .points to a
            # different size first requires clearing those companion
            # arrays or pyvista raises InvalidMeshWarning before our
            # topology-change warning fires.
            sphere.clear_data()
            sphere.points = cube_points

    out = tmp_path / "topology.blend"
    with pytest.warns(UserWarning, match=r"point count changed"):
        pl.blender.export_animation_blend(
            str(out), update, frames=range(4), fps=24, bake_deformation=True
        )

    bpy.ops.wm.open_mainfile(filepath=str(out))
    mesh_data = _first_mesh_data()
    if mesh_data is None:
        pytest.fail("reloaded .blend has no mesh data-block")
    if mesh_data.shape_keys is not None:
        pytest.fail(
            "topology-unstable actor should have no shape keys; got "
            f"{[k.name for k in mesh_data.shape_keys.key_blocks]}"
        )


@pytest.mark.bpy
def test_export_animation_blend_mdd_backend_writes_sidecar_and_modifier(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_deformation='mdd'`` writes an MDD sidecar + MESH_CACHE mod."""
    n_frames = 5
    out = tmp_path / "wave.blend"
    _bake_deformation_animation(
        offscreen_plotter, out, n_frames=n_frames, bake_deformation="mdd"
    )

    mesh_data = _first_mesh_data()
    if mesh_data is None:
        pytest.fail("reloaded .blend has no mesh data-block")
    # MDD path should NOT create shape keys.
    if mesh_data.shape_keys is not None:
        pytest.fail(
            "bake_deformation='mdd' should not create shape keys; got "
            f"{[k.name for k in mesh_data.shape_keys.key_blocks]}"
        )

    mesh_obj = next(
        (obj for obj in bpy.data.objects if obj.data is mesh_data),
        None,
    )
    if mesh_obj is None:
        pytest.fail("could not find object wrapping the mesh data-block")
    mesh_cache_mods = [m for m in mesh_obj.modifiers if m.type == "MESH_CACHE"]
    if len(mesh_cache_mods) != 1:
        pytest.fail(
            f"expected exactly one MESH_CACHE modifier, got {len(mesh_cache_mods)}"
        )
    # Cast to the concrete subtype so ty resolves cache_format / filepath
    # (fake-bpy-module's ``Object.modifiers`` typed return is the abstract
    # ``Modifier``, which doesn't carry the MeshCache-specific attributes).
    mod = cast("bpy.types.MeshCacheModifier", mesh_cache_mods[0])
    if mod.cache_format != "MDD":
        pytest.fail(f"modifier cache_format={mod.cache_format!r}, expected 'MDD'")

    resolved = Path(bpy.path.abspath(mod.filepath))
    if not resolved.exists() or resolved.stat().st_size == 0:
        pytest.fail(
            f"MDD sidecar missing or empty at {resolved} (modifier filepath: "
            f"{mod.filepath!r})"
        )
    # Sanity-check the MDD header: big-endian uint32 frame count, then
    # uint32 vertex count. Frame count must equal the requested N.
    with resolved.open("rb") as fh:
        n_frames_mdd = int.from_bytes(fh.read(4), "big")
    if n_frames_mdd != n_frames:
        pytest.fail(f"MDD header n_frames={n_frames_mdd}, expected {n_frames}")


@pytest.mark.bpy
def test_export_animation_blend_true_defaults_to_mdd(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_deformation=True`` resolves to MDD, not Shape Keys."""
    out = tmp_path / "default_true.blend"
    _bake_deformation_animation(
        offscreen_plotter, out, n_frames=3, bake_deformation=True
    )
    mesh_data = _first_mesh_data()
    if mesh_data is None:
        pytest.fail("reloaded .blend has no mesh data-block")
    if mesh_data.shape_keys is not None:
        pytest.fail("bake_deformation=True should default to MDD, not Shape Keys")
    mesh_obj = next(
        (obj for obj in bpy.data.objects if obj.data is mesh_data),
        None,
    )
    if mesh_obj is None or not any(m.type == "MESH_CACHE" for m in mesh_obj.modifiers):
        pytest.fail("bake_deformation=True did not add a MESH_CACHE modifier")


@pytest.mark.bpy
def test_export_animation_blend_bake_camera_false_skips_camera_keyframes(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_camera=False`` leaves the camera static."""
    out = tmp_path / "no_camera.blend"
    _bake_deformation_animation(
        offscreen_plotter,
        out,
        n_frames=4,
        bake_deformation=False,
        bake_camera=False,
    )
    scene = bpy.context.scene
    if scene is None or scene.camera is None:
        pytest.fail("reloaded .blend missing scene or camera")
    if scene.camera.animation_data is not None:
        pytest.fail(
            "bake_camera=False should leave the camera with no animation_data, "
            f"got {scene.camera.animation_data!r}"
        )


def test_export_animation_blend_rejects_unknown_deformation_mode(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """Invalid string for ``bake_deformation`` raises ValueError."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red")
    updater = pl.blender.orbit_camera(n_frames=4)
    with pytest.raises(ValueError, match=r"bake_deformation=.*invalid"):
        pl.blender.export_animation_blend(
            str(tmp_path / "bad.blend"),
            updater,
            frames=range(4),
            fps=24,
            bake_deformation="alembic",
        )


def _bake_scalar_animation(
    pl: pv.Plotter,
    out_path: Path,
    *,
    n_frames: int,
    bake_scalars: bool,
) -> None:
    """Add a scalar-coloured plane, vary its scalars per frame, export, reload.

    Used by the scalar-bake tests. ``plane["heat"]`` is mutated by
    a sine wave dependent on ``frame`` so the bridge has something
    non-constant to bake.
    """
    plane = pv.Plane(i_resolution=10, j_resolution=10)
    plane["heat"] = np.zeros(plane.n_points, dtype=np.float32)
    pl.add_mesh(
        plane,
        scalars="heat",
        cmap="viridis",
        show_scalar_bar=False,
        clim=[-1.0, 1.0],
    )
    pl.camera_position = [(2.0, -2.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def update(frame: int) -> None:
        plane["heat"] = np.sin(plane.points[:, 0] * 2.0 + frame * 0.3).astype(
            np.float32
        )

    pl.blender.export_animation_blend(
        str(out_path),
        update,
        frames=range(n_frames),
        fps=24,
        bake_camera=False,
        bake_scalars=bake_scalars,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out_path))


@pytest.mark.bpy
def test_export_animation_blend_bake_scalars_adds_png_and_nodes_modifier(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_scalars=True`` adds a Nodes modifier + packed-PNG image."""
    out = tmp_path / "heat.blend"
    _bake_scalar_animation(offscreen_plotter, out, n_frames=5, bake_scalars=True)

    mesh_obj = next(
        (obj for obj in bpy.data.objects if obj.type == "MESH"),
        None,
    )
    if mesh_obj is None:
        pytest.fail("reloaded .blend has no mesh object")
    nodes_mods = [m for m in mesh_obj.modifiers if m.type == "NODES"]
    if not nodes_mods:
        pytest.fail("bake_scalars=True did not add a NODES modifier")

    # The image is packed *into* the .blend (single source of truth);
    # the external PNG should have been cleaned up after save.
    leftover_sidecars = list(out.parent.glob(f"{out.stem}__*_scalars.png"))
    if leftover_sidecars:
        pytest.fail(
            f"external PNG sidecar should be removed after packing; got "
            f"{leftover_sidecars!r}"
        )

    scalar_images = [
        img for img in bpy.data.images if img.name.endswith("_scalars.png")
    ]
    if len(scalar_images) != 1:
        pytest.fail(
            f"expected exactly one scalar image in the .blend, got "
            f"{[img.name for img in scalar_images]}"
        )
    if scalar_images[0].packed_file is None:
        pytest.fail(
            f"scalar image {scalar_images[0].name!r} was not packed into the .blend"
        )


@pytest.mark.bpy
def test_export_animation_blend_bake_scalars_drives_color_attribute(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """Scrubbing the timeline changes the ``scalars`` color attribute."""
    out = tmp_path / "heat_drives.blend"
    _bake_scalar_animation(offscreen_plotter, out, n_frames=5, bake_scalars=True)

    scene = bpy.context.scene
    if scene is None:
        pytest.fail("reloaded .blend has no scene")
    mesh_obj = next(
        (obj for obj in bpy.data.objects if obj.type == "MESH"),
        None,
    )
    if mesh_obj is None:
        pytest.fail("reloaded .blend has no mesh object")
    depsgraph = bpy.context.evaluated_depsgraph_get()

    def colour_at_frame(frame: int) -> np.ndarray:
        scene.frame_set(frame)
        evaluated = mesh_obj.evaluated_get(depsgraph)
        # fake-bpy-module types ``Object.data`` as a wide union; narrow
        # to the concrete Mesh subtype before touching color_attributes.
        eval_mesh = cast("bpy.types.Mesh", evaluated.data)
        ca = cast(
            "bpy.types.ByteColorAttribute | bpy.types.FloatColorAttribute | None",
            eval_mesh.color_attributes.get("scalars"),
        )
        if ca is None:
            pytest.fail("evaluated mesh has no 'scalars' color attribute")
        buffer = np.zeros(len(ca.data) * 4, dtype=np.float32)
        ca.data.foreach_get("color", buffer)
        return buffer.reshape(-1, 4)

    f0 = colour_at_frame(0)
    f4 = colour_at_frame(4)
    if np.allclose(f0, f4):
        pytest.fail("scalars color attribute did not change across frames")


@pytest.mark.bpy
def test_export_animation_blend_bake_scalars_default_no_modifier(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """Default ``bake_scalars=False`` leaves the mesh modifier-free."""
    out = tmp_path / "no_heat.blend"
    _bake_scalar_animation(offscreen_plotter, out, n_frames=4, bake_scalars=False)

    mesh_obj = next(
        (obj for obj in bpy.data.objects if obj.type == "MESH"),
        None,
    )
    if mesh_obj is None:
        pytest.fail("reloaded .blend has no mesh object")
    nodes_mods = [m for m in mesh_obj.modifiers if m.type == "NODES"]
    if nodes_mods:
        pytest.fail(
            f"bake_scalars=False should not add any NODES modifier; got "
            f"{[m.name for m in nodes_mods]}"
        )
    sidecars = list(out.parent.glob(f"{out.stem}__*_scalars.png"))
    if sidecars:
        pytest.fail(
            f"bake_scalars=False should not produce a PNG sidecar; got {sidecars!r}"
        )


@pytest.mark.bpy
def test_export_animation_blend_bake_lights_keyframes_intensity(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_lights=True`` keyframes a moving / dimming light."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red")
    pl.camera_position = [(3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    light = pv.Light(position=(5, 0, 0), light_type="scene light", intensity=1.0)
    pl.add_light(light)

    def update(frame: int) -> None:
        light.position = (5.0 * np.cos(frame * 0.5), 5.0 * np.sin(frame * 0.5), 3.0)
        light.intensity = 0.5 + 0.5 * np.sin(frame * 0.3)

    out = tmp_path / "lights.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(6),
        fps=24,
        bake_camera=False,
        bake_lights=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    # The vtkLightKit defaults populate PVLight_0 ... 4; our explicit light
    # lands at PVLight_5 and is the only one with animation_data.
    animated_lights = [
        obj
        for obj in bpy.data.objects
        if obj.type == "LIGHT" and obj.animation_data is not None
    ]
    if len(animated_lights) != 1:
        pytest.fail(
            f"expected exactly one animated light, got "
            f"{[o.name for o in animated_lights]}"
        )
    light_obj = animated_lights[0]
    obj_anim = light_obj.animation_data
    if obj_anim is None or obj_anim.action is None:
        pytest.fail("animated light has no action despite animation_data set")
    obj_fcurves = _action_fcurves(obj_anim.action)
    if not any(
        fc.data_path == "location" or fc.data_path.endswith("].location")
        for fc in obj_fcurves
    ):
        pytest.fail(
            f"animated light has no location fcurve; got "
            f"{[fc.data_path for fc in obj_fcurves]}"
        )

    light_data = cast("bpy.types.Light", light_obj.data)
    data_anim = light_data.animation_data
    if data_anim is None or data_anim.action is None:
        pytest.fail("light data-block has no animation action — energy not keyframed")
    data_fcurves = _action_fcurves(data_anim.action)
    energy_curves = [
        fc
        for fc in data_fcurves
        if fc.data_path == "energy" or fc.data_path.endswith("].energy")
    ]
    if not energy_curves:
        pytest.fail(
            f"animated light has no energy fcurve; got "
            f"{[fc.data_path for fc in data_fcurves]}"
        )


@pytest.mark.bpy
def test_export_animation_blend_bake_lights_default_no_animation(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """Default ``bake_lights=False`` leaves lights without animation_data."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red")
    pl.camera_position = [(3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    light = pv.Light(position=(5, 0, 0), light_type="scene light", intensity=1.0)
    pl.add_light(light)

    def update(frame: int) -> None:
        light.intensity = 0.5 + 0.5 * np.sin(frame * 0.3)

    out = tmp_path / "no_light_anim.blend"
    pl.blender.export_animation_blend(
        str(out), update, frames=range(4), fps=24, bake_camera=False
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    animated_lights = [
        obj
        for obj in bpy.data.objects
        if obj.type == "LIGHT" and obj.animation_data is not None
    ]
    if animated_lights:
        pytest.fail(
            f"bake_lights=False (default) should leave lights static; got "
            f"{[o.name for o in animated_lights]}"
        )


@pytest.mark.bpy
def test_export_animation_blend_bake_scalars_handles_cell_data(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """cell-data scalars route through a FACE-domain GN store."""
    pl = offscreen_plotter
    plane = pv.Plane(i_resolution=10, j_resolution=10)
    plane.cell_data["heat"] = np.zeros(plane.n_cells, dtype=np.float32)
    pl.add_mesh(
        plane,
        scalars="heat",
        cmap="viridis",
        show_scalar_bar=False,
        clim=[-1.0, 1.0],
    )
    pl.camera_position = [(2.0, -2.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def update(frame: int) -> None:
        plane.cell_data["heat"] = np.sin(
            np.arange(plane.n_cells) * 0.2 + frame * 0.3
        ).astype(np.float32)

    out = tmp_path / "cell_scalars.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(5),
        fps=24,
        bake_camera=False,
        bake_scalars=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    mesh_obj = next(
        (obj for obj in bpy.data.objects if obj.type == "MESH"),
        None,
    )
    if mesh_obj is None:
        pytest.fail("reloaded .blend has no mesh object")
    _assert_face_domain_scalar_modifier(mesh_obj)
    scene = bpy.context.scene
    if scene is None:
        pytest.fail("reloaded .blend has no active scene")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    f0 = _sample_face_scalars(mesh_obj, scene, depsgraph, frame=0)
    f4 = _sample_face_scalars(mesh_obj, scene, depsgraph, frame=4)
    if np.allclose(f0, f4):
        pytest.fail("cell-data scalars did not vary across frames")


@pytest.mark.bpy
def test_user_matrix_translates_to_bpy_matrix_world(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``actor.user_matrix`` propagates to ``bpy_obj.matrix_world``."""
    pl = offscreen_plotter
    sphere = pv.Sphere()
    actor = pl.add_mesh(sphere, color="red")
    user_matrix = np.eye(4)
    user_matrix[0, 3] = 2.0
    user_matrix[2, 3] = 1.5
    actor.user_matrix = user_matrix

    pl.camera_position = [(6.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    pl.blender.render(str(tmp_path / "user_matrix.png"), samples=4)

    mesh_data = _first_mesh_data()
    if mesh_data is None:
        pytest.fail("static render did not produce a mesh data-block")
    mesh_obj = next(
        (obj for obj in bpy.data.objects if obj.data is mesh_data),
        None,
    )
    if mesh_obj is None:
        pytest.fail("could not find object wrapping the mesh data-block")
    bpy_matrix = np.array(mesh_obj.matrix_world)
    if not np.allclose(bpy_matrix, user_matrix, atol=1e-6):
        pytest.fail(
            f"bpy obj.matrix_world\n{bpy_matrix}\n"
            f"does not match actor.user_matrix\n{user_matrix}"
        )


def _extract_left_right_bar_bands(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the per-tile scalar-bar pixel rect out of a 2-tile composite.

    PyVista's default scalar-bar viewport is
    ``(x0=0.35, y0=0.05, width=0.3, height=0.05)``. In PIL coords
    (y-down) that's a thin band at the bottom of each tile, away from
    the rendered objects. This helper returns the band for the left
    and right tile so callers can compare colour content.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(left_band, right_band)`` each shape
        ``(band_height, band_width, 3)``.

    """
    height_total, width_total = arr.shape[:2]
    half = width_total // 2
    y_top = round((1.0 - 0.10) * height_total)
    y_bot = round((1.0 - 0.05) * height_total)
    x_lo = round(0.35 * half)
    x_hi = round(0.65 * half)
    return (
        arr[y_top:y_bot, x_lo:x_hi],
        arr[y_top:y_bot, half + x_lo : half + x_hi],
    )


_Z_AXIS_RGB = (52, 152, 219)  # teal — matches axes.py's _AXIS_COLORS[2]


def _count_z_axis_pixels(rgba: np.ndarray) -> int:
    """Count opaque teal-Z-axis pixels in ``rgba`` (broad colour match).

    The match window is wide so anti-aliased edges still register.

    Returns
    -------
    int
        Number of pixels in ``rgba`` whose colour is within ``tolerance``
        of the canonical Z-axis teal and whose alpha exceeds the opaque
        threshold.

    """
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    r, g, b, a = (rgba_uint8[..., i] for i in range(4))
    target_r, target_g, target_b = _Z_AXIS_RGB
    tolerance = 60
    opaque_threshold = 200
    return int(
        (
            (a > opaque_threshold)
            & (np.abs(r.astype(int) - target_r) < tolerance)
            & (np.abs(g.astype(int) - target_g) < tolerance)
            & (np.abs(b.astype(int) - target_b) < tolerance)
        ).sum()
    )


def test_axes_overlay_renders_collapsed_z_axis_as_glyph(
    offscreen_plotter: pv.Plotter,
) -> None:
    """Regression: top-down view collapses Z → glyph, not a degenerate arrow."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere())
    # Force a top-down camera so the Z axis projects to ~zero screen-length.
    pl.camera_position = [(0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    cast("Callable[[], None]", pl.show_axes)()

    width, height = 400, 400
    rgba = pl.blender.render_hud_overlay("axes", width=width, height=height)
    if rgba is None:
        pytest.fail("axes overlay returned None despite show_axes() being called")

    # Axes widget viewport is the lower-left 20%; in PIL coords (y-down)
    # that's the bottom-left 20% rect. Sample the band around the origin
    # and look for teal Z-axis pixels (broad match against _Z_AXIS_RGB).
    band_y0, band_y1 = int(height * 0.85), int(height * 0.95)
    band_x0, band_x1 = int(width * 0.05), int(width * 0.15)
    band = rgba[band_y0:band_y1, band_x0:band_x1]
    z_pixels = _count_z_axis_pixels(band)
    min_z_pixels = 10
    if z_pixels < min_z_pixels:
        pytest.fail(
            f"top-down view: Z-axis glyph missing in lower-left band "
            f"({z_pixels} teal pixels < {min_z_pixels} threshold)"
        )


@pytest.mark.bpy
def test_subplot_hud_camera_matches_rendered_pose(tmp_path: Path) -> None:
    """HUD axes triad reads the same camera Cycles rendered with.

    Regression for the ``pyvista.Plotter.camera`` lazy-reset bug:
    accessing ``plotter.camera`` for the first time on a renderer
    whose ``camera.is_set`` is False resets the pose to the iso
    default. The bridge sets ``is_set`` after its own
    ``reset_camera()`` so the HUD and Cycles see identical state.
    """
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[600, 300])
    pl.subplot(0, 0)
    pl.add_mesh(pv.examples.load_random_hills(), cmap="terrain")
    # pyvista decorates ``show_axes`` with ``functools.wraps`` in a way
    # that confuses ty's stub resolver. The runtime call is fine.
    cast("Callable[[], None]", pl.show_axes)()
    pl.subplot(0, 1)
    pl.add_mesh(pv.Sphere())

    pl.blender.render(str(tmp_path / "subplot_camera_match.png"), samples=4)

    # After render(), the left renderer's camera should still reflect
    # the bridge-configured pose, NOT the iso default the lazy getter
    # would have introduced. ``load_random_hills`` is X-Y dominant
    # (Z << X, Y), so ``reset_camera`` produces a top-down view where
    # the camera sits well above the focal point on the Z axis. Iso
    # would put X, Y, and Z all comparable.
    cam = pl.renderers[0].camera
    pos = cam.position
    if not pl.renderers[0].camera.is_set:
        pytest.fail("bridge did not pin renderer.camera.is_set")
    # Top-down sanity: |z| dominates |x|, |y| by an order of magnitude.
    threshold = 5.0
    if abs(pos[2]) < threshold * max(abs(pos[0]), abs(pos[1]), 1.0):
        pytest.fail(
            f"renderer[0] camera position {pos} doesn't look top-down; "
            f"lazy-reset may have introduced an iso pose"
        )
    pl.close()


@pytest.mark.bpy
def test_subplot_scalar_bars_filtered_per_tile(tmp_path: Path) -> None:
    """Each tile draws only the scalar bars that belong to its renderer.

    Left tile gets a viridis bar; right tile gets an inferno bar. The
    bar overlay is rendered at tile resolution; we slice the composite
    image to each tile's pixel rect and verify both bars are present
    (each in the right tile) without cross-contamination.
    """
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[800, 400])
    pl.subplot(0, 0)
    m1 = pv.Sphere()
    m1["height"] = m1.points[:, 2]
    pl.add_mesh(
        m1,
        scalars="height",
        cmap="viridis",
        show_scalar_bar=True,
        scalar_bar_args={"title": "Sphere height", "color": "white"},
    )
    pl.subplot(0, 1)
    m2 = pv.Cube()
    m2.cell_data["heat"] = [1, 2, 3, 4, 5, 6]
    pl.add_mesh(
        m2,
        scalars="heat",
        cmap="inferno",
        show_scalar_bar=True,
        scalar_bar_args={"title": "Cube heat", "color": "white"},
    )

    out = tmp_path / "per_tile_bars.png"
    pl.blender.render(str(out), samples=4)
    pl.close()

    with Image.open(out) as img:
        arr = np.asarray(img.convert("RGB"))

    left_bar, right_bar = _extract_left_right_bar_bands(arr)

    # If the filter is broken (every bar drawn on every tile) the two
    # bands would contain identical pixels (or the same two stacked
    # bars). With the filter working, left has the viridis gradient
    # and right has the inferno gradient — vastly different colour
    # profiles. The cheapest robust check is the mean per-channel diff.
    mean_diff = float(
        np.abs(left_bar.mean(axis=(0, 1)) - right_bar.mean(axis=(0, 1))).sum()
    )
    min_channel_diff = 40.0  # viridis vs inferno mean differs by ~80 in tests
    if mean_diff < min_channel_diff:
        pytest.fail(
            f"left/right bar mean colour diff {mean_diff:.1f} below threshold "
            f"{min_channel_diff}; per-tile scalar-bar filter likely not in effect"
        )


@pytest.mark.bpy
def test_subplot_renders_text_in_each_tile(tmp_path: Path) -> None:
    """per-tile HUD compositing puts the right text over each viewport."""
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[600, 300])
    pl.subplot(0, 0)
    pl.add_mesh(pv.Sphere(), color="red")
    pl.add_text("Left", position="upper_edge", font_size=14, color="white")
    pl.subplot(0, 1)
    pl.add_mesh(pv.Cube(), color="blue")
    pl.add_text("Right", position="upper_edge", font_size=14, color="white")

    out = tmp_path / "per_tile_hud.png"
    pl.blender.render(str(out), samples=4)
    pl.close()

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("subplot render produced no output")
    with Image.open(out) as img:
        arr = np.asarray(img.convert("RGB"))

    # Both top edges should carry near-white pixels from the rendered
    # text overlays. Threshold is loose because Cycles can introduce
    # noise underneath the text — we just want "the white text is
    # present in both tiles", not pixel-exact equality.
    expected_brightness = 200
    half = arr.shape[1] // 2
    if arr[5, :half].mean() < expected_brightness:
        pytest.fail(
            f"left tile top row not bright enough ({arr[5, :half].mean():.1f}); "
            f"per-tile HUD text overlay missing"
        )
    if arr[5, half:].mean() < expected_brightness:
        pytest.fail(
            f"right tile top row not bright enough ({arr[5, half:].mean():.1f}); "
            f"per-tile HUD text overlay missing"
        )


@pytest.mark.bpy
def test_export_animation_blend_bake_transforms_orbits_actor(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_transforms=True`` keyframes a moving actor's user_matrix."""
    pl = offscreen_plotter
    sphere = pv.Sphere(radius=0.5)
    actor = pl.add_mesh(sphere, color="red")
    pl.camera_position = [(6.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def update(frame: int) -> None:
        angle = frame * 0.4
        m = np.eye(4)
        m[0, 3] = 2.0 * np.cos(angle)
        m[1, 3] = 2.0 * np.sin(angle)
        actor.user_matrix = m

    out = tmp_path / "transforms.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(6),
        fps=24,
        bake_camera=False,
        bake_transforms=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    scene = bpy.context.scene
    if scene is None:
        pytest.fail("reloaded .blend has no scene")
    mesh_obj = next((obj for obj in bpy.data.objects if obj.type == "MESH"), None)
    if mesh_obj is None:
        pytest.fail("reloaded .blend has no mesh object")
    anim = mesh_obj.animation_data
    if anim is None or anim.action is None:
        pytest.fail("transform-baked actor has no animation_data.action")

    fcurves = _action_fcurves(anim.action)
    location_curves = [fc for fc in fcurves if fc.data_path == "location"]
    expected_axes = 3
    if len(location_curves) != expected_axes:
        pytest.fail(
            f"expected {expected_axes} location fcurves, got {len(location_curves)}"
        )

    # Sanity: evaluate position at frame 0 vs frame 4; the actor orbits,
    # so the X channel must differ significantly between frames.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene.frame_set(0)
    x0 = mesh_obj.evaluated_get(depsgraph).location[0]
    scene.frame_set(4)
    x4 = mesh_obj.evaluated_get(depsgraph).location[0]
    if abs(x0 - x4) < 1.0:
        pytest.fail(f"transform-baked actor barely moved: x(0)={x0:.3f} x(4)={x4:.3f}")


@pytest.mark.bpy
def test_export_animation_blend_bake_transforms_skips_static_actor(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A static ``user_matrix`` produces no animation_data on the actor."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red")
    pl.camera_position = [(3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def update(_frame: int) -> None:
        # Intentionally no actor mutation — user_matrix stays identity.
        return

    out = tmp_path / "no_transform_anim.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(4),
        fps=24,
        bake_camera=False,
        bake_transforms=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))
    mesh_obj = next((obj for obj in bpy.data.objects if obj.type == "MESH"), None)
    if mesh_obj is None:
        pytest.fail("reloaded .blend has no mesh object")
    if mesh_obj.animation_data is not None:
        pytest.fail(
            "static actor should have no animation_data; got "
            f"{mesh_obj.animation_data!r}"
        )


@pytest.mark.bpy
def test_export_animation_blend_bake_materials_keyframes_bsdf(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_materials=True`` keyframes varying BSDF inputs."""
    pl = offscreen_plotter
    actor = pl.add_mesh(pv.Sphere(), color="red", pbr=True, metallic=0.0, roughness=0.5)
    pl.camera_position = [(3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def update(frame: int) -> None:
        actor.prop.roughness = 0.1 + 0.4 * abs(np.sin(frame * 0.5))
        actor.prop.metallic = abs(np.sin(frame * 0.5))

    out = tmp_path / "materials.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(6),
        fps=24,
        bake_camera=False,
        bake_materials=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    mats_with_actions = [
        m
        for m in bpy.data.materials
        if m.node_tree is not None and m.node_tree.animation_data is not None
    ]
    if not mats_with_actions:
        pytest.fail("no material has animation_data after bake_materials=True")

    mat = mats_with_actions[0]
    node_tree = mat.node_tree
    if node_tree is None:
        pytest.fail("material has no node_tree (filter above already excludes this)")
    anim = node_tree.animation_data
    if anim is None or anim.action is None:
        pytest.fail("material node_tree has no action")
    fcurves = _action_fcurves(anim.action)
    if not fcurves:
        pytest.fail(f"action {anim.action.name!r} has no fcurves")
    # Each keyframed input should carry exactly 6 keys (one per frame).
    expected_keys = 6
    for fc in fcurves:
        if len(fc.keyframe_points) != expected_keys:
            pytest.fail(
                f"fcurve {fc.data_path}[{fc.array_index}] has "
                f"{len(fc.keyframe_points)} keys, expected {expected_keys}"
            )


@pytest.mark.bpy
def test_export_animation_blend_bake_materials_skips_static(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A static material produces no node_tree animation_data."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red", pbr=True, roughness=0.4)
    pl.camera_position = [(3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def update(_frame: int) -> None:
        return  # no material mutation

    out = tmp_path / "static_material.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(4),
        fps=24,
        bake_camera=False,
        bake_materials=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        if mat.node_tree.animation_data is not None:
            pytest.fail(
                f"static material {mat.name!r} should have no animation_data; "
                f"got {mat.node_tree.animation_data!r}"
            )


@pytest.mark.bpy
def test_render_tessellates_quadratic_cells(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A VTK_QUADRATIC_TRIANGLE actor is tessellated before extract_surface.

    Without the pre-pass, ``extract_surface()`` would linearise the
    curved face into a single flat triangle (3 verts, 1 face). With
    ``config.tessellation_subdivide=3`` the bridge runs
    :meth:`pv.DataSet.tessellate` first and the bpy mesh ends up with
    far more vertices than the source's 6 high-order nodes.
    """
    import vtkmodules.all as vtk  # noqa: PLC0415

    pl = offscreen_plotter
    corners = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    midedges = np.array(
        [[0.5, 0.1, 0.3], [0.75, 0.5, 0.3], [0.25, 0.5, 0.3]], dtype=np.float32
    )
    points = np.vstack([corners, midedges])
    quad_tri = pv.UnstructuredGrid(
        [6, 0, 1, 2, 3, 4, 5],
        [vtk.VTK_QUADRATIC_TRIANGLE],
        points,
    )
    pl.add_mesh(quad_tri, color="orange")
    pl.camera_position = [(2.0, 2.0, 2.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "quadratic.png"
    pl.blender.render(str(out), samples=4)
    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("quadratic-cell render produced no output")

    # The bpy mesh should carry the tessellated geometry — many more
    # vertices than the 6-node high-order cell could have produced
    # via plain extract_surface (which would have linearised to 3).
    meshes = [m for m in bpy.data.meshes if m.name.startswith("UnstructuredGrid")]
    if not meshes:
        pytest.fail(
            f"no UnstructuredGrid mesh data-block; got: "
            f"{[m.name for m in bpy.data.meshes]}"
        )
    n_verts = len(meshes[0].vertices)
    min_tess_verts = 10  # tessellated curve has many; linearised would be 3
    if n_verts < min_tess_verts:
        pytest.fail(
            f"quadratic cell was not tessellated: bpy mesh has {n_verts} "
            f"vertices (expected >= {min_tess_verts} from "
            f"tessellate(max_n_subdivide=3))"
        )


@pytest.mark.bpy
def test_render_tessellate_disabled_keeps_linearised_surface(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``config.tessellation_subdivide=0`` skips the pre-pass entirely.

    With the knob set to zero, ``extract_surface()`` runs on the raw
    high-order cell and produces the linearised single triangle.
    """
    import vtkmodules.all as vtk  # noqa: PLC0415

    from pyvista_blender import config  # noqa: PLC0415

    pl = offscreen_plotter
    corners = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    midedges = np.array(
        [[0.5, 0.1, 0.3], [0.75, 0.5, 0.3], [0.25, 0.5, 0.3]], dtype=np.float32
    )
    points = np.vstack([corners, midedges])
    quad_tri = pv.UnstructuredGrid(
        [6, 0, 1, 2, 3, 4, 5],
        [vtk.VTK_QUADRATIC_TRIANGLE],
        points,
    )
    pl.add_mesh(quad_tri, color="orange")
    pl.camera_position = [(2.0, 2.0, 2.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)]

    original_subdivide = config.tessellation_subdivide
    config.tessellation_subdivide = 0
    try:
        out = tmp_path / "linearised.png"
        pl.blender.render(str(out), samples=4)
    finally:
        config.tessellation_subdivide = original_subdivide

    meshes = [m for m in bpy.data.meshes if m.name.startswith("UnstructuredGrid")]
    if not meshes:
        pytest.fail("no UnstructuredGrid mesh data-block")
    n_verts = len(meshes[0].vertices)
    # extract_surface() on a single quadratic triangle leaves the 6
    # cell nodes intact (corners + mid-edges) but produces only linear
    # tris — nothing close to the tessellated >10-vert refinement.
    max_linear_verts = 8
    if n_verts > max_linear_verts:
        pytest.fail(
            f"tessellation_subdivide=0 should leave the linearised "
            f"single cell (~6 verts); got {n_verts}"
        )


@pytest.mark.bpy
def test_render_points_style_uses_point_cloud_data(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``style="points"`` produces a bpy ``PointCloud`` (not a Mesh).

    The bridge dispatches points-styled actors to the point cloud
    translator. The expected artifacts after a render: a
    ``bpy.data.pointclouds`` data-block whose point count matches the
    source dataset, and a ``*_points_mat`` material on it.
    """
    pl = offscreen_plotter
    n_points = 500
    rng = np.random.default_rng(seed=42)
    points = rng.uniform(-1.0, 1.0, size=(n_points, 3)).astype(np.float32)
    cloud = pv.PolyData(points)
    cloud["height"] = points[:, 2]
    pl.add_mesh(cloud, style="points", scalars="height", cmap="viridis", point_size=8)
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "points_style.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("style='points' render produced no output")

    pcs = list(bpy.data.pointclouds)
    if not pcs:
        pytest.fail("style='points' did not produce a PointCloud data-block")
    if len(pcs[0].points) != n_points:
        pytest.fail(f"PointCloud has {len(pcs[0].points)} points, expected {n_points}")

    point_mats = [m for m in bpy.data.materials if m.name.endswith("_points_mat")]
    if not point_mats:
        pytest.fail("style='points' did not produce a *_points_mat material")
    # No mesh data-block should have leaked through for the point actor.
    point_meshes = [
        m for m in bpy.data.meshes if "PolyData" in m.name and "vol" not in m.name
    ]
    if point_meshes:
        pytest.fail(
            f"style='points' actor leaked into bpy.data.meshes: "
            f"{[m.name for m in point_meshes]}"
        )


@pytest.mark.bpy
def test_render_points_gaussian_style_uses_splat_material(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``style="points_gaussian"`` builds the appropriate splat shader.

    The bridge dispatches PointGaussianMapper actors to the same
    point-cloud translator, but with the gaussian shader graph instead
    of the opaque Principled BSDF. Defaults to a Principled BSDF
    foreground (PyVista's ``emissive=False`` default); see the
    emissive variant below for the additive-emission path.
    """
    pl = offscreen_plotter
    rng = np.random.default_rng(seed=7)
    points = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
    cloud = pv.PolyData(points)
    pl.add_mesh(cloud, style="points_gaussian", color="orange", point_size=12)
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "points_gaussian.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("style='points_gaussian' render produced no output")

    gauss_mats = [m for m in bpy.data.materials if m.name.endswith("_gauss_mat")]
    if not gauss_mats:
        pytest.fail(
            "style='points_gaussian' did not produce a *_gauss_mat material; "
            f"materials: {[m.name for m in bpy.data.materials]}"
        )
    mat = gauss_mats[0]
    if mat.node_tree is None:
        pytest.fail("gaussian splat material has no node_tree")
    node_types = {n.bl_idname for n in mat.node_tree.nodes}
    # Default emissive=False path: Transparent + Principled BSDF mix.
    expected = {
        "ShaderNodeBsdfTransparent",
        "ShaderNodeBsdfPrincipled",
        "ShaderNodeMixShader",
    }
    missing = expected - node_types
    if missing:
        pytest.fail(
            f"gaussian splat shader missing required nodes: {sorted(missing)}; "
            f"present: {sorted(node_types)}"
        )
    if "ShaderNodeEmission" in node_types:
        pytest.fail(
            "default emissive=False path should not build an Emission node; "
            f"present: {sorted(node_types)}"
        )


@pytest.mark.bpy
def test_render_points_style_render_as_spheres_uses_principled_bsdf(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``prop.render_points_as_spheres=True`` switches to a PBR sphere.

    Mirrors VTK's flag of the same name. The default (False) builds
    flat Emission dots; setting it to True builds a Principled BSDF
    so the dots shade like tiny lit spheres.
    """
    pl = offscreen_plotter
    rng = np.random.default_rng(seed=13)
    points = rng.uniform(-1.0, 1.0, size=(150, 3)).astype(np.float32)
    cloud = pv.PolyData(points)
    actor = pl.add_mesh(cloud, style="points", color="cyan", point_size=10)
    actor.prop.render_points_as_spheres = True
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "points_as_spheres.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("points-as-spheres render produced no output")

    point_mats = [m for m in bpy.data.materials if m.name.endswith("_points_mat")]
    mat = point_mats[0]
    if mat.node_tree is None:
        pytest.fail("points material has no node_tree")
    node_types = {n.bl_idname for n in mat.node_tree.nodes}
    if "ShaderNodeBsdfPrincipled" not in node_types:
        pytest.fail(
            "render_points_as_spheres=True should build a Principled BSDF; "
            f"present: {sorted(node_types)}"
        )
    if "ShaderNodeEmission" in node_types:
        pytest.fail(
            "render_points_as_spheres=True should not also keep an Emission "
            f"node; present: {sorted(node_types)}"
        )


@pytest.mark.bpy
def test_render_points_gaussian_render_as_spheres_drops_transparent_mix(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``render_points_as_spheres=True`` makes gaussian splats opaque.

    PyVista's GL output with this flag draws each splat as a crisp PBR
    sphere (no alpha falloff); the bridge mirrors that by dropping the
    Transparent + Mix path and feeding the Principled BSDF straight
    into the material output.
    """
    pl = offscreen_plotter
    rng = np.random.default_rng(seed=21)
    points = rng.uniform(-1.0, 1.0, size=(150, 3)).astype(np.float32)
    cloud = pv.PolyData(points)
    actor = pl.add_mesh(
        cloud,
        style="points_gaussian",
        color="orange",
        point_size=12,
        emissive=False,
        render_points_as_spheres=True,
    )
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "gauss_spheres.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("gaussian-as-spheres render produced no output")

    gauss_mats = [m for m in bpy.data.materials if m.name.endswith("_gauss_mat")]
    mat = gauss_mats[0]
    if mat.node_tree is None:
        pytest.fail("gaussian material has no node_tree")
    node_types = {n.bl_idname for n in mat.node_tree.nodes}
    if "ShaderNodeBsdfPrincipled" not in node_types:
        pytest.fail(
            "render_points_as_spheres=True should keep the Principled BSDF; "
            f"present: {sorted(node_types)}"
        )
    if "ShaderNodeBsdfTransparent" in node_types or "ShaderNodeMixShader" in node_types:
        pytest.fail(
            "render_points_as_spheres=True should drop the soft-splat "
            f"Transparent + Mix branch; present: {sorted(node_types)}"
        )
    # PyVista's ``render_points_as_spheres=True`` for gaussian doesn't
    # set ``prop.render_points_as_spheres`` — it calls
    # ``mapper.use_circular_splat`` instead, which the bridge detects
    # via ``mapper.GetSplatShaderCode() != None``. Verify the actor
    # actually ended up in that state so the test catches a future
    # pyvista API change that breaks the signal.
    splat_code = getattr(actor.mapper, "GetSplatShaderCode", lambda: None)()
    if splat_code is None and not actor.prop.render_points_as_spheres:
        pytest.fail(
            "neither mapper.GetSplatShaderCode nor prop.render_points_as_spheres "
            "reflects the hard-sphere intent; pyvista API may have changed"
        )


@pytest.mark.bpy
def test_render_points_gaussian_emissive_uses_emission_shader(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``mapper.emissive=True`` switches the foreground to Emission.

    Honours VTK's :class:`PointGaussianMapper` emissive flag — when set,
    the splats become self-lit light blobs (additive look against dark
    backgrounds) instead of the default scene-lit translucent spheres.
    """
    pl = offscreen_plotter
    rng = np.random.default_rng(seed=9)
    points = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
    cloud = pv.PolyData(points)
    actor = pl.add_mesh(cloud, style="points_gaussian", color="white", point_size=12)
    actor.mapper.emissive = True
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "points_gaussian_emissive.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("emissive gaussian render produced no output")

    gauss_mats = [m for m in bpy.data.materials if m.name.endswith("_gauss_mat")]
    mat = gauss_mats[0]
    if mat.node_tree is None:
        pytest.fail("emissive gaussian material has no node_tree")
    node_types = {n.bl_idname for n in mat.node_tree.nodes}
    if "ShaderNodeEmission" not in node_types:
        pytest.fail(
            "emissive=True should build an Emission foreground; "
            f"present: {sorted(node_types)}"
        )


@pytest.mark.bpy
def test_render_points_gaussian_honors_mapper_scale_array(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``mapper.scale_array`` drives the per-point radius attribute.

    PyVista's :class:`PointGaussianMapper` exposes a ``scale_array``
    that VTK uses to size each splat. The bridge mirrors that contract
    by multiplying the base radius by the per-point value, so the
    resulting PointCloud carries varying radii.
    """
    pl = offscreen_plotter
    rng = np.random.default_rng(seed=11)
    points = rng.uniform(-1.0, 1.0, size=(300, 3)).astype(np.float32)
    cloud = pv.PolyData(points)
    cloud["r"] = np.linalg.norm(points, axis=1).astype(np.float32)
    actor = pl.add_mesh(
        cloud, style="points_gaussian", scalars="r", cmap="viridis", point_size=10
    )
    actor.mapper.scale_array = "r"
    pl.camera_position = [(3.0, 3.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "scale_array.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("scale_array render produced no output")

    pcs = list(bpy.data.pointclouds)
    if not pcs:
        pytest.fail("scale_array path did not produce a PointCloud")
    radius_attr = cast(
        "bpy.types.FloatAttribute | None", pcs[0].attributes.get("radius")
    )
    if radius_attr is None:
        pytest.fail("PointCloud is missing the 'radius' attribute")
    radii = np.zeros(len(pcs[0].points), dtype=np.float32)
    radius_attr.data.foreach_get("value", radii)
    min_radius_variation = 1e-6
    if radii.std() < min_radius_variation:
        pytest.fail(
            f"scale_array did not introduce per-point radius variation "
            f"(std={radii.std():.2e}); radii: min={radii.min():.4f} "
            f"max={radii.max():.4f}"
        )


@pytest.mark.bpy
def test_render_points_style_falls_back_to_flat_color(
    offscreen_plotter: pv.Plotter,
    tmp_path: Path,
) -> None:
    """A points-style actor without scalars renders with the prop colour.

    No ``scalars`` color attribute should appear; the BSDF's Base Color
    is set directly from ``prop.color``.
    """
    pl = offscreen_plotter
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float32
    )
    pl.add_mesh(pv.PolyData(points), style="points", color="red", point_size=10)
    pl.camera_position = [(2.0, 2.0, 2.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "flat_points.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("flat-colour points render produced no output")
    pcs = list(bpy.data.pointclouds)
    if not pcs:
        pytest.fail("flat-colour points style did not produce a PointCloud")
    scalar_attr = pcs[0].color_attributes.get("scalars")
    if scalar_attr is not None:
        pytest.fail("flat-colour points actor should not carry a 'scalars' color attr")


@pytest.mark.bpy
def test_render_volume_writes_png(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``pl.add_volume(...)`` renders via cube + Volume shader (no VDB)."""
    pl = offscreen_plotter
    grid = pv.ImageData(dimensions=(20, 20, 20), spacing=(0.1, 0.1, 0.1))
    x, y, z = grid.points.T
    grid["density"] = (
        np.sin(3 * x) * np.cos(3 * y) * np.sin(3 * z) * 0.5 + 0.5
    ).astype(np.float32)
    pl.add_volume(
        grid,
        scalars="density",
        cmap="inferno",
        opacity="linear",
        show_scalar_bar=False,
    )
    pl.camera_position = [(4.0, -3.0, 3.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "volume.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("volume render produced no output")

    # The bridge built a closed cube mesh + Volume Principled material;
    # the material's packed atlas image lives in bpy.data.images.
    vol_mats = [m for m in bpy.data.materials if m.name.endswith("_vol_mat")]
    if not vol_mats:
        pytest.fail("volume render did not produce a *_vol_mat material")
    atlas_images = [
        img for img in bpy.data.images if img.name.startswith("pvblender_volume_")
    ]
    if not atlas_images:
        pytest.fail("volume render did not produce a packed atlas image")
    if atlas_images[0].packed_file is None:
        pytest.fail("volume atlas image is not packed into the .blend")


def _has_volume_artifacts() -> bool:
    """Return True when bpy carries a ``*_vol_mat`` material and a packed atlas.

    Returns
    -------
    bool
        ``True`` once the volume translator has produced its material
        and packed atlas image in the active bpy session.

    """
    vol_mats = [m for m in bpy.data.materials if m.name.endswith("_vol_mat")]
    atlas_images = [
        img for img in bpy.data.images if img.name.startswith("pvblender_volume_")
    ]
    if not vol_mats or not atlas_images:
        return False
    return atlas_images[0].packed_file is not None


def test_pl_blender_add_volume_registers_live_dataset(
    offscreen_plotter: pv.Plotter,
) -> None:
    """``pl.blender.add_volume`` pins the original dataset on the component.

    The bridge's volume registry should now map the actor's identity
    to the user's grid (not pyvista's internal copy), so mutations
    on the original grid propagate through render and animation paths.
    """
    pl = offscreen_plotter
    grid = pv.ImageData(dimensions=(10, 10, 10), spacing=(0.1, 0.1, 0.1))
    grid["density"] = np.zeros(grid.n_points, dtype=np.float32)
    actor = pl.blender.add_volume(grid, scalars="density", cmap="viridis")

    # Mutate the user's grid; pyvista internally copies on add_volume,
    # so the actor's mapper dataset will NOT see this. The registered
    # source on the component WILL see this — that's the whole point.
    grid["density"] = np.ones(grid.n_points, dtype=np.float32)

    sources: dict[str, object] = pl.blender.volume_sources
    if not sources:
        pytest.fail("pl.blender.add_volume did not populate the live-dataset registry")
    actor_keys = list(sources.keys())
    if len(actor_keys) != 1:
        pytest.fail(
            f"expected 1 registered volume source, got {len(actor_keys)}: {actor_keys}"
        )
    registered = sources[actor_keys[0]]
    if registered is not grid:
        pytest.fail(
            f"registered source is {registered!r}, expected the user's grid "
            f"({grid!r}) — copies break the per-frame mutation contract"
        )
    # Sanity check: pyvista did copy internally, so the actor's mapper
    # dataset diverges from the user's grid — confirming the registry
    # is the only path that surfaces the latest scalars.
    if actor.mapper.dataset is grid:
        pytest.fail(
            "pyvista.add_volume no longer copies the input — registry "
            "redundant; revisit the ergonomic fix"
        )


@pytest.mark.bpy
def test_pl_blender_add_volume_drives_animation_without_actor_indirection(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """per-frame mutations of the original grid drive the animation bake.

    The user mutates ``grid[scalars]`` directly (the natural API)
    instead of routing through ``actor.mapper.dataset``. ``bake_volume``
    should still produce a frame-indexed Value node and a multi-frame
    atlas because the bridge reads from the registered live dataset.
    """
    pl = offscreen_plotter
    grid = pv.ImageData(dimensions=(16, 16, 16), spacing=(0.1, 0.1, 0.1))
    grid["density"] = np.zeros(grid.n_points, dtype=np.float32)
    pl.blender.add_volume(grid, scalars="density", cmap="inferno", opacity="linear")
    pl.camera_position = [(4.0, -3.0, 3.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)]

    n_frames = 3

    def update(frame: int) -> None:
        # Natural pattern: mutate the user's own grid.
        x, y, z = grid.points.T
        grid["density"] = (
            np.sin(2.0 * x + 0.3 * frame) * np.cos(2.0 * y) * np.sin(2.0 * z) * 0.5
            + 0.5
        ).astype(np.float32)

    out = tmp_path / "volume_grid_anim.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(n_frames),
        fps=24,
        bake_camera=False,
        bake_volume=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    vol_mats = [m for m in bpy.data.materials if m.name.endswith("_vol_mat")]
    if not vol_mats:
        pytest.fail("animation export lost the volume material")
    mat = vol_mats[0]
    if mat.node_tree is None:
        pytest.fail("volume material has no node_tree after export")
    value_node = mat.node_tree.nodes.get("pvb_volume_frame_value")
    if value_node is None:
        pytest.fail(
            "live grid mutations did not register as per-frame variation; "
            "the live-dataset registry is not feeding the volume sampler"
        )


@pytest.mark.bpy
def test_export_animation_blend_bake_glyphs_injects_set_position(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_glyphs=True`` injects a Set Position node when points move.

    The bridge bakes per-frame glyph positions into a float-buffer image
    packed inside the .blend, then splices a sampler + Set Position
    pipeline upstream of the existing instancer in the glyph's GN tree.
    The image data-block lives in ``bpy.data.images`` named
    ``<obj>__positions.exr``.
    """
    pl = offscreen_plotter
    base_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    cloud = pv.PolyData(base_points)
    pl.blender.add_glyph(cloud, geom=pv.Cone(), factor=0.3)
    pl.camera_position = [(3.0, -3.0, 2.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)]

    def update(frame: int) -> None:
        cloud.points = base_points + np.array([0.0, 0.0, 0.1 * frame], dtype=np.float32)

    n_frames = 5
    out = tmp_path / "glyph_anim.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(n_frames),
        fps=24,
        bake_camera=False,
        bake_glyphs=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    # The image data-block is packed inside the .blend.
    position_images = [
        img for img in bpy.data.images if img.name.endswith("__positions.exr")
    ]
    if not position_images:
        pytest.fail(
            "bake_glyphs=True did not produce a *__positions.exr image; "
            f"images: {[img.name for img in bpy.data.images]}"
        )
    img = position_images[0]
    if img.packed_file is None:
        pytest.fail(f"position image {img.name!r} is not packed into the .blend")
    expected_size = (3, n_frames)  # (N_points, N_frames)
    if tuple(img.size) != expected_size:
        pytest.fail(
            f"position image size {tuple(img.size)} != expected {expected_size}"
        )

    # A Set Position node should have been spliced into the glyph's GN tree.
    glyph_trees = [t for t in bpy.data.node_groups if t.name.endswith("_GN")]
    if not glyph_trees:
        pytest.fail("glyph GN tree missing after bake_glyphs export")
    tree = glyph_trees[0]
    node_types = {n.bl_idname for n in tree.nodes}
    expected_nodes = {
        "GeometryNodeSetPosition",
        "GeometryNodeImageTexture",
        "GeometryNodeInputSceneTime",
        "GeometryNodeInstanceOnPoints",
    }
    missing = expected_nodes - node_types
    if missing:
        pytest.fail(
            f"bake_glyphs sub-graph missing nodes: {sorted(missing)}; "
            f"present: {sorted(node_types)}"
        )


@pytest.mark.bpy
def test_export_animation_blend_bake_glyphs_skips_constant_positions(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A glyph whose source data is static skips the bake entirely.

    No image, no Set Position node — the GN tree stays in the same
    shape as a static render.
    """
    pl = offscreen_plotter
    cloud = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    pl.blender.add_glyph(cloud, geom=pv.Cone(), factor=0.3)
    pl.camera_position = [(3.0, -3.0, 2.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)]

    def update(_frame: int) -> None:
        return  # constant points

    out = tmp_path / "glyph_static.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(4),
        fps=24,
        bake_camera=False,
        bake_glyphs=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    position_images = [
        img for img in bpy.data.images if img.name.endswith("__positions.exr")
    ]
    if position_images:
        pytest.fail(
            f"constant glyph should not produce a positions image; got "
            f"{[img.name for img in position_images]}"
        )
    glyph_trees = [t for t in bpy.data.node_groups if t.name.endswith("_GN")]
    if not glyph_trees:
        pytest.fail("glyph GN tree missing after export")
    node_types = {n.bl_idname for n in glyph_trees[0].nodes}
    if "GeometryNodeSetPosition" in node_types:
        pytest.fail("constant glyph should not have a Set Position spliced in")


@pytest.mark.bpy
def test_export_animation_blend_bakes_volume_atlas(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """``bake_volume=True`` stacks per-frame volume scalars in one atlas.

    The material's atlas image grows vertically by a factor of
    ``n_frames``, and a ``ShaderNodeValue`` named
    ``pvb_volume_frame_value`` carries a keyframed action so playback
    scrolls through the frame bands.
    """
    pl = offscreen_plotter
    grid = pv.ImageData(dimensions=(20, 20, 20), spacing=(0.1, 0.1, 0.1))
    grid["density"] = np.zeros(grid.n_points, dtype=np.float32)
    vol = cast(
        "pv.Volume",
        pl.add_volume(grid, scalars="density", cmap="inferno", opacity="linear"),
    )
    pl.camera_position = [(4.0, -3.0, 3.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)]

    n_frames = 4

    def update(frame: int) -> None:
        # pyvista.add_volume copies the input dataset, so mutate the
        # mapper's dataset directly to drive per-frame scalar changes.
        ds = vol.mapper.dataset
        x, y, z = ds.points.T
        ds["density"] = (
            np.sin(2.0 * x + 0.3 * frame) * np.cos(2.0 * y) * np.sin(2.0 * z) * 0.5
            + 0.5
        ).astype(np.float32)

    out = tmp_path / "volume_anim.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(n_frames),
        fps=24,
        bake_camera=False,
        bake_volume=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    vol_mats = [m for m in bpy.data.materials if m.name.endswith("_vol_mat")]
    if not vol_mats:
        pytest.fail("animation export lost the volume material")
    mat = vol_mats[0]
    if mat.node_tree is None:
        pytest.fail("volume material has no node_tree after export")
    value_node = mat.node_tree.nodes.get("pvb_volume_frame_value")
    if value_node is None:
        pytest.fail(
            "bake_volume=True did not inject a frame-indexed Value node; "
            f"nodes: {[n.name for n in mat.node_tree.nodes]}"
        )
    nt_anim = mat.node_tree.animation_data
    if nt_anim is None or nt_anim.action is None:
        pytest.fail(
            "Value node has no animation_data.action — bake_volume keyframes missing"
        )
    fcurves = _action_fcurves(nt_anim.action)
    value_curves = [fc for fc in fcurves if "pvb_volume_frame_value" in fc.data_path]
    if len(value_curves) != 1:
        pytest.fail(
            f"expected exactly one Value-node fcurve, got {len(value_curves)}: "
            f"{[fc.data_path for fc in fcurves]}"
        )
    keyframes = value_curves[0].keyframe_points
    if len(keyframes) != n_frames:
        pytest.fail(
            f"expected {n_frames} keyframes on the Value node, got {len(keyframes)}"
        )

    atlas_images = [
        img for img in bpy.data.images if img.name.startswith("pvblender_volume_")
    ]
    if not atlas_images:
        pytest.fail("animation export lost the packed atlas image")
    # Static atlas was (ny=20, nx*nz=400); animated stacks n_frames vertically.
    expected_height = 20 * n_frames
    if atlas_images[0].size[1] != expected_height:
        pytest.fail(
            f"animated atlas height {atlas_images[0].size[1]} != "
            f"{expected_height} (ny * n_frames)"
        )


@pytest.mark.bpy
def test_export_animation_blend_skips_static_volume_bake(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A volume whose scalars are constant across frames stays static.

    No Value node is injected; the material keeps its single-
    frame shader graph.
    """
    pl = offscreen_plotter
    grid = pv.ImageData(dimensions=(20, 20, 20), spacing=(0.1, 0.1, 0.1))
    grid["density"] = np.linspace(0.0, 1.0, grid.n_points, dtype=np.float32)
    pl.add_volume(grid, scalars="density", cmap="inferno", opacity="linear")
    pl.camera_position = [(4.0, -3.0, 3.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)]

    def update(_frame: int) -> None:
        return  # constant scalars — no-op updater

    out = tmp_path / "volume_static.blend"
    pl.blender.export_animation_blend(
        str(out),
        update,
        frames=range(4),
        fps=24,
        bake_camera=False,
        bake_volume=True,
    )
    bpy.ops.wm.open_mainfile(filepath=str(out))

    vol_mats = [m for m in bpy.data.materials if m.name.endswith("_vol_mat")]
    if not vol_mats:
        pytest.fail("animation export lost the volume material")
    mat = vol_mats[0]
    if mat.node_tree is None:
        pytest.fail("volume material has no node_tree after export")
    value_node = mat.node_tree.nodes.get("pvb_volume_frame_value")
    if value_node is not None:
        pytest.fail("constant-scalar volume should not get a frame-indexed Value node")


@pytest.mark.bpy
def test_render_volume_resamples_unstructured_grid(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """An UnstructuredGrid input is resampled to ImageData and rendered."""
    pl = offscreen_plotter
    # Build an UnstructuredGrid by casting an ImageData — gives us a
    # known scalar field on tet cells without depending on external data.
    base = pv.ImageData(dimensions=(10, 10, 10), spacing=(0.1, 0.1, 0.1))
    base["density"] = np.linspace(0.0, 1.0, base.n_points, dtype=np.float32)
    ug = base.cast_to_unstructured_grid()
    pl.add_volume(ug, scalars="density", cmap="viridis", opacity="linear")
    pl.camera_position = [(2.0, -2.0, 2.0), (0.5, 0.5, 0.5), (0.0, 0.0, 1.0)]

    out = tmp_path / "ugrid_volume.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("UnstructuredGrid volume render produced no output")
    if not _has_volume_artifacts():
        pytest.fail("UnstructuredGrid volume did not produce *_vol_mat + packed atlas")


@pytest.mark.bpy
def test_render_volume_resamples_structured_grid(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A StructuredGrid input flows through resample → atlas → render."""
    pl = offscreen_plotter
    i, j, k = np.mgrid[0:10, 0:10, 0:10].astype(np.float64)
    sg = pv.StructuredGrid(i, j, k)
    sg["temp"] = (np.sin(0.5 * sg.points[:, 0]) + 1.0).astype(np.float32)
    pl.add_volume(sg, scalars="temp", cmap="inferno", opacity="linear")
    pl.camera_position = [(20.0, -20.0, 20.0), (5.0, 5.0, 5.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "sgrid_volume.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("StructuredGrid volume render produced no output")
    if not _has_volume_artifacts():
        pytest.fail("StructuredGrid volume did not produce *_vol_mat + packed atlas")


@pytest.mark.bpy
def test_render_volume_resamples_rectilinear_grid(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """A RectilinearGrid input flows through resample → atlas → render."""
    pl = offscreen_plotter
    # Non-uniform spacing along x to exercise the resampling path
    # (uniform spacing would degenerate to an ImageData equivalent).
    xs = np.array([0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.8, 3.7, 4.7, 5.8])
    ys = np.linspace(0.0, 4.0, 10)
    zs = np.linspace(0.0, 4.0, 10)
    rg = pv.RectilinearGrid(xs, ys, zs)
    rg["density"] = (np.sin(rg.points[:, 0]) * np.cos(rg.points[:, 1]) + 1.0).astype(
        np.float32
    )
    pl.add_volume(rg, scalars="density", cmap="plasma", opacity="linear")
    pl.camera_position = [(10.0, -8.0, 8.0), (3.0, 2.0, 2.0), (0.0, 0.0, 1.0)]

    out = tmp_path / "rgrid_volume.png"
    pl.blender.render(str(out), samples=4)

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("RectilinearGrid volume render produced no output")
    if not _has_volume_artifacts():
        pytest.fail("RectilinearGrid volume did not produce *_vol_mat + packed atlas")


def test_export_animation_blend_rejects_empty_frames(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """An empty ``frames`` iterable raises rather than writing a silent stub."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red")
    updater = pl.blender.orbit_camera(n_frames=4)
    with pytest.raises(ValueError, match=r"frames is empty"):
        pl.blender.export_animation_blend(
            str(tmp_path / "empty.blend"), updater, frames=[], fps=30
        )


def test_orbit_camera_accessor_returns_updater(
    offscreen_plotter: pv.Plotter,
) -> None:
    """``pl.blender.orbit_camera(...)`` returns a callable that mutates the camera."""
    pl = offscreen_plotter
    pl.camera_position = [(4.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    initial_position = tuple(pl.camera.position)

    update = pl.blender.orbit_camera(n_frames=4)
    if not callable(update):
        pytest.fail(f"orbit_camera() returned {update!r}, expected a callable")

    update(0)
    if tuple(pl.camera.position) != pytest.approx(initial_position):
        pytest.fail("frame 0 of the orbit did not reproduce the starting pose")

    update(1)
    if tuple(pl.camera.position) == pytest.approx(initial_position):
        pytest.fail("frame 1 of the orbit did not move the camera")


def test_render_hud_overlay_unknown_kind_raises(
    offscreen_plotter: pv.Plotter,
) -> None:
    """An unknown overlay name raises with a list of valid options."""
    pl = offscreen_plotter
    with pytest.raises(ValueError, match=r"known overlays:.*scalar_bar"):
        pl.blender.render_hud_overlay("nope", width=64, height=64)


def test_render_hud_overlay_scalar_bar_returns_rgba(
    offscreen_plotter: pv.Plotter,
) -> None:
    """``scalar_bar`` overlay returns an ``(H, W, 4)`` array when bars exist."""
    pl = offscreen_plotter
    sphere = pv.Sphere()
    sphere["z"] = sphere.points[:, 2]
    pl.add_mesh(sphere, scalars="z", cmap="viridis", show_scalar_bar=True)

    rgba = pl.blender.render_hud_overlay("scalar_bar", width=320, height=240)
    if rgba is None:
        pytest.fail("scalar_bar producer returned None despite a visible bar")
    expected_shape = (240, 320, 4)
    if rgba.shape != expected_shape:
        pytest.fail(f"expected shape {expected_shape}, got {rgba.shape}")


def test_render_hud_overlay_returns_none_when_empty(
    offscreen_plotter: pv.Plotter,
) -> None:
    """An empty plotter yields ``None`` rather than a blank array."""
    pl = offscreen_plotter
    result = pl.blender.render_hud_overlay("scalar_bar", width=64, height=64)
    if result is not None:
        pytest.fail(f"expected None on an empty plotter, got shape {result.shape}")


def test_show_requires_on_screen_plotter(offscreen_plotter: pv.Plotter) -> None:
    """``show()`` needs a real VTK event loop; off-screen plotters can't host one."""
    pl = offscreen_plotter
    with pytest.raises(RuntimeError, match=r"off_screen=True has no VTK event loop"):
        pl.blender.show()


@pytest.mark.bpy
def test_scene_cache_reuses_blocks_across_renders(
    offscreen_plotter: pv.Plotter,
) -> None:
    """Two consecutive build_scene_from_plotter calls reuse mesh + material."""
    pl = offscreen_plotter
    sphere = pv.Sphere()
    pl.add_mesh(sphere, color="red")

    cache_1 = build_scene_from_plotter(pl, cache=None)
    if not cache_1.objects:
        pytest.fail("first build did not populate the object cache")
    if not cache_1.materials:
        pytest.fail("first build did not populate the material cache")
    obj_names_1 = dict(cache_1.objects)
    mat_names_1 = dict(cache_1.materials)

    cache_2 = build_scene_from_plotter(pl, cache=cache_1)
    if cache_2 is not cache_1:
        pytest.fail("second build returned a new cache instead of mutating")
    if dict(cache_2.objects) != obj_names_1:
        pytest.fail(
            f"object cache changed: {obj_names_1!r} -> {dict(cache_2.objects)!r}"
        )
    if dict(cache_2.materials) != mat_names_1:
        pytest.fail(
            f"material cache changed: {mat_names_1!r} -> {dict(cache_2.materials)!r}"
        )


@pytest.mark.bpy
def test_scene_cache_evicts_removed_actor(
    offscreen_plotter: pv.Plotter,
) -> None:
    """Removing a mesh between renders drops its cache entry."""
    pl = offscreen_plotter
    pl.add_mesh(pv.Sphere(), color="red", name="sphere")
    pl.add_mesh(pv.Cube(), color="blue", name="cube")
    expected_objects = 2

    cache = build_scene_from_plotter(pl, cache=None)
    n_objects_first = len(cache.objects)
    if n_objects_first != expected_objects:
        pytest.fail(
            f"expected {expected_objects} cached objects, got {n_objects_first}"
        )
    objects_before_remove = set(bpy.data.objects.keys())

    pl.renderer.remove_actor("cube")
    cache = build_scene_from_plotter(pl, cache=cache)
    if len(cache.objects) != 1:
        pytest.fail(f"expected 1 cached object after removal, got {len(cache.objects)}")
    objects_after_remove = set(bpy.data.objects.keys())
    survivors = objects_before_remove & objects_after_remove
    if "cube" in (objects_before_remove - survivors):
        # The cube was removed from bpy.data.objects as expected; nothing to do.
        return
    if "cube" in objects_after_remove:
        pytest.fail("cube bpy object survived eviction")
