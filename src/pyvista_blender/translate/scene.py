# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Orchestrator: walk a ``pv.Plotter`` and reconcile a Blender scene.

The first call builds the scene from scratch (after wiping bpy's factory
state). Subsequent calls *reconcile* against an identity-keyed cache: a
mesh whose source PyVista dataset object is unchanged keeps its
``bpy.types.Mesh`` data block; a material whose source ``pv.Property``
hasn't been replaced keeps its node tree. Camera, lights, and world
shader are cheap and always rebuilt fresh.

This is the "Level 1" identity cache documented in ``docs/architecture.md``.
Per-frame animation rides on it: vertex / scalar updates mutate a
long-lived bpy mesh in place rather than rebuilding it.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, cast

import bpy
import pyvista as pv

from pyvista_blender.translate import (
    background,
    camera,
    glyph,
    light,
    material,
    mesh,
    point_cloud,
    volume,
    wireframe,
)

if TYPE_CHECKING:
    from pyvista_blender._glyph import GlyphSpec

#: Signature PyVista's :meth:`pv.DataSet.glyph` leaves on its baked output.
#: When we see this point-data array on an actor's dataset we know the
#: user pre-baked their glyphs into a single merged polydata — slower
#: and heavier than routing through ``pl.blender.add_glyph(...)``. We
#: render the baked mesh as-is (so behaviour is unchanged) but warn the
#: user once per actor so they can opt into the GN-instanced path.
_BAKED_GLYPH_MARKER = "GlyphVector"

#: Strip the ``(Addr=0x...)`` suffix PyVista auto-generates for unnamed
#: actors. The dataset class name (``PolyData``, ``UnstructuredGrid``,
#: ...) stays; bpy's name-collision handler appends ``.001`` / ``.002``
#: when several mapped objects share the same shortened name. Names a
#: user passed explicitly via ``pl.add_mesh(..., name="hills")`` don't
#: match this pattern and pass through untouched.
_PV_ACTOR_NAME_RE = re.compile(r"^(?P<base>.+)\(Addr=0x[0-9a-fA-F]+\)$")

__all__ = ["SceneCache", "build_scene_from_plotter"]


def _bpy_friendly_name(actor_name: str) -> str:
    """Trim PyVista's auto-generated ``(Addr=...)`` suffix from an actor name.

    Parameters
    ----------
    actor_name
        Raw key from ``renderer.actors`` — either user-supplied
        (``pl.add_mesh(..., name="hills")``) or PyVista's default
        ``<DatasetClassName>(Addr=0x...)`` pattern.

    Returns
    -------
    str
        Trimmed name for use as a bpy object label. Unchanged if
        ``actor_name`` doesn't match the auto-generated pattern.

    """
    match = _PV_ACTOR_NAME_RE.match(actor_name)
    if match is None:
        return actor_name
    return match.group("base").rstrip()


def vtk_identity(vtk_wrapped: object) -> str:
    """Return a stable cross-call identity for a VTK-wrapped PyVista object.

    PyVista's :class:`pv.Actor` is held alive by ``renderer.actors`` and
    its SWIG ``__this__`` handle encodes the raw VTK pointer plus type
    tag, so it stays the same as long as the underlying VTK actor lives.
    Importantly, ``actor.mapper.dataset`` is *not* a safe key — calling
    ``dataset.extract_surface(...)`` (which the mesh translator does)
    can rebind the mapper's input to a new VTK polydata, so the same
    actor would hash differently before vs. after the translation step.
    The actor itself is the right granularity. Falls back to ``id()``
    for the rare non-VTK case.

    Returns
    -------
    str
        The SWIG ``__this__`` string when present, otherwise the ``id()``
        rendered as a string for type uniformity.

    """
    this = getattr(vtk_wrapped, "__this__", None)
    if isinstance(this, str):
        return this
    return str(id(vtk_wrapped))


class SceneCache:
    """Identity-keyed map from PyVista actors / properties to bpy data names.

    Stores names (strings) rather than bpy object references so the cache
    survives if the underlying data block is removed out-of-band — a
    missing ``bpy.data.X.get(name)`` lookup simply rebuilds. Keys are
    SWIG handles (see :func:`vtk_identity`) so they stay stable across
    PyVista's wrapper recreation.
    """

    __slots__ = (
        "glyphs",
        "materials",
        "objects",
        "point_clouds",
        "volumes",
        "wire_materials",
        "wires",
    )

    def __init__(self) -> None:
        """Initialise empty mesh, material, wireframe, glyph, and volume maps."""
        # __this__(pv.Actor) -> bpy.types.Object.name
        self.objects: dict[str, str] = {}
        # __this__(pv.Property) -> bpy.types.Material.name
        self.materials: dict[str, str] = {}
        # __this__(pv.Actor) -> bpy.types.Object.name of the wire overlay
        self.wires: dict[str, str] = {}
        # __this__(pv.Property) -> bpy.types.Material.name of the wire shader
        self.wire_materials: dict[str, str] = {}
        # GlyphSpec ordinal -> (points_obj.name, geom_obj.name, node_group.name)
        # The geom + node group are rebuilt every render (cheap, small mesh)
        # so the cache only tracks the names for eviction on subsequent calls.
        self.glyphs: dict[int, tuple[str, str, str]] = {}
        # __this__(pv.Volume) -> bpy.types.Object.name of the carrier.
        # Volumes are rebuilt every render (no refresh path), so this
        # just lets us purge yesterday's carriers before rebuilding to
        # keep ``bpy.data.objects`` clean.
        self.volumes: dict[str, str] = {}
        # __this__(pv.Actor) -> bpy.types.Object.name of the point cloud.
        # Same rebuild-every-render strategy as volumes; the cache only
        # tracks names so purges before rebuild keep
        # ``bpy.data.pointclouds`` clean.
        self.point_clouds: dict[str, str] = {}

    def is_empty(self) -> bool:
        """Return whether nothing is cached yet.

        Returns
        -------
        bool
            ``True`` if every map is empty, i.e. the next render is the
            first one for this component.

        """
        return not (
            self.objects
            or self.materials
            or self.wires
            or self.wire_materials
            or self.glyphs
            or self.volumes
            or self.point_clouds
        )


def build_scene_from_plotter(
    plotter: pv.BasePlotter,
    cache: SceneCache | None = None,
    glyphs: list[GlyphSpec] | None = None,
    volume_sources: dict[str, pv.DataSet] | None = None,
) -> SceneCache:
    """Reconcile the active bpy scene against ``plotter`` state.

    Parameters
    ----------
    plotter
        Source PyVista plotter whose renderers are walked.
    cache
        Identity-keyed reuse map carried across render calls. ``None``
        means "fresh start"; a new :class:`SceneCache` is allocated and
        the bpy scene is wiped to factory state first.
    glyphs
        Glyph specs registered via ``pl.blender.add_glyph(...)``. Each
        spec is materialised as a Geometry-Nodes-instanced bpy object.
        ``None`` is treated as "no glyphs".
    volume_sources
        Live-dataset overrides registered via
        ``pl.blender.add_volume(...)``. Maps each volume actor's
        ``vtk_identity`` to the *original* :class:`pv.DataSet` the user
        is mutating. The volume translator reads scalars from this
        dataset instead of ``actor.mapper.dataset`` (which pyvista
        copies on ``add_volume``). ``None`` is treated as "no
        overrides" — the translator reads the actor's copied dataset.

    Returns
    -------
    SceneCache
        The same cache, with stale entries evicted and new entries added.
        Caller is expected to retain it for the next render.

    """
    if cache is None:
        cache = SceneCache()
    if cache.is_empty():
        bpy.ops.wm.read_factory_settings(use_empty=True)
    else:
        # Cameras, lights, and the world shader are rebuilt every call;
        # purge the previous ones so they don't accumulate in bpy.data.
        _purge_transient_objects()
        _purge_cached_glyphs(cache)
        _purge_cached_volumes(cache)
        _purge_cached_point_clouds(cache)

    seen_actors: set[str] = set()
    seen_props: set[str] = set()

    warned_baked_glyphs: set[str] = set()
    for renderer in plotter.renderers:
        for raw_name, actor in renderer.actors.items():
            if not getattr(actor, "visibility", True):
                continue
            actor_name = _bpy_friendly_name(raw_name)
            if isinstance(actor, pv.Actor):
                actor_key = vtk_identity(actor)
                point_cloud_mode = _detect_point_cloud_mode(actor)
                if point_cloud_mode is not None:
                    pc_obj = point_cloud.translate_point_cloud(
                        actor, actor_name, mode=point_cloud_mode
                    )
                    cache.point_clouds[actor_key] = pc_obj.name
                    continue
                prop_key = vtk_identity(actor.prop)
                _maybe_warn_baked_glyph(actor, actor_key, warned_baked_glyphs)
                obj = _reconcile_actor_mesh(actor, actor_name, actor_key, cache)
                _reconcile_actor_material(actor, actor_name, prop_key, obj, cache)
                _reconcile_actor_wireframe(
                    actor,
                    actor_name,
                    actor_key,
                    prop_key,
                    surface_obj=obj,
                    cache=cache,
                )
                seen_actors.add(actor_key)
                seen_props.add(prop_key)
            elif isinstance(actor, pv.Volume):
                # Volumes are rebuilt from scratch every render (the GN
                # graph + packed atlas is small enough that caching
                # would add more code than it saves).
                actor_key = vtk_identity(actor)
                live_dataset = volume_sources.get(actor_key) if volume_sources else None
                vol_obj = volume.translate_volume(
                    actor, actor_name, live_dataset=live_dataset
                )
                cache.volumes[actor_key] = vol_obj.name

    _evict_stale_objects(cache, seen_actors)
    _evict_stale_materials(cache, seen_props)
    _evict_stale_wires(cache, seen_actors)
    _evict_stale_wire_materials(cache, seen_props)

    camera.translate_camera(plotter.camera, tuple(plotter.window_size))
    light.translate_lights(plotter)
    background.translate_background(plotter)

    for ordinal, spec in enumerate(glyphs or []):
        points_obj = glyph.translate_glyph(spec, ordinal)
        cache.glyphs[ordinal] = (
            points_obj.name,
            f"{spec.name or f'PVGlyph_{ordinal}'}_geom",
            f"{spec.name or f'PVGlyph_{ordinal}'}_GN",
        )

    return cache


def _reconcile_actor_mesh(
    actor: pv.Actor,
    actor_name: str,
    actor_key: str,
    cache: SceneCache,
) -> bpy.types.Object:
    """Return a bpy ``Object`` for ``actor``, reusing the cache when possible.

    On a cache hit, the existing mesh's vertex positions and scalars are
    refreshed in place so per-frame mutations (vertex deformation,
    scalar field updates) propagate without re-allocating the bpy mesh
    data block, the foundation for per-frame animation.

    Returns
    -------
    bpy.types.Object
        The cached object on a hit (refreshed), or a freshly translated
        one on a miss / topology change.

    """
    cached_name = cache.objects.get(actor_key)
    obj = bpy.data.objects.get(cached_name) if cached_name else None
    if obj is None:
        obj = mesh.translate_actor_mesh(actor, name=actor_name)
        cache.objects[actor_key] = obj.name
        return obj

    if mesh.refresh_actor_mesh(actor, obj):
        return obj

    # Topology changed between renders — rebuild from scratch.
    old_mesh = obj.data
    bpy.data.objects.remove(obj)
    if old_mesh is not None and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    new_obj = mesh.translate_actor_mesh(actor, name=actor_name)
    cache.objects[actor_key] = new_obj.name
    return new_obj


def _reconcile_actor_material(
    actor: pv.Actor,
    actor_name: str,
    prop_key: str,
    obj: bpy.types.Object,
    cache: SceneCache,
) -> None:
    """Attach a material to ``obj``, reusing the cache when possible."""
    cached_name = cache.materials.get(prop_key)
    mat = bpy.data.materials.get(cached_name) if cached_name else None
    if mat is None:
        has_scalars = "scalars" in obj.data.color_attributes
        mat = material.make_material(
            actor, name=f"{actor_name}_mat", has_scalars=has_scalars
        )
        cache.materials[prop_key] = mat.name

    materials = obj.data.materials
    if not materials:
        materials.append(mat)
    elif materials[0] is not mat:
        materials.clear()
        materials.append(mat)


def _reconcile_actor_wireframe(
    actor: pv.Actor,
    actor_name: str,
    actor_key: str,
    prop_key: str,
    *,
    surface_obj: bpy.types.Object,
    cache: SceneCache,
) -> None:
    """Add or remove a wireframe overlay for ``actor`` per its style flags.

    Three branches: pure wireframe (``style="Wireframe"``) hides the fill
    surface and shows the wire only; surface + edges (``show_edges=True``)
    keeps both; everything else clears any previously cached wire.
    """
    prop = actor.prop
    needs_wire = wireframe.actor_needs_wire(actor)
    wire_only = str(getattr(prop, "style", "")).lower() == "wireframe"

    # Reset render visibility every call so previous-render state doesn't leak.
    surface_obj.hide_render = wire_only

    cached_wire_name = cache.wires.get(actor_key)
    if not needs_wire:
        if cached_wire_name is not None:
            _remove_wire_object(cached_wire_name)
            cache.wires.pop(actor_key, None)
        return

    wire_obj = bpy.data.objects.get(cached_wire_name) if cached_wire_name else None
    if wire_obj is None:
        wire_obj = wireframe.make_wire_object(surface_obj, actor_name, prop)
        cache.wires[actor_key] = wire_obj.name
    else:
        # Thickness can change between renders if the user mutated line_width
        # in-place. Modifier lookups are cheap; refresh unconditionally.
        modifier = wire_obj.modifiers.get("PVWireframe")
        if modifier is not None:
            modifier.thickness = wireframe.thickness_for(prop)

    cached_wire_mat_name = cache.wire_materials.get(prop_key)
    wire_mat = (
        bpy.data.materials.get(cached_wire_mat_name) if cached_wire_mat_name else None
    )
    if wire_mat is None:
        wire_mat = wireframe.make_wire_material(f"{actor_name}_wire_mat", prop)
        cache.wire_materials[prop_key] = wire_mat.name

    wire_materials = wire_obj.data.materials
    if not wire_materials:
        wire_materials.append(wire_mat)
    elif wire_materials[0] is not wire_mat:
        wire_materials.clear()
        wire_materials.append(wire_mat)


def _remove_wire_object(name: str) -> None:
    """Remove a previously cached wireframe overlay and its mesh."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    mesh_data = obj.data
    bpy.data.objects.remove(obj)
    if mesh_data is not None and mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)


def _maybe_warn_baked_glyph(actor: pv.Actor, actor_key: str, warned: set[str]) -> None:
    """Emit a one-time warning when the actor wraps a pre-baked glyph polydata.

    Detection is signature-based: ``pv.DataSet.glyph(...)`` leaves a
    ``GlyphVector`` point-data array on its output. We can't reconstruct
    the original source points + glyph mesh from that baked output
    (auto-unbake is genuinely lossy), so we render it as-is but suggest
    the lighter-weight ``pl.blender.add_glyph(...)`` path that hosts the
    glyph geometry once and instances at render time.
    """
    if actor_key in warned:
        return
    dataset = actor.mapper.dataset
    if not hasattr(dataset, "point_data"):
        return
    if _BAKED_GLYPH_MARKER not in dataset.point_data:
        return
    warned.add(actor_key)
    warnings.warn(
        "Detected a pre-baked glyph polydata "
        "(point-data carries 'GlyphVector'). Rendering as a single "
        "merged mesh — for N source points and a V-vertex glyph this "
        "uploads N*V vertices to Cycles. Consider replacing "
        "`add_mesh(source.glyph(...))` with `pl.blender.add_glyph("
        "source, geom, orient=..., scale=..., factor=...)` to instance "
        "via Geometry Nodes instead (uploads N+V vertices).",
        UserWarning,
        stacklevel=4,
    )


def _purge_transient_objects() -> None:
    """Remove camera / light objects so they can be rebuilt cleanly."""
    transient_types = {"CAMERA", "LIGHT"}
    for obj in list(bpy.data.objects):
        if obj.type in transient_types:
            bpy.data.objects.remove(obj)
    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)
    for light_data in list(bpy.data.lights):
        if light_data.users == 0:
            bpy.data.lights.remove(light_data)


def _purge_cached_glyphs(cache: SceneCache) -> None:
    """Drop every glyph object + geom + node group cached from a prior render.

    Glyph instancer bpy data is rebuilt from scratch each render (the
    geometry is small and the node-group authoring API is awkward to
    diff). This clears the previous artefacts so they don't accumulate
    in ``bpy.data``.
    """
    for points_name, geom_name, tree_name in cache.glyphs.values():
        for obj_name in (points_name, geom_name):
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                obj_data = obj.data
                bpy.data.objects.remove(obj)
                if obj_data is not None and obj_data.users == 0:
                    bpy.data.meshes.remove(obj_data)
        tree = bpy.data.node_groups.get(tree_name)
        if tree is not None and tree.users == 0:
            bpy.data.node_groups.remove(tree)
    cache.glyphs.clear()


def _detect_point_cloud_mode(actor: pv.Actor) -> str | None:
    """Decide whether ``actor`` should render as a point cloud, and how.

    Returns
    -------
    str or None
        ``"gaussian"`` when the actor's mapper is a
        :class:`pyvista.PointGaussianMapper` (set up by
        ``add_mesh(..., style="points_gaussian")``); ``"points"`` when
        the actor's property carries ``style="Points"`` (regular point
        rendering); ``None`` for the default surface-mesh path.

    """
    point_gaussian_mapper = getattr(pv, "PointGaussianMapper", None)
    if point_gaussian_mapper is not None and isinstance(
        actor.mapper, point_gaussian_mapper
    ):
        return "gaussian"
    style = str(getattr(actor.prop, "style", "")).lower()
    if style == "points":
        return "points"
    return None


def _purge_cached_volumes(cache: SceneCache) -> None:
    """Drop every volume carrier object cached from a prior render.

    Volumes are rebuilt from scratch each call (the GN graph + packed
    atlas image are small enough that diffing isn't worth the
    complexity). This clears the previous carriers so they don't
    accumulate in ``bpy.data``.
    """
    for obj_name in cache.volumes.values():
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        obj_data = obj.data
        bpy.data.objects.remove(obj)
        if obj_data is not None and obj_data.users == 0:
            bpy.data.meshes.remove(obj_data)
    cache.volumes.clear()


def _purge_cached_point_clouds(cache: SceneCache) -> None:
    """Drop every PointCloud object cached from a prior render.

    Point clouds are rebuilt from scratch each call (the per-point
    foreach_set is cheap relative to a render, and tracking diffs
    against a moving scalar field would add code without payoff). The
    data-block lives in ``bpy.data.pointclouds`` rather than
    ``bpy.data.meshes``, so this purge dispatches there.
    """
    for obj_name in cache.point_clouds.values():
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        obj_data = obj.data
        bpy.data.objects.remove(obj)
        if obj_data is not None and obj_data.users == 0:
            bpy.data.pointclouds.remove(cast("bpy.types.PointCloud", obj_data))
    cache.point_clouds.clear()


def _evict_stale_objects(cache: SceneCache, seen: set[str]) -> None:
    """Drop cached objects whose source dataset is no longer in the scene."""
    for key in list(cache.objects):
        if key in seen:
            continue
        name = cache.objects.pop(key)
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        mesh_data = obj.data
        bpy.data.objects.remove(obj)
        if mesh_data is not None and mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)


def _evict_stale_materials(cache: SceneCache, seen: set[str]) -> None:
    """Drop cached materials whose source property is no longer in the scene."""
    for key in list(cache.materials):
        if key in seen:
            continue
        name = cache.materials.pop(key)
        mat = bpy.data.materials.get(name)
        if mat is not None and mat.users == 0:
            bpy.data.materials.remove(mat)


def _evict_stale_wires(cache: SceneCache, seen: set[str]) -> None:
    """Drop cached wireframe overlays whose actor is no longer in the scene."""
    for key in list(cache.wires):
        if key in seen:
            continue
        _remove_wire_object(cache.wires.pop(key))


def _evict_stale_wire_materials(cache: SceneCache, seen: set[str]) -> None:
    """Drop cached wire shaders whose source property is no longer present."""
    for key in list(cache.wire_materials):
        if key in seen:
            continue
        name = cache.wire_materials.pop(key)
        mat = bpy.data.materials.get(name)
        if mat is not None and mat.users == 0:
            bpy.data.materials.remove(mat)
