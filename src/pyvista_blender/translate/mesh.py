# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate a PyVista actor's dataset into a ``bpy.types.Mesh`` + ``Object``.

For ``UnstructuredGrid`` and similar volumetric datasets, the surface is
extracted before triangulation. Scalars are uploaded to a ``FLOAT_COLOR``
attribute named ``"scalars"`` whose domain depends on where the scalar
lives on the source dataset:

* **POINT** (per-vertex) — interpolated across the face, smooth shading.
* **CORNER** (per-loop) — each loop reads the parent cell's flat colour,
  giving the per-face cell visualisation that FEA / CFD users expect.

The bpy ``ShaderNodeAttribute("scalars")`` automatically picks up
whichever domain is present.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import bpy
import matplotlib as mpl
import numpy as np
from mathutils import Matrix

from pyvista_blender import config

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["refresh_actor_mesh", "translate_actor_mesh"]


def _actor_world_matrix(actor: pv.Actor) -> Matrix:
    """Convert an actor's ``user_matrix`` (numpy 4x4) to a ``mathutils.Matrix``.

    The bridge translates surfaces in *local* coordinates (vertex
    positions come straight off ``dataset.points``), so the actor's
    user transform — typically used to place or animate a static
    mesh in a scene — has to live on the bpy object's
    ``matrix_world`` rather than baked into the vertex data. This
    matches the VTK convention where ``vtkActor.SetUserMatrix`` is
    applied at render time, not at the polydata level.

    Returns
    -------
    mathutils.Matrix
        4x4 matrix derived from ``actor.user_matrix``; the identity
        matrix when the actor exposes no ``user_matrix`` attribute or
        the array is non-finite.

    """
    user_matrix = getattr(actor, "user_matrix", None)
    if user_matrix is None:
        return Matrix.Identity(4)
    arr = np.asarray(user_matrix, dtype=np.float64)
    expected_shape = (4, 4)
    if arr.shape != expected_shape or not np.isfinite(arr).all():
        return Matrix.Identity(4)
    return Matrix(arr.tolist())


def translate_actor_mesh(
    actor: pv.Actor,
    name: str,
) -> bpy.types.Object:
    """Build a bpy ``Object`` wrapping a triangulated surface of ``actor``'s dataset.

    Parameters
    ----------
    actor
        The PyVista actor whose underlying dataset to translate.
    name
        Name for both the mesh data-block and the wrapping object.

    Returns
    -------
    bpy.types.Object
        The new Blender object, linked into the active scene's collection.

    """
    surface = _surface_with_point_scalars(actor)
    mesh_data = bpy.data.meshes.new(name)
    _build_mesh_topology(surface, mesh_data)
    _maybe_attach_scalars(surface, actor.mapper, mesh_data)

    obj = bpy.data.objects.new(name, mesh_data)
    obj.matrix_world = _actor_world_matrix(actor)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def refresh_actor_mesh(actor: pv.Actor, obj: bpy.types.Object) -> bool:
    """Re-upload vertex positions and scalars onto a cached bpy mesh.

    Used for per-frame updates in :func:`pl.blender.animate`: when the
    actor's source dataset has been mutated in place (vertex translation,
    scalar field update), this refreshes the bpy mesh data without
    re-allocating it. Falls back to a full rebuild when the topology has
    changed (different vertex / triangle count).

    Parameters
    ----------
    actor
        The PyVista actor whose dataset to re-extract.
    obj
        The previously cached bpy object linked to a mesh data block.

    Returns
    -------
    bool
        ``True`` when the existing mesh was refreshed in place, ``False``
        when topology changed and the caller should replace the object
        wholesale (rare; only happens if the user added / removed
        geometry between frames).

    """
    surface = _surface_with_point_scalars(actor)
    mesh_data = obj.data

    verts = np.ascontiguousarray(surface.points, dtype=np.float32)
    tris = np.ascontiguousarray(surface.regular_faces, dtype=np.int32)
    if len(mesh_data.vertices) != len(verts) or len(mesh_data.polygons) != len(tris):
        return False

    mesh_data.vertices.foreach_set("co", verts.ravel())
    _refresh_scalars(surface, actor.mapper, mesh_data)
    mesh_data.update()
    # Refresh the world transform too — the user may have animated
    # ``actor.user_matrix`` across frames; we want each per-frame
    # render to honour the latest pose without rebuilding the mesh.
    obj.matrix_world = _actor_world_matrix(actor)
    return True


def _build_mesh_topology(surface: pv.PolyData, mesh_data: bpy.types.Mesh) -> None:
    """Populate vertex / loop / polygon arrays on a freshly-created mesh."""
    verts = np.ascontiguousarray(surface.points, dtype=np.float32)
    tris = np.ascontiguousarray(surface.regular_faces, dtype=np.int32)
    n_verts = len(verts)
    n_tris = len(tris)

    mesh_data.vertices.add(n_verts)
    mesh_data.vertices.foreach_set("co", verts.ravel())

    n_loops = n_tris * 3
    mesh_data.loops.add(n_loops)
    mesh_data.loops.foreach_set("vertex_index", tris.ravel())

    mesh_data.polygons.add(n_tris)
    mesh_data.polygons.foreach_set("loop_start", np.arange(n_tris, dtype=np.int32) * 3)
    mesh_data.polygons.foreach_set("loop_total", np.full(n_tris, 3, dtype=np.int32))

    mesh_data.update(calc_edges=True)
    mesh_data.validate()

    # bpy 5.x defaults polygons to use_smooth=True, which averages normals
    # across every shared vertex and rounds off corners on low-poly hulls
    # (e.g. pv.Cube → 8 shared vertices). Mirror VTK's feature_angle by
    # splitting normals at edges whose neighbouring faces meet at a sharper
    # angle than the configured threshold. Cycles / Eevee Next read the
    # resulting `sharp_edge` attribute directly when shading.
    angle_deg = float(config.sharp_edge_angle)
    mesh_data.set_sharp_from_angle(angle=math.radians(angle_deg))


def _refresh_scalars(
    surface: pv.PolyData,
    mapper: object,
    mesh_data: bpy.types.Mesh,
) -> None:
    """Update an existing ``"scalars"`` colour attribute, or create one if absent.

    Mirrors :func:`_maybe_attach_scalars` but writes through to an
    existing FLOAT_COLOR attribute (created on first frame, mutated on
    subsequent frames) so per-frame scalar changes don't leak bpy data
    blocks.
    """
    if not getattr(mapper, "scalar_visibility", False):
        return

    rgba_linear, domain = _resolve_scalar_colors(surface, mapper)
    if rgba_linear is None or domain is None:
        return

    attr = mesh_data.color_attributes.get("scalars")
    if attr is None or attr.domain != domain:
        if attr is not None:
            mesh_data.color_attributes.remove(attr)
        attr = mesh_data.color_attributes.new(
            name="scalars", type="FLOAT_COLOR", domain=domain
        )
    attr.data.foreach_set("color", rgba_linear.ravel())


def _surface_with_point_scalars(actor: pv.Actor) -> pv.PolyData:
    """Extract a triangulated surface, preserving the source scalar domain.

    Returns
    -------
    pv.PolyData
        Triangulated boundary surface. Point scalars stay on
        ``point_data``; cell scalars stay on ``cell_data`` and are
        replicated per resulting triangle by VTK's triangulate filter
        (the bridge then attaches them on the CORNER domain).

    """
    dataset = actor.mapper.dataset
    dataset = _maybe_tessellate(dataset)
    if hasattr(dataset, "extract_surface"):
        # Pin the algorithm explicitly so PyVista's planned default-change
        # (dataset_surface → None) doesn't shift the bridge's output later.
        surface = dataset.extract_surface(algorithm="dataset_surface")
    else:
        surface = dataset.copy()

    return surface.triangulate()


#: VTK cell-type IDs that carry curved geometry and benefit from
#: :meth:`pv.DataSet.tessellate` before surface extraction. Sourced from
#: VTK's ``vtkCellType.h``:
#:
#: * 21 - 37: quadratic / biquadratic / triquadratic / cubic family
#: * 68 - 74: Lagrange family
#: * 75 - 81: Bezier family
_HIGH_ORDER_CELL_TYPES = frozenset(range(21, 38)) | frozenset(range(68, 82))


def _maybe_tessellate(dataset: pv.DataSet) -> pv.DataSet:
    """Refine high-order cells into linear simplices before surface extraction.

    ``extract_surface()`` linearises curved cell faces into flat
    triangles, which loses the curvature carried by quadratic /
    Lagrange / Bezier cells (VTK cell types in
    :data:`_HIGH_ORDER_CELL_TYPES`). When any such cell appears in
    ``dataset.celltypes``, the bridge runs
    :meth:`pv.DataSet.tessellate` first with
    ``max_n_subdivide=config.tessellation_subdivide``. Datasets with
    only linear cells skip the filter entirely so we don't pay
    VTK's per-cell adaptive-subdivision cost for nothing.

    Returns
    -------
    pv.DataSet
        Either the original ``dataset`` (no high-order cells, or
        ``tessellation_subdivide`` is 0), or the tessellated output
        with the same point + cell data.

    """
    subdivide = int(getattr(config, "tessellation_subdivide", 0))
    if subdivide <= 0:
        return dataset
    celltypes = getattr(dataset, "celltypes", None)
    if celltypes is None or len(celltypes) == 0:
        return dataset
    unique_types = {int(t) for t in np.unique(np.asarray(celltypes))}
    if not (unique_types & _HIGH_ORDER_CELL_TYPES):
        return dataset
    tessellate = getattr(dataset, "tessellate", None)
    if tessellate is None:
        return dataset
    return tessellate(max_n_subdivide=subdivide)


def _maybe_attach_scalars(
    surface: pv.PolyData,
    mapper: object,
    mesh_data: bpy.types.Mesh,
) -> None:
    """Bake the active scalar field through the LUT into a colour attribute.

    Picks POINT or CORNER domain depending on where the scalar lives:
    point-data scalars get smooth interpolation across the face; cell-
    data scalars stay flat per face (each loop reads its parent cell's
    value) for the FEA / CFD per-element visualisation.
    """
    if not getattr(mapper, "scalar_visibility", False):
        return

    rgba_linear, domain = _resolve_scalar_colors(surface, mapper)
    if rgba_linear is None or domain is None:
        return

    attr = mesh_data.color_attributes.new(
        name="scalars", type="FLOAT_COLOR", domain=domain
    )
    attr.data.foreach_set("color", rgba_linear.ravel())


def _resolve_scalar_colors(
    surface: pv.PolyData, mapper: object
) -> tuple[np.ndarray | None, str | None]:
    """Compute the per-vertex / per-loop colours for the active scalar.

    Returns
    -------
    tuple of (np.ndarray or None, str or None)
        ``(rgba_linear, domain)`` ready for ``foreach_set("color", ...)``,
        or ``(None, None)`` when no usable scalar field is present.

    """
    array_name, domain = _resolve_active_scalar(surface, mapper)
    if array_name is None or domain is None:
        return None, None

    if domain == "POINT":
        raw = np.asarray(surface.point_data[array_name], dtype=np.float32)
    else:
        cell_values = np.asarray(surface.cell_data[array_name], dtype=np.float32)
        raw = np.repeat(cell_values, 3)

    lo, hi = getattr(mapper, "scalar_range", (0.0, 1.0))
    if hi <= lo:
        return None, None

    normalized = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)

    cmap_name = (
        getattr(getattr(mapper, "lookup_table", None), "cmap", None) or "viridis"
    )
    cmap = mpl.colormaps.get_cmap(cmap_name)
    rgba_srgb = np.asarray(cmap(normalized), dtype=np.float32)
    return _srgb_to_linear(rgba_srgb), domain


def _resolve_active_scalar(
    surface: pv.PolyData, mapper: object
) -> tuple[str | None, str | None]:
    """Find the active scalar name and which bpy domain it should land on.

    Returns
    -------
    tuple of (str or None, str or None)
        ``(array_name, "POINT")`` for point-data scalars,
        ``(array_name, "CORNER")`` for cell-data scalars, or
        ``(None, None)`` when no usable scalar is present.

    """
    array_name = getattr(mapper, "array_name", None)
    if not array_name:
        # CompositePolyDataMapper (PyVista's MultiBlock mapper) leaves
        # array_name blank; fall back to whichever array extract_surface
        # promoted to active.
        array_name = getattr(surface, "active_scalars_name", None)
    if not array_name:
        return None, None

    if array_name in surface.point_data:
        return array_name, "POINT"
    if array_name in surface.cell_data:
        return array_name, "CORNER"
    return None, None


def _srgb_to_linear(rgba: np.ndarray) -> np.ndarray:
    """Convert sRGB-encoded RGBA values to scene-linear (alpha untouched).

    Returns
    -------
    np.ndarray
        A new array with the same shape as ``rgba``; the RGB channels are
        linearised, alpha is copied through.

    """
    out = rgba.copy()
    rgb = out[..., :3]
    threshold = 0.04045
    low = rgb / 12.92
    high = ((rgb + 0.055) / 1.055) ** 2.4
    out[..., :3] = np.where(rgb <= threshold, low, high)
    return out
