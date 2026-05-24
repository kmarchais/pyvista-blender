# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate a PyVista volume actor into a closed-cube + Volume shader.

``plotter.add_volume(grid, scalars=..., cmap=..., opacity=...)`` yields
a :class:`pyvista.Volume` (a ``vtkVolume`` subclass), not a
:class:`pyvista.Actor`. The mesh dispatch in :mod:`scene` skips non-
Actor entries, so volumes are routed here separately.

The bridge avoids the OpenVDB blocker (no pyopenvdb wheels for Python
3.11+) by drawing the volume **inside a closed cube mesh** with a
Cycles Volume shader. Cycles natively treats the interior of a closed
mesh that carries only a Volume output as a continuous medium. Per-
voxel scalar data lives in a 2D atlas image (slices stacked
horizontally); the material's shader graph computes the right atlas
pixel from each shading point's world position. The atlas is packed
into the ``.blend`` via :meth:`bpy.types.Image.pack` so the file
stays self-contained.

Non-ImageData grids (UnstructuredGrid, StructuredGrid,
RectilinearGrid) are resampled to a regular ImageData via
:class:`vtkResampleToImage` before the atlas is baked. The
resampler infills points that fall outside the source dataset with the
nearest value, and the bridge masks those out using the
``vtkValidPointMask`` array so empty space stays empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import bpy
import imageio.v3 as iio
import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkResampleToImage

__all__ = [
    "DEFAULT_RESAMPLE_DIMS",
    "NODE_NAME_COMBINE",
    "NODE_NAME_FRAME_VALUE",
    "NODE_NAME_IMAGE",
    "NODE_NAME_SEP",
    "bake_atlas_image",
    "build_animated_atlas",
    "inject_frame_offset",
    "resolve_array_name",
    "resolve_image_data",
    "resolve_scalar_array",
    "translate_volume",
]

DEFAULT_RESAMPLE_DIMS = (64, 64, 64)
NODE_NAME_COMBINE = "pvb_volume_combine"
NODE_NAME_SEP = "pvb_volume_sep"
NODE_NAME_IMAGE = "pvb_volume_image"
NODE_NAME_FRAME_VALUE = "pvb_volume_frame_value"

_RGBA = tuple[float, float, float, float]


def translate_volume(
    actor: pv.Volume,
    name: str,
    *,
    live_dataset: pv.DataSet | None = None,
) -> bpy.types.Object:
    """Build a bpy ``Object`` whose closed cube interior renders as ``actor``.

    Parameters
    ----------
    actor
        PyVista volume actor. ImageData inputs are used directly;
        other grid types (UnstructuredGrid, StructuredGrid,
        RectilinearGrid) are resampled to ImageData via
        :class:`vtkResampleToImage`.
    name
        Name for the bpy object and material.
    live_dataset
        Optional override for ``actor.mapper.dataset``. PyVista's
        :meth:`pl.add_volume` copies the input grid, so the actor's
        mapper dataset is a snapshot at registration time. The bridge
        accepts the *original* user-owned grid here (registered via
        ``pl.blender.add_volume``) so mutations to ``dataset[scalars]
        = ...`` flow through to the rendered output. ``None`` (default)
        falls back to ``actor.mapper.dataset``.

    Returns
    -------
    bpy.types.Object
        The carrier object linked into the active scene's collection.
        Its mesh is a closed cube spanning the volume's world-space
        bounds; the material drives Volume Principled emission +
        density from a packed atlas image keyed by world position.

    """
    array_name = _resolve_array_name(actor)
    source_dataset = live_dataset if live_dataset is not None else actor.mapper.dataset
    image_data = _ensure_image_data(source_dataset, array_name)
    nx, ny, nz = (int(d) for d in image_data.dimensions)
    bounds = image_data.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    bbox_min = (float(bounds[0]), float(bounds[2]), float(bounds[4]))
    bbox_max = (float(bounds[1]), float(bounds[3]), float(bounds[5]))
    values = _resolve_scalar_array(image_data, array_name, nx, ny, nz)

    image = _bake_volume_atlas(values, name)
    mesh = _build_cube_mesh(name, bbox_min, bbox_max)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    material = _build_volume_material(
        actor,
        name,
        image,
        nx=nx,
        nz=nz,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
    mesh.materials.append(material)
    return obj


def resolve_array_name(actor: pv.Volume) -> str | None:
    """Public alias for :func:`_resolve_array_name`.

    Returns
    -------
    str or None
        Same value as :func:`_resolve_array_name`. Exposed so the
        animation baker can resample volumes the same way the static
        translator does.

    """
    return _resolve_array_name(actor)


def resolve_image_data(
    dataset: pv.DataSet,
    array_name: str | None,
    *,
    sampling_dims: tuple[int, int, int] = DEFAULT_RESAMPLE_DIMS,
) -> pv.ImageData:
    """Public alias for :func:`_ensure_image_data`.

    Returns
    -------
    pv.ImageData
        Same value as :func:`_ensure_image_data`. Exposed so the
        animation baker can build per-frame atlases from the same
        regular grid the static translator uses.

    """
    return _ensure_image_data(dataset, array_name, sampling_dims=sampling_dims)


def resolve_scalar_array(
    image_data: pv.ImageData, array_name: str | None
) -> np.ndarray:
    """Return the ImageData's active scalar field as a ``(nz, ny, nx)`` array.

    Returns
    -------
    np.ndarray
        Float32 ``(nz, ny, nx)`` view. Convenience wrapper that resolves
        the dimensions from the ImageData and dispatches to
        :func:`_resolve_scalar_array`.

    """
    nx, ny, nz = (int(d) for d in image_data.dimensions)
    return _resolve_scalar_array(image_data, array_name, nx, ny, nz)


def bake_atlas_image(values: np.ndarray, name: str) -> bpy.types.Image:
    """Public alias for :func:`_bake_volume_atlas`.

    Returns
    -------
    bpy.types.Image
        Same value as :func:`_bake_volume_atlas`. Exposed so callers
        outside this module can build atlases for arbitrary scalar
        fields (e.g. per-frame animation bakes).

    """
    return _bake_volume_atlas(values, name)


def build_animated_atlas(
    frames: list[np.ndarray], name: str
) -> tuple[bpy.types.Image, int]:
    """Stack per-frame ``(nz, ny, nx)`` scalar arrays into one atlas image.

    Frames are stacked vertically — each frame contributes an
    ``(ny, nx * nz)`` slice band, so the final atlas is shape
    ``(ny * n_frames, nx * nz)``. The shader graph picks the right
    band using a keyframed Value node (see :func:`inject_frame_offset`).

    Parameters
    ----------
    frames
        Per-frame scalar grids, all with the same ``(nz, ny, nx)``
        shape. Frame order in the list is the order along the atlas
        V axis.
    name
        Base name for the packed image data-block.

    Returns
    -------
    tuple of (bpy.types.Image, int)
        ``(image, n_frames)``. The image is packed inside the .blend
        via :meth:`bpy.types.Image.pack`. ``n_frames`` is returned for
        the shader graph builder's convenience.

    Raises
    ------
    ValueError
        When ``frames`` is empty or per-frame shapes disagree.

    """
    if not frames:
        msg = "build_animated_atlas: empty frame list"
        raise ValueError(msg)
    shape0 = frames[0].shape
    if any(f.shape != shape0 for f in frames[1:]):
        msg = (
            f"build_animated_atlas: per-frame shapes disagree "
            f"({[tuple(f.shape) for f in frames]!r})"
        )
        raise ValueError(msg)

    stacked = np.stack(frames, axis=0)  # (n_frames, nz, ny, nx)
    normalized = _normalize_with_nan(stacked)
    atlas = _pack_animated_atlas(normalized)
    byte_image = (atlas * 255.0).astype(np.uint8)

    tmp_path = Path(bpy.app.tempdir) / f"pvblender_volume_{name}.png"
    iio.imwrite(tmp_path, byte_image)
    existing = bpy.data.images.get(tmp_path.name)
    if existing is not None:
        bpy.data.images.remove(existing)
    image = bpy.data.images.load(str(tmp_path))
    image.pack()
    tmp_path.unlink(missing_ok=True)
    return image, normalized.shape[0]


def _normalize_with_nan(stacked: np.ndarray) -> np.ndarray:
    """Rescale ``stacked`` to ``[0, 1]`` per the global (NaN-ignoring) range.

    Returns
    -------
    np.ndarray
        Same shape as ``stacked``; NaNs replaced with the minimum so
        they read as the empty end of the opacity transfer function.

    """
    if np.all(np.isnan(stacked)):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(stacked))
        vmax = float(np.nanmax(stacked))
    if not (vmax > vmin):
        vmax = vmin + 1.0
    safe = np.where(np.isnan(stacked), vmin, stacked)
    return np.clip((safe - vmin) / (vmax - vmin), 0.0, 1.0)


def _pack_animated_atlas(normalized: np.ndarray) -> np.ndarray:
    """Pack ``(n_frames, nz, ny, nx)`` into a ``(ny*n_frames, nx*nz)`` atlas.

    Returns
    -------
    np.ndarray
        Float32 atlas; frame ``f`` occupies rows
        ``[f*ny : (f+1)*ny]`` and slice ``k`` occupies columns
        ``[k*nx : (k+1)*nx]``.

    """
    n_frames, nz, ny, nx = normalized.shape
    atlas = np.zeros((ny * n_frames, nx * nz), dtype=np.float32)
    for f in range(n_frames):
        for k in range(nz):
            atlas[f * ny : (f + 1) * ny, k * nx : (k + 1) * nx] = normalized[f, k]
    return atlas


def inject_frame_offset(
    material: bpy.types.Material,
    n_frames: int,
) -> bpy.types.Node:
    """Splice a keyframable frame-index offset into a volume material's V-coord.

    The static material wires ``sep.Y → combine.inputs[1]`` (atlas V =
    local Y). For animation, the atlas grows vertically by a factor of
    ``n_frames``, so atlas V must become
    ``(frame_index + local_y) / n_frames``. This helper finds the
    existing ``sep`` and ``combine`` nodes (placed under stable names
    by :func:`_build_volume_material`), inserts the offset math, and
    returns the new ``ShaderNodeValue`` so the caller can keyframe
    ``value`` per frame.

    Parameters
    ----------
    material
        Volume material previously built by :func:`translate_volume`.
        Its node tree is expected to contain nodes named
        :data:`NODE_NAME_SEP` and :data:`NODE_NAME_COMBINE`.
    n_frames
        Number of frames stacked in the atlas. Used as the divisor.

    Returns
    -------
    bpy.types.Node
        The new ``ShaderNodeValue``. Caller keyframes
        ``node.outputs[0].default_value`` per frame so playback
        scrolls through the atlas.

    Raises
    ------
    RuntimeError
        When the named ``sep`` or ``combine`` nodes are missing from
        the material's node tree (defensive — the static path always
        creates both).

    """
    nt = material.node_tree
    if nt is None:
        msg = f"material {material.name!r} has no node_tree"
        raise RuntimeError(msg)
    sep = nt.nodes.get(NODE_NAME_SEP)
    combine = nt.nodes.get(NODE_NAME_COMBINE)
    if sep is None or combine is None:
        msg = (
            f"material {material.name!r} is missing the volume shader skeleton "
            f"(sep={sep!r}, combine={combine!r}); was it built by translate_volume?"
        )
        raise RuntimeError(msg)

    # Drop the existing direct sep.Y → combine.V link.
    for link in list(nt.links):
        if link.to_node is combine and link.to_socket is combine.inputs[1]:
            nt.links.remove(link)

    value_node = nt.nodes.new("ShaderNodeValue")
    value_node.name = NODE_NAME_FRAME_VALUE
    value_node.label = "Frame index"
    value_node.location = (-700, 200)

    add = _math(nt, "ADD")
    add.location = (-500, 150)
    nt.links.new(value_node.outputs[0], add.inputs[0])
    nt.links.new(sep.outputs["Y"], add.inputs[1])

    divide = _math_const(nt, "DIVIDE", float(n_frames))
    divide.location = (-400, 150)
    nt.links.new(add.outputs[0], divide.inputs[0])
    nt.links.new(divide.outputs[0], combine.inputs[1])
    return value_node


def _resolve_array_name(actor: pv.Volume) -> str | None:
    """Pick the array name the actor's mapper renders.

    Returns
    -------
    str or None
        The active mapper array name, falling back to the dataset's
        ``active_scalars_name``. ``None`` when neither resolves.

    """
    dataset = actor.mapper.dataset
    return getattr(actor.mapper, "array_name", None) or dataset.active_scalars_name


def _ensure_image_data(
    dataset: pv.DataSet,
    array_name: str | None,
    *,
    sampling_dims: tuple[int, int, int] = DEFAULT_RESAMPLE_DIMS,
) -> pv.ImageData:
    """Return ``dataset`` if it is ImageData, else resample to a regular grid.

    The resampler infills points that fall outside the source mesh's
    cells with the nearest value; the bridge masks those points to
    NaN using the auxiliary ``vtkValidPointMask`` array so the cube's
    interior reads as empty space outside the source's footprint.

    Parameters
    ----------
    dataset
        The source dataset attached to the actor's mapper.
    array_name
        The scalar array the bridge will render. Used to drive the
        resampler so VTK doesn't drop it.
    sampling_dims
        Output ImageData dimensions ``(nx, ny, nz)``. Defaults to a
        64-cube; tune up for sharper interior detail at the cost of
        atlas memory.

    Returns
    -------
    pv.ImageData
        Either the input itself (zero-copy) or a freshly resampled
        regular grid carrying ``array_name`` in its point data, with
        invalid points replaced by NaN.

    """
    if isinstance(dataset, pv.ImageData):
        return dataset

    resampler = vtkResampleToImage()
    resampler.SetInputDataObject(dataset)
    resampler.SetSamplingDimensions(*sampling_dims)
    resampler.SetUseInputBounds(True)  # noqa: FBT003
    resampler.Update()
    image = cast("pv.ImageData", pv.wrap(resampler.GetOutput()))

    if array_name is not None and array_name in image.point_data:
        mask = np.asarray(
            image.point_data.get("vtkValidPointMask", np.ones(image.n_points)),
            dtype=bool,
        )
        scalars = np.asarray(image.point_data[array_name], dtype=np.float32).copy()
        scalars[~mask] = np.nan
        image.point_data[array_name] = scalars
        image.set_active_scalars(array_name)
    return image


def _resolve_scalar_array(
    image_data: pv.ImageData,
    array_name: str | None,
    nx: int,
    ny: int,
    nz: int,
) -> np.ndarray:
    """Return the ImageData's scalar field reshaped to ``(nz, ny, nx)``.

    Returns
    -------
    np.ndarray
        Float32 ``(nz, ny, nx)`` view of the active scalars; invalid
        points (from resampling outside the source bounds) carry NaN.

    Raises
    ------
    ValueError
        When the source scalar array's size doesn't match
        ``nx * ny * nz``. Point-data scalars only.

    """
    if array_name is None or array_name not in image_data.point_data:
        scalars = np.asarray(image_data.active_scalars, dtype=np.float32)
    else:
        scalars = np.asarray(image_data.point_data[array_name], dtype=np.float32)
    expected = nx * ny * nz
    if scalars.size != expected:
        msg = (
            f"volume scalar array has {scalars.size} entries; expected "
            f"{expected} = nx({nx}) * ny({ny}) * nz({nz}). Volume rendering "
            f"requires point-data scalars on a regular ImageData grid."
        )
        raise ValueError(msg)
    return scalars.reshape((nz, ny, nx))


def _build_cube_mesh(
    name: str,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> bpy.types.Mesh:
    """Construct a closed cube spanning ``[bbox_min, bbox_max]``.

    Faces are wound so their normals point outward — Cycles uses the
    surface normal to decide which side is the volume's interior, so
    flipped winding renders the volume as empty space.

    Returns
    -------
    bpy.types.Mesh
        The new mesh data-block, ready to bind to an object.

    """
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    verts = [
        (x0, y0, z0),  # 0
        (x1, y0, z0),  # 1
        (x1, y1, z0),  # 2
        (x0, y1, z0),  # 3
        (x0, y0, z1),  # 4
        (x1, y0, z1),  # 5
        (x1, y1, z1),  # 6
        (x0, y1, z1),  # 7
    ]
    # All faces wound CCW when viewed from outside (normals out).
    faces = [
        (0, 3, 2, 1),  # bottom (-Z)
        (4, 5, 6, 7),  # top (+Z)
        (0, 1, 5, 4),  # front (-Y)
        (2, 3, 7, 6),  # back (+Y)
        (1, 2, 6, 5),  # right (+X)
        (0, 4, 7, 3),  # left (-X)
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _bake_volume_atlas(values: np.ndarray, name: str) -> bpy.types.Image:
    """Pack a 3D scalar grid into a 2D atlas image and return the bpy Image.

    Layout: the ``nz`` slices are stacked horizontally, so the atlas
    has shape ``(ny, nx * nz)``. The shader computes the slice index
    from the normalised world Z, then samples the appropriate column
    band. Single-channel L8 keeps the image small; the colormap is
    applied downstream by a ColorRamp inside the material.

    The image is **packed into the .blend** so the file stays
    self-contained.

    Returns
    -------
    bpy.types.Image
        Freshly loaded + packed image; the temp file on disk has
        already been removed.

    """
    nz, ny, nx = values.shape
    if np.all(np.isnan(values)):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not (vmax > vmin):
        vmax = vmin + 1.0
    # Resampled grids may carry NaN for points outside the source mesh;
    # collapse those to the minimum so they read as the empty end of
    # the opacity transfer function.
    safe_values = np.where(np.isnan(values), vmin, values)
    normalized = np.clip((safe_values - vmin) / (vmax - vmin), 0.0, 1.0)
    atlas = np.zeros((ny, nx * nz), dtype=np.float32)
    for k in range(nz):
        atlas[:, k * nx : (k + 1) * nx] = normalized[k]
    byte_image = (atlas * 255.0).astype(np.uint8)

    tmp_path = Path(bpy.app.tempdir) / f"pvblender_volume_{name}.png"
    iio.imwrite(tmp_path, byte_image)
    existing = bpy.data.images.get(tmp_path.name)
    if existing is not None:
        bpy.data.images.remove(existing)
    image = bpy.data.images.load(str(tmp_path))
    image.pack()
    tmp_path.unlink(missing_ok=True)
    return image


def _build_volume_material(
    actor: pv.Volume,
    name: str,
    image: bpy.types.Image,
    *,
    nx: int,
    nz: int,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> bpy.types.Material:
    """Wire Position → atlas-UV → Image Texture → Volume Principled.

    The shader graph turns the shading point's world position into a
    normalised ``(u, v, w)`` in ``[0, 1]³``, picks the right horizontal
    slice band via ``w``, looks up the atlas, then routes that scalar
    through two ColorRamps (colour + opacity) into Volume Principled.
    Emission Color is driven by the colour ramp so the volume is
    self-lit and visible without scene lighting; the absorption
    ``Color`` carries the same colour so a tasteful tint shows up
    when scene lights *are* present.

    Returns
    -------
    bpy.types.Material
        The new material with its Volume shader graph ready to bind
        to the cube mesh.

    Raises
    ------
    RuntimeError
        When the freshly-created material exposes no ``node_tree``
        (Blender invariant; included for ty narrowing).

    """
    mat = bpy.data.materials.new(f"{name}_vol_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        msg = f"material {mat.name!r} has no node_tree"
        raise RuntimeError(msg)
    nt.nodes.clear()

    color_stops, alpha_stops = _sample_lut(actor)
    density_socket = _build_density_lookup(
        nt,
        image,
        nx=nx,
        nz=nz,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )

    color_ramp = nt.nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (-400, 100)
    _fill_color_ramp(color_ramp, color_stops)
    nt.links.new(density_socket, color_ramp.inputs["Fac"])

    alpha_ramp = nt.nodes.new("ShaderNodeValToRGB")
    alpha_ramp.location = (-400, -150)
    _fill_color_ramp(alpha_ramp, alpha_stops)
    nt.links.new(density_socket, alpha_ramp.inputs["Fac"])

    density_scale = nt.nodes.new("ShaderNodeMath")
    density_scale.operation = "MULTIPLY"
    density_scale.location = (-150, -150)
    density_scale.inputs[1].default_value = _resolve_volume_density_scale(actor)
    nt.links.new(alpha_ramp.outputs["Color"], density_scale.inputs[0])

    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.location = (100, 0)
    nt.links.new(color_ramp.outputs["Color"], vol.inputs["Color"])
    nt.links.new(color_ramp.outputs["Color"], vol.inputs["Emission Color"])
    nt.links.new(density_scale.outputs[0], vol.inputs["Density"])
    vol.inputs["Emission Strength"].default_value = 1.0

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    return mat


def _build_density_lookup(
    nt: bpy.types.NodeTree,
    image: bpy.types.Image,
    *,
    nx: int,
    nz: int,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> bpy.types.NodeSocket:
    """Wire Position → normalised UVW → atlas-UV → Image Texture.

    The atlas packs ``nz`` horizontal slices of size ``(ny, nx)``, so
    the final atlas U is ``(slice_index + u) / nz`` and atlas V is
    just the local-Y component. The slice index is ``floor(w * nz)``
    clamped to ``[0, nz-1]``.

    Returns
    -------
    bpy.types.NodeSocket
        The Image Texture's ``Color`` output (a scalar in ``[0, 1]``
        since the atlas is single-channel L8); ready to feed
        downstream ColorRamps.

    """
    span = (
        max(bbox_max[0] - bbox_min[0], 1e-9),
        max(bbox_max[1] - bbox_min[1], 1e-9),
        max(bbox_max[2] - bbox_min[2], 1e-9),
    )
    position = nt.nodes.new("ShaderNodeNewGeometry")
    position.location = (-1400, 0)

    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.location = (-1200, 0)
    sub.inputs[1].default_value = bbox_min
    nt.links.new(position.outputs["Position"], sub.inputs[0])

    div = nt.nodes.new("ShaderNodeVectorMath")
    div.operation = "DIVIDE"
    div.location = (-1000, 0)
    div.inputs[1].default_value = span
    nt.links.new(sub.outputs[0], div.inputs[0])

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-800, 0)
    nt.links.new(div.outputs[0], sep.inputs[0])

    # k = floor(w * nz), clamped to [0, nz - 1].
    w_scaled = _math_const(nt, "MULTIPLY", float(nz))
    w_scaled.location = (-600, -150)
    nt.links.new(sep.outputs["Z"], w_scaled.inputs[0])
    w_floor = _math_unary(nt, "FLOOR")
    w_floor.location = (-450, -150)
    nt.links.new(w_scaled.outputs[0], w_floor.inputs[0])
    w_clamp = _math_const(nt, "MINIMUM", float(nz - 1))
    w_clamp.location = (-300, -150)
    nt.links.new(w_floor.outputs[0], w_clamp.inputs[0])

    # Atlas U = (slice_index + local_u) / nz; atlas V = local_v.
    u_offset = _math(nt, "ADD")
    u_offset.location = (-600, 0)
    nt.links.new(w_clamp.outputs[0], u_offset.inputs[0])
    nt.links.new(sep.outputs["X"], u_offset.inputs[1])
    u_atlas = _math_const(nt, "DIVIDE", float(nz))
    u_atlas.location = (-450, 0)
    nt.links.new(u_offset.outputs[0], u_atlas.inputs[0])

    combine = nt.nodes.new("ShaderNodeCombineXYZ")
    combine.location = (-300, 0)
    combine.name = NODE_NAME_COMBINE
    nt.links.new(u_atlas.outputs[0], combine.inputs[0])
    nt.links.new(sep.outputs["Y"], combine.inputs[1])

    sep.name = NODE_NAME_SEP

    img_node = cast("bpy.types.ShaderNodeTexImage", nt.nodes.new("ShaderNodeTexImage"))
    img_node.location = (-150, 0)
    img_node.name = NODE_NAME_IMAGE
    img_node.image = image
    img_node.interpolation = "Linear"
    img_node.extension = "EXTEND"
    nt.links.new(combine.outputs[0], img_node.inputs["Vector"])
    del nx  # name kept in signature for the comment + future use
    return img_node.outputs["Color"]


def _sample_lut(
    actor: pv.Volume,
) -> tuple[list[tuple[float, _RGBA]], list[tuple[float, _RGBA]]]:
    """Sample PyVista's volume LUT into colour and opacity ramp stops.

    PyVista bakes the colormap **and** the opacity transfer function
    into ``vol.mapper.lookup_table`` as a 256-entry RGBA table — the
    RGB channels carry the colormap colour, the A channel carries the
    opacity transfer function. The matplotlib ``cmap`` attribute is
    ``None`` for volume LUTs (unlike the actor case), so we read the
    baked table directly via :meth:`vtkLookupTable.GetTableValue`.

    Returns
    -------
    tuple
        ``(color_stops, alpha_stops)``; each a list of
        ``(position, (r, g, b, alpha))`` tuples ready for
        :func:`_fill_color_ramp`.

    """
    lut = actor.mapper.lookup_table
    n_table = int(lut.GetNumberOfTableValues())
    n_stops = min(n_table, 16)
    indices = np.linspace(0, n_table - 1, n_stops).astype(int)
    color_stops: list[tuple[float, _RGBA]] = []
    alpha_stops: list[tuple[float, _RGBA]] = []
    for i, idx in enumerate(indices):
        position = float(i / max(n_stops - 1, 1))
        r, g, b, a = lut.GetTableValue(int(idx))
        color_stops.append((position, (float(r), float(g), float(b), 1.0)))
        alpha_stops.append((position, (float(a), float(a), float(a), 1.0)))
    return color_stops, alpha_stops


def _fill_color_ramp(
    ramp_node: bpy.types.Node,
    stops: list[tuple[float, _RGBA]],
) -> None:
    """Replace ``ramp_node``'s elements with ``stops``."""
    ramp = cast("bpy.types.ColorRamp", ramp_node.color_ramp)  # type: ignore[attr-defined]
    elements = ramp.elements
    while len(elements) > 1:
        elements.remove(elements[-1])
    if not stops:
        return
    first_pos, first_col = stops[0]
    elements[0].position = first_pos
    elements[0].color = first_col
    for pos, col in stops[1:]:
        elem = elements.new(pos)
        elem.color = col


def _resolve_volume_density_scale(actor: pv.Volume) -> float:
    """Pick a density multiplier that produces a visible (but not opaque) cloud.

    Volume Principled's Density is in inverse-distance-along-ray units;
    a value around 4-8 reads as a translucent cloud for unit-sized
    bounds. ``opacity_unit_distance`` from PyVista lets advanced users
    tune the falloff; we invert it so a smaller unit distance produces
    a denser cloud, matching PyVista's preview semantics.

    Returns
    -------
    float
        The multiplier the material's density-side ``MULTIPLY`` math
        node will apply to the alpha ramp output.

    """
    return 4.0 / max(float(getattr(actor.prop, "opacity_unit_distance", 1.0)), 1e-3)


def _math_const(nt: bpy.types.NodeTree, operation: str, const: float) -> bpy.types.Node:
    """Add a math node with the second operand baked to ``const``.

    Returns
    -------
    bpy.types.Node
        The new ``ShaderNodeMath``; input ``[0]`` is free for the
        caller to wire.

    """
    node = nt.nodes.new("ShaderNodeMath")
    node.operation = operation
    node.inputs[1].default_value = const
    return node


def _math(nt: bpy.types.NodeTree, operation: str) -> bpy.types.Node:
    """Add a binary math node leaving both inputs free for the caller to wire.

    Returns
    -------
    bpy.types.Node
        The new ``ShaderNodeMath``; both inputs are free.

    """
    node = nt.nodes.new("ShaderNodeMath")
    node.operation = operation
    return node


def _math_unary(nt: bpy.types.NodeTree, operation: str) -> bpy.types.Node:
    """Add a single-operand math node (used for FLOOR / ABS / SIGN).

    Returns
    -------
    bpy.types.Node
        The new ``ShaderNodeMath``; only input ``[0]`` is meaningful.

    """
    node = nt.nodes.new("ShaderNodeMath")
    node.operation = operation
    return node
