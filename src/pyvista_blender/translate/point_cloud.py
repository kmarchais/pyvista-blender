# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate a PyVista point-style actor into a Cycles ``PointCloud`` object.

PyVista's ``style="points"`` and ``style="points_gaussian"`` render the
actor's dataset as discrete points rather than as a triangulated
surface. The bridge maps both to Blender's native
:class:`bpy.types.PointCloud` primitive, which Cycles draws as
per-point spheres without the overhead of explicit instancing — render
cost scales with ``N_points`` instead of ``N_points * N_geom_verts``.

Two render modes, distinguished by the actor's mapper:

* ``style="points"`` (``vtkDataSetMapper`` with ``prop.style="Points"``)
  → opaque Principled BSDF spheres coloured by the actor's flat
  colour or by the active scalar field (per-point ``COLOR POINT``
  attribute).
* ``style="points_gaussian"`` (``vtkPointGaussianMapper``) → Transparent
  BSDF mixed with Emission via a Gaussian alpha falloff sampled inside
  each point's bounding sphere. Each point becomes a soft splat — the
  conventional rendering for particle systems / molecular surfaces /
  Gaussian Splatting datasets.

PyVista's ``prop.point_size`` is in screen pixels; Cycles renders
point clouds in world-space radius. The bridge converts via
:data:`POINT_SIZE_TO_WORLD_RADIUS`, a conservative pixel-to-world
factor tuned so the default ``point_size=5`` reads as a small but
visible sphere on unit-bound data. Users can override via the actor
property or by mutating the radius attribute on the returned
PointCloud after the bridge builds it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import bpy
import matplotlib as mpl
import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["translate_point_cloud"]

#: Pixel → world-space radius conversion factor.
#:
#: PyVista's ``point_size`` is an integer in screen pixels (3 - 20 is
#: typical). Cycles' PointCloud uses world-space radii. There's no
#: camera-independent exact conversion, but the bridge picks a factor
#: that produces a visible splat for unit-bound data at the default
#: ``point_size=5``. Users with very small or very large coordinate
#: extents can override via the radius attribute on the returned
#: PointCloud or via :class:`pyvista_blender.config`.
POINT_SIZE_TO_WORLD_RADIUS = 0.005


def translate_point_cloud(
    actor: pv.Actor,
    name: str,
    *,
    mode: str,
) -> bpy.types.Object:
    """Build a bpy ``Object`` whose ``PointCloud`` data renders as ``actor``.

    Parameters
    ----------
    actor
        PyVista actor whose ``prop.style == "Points"`` (or whose mapper
        is a ``vtkPointGaussianMapper`` for the gaussian variant).
    name
        Name for the bpy object, point-cloud data-block, and material.
    mode
        ``"points"`` (opaque sphere shader) or ``"gaussian"`` (gaussian-
        falloff splat shader). Selected upstream by :mod:`scene` based
        on the actor's mapper class.

    Returns
    -------
    bpy.types.Object
        The point-cloud object linked into the active scene's
        collection. Its data is a :class:`bpy.types.PointCloud`
        whose ``position`` + ``radius`` attributes carry per-point
        positions and radii, with an optional ``scalars`` Color
        attribute when the actor renders a scalar field.

    Raises
    ------
    ValueError
        When ``mode`` is not one of ``"points"`` / ``"gaussian"``.

    """
    if mode not in {"points", "gaussian"}:
        msg = f"point cloud mode {mode!r} must be 'points' or 'gaussian'"
        raise ValueError(msg)

    dataset = actor.mapper.dataset
    points = np.asarray(dataset.points, dtype=np.float32)
    n_points = points.shape[0]

    pc = _build_point_cloud_data(name, points, actor)
    rgba = _resolve_scalar_colors(dataset, actor.mapper, n_points)
    if rgba is not None:
        _attach_color_attribute(pc, rgba)

    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)

    material = _build_point_cloud_material(
        actor, name, mode=mode, has_scalars=rgba is not None
    )
    pc.materials.append(material)
    return obj


def _build_point_cloud_data(
    name: str,
    points: np.ndarray,
    actor: pv.Actor,
) -> bpy.types.PointCloud:
    """Allocate a PointCloud data-block and set per-point positions + radii.

    Returns
    -------
    bpy.types.PointCloud
        The freshly created data-block, sized to ``points.shape[0]``,
        with ``position`` populated and a ``radius`` attribute set
        from the actor's ``point_size`` (optionally scaled per-point
        when the mapper carries a ``scale_array``).

    """
    pc = cast("bpy.types.PointCloud", bpy.data.pointclouds.new(name=name))
    n_points = points.shape[0]
    pc.resize(n_points)
    pc.attributes["position"].data.foreach_set("vector", points.ravel())

    radius_attr = pc.attributes.get("radius") or pc.attributes.new(
        name="radius", type="FLOAT", domain="POINT"
    )
    radii = _resolve_per_point_radii(actor, n_points)
    radius_attr.data.foreach_set("value", radii)
    return pc


def _resolve_per_point_radii(actor: pv.Actor, n_points: int) -> np.ndarray:
    """Resolve the per-point world-space radius array.

    Starts from a uniform value derived from ``prop.point_size`` and
    multiplies by a per-point factor when the mapper's ``scale_array``
    names a point-data field. PyVista's :class:`PointGaussianMapper`
    exposes a ``scale_array`` (default ``None``) that VTK uses to size
    each splat; the bridge mirrors that contract so users who set
    ``actor.mapper.scale_array = "my_field"`` get per-point sizing for
    free.

    Returns
    -------
    np.ndarray
        Float32 ``(n_points,)`` array of world-space radii.

    """
    point_size = float(getattr(actor.prop, "point_size", 5.0))
    base_radius = point_size * POINT_SIZE_TO_WORLD_RADIUS

    scale_array_name = getattr(actor.mapper, "scale_array", None)
    dataset = actor.mapper.dataset
    if (
        scale_array_name
        and dataset is not None
        and scale_array_name in dataset.point_data
    ):
        raw = np.asarray(dataset.point_data[scale_array_name], dtype=np.float32)
        if raw.size == n_points:
            # VTK's PointGaussianMapper multiplies the splat size by
            # ``scale_factor * scale_array_value``; for the bridge a
            # symmetric default is "treat scale_array values as a
            # per-point multiplier on the base radius, normalized so
            # the array's mean reads as the unscaled splat size".
            mean = float(np.mean(np.abs(raw)))
            if mean > 0.0:
                multipliers = np.abs(raw) / mean
                return (base_radius * multipliers).astype(np.float32)
    return np.full(n_points, base_radius, dtype=np.float32)


def _attach_color_attribute(pc: bpy.types.PointCloud, rgba: np.ndarray) -> None:
    """Add a POINT-domain ``"scalars"`` color attribute carrying per-point RGBA."""
    attr = pc.color_attributes.new(name="scalars", type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", rgba.ravel())


def _resolve_scalar_colors(
    dataset: pv.DataSet,
    mapper: object,
    n_points: int,
) -> np.ndarray | None:
    """Bake the active scalar field through the LUT into per-point RGBA.

    Mirrors :func:`pyvista_blender.translate.mesh._resolve_scalar_colors`
    but is constrained to POINT-domain scalars — the point cloud has no
    "cell" concept, so cell-data scalars are not surfaced here.

    Returns
    -------
    np.ndarray or None
        ``(n_points, 4)`` float32 RGBA in linear space, or ``None``
        when the mapper has no usable scalar visibility or the active
        array doesn't live on point data.

    """
    if not getattr(mapper, "scalar_visibility", False):
        return None

    # PointGaussianMapper precomputes per-point RGBA bytes (uint8 0-255)
    # via its own colormap pass and stamps them onto the dataset as the
    # ``__rgba__`` point-data array; reading those directly skips the
    # bridge's own colormap baking and reproduces VTK's exact look.
    # Skip if the precomputed buffer is malformed (wrong size / dtype).
    precomputed = _try_read_precomputed_rgba(dataset, n_points)
    if precomputed is not None:
        return precomputed

    # Fall back to the actor's nominal scalar field (DataSetMapper
    # path: array_name carries the user-facing name) through the
    # cmap → RGBA pipeline.
    candidates = [
        n
        for n in (
            getattr(mapper, "array_name", None),
            getattr(dataset, "active_scalars_name", None),
        )
        if n and n != "__rgba__"
    ]
    array_name = next(
        (n for n in candidates if n in dataset.point_data),
        None,
    )
    if array_name is None:
        return None

    raw = np.asarray(dataset.point_data[array_name], dtype=np.float32)
    if raw.size != n_points:
        return None

    lo, hi = getattr(mapper, "scalar_range", (0.0, 1.0))
    if hi <= lo:
        return None

    normalized = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    cmap_name = (
        getattr(getattr(mapper, "lookup_table", None), "cmap", None) or "viridis"
    )
    cmap = mpl.colormaps.get_cmap(cmap_name)
    rgba_srgb = np.asarray(cmap(normalized), dtype=np.float32)
    return _srgb_to_linear(rgba_srgb)


def _try_read_precomputed_rgba(dataset: pv.DataSet, n_points: int) -> np.ndarray | None:
    """Return ``dataset.point_data["__rgba__"]`` as linear RGBA when present.

    VTK's :class:`vtkPointGaussianMapper` stamps a uint8 (n, 4) array
    onto the dataset's point data under the reserved name
    ``__rgba__`` — its own colormap output, baked once per render.
    Reading it directly reproduces VTK's preview look exactly and
    skips a redundant colormap pass on the bridge side.

    Returns
    -------
    np.ndarray or None
        ``(n_points, 4)`` float32 RGBA in linear space, or ``None``
        when the buffer is missing or its shape doesn't match.

    """
    raw = dataset.point_data.get("__rgba__")
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.ndim == 2:  # noqa: PLR2004
        if arr.shape != (n_points, 4):
            return None
        flat = arr
    elif arr.ndim == 1 and arr.size == n_points * 4:
        flat = arr.reshape(n_points, 4)
    else:
        return None
    # VTK ships ``__rgba__`` as either uint8 in [0, 255] or float in
    # [0, 1] depending on the mapper configuration; detect by dtype
    # rather than range (a uint8 buffer happening to land in [0, 1]
    # would be all-black anyway).
    if np.issubdtype(arr.dtype, np.integer):
        rgba_srgb = flat.astype(np.float32) / 255.0
    else:
        rgba_srgb = flat.astype(np.float32)
    return _srgb_to_linear(rgba_srgb)


def _srgb_to_linear(rgba_srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB-encoded RGBA values to Cycles' linear working space.

    Returns
    -------
    np.ndarray
        Same shape as ``rgba_srgb``, alpha channel passed through
        unmodified.

    """
    rgb = rgba_srgb[..., :3]
    a = rgba_srgb[..., 3:4]
    threshold = 0.04045
    low = rgb / 12.92
    high = np.power((rgb + 0.055) / 1.055, 2.4)
    linear_rgb = np.where(rgb <= threshold, low, high).astype(np.float32)
    return np.concatenate([linear_rgb, a], axis=-1)


def _build_point_cloud_material(
    actor: pv.Actor,
    name: str,
    *,
    mode: str,
    has_scalars: bool,
) -> bpy.types.Material:
    """Build the Cycles material for a point-cloud actor.

    Returns
    -------
    bpy.types.Material
        Material with a Principled BSDF (``mode="points"``) or a
        Transparent + Emission mix with gaussian alpha
        (``mode="gaussian"``).

    Raises
    ------
    RuntimeError
        When the freshly-created material exposes no ``node_tree``
        (Blender invariant; included for ty narrowing).

    """
    mat_name = f"{name}_points_mat" if mode == "points" else f"{name}_gauss_mat"
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        msg = f"material {mat.name!r} has no node_tree"
        raise RuntimeError(msg)
    nt.nodes.clear()

    if mode == "points":
        _build_opaque_point_shader(nt, actor, has_scalars=has_scalars)
    else:
        _build_gaussian_splat_shader(nt, actor, has_scalars=has_scalars)
    return mat


def _build_opaque_point_shader(
    nt: bpy.types.NodeTree,
    actor: pv.Actor,
    *,
    has_scalars: bool,
) -> None:
    """Wire the foreground shader for ``style="points"``.

    The branch depends on :attr:`pv.Property.render_points_as_spheres`,
    which mirrors VTK's flag of the same name:

    * ``True`` — render as PBR-shaded spheres (Principled BSDF), lit
      by scene lights. Visually identical to small Sphere actors.
    * ``False`` (PyVista default) — render as flat self-emissive dots
      (Emission shader). VTK's GL preview draws unshaded square
      sprites for this mode; the bridge approximates with a uniform
      Cycles emission so the points read the same colour everywhere
      without depending on scene lighting.
    """
    render_as_spheres = bool(getattr(actor.prop, "render_points_as_spheres", False))

    if render_as_spheres:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        _wire_color_input(nt, bsdf.inputs["Base Color"], actor, has_scalars=has_scalars)
        bsdf.inputs["Roughness"].default_value = 0.5
        foreground = bsdf.outputs["BSDF"]
    else:
        emission = nt.nodes.new("ShaderNodeEmission")
        emission.location = (0, 0)
        _wire_color_input(nt, emission.inputs["Color"], actor, has_scalars=has_scalars)
        emission.inputs["Strength"].default_value = 1.0
        foreground = emission.outputs["Emission"]

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    nt.links.new(foreground, out.inputs["Surface"])


def _build_gaussian_splat_shader(
    nt: bpy.types.NodeTree,
    actor: pv.Actor,
    *,
    has_scalars: bool,
) -> None:
    """Wire the foreground shader for ``style="points_gaussian"``.

    Two visual modes, branched on whether the user opted into the
    "hard sphere" splat variant. PyVista signals this through
    :meth:`vtkPointGaussianMapper.use_circular_splat` (called when
    ``add_mesh(..., style="points_gaussian", render_points_as_spheres=True)``);
    that method sets a non-``None`` fragment shader code on the
    mapper, which the bridge reads via ``GetSplatShaderCode()``.
    PyVista also honours ``prop.render_points_as_spheres=True``
    directly when the user sets it without the ``add_mesh`` shortcut,
    so both signals route to the same hard-sphere path.

    * **Hard sphere mode** — no alpha falloff, no transparent mix.
      The foreground shader feeds straight into the material output.
      Matches PyVista's GL look with ``render_points_as_spheres=True``
      (crisp PBR-shaded balls).
    * **Soft splat mode** (default) — foreground shader mixed with a
      Transparent BSDF through the camera-facing falloff
      ``max(0, N · V)^k``. Reads as a soft camera-facing blob,
      matching PyVista's default ``style="points_gaussian"`` sprite
      look.

    ``mapper.emissive`` controls the foreground in both modes:

    * ``False`` (PyVista default) → Principled BSDF (scene-lit).
    * ``True`` → Emission shader (self-lit additive blobs).
    """
    emissive = bool(getattr(actor.mapper, "emissive", False))
    if _wants_hard_sphere_splat(actor):
        foreground = _build_splat_foreground(
            nt, actor, emissive=emissive, has_scalars=has_scalars
        )
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (650, 0)
        nt.links.new(foreground, out.inputs["Surface"])
        return

    alpha_socket = _wire_camera_facing_falloff(nt)
    foreground = _build_splat_foreground(
        nt, actor, emissive=emissive, has_scalars=has_scalars
    )
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (200, -100)

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (450, 0)
    nt.links.new(alpha_socket, mix.inputs["Fac"])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    nt.links.new(foreground, mix.inputs[2])

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (650, 0)
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def _wants_hard_sphere_splat(actor: pv.Actor) -> bool:
    """Return True when the gaussian actor should render as opaque spheres.

    PyVista's ``add_mesh(..., style="points_gaussian",
    render_points_as_spheres=True)`` doesn't set
    ``prop.render_points_as_spheres`` — it calls
    :meth:`vtkPointGaussianMapper.use_circular_splat`, which installs
    a custom fragment shader on the mapper. Reading
    ``GetSplatShaderCode()`` lets the bridge detect that intent and
    skip the soft-splat alpha falloff. Users who flip the property
    directly (without the ``add_mesh`` shortcut) get the same
    behaviour via the fallback check on the prop.

    Returns
    -------
    bool
        ``True`` when either signal indicates the hard-sphere variant.

    """
    mapper = actor.mapper
    get_splat = getattr(mapper, "GetSplatShaderCode", None)
    if callable(get_splat) and get_splat() is not None:
        return True
    return bool(getattr(actor.prop, "render_points_as_spheres", False))


def _build_splat_foreground(
    nt: bpy.types.NodeTree,
    actor: pv.Actor,
    *,
    emissive: bool,
    has_scalars: bool,
) -> bpy.types.NodeSocket:
    """Build the lit-or-emissive foreground shader for the gaussian splat mix.

    Returns
    -------
    bpy.types.NodeSocket
        The shader output socket to feed into the Mix Shader's bright
        input.

    """
    if emissive:
        emission = nt.nodes.new("ShaderNodeEmission")
        emission.location = (200, 100)
        _wire_color_input(nt, emission.inputs["Color"], actor, has_scalars=has_scalars)
        # Above-unit emission compensates for the camera-facing alpha
        # falloff that fades each splat toward its rim — without it,
        # the splats read dimmer than VTK's flat additive blend.
        emission.inputs["Strength"].default_value = 2.0
        return emission.outputs["Emission"]

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (200, 100)
    _wire_color_input(nt, bsdf.inputs["Base Color"], actor, has_scalars=has_scalars)
    # Diffuse-like splats with a touch of roughness so scene lights
    # land without specular hot-spots that would defeat the soft look.
    bsdf.inputs["Roughness"].default_value = 0.7
    return bsdf.outputs["BSDF"]


def _wire_camera_facing_falloff(
    nt: bpy.types.NodeTree,
) -> bpy.types.NodeSocket:
    """Build ``max(0, N · V)^k`` from the shading normal and incoming ray.

    For a Cycles point-cloud sphere the shading normal points
    radially outward from each point's centre, so the dot with the
    view direction is 1 at the splat's centre (as the camera sees it)
    and 0 at the silhouette. Raising to the fourth power compresses
    the falloff into a tighter, gaussian-like splat.

    Returns
    -------
    bpy.types.NodeSocket
        Math-node output carrying the alpha factor in ``[0, 1]``.

    """
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-800, 0)

    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.location = (-600, 0)
    nt.links.new(geom.outputs["Normal"], dot.inputs[0])
    nt.links.new(geom.outputs["Incoming"], dot.inputs[1])

    # max(0, dot) — silhouette pixels can carry negative values when
    # the surface normal flips into the camera; clamping avoids
    # the resulting non-monotonic alpha curve.
    clamp = nt.nodes.new("ShaderNodeMath")
    clamp.operation = "MAXIMUM"
    clamp.location = (-400, 0)
    clamp.inputs[1].default_value = 0.0
    nt.links.new(dot.outputs["Value"], clamp.inputs[0])

    power = nt.nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    power.location = (-200, 0)
    power.inputs[1].default_value = 4.0
    nt.links.new(clamp.outputs[0], power.inputs[0])
    return power.outputs[0]


def _wire_color_input(
    nt: bpy.types.NodeTree,
    socket: bpy.types.NodeSocket,
    actor: pv.Actor,
    *,
    has_scalars: bool,
) -> None:
    """Feed the actor's scalar attribute or flat colour into ``socket``."""
    if has_scalars:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "scalars"
        attr.location = (socket.node.location.x - 250, socket.node.location.y)
        nt.links.new(attr.outputs["Color"], socket)
    else:
        r, g, b = actor.prop.color.float_rgb
        socket.default_value = (r, g, b, 1.0)
