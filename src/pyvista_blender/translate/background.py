# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate the renderer's background / environment into the bpy World shader.

Three branches keyed off PyVista's renderer state:

* **Solid colour** (``set_background(color)``) — single ``Background`` node
  fed by the linearised colour.
* **Vertical gradient** (``set_background(bottom, top=...)``) — a
  ``TexCoord -> Mapping -> Gradient -> MixRGB -> Background`` chain. The
  ``Generated`` v-axis goes 0 at the bottom and 1 at the top of the
  framed view, matching VTK's gradient orientation.
* **Environment texture** (``set_environment_texture(...)``) — the
  ``vtkTexture``'s image data is written to a temporary EXR file and
  loaded via ``ShaderNodeTexEnvironment``. Cycles uses it for both
  camera-visible background *and* IBL on PBR materials, so the world
  shader is left visible to glossy / diffuse rays in this branch
  (mirrors VTK's ``UseImageBasedLighting`` behaviour).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import bpy
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["translate_background"]


def translate_background(source: pv.BasePlotter | object) -> None:
    """Configure the World shader from a plotter's active renderer or a renderer.

    Parameters
    ----------
    source
        Either a :class:`pyvista.BasePlotter` (uses
        ``plotter.renderer``) or a ``pv.Renderer`` directly (used by
        the subplot tile path so each tile gets its own background:
        solid / gradient / HDRI as configured per subplot).

    Notes
    -----
    ``read_factory_settings(use_empty=True)`` leaves the scene without a
    World data block, so we create one when needed.

    """
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.location = (300, 0)

    # If ``source`` exposes a ``renderer`` attribute we assume it's a
    # plotter and grab the active renderer; otherwise assume ``source``
    # is already a renderer (subplot tile path).
    renderer = source.renderer if hasattr(source, "renderer") else source  # type: ignore[attr-defined]
    env_tex = _renderer_env_texture(renderer)
    if env_tex is not None:
        bg_socket = _build_environment(tree, env_tex)
        _allow_world_in_shading_rays(world)
    elif _renderer_has_gradient(renderer):
        bg_socket = _build_gradient(tree, renderer)
        _hide_world_from_shading_rays(world)
    else:
        bg_socket = _build_solid(tree, renderer)
        _hide_world_from_shading_rays(world)

    if bg_socket is not None:
        tree.links.new(bg_socket, output.inputs["Surface"])


def _build_solid(
    tree: bpy.types.NodeTree, renderer: object
) -> bpy.types.NodeSocket | None:
    """Build a single Background node fed by the renderer's solid colour.

    Returns
    -------
    bpy.types.NodeSocket or None
        The Background node's output; ``None`` if no background colour is set.

    """
    bg_color = getattr(renderer, "background_color", None)
    if bg_color is None:
        return None
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.location = (0, 0)
    bg.inputs["Color"].default_value = (*_srgb_to_linear_rgb(bg_color.float_rgb), 1.0)
    return bg.outputs["Background"]


def _build_gradient(tree: bpy.types.NodeTree, renderer: object) -> bpy.types.NodeSocket:
    """Build a Generated-coord vertical gradient between bottom and top colours.

    Returns
    -------
    bpy.types.NodeSocket
        The terminal Background node's output socket.

    """
    bottom = _srgb_to_linear_rgb(renderer.GetBackground())
    top = _srgb_to_linear_rgb(renderer.GetBackground2())

    # ``TexCoord > Camera`` returns the surface point's camera-local
    # coordinate; for the world shader (evaluated per camera ray) this is
    # the ray direction in camera space, so its Y component runs from
    # negative at the bottom of the framed view to positive at the top.
    # Rescale by tan(half_fov) so the gradient stops always land on the
    # screen edges regardless of camera FOV.
    tex_coord = tree.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-700, 0)

    sep = tree.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-500, 0)
    tree.links.new(tex_coord.outputs["Camera"], sep.inputs["Vector"])

    # Map y → 0.5 * y / tan(half_fov) + 0.5 so y at the bottom of the
    # frame → 0 and at the top → 1. Camera-space Y at the bottom edge is
    # -tan(fov/2); at the top edge it's +tan(fov/2). We bake half_fov as
    # a Math node so the world shader stays self-contained (no per-frame
    # driver).
    cam = bpy.context.scene.camera
    fov = float(cam.data.angle) if cam is not None and cam.data.type == "PERSP" else 1.0
    half_tan = max(math.tan(fov / 2.0), 1e-6)

    rescale = tree.nodes.new("ShaderNodeMath")
    rescale.operation = "MULTIPLY_ADD"
    rescale.location = (-300, 0)
    rescale.inputs[1].default_value = 0.5 / half_tan
    rescale.inputs[2].default_value = 0.5
    tree.links.new(sep.outputs["Y"], rescale.inputs[0])

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-100, 0)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (*bottom, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (*top, 1.0)
    tree.links.new(rescale.outputs["Value"], ramp.inputs["Fac"])

    bg = tree.nodes.new("ShaderNodeBackground")
    bg.location = (100, 0)
    tree.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    return bg.outputs["Background"]


def _build_environment(
    tree: bpy.types.NodeTree, env_tex: object
) -> bpy.types.NodeSocket | None:
    """Build an Environment Texture node fed by the renderer's IBL image.

    Returns
    -------
    bpy.types.NodeSocket or None
        The Background node's output; ``None`` if the texture has no data.

    """
    image = _load_env_image(env_tex)
    if image is None:
        return None

    tex_coord = tree.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-500, 0)

    env_node = tree.nodes.new("ShaderNodeTexEnvironment")
    env_node.location = (-200, 0)
    env_node.image = image
    tree.links.new(tex_coord.outputs["Generated"], env_node.inputs["Vector"])

    bg = tree.nodes.new("ShaderNodeBackground")
    bg.location = (100, 0)
    tree.links.new(env_node.outputs["Color"], bg.inputs["Color"])
    return bg.outputs["Background"]


def _renderer_has_gradient(renderer: object) -> bool:
    """Return whether the renderer is in gradient-background mode.

    Returns
    -------
    bool
        ``True`` when ``set_background(color, top=...)`` was used.

    """
    getter = getattr(renderer, "GetGradientBackground", None)
    return bool(getter()) if getter is not None else False


def _renderer_env_texture(renderer: object) -> object | None:
    """Return the renderer's environment texture, or ``None`` if unset.

    Returns
    -------
    object or None
        The ``vtkTexture`` instance when ``set_environment_texture(...)``
        was used, otherwise ``None``.

    """
    if not bool(getattr(renderer, "GetUseImageBasedLighting", lambda: False)()):
        return None
    getter = getattr(renderer, "GetEnvironmentTexture", None)
    return getter() if getter is not None else None


def _load_env_image(env_tex: object) -> bpy.types.Image | None:
    """Materialise the vtkTexture's image data as a ``bpy.types.Image``.

    The data is written to a tmp PNG and loaded with ``pack=True`` so the
    image is self-contained in bpy memory afterwards; the tmp file is
    removed immediately.

    Returns
    -------
    bpy.types.Image or None
        The loaded image; ``None`` if the texture has no usable data.

    """
    pixels = _read_vtk_texture_rgba(env_tex)
    if pixels is None:
        return None
    height, width, _ = pixels.shape

    image = bpy.data.images.new(
        name="PVEnvTexture", width=width, height=height, alpha=True
    )
    # bpy.types.Image.pixels is a flat float buffer in row order from BOTTOM
    # to top; VTK delivers top-to-bottom (origin at top-left). Flip the rows.
    flipped = pixels[::-1, :, :].astype(np.float32).ravel()
    image.pixels.foreach_set(flipped)
    image.update()
    return image


def _read_vtk_texture_rgba(env_tex: object) -> np.ndarray | None:
    """Pull RGBA pixel data out of a ``vtkTexture`` as a numpy array.

    Returns
    -------
    np.ndarray or None
        Shape ``(H, W, 4)`` of float32 in ``[0, 1]``; ``None`` when the
        texture has no input image (programmatically synthesised
        textures, unlikely in user scenes).

    """
    image_data = env_tex.GetInputDataObject(0, 0)
    if image_data is None:
        return None
    dims = image_data.GetDimensions()
    width, height = int(dims[0]), int(dims[1])
    scalars = image_data.GetPointData().GetScalars()
    if scalars is None:
        return None

    arr = vtk_to_numpy(scalars).reshape(height, width, -1)
    rgba = np.ones((height, width, 4), dtype=np.float32)
    n_components = arr.shape[2]
    rgba[..., : min(n_components, 4)] = (
        arr[..., : min(n_components, 4)].astype(np.float32) / 255.0
    )
    if n_components == 1:
        # Greyscale source: replicate across RGB.
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = arr[..., 0] / 255.0
    return rgba


def _hide_world_from_shading_rays(world: bpy.types.World) -> None:
    """Hide the world shader from diffuse/glossy/transmission/scatter rays.

    Matches VTK's behaviour for solid and gradient backgrounds: the colour
    is camera-visible but does not act as an ambient fill.
    """
    visibility = world.cycles_visibility
    visibility.diffuse = False
    visibility.glossy = False
    visibility.transmission = False
    visibility.scatter = False


def _allow_world_in_shading_rays(world: bpy.types.World) -> None:
    """Re-enable world contribution to shading rays for IBL."""
    visibility = world.cycles_visibility
    visibility.diffuse = True
    visibility.glossy = True
    visibility.transmission = True
    visibility.scatter = True


def _srgb_to_linear_rgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert an sRGB-encoded RGB triple to scene-linear.

    Returns
    -------
    tuple of float
        Linearised RGB; alpha is not consumed here.

    """
    return tuple(_srgb_to_linear(float(c)) for c in rgb)


def _srgb_to_linear(value: float) -> float:
    """Convert a single sRGB channel in [0, 1] to scene-linear.

    Returns
    -------
    float
        The linearised value, using the standard piecewise sRGB curve.

    """
    threshold = 0.04045
    if value <= threshold:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4
