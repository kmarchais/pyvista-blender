# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Internal option bundles shared across the render entry points.

Kept bpy-free so :mod:`_component` and :mod:`interactive.overlay` can
import these symbols at module top without pulling in the ~200 MB
``bpy`` wheel (which only loads when a render actually runs, via the
lazy import inside :func:`do_render` / :func:`do_animate` /
:func:`do_export_animation_blend`).

Each dataclass / NamedTuple bundles a logical group of kwargs so the
``do_*`` entry points and their helpers stay under ruff's
argument-count threshold without dropping any per-call setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pyvista as pv
    from PIL import Image as PILImage

    from pyvista_blender._glyph import GlyphSpec
    from pyvista_blender.translate.scene import SceneCache


@dataclass(frozen=True, slots=True)
class _EngineParams:
    """Bundle the render-engine kwargs forwarded to ``configure_engine``.

    Carries fully-resolved (post-three-tier-resolution) values for
    every per-call render setting. The dict-splat counterpart is
    :class:`_RenderKwargs`, which uses ``total=False`` to express the
    partial-state shape that ``pl.blender.render(**kwargs)`` callers
    work with.
    """

    engine: str
    device: str
    samples: int
    denoise: bool
    transparent_bg: bool


class _RenderKwargs(TypedDict, total=False):  # noqa: PYI049
    """Per-key types for the kwargs accepted by ``pl.blender.render``.

    Used by callers that build a dict and splat it (the Trame web app,
    the Jupyter handler). ``total=False`` so each key is optional and
    the resolver / component / module defaults fill in the rest. The
    dict-splat counterpart of :class:`_EngineParams`, which carries
    the same five fields as a frozen dataclass for internal
    function-to-function passing.

    The ``# noqa: PYI049`` is required because the TypedDict isn't
    consumed inside this module — only by ``_component.py`` and the
    Trame web app — but it belongs alongside the other option types.
    """

    engine: str
    device: str
    samples: int
    denoise: bool
    transparent_bg: bool


@dataclass(frozen=True, slots=True)
class _PlotterSources:
    """Per-call registries from the component that translators consume.

    Bundled together because every ``do_*`` entry point and the
    interactive overlay's ``render_and_blit`` thread the same two
    optional collections through. Kept separate from the mutable
    :class:`SceneCache` so callers can swap in an updated cache
    without rebuilding the bundle.
    """

    glyphs: list[GlyphSpec] | None = None
    volume_sources: dict[str, pv.DataSet] | None = None


#: Module-level empty :class:`_PlotterSources` for use as a default
#: argument. Avoids ruff's B008 "function call in argument default"
#: lint, while staying safe because the dataclass is frozen.
_EMPTY_SOURCES = _PlotterSources()


@dataclass(frozen=True, slots=True)
class _ActorSampleBuckets:
    """Per-channel sample collectors threaded through :func:`_sample_one_actor`.

    Each dict accumulates per-frame samples for one bake channel
    (keyed by ``vtk_identity`` of the actor). Bundling them keeps the
    helper's signature short across the per-frame walk.
    """

    vertex: dict[str, list[tuple[int, np.ndarray]]]
    scalar: dict[str, list[tuple[int, np.ndarray]]]
    transform: dict[str, list[tuple[int, np.ndarray]]]
    material: dict[str, list[tuple[int, _MaterialSnapshot]]]


@dataclass(frozen=True, slots=True)
class _SubplotTileContext:
    """Per-call state threaded into each subplot tile render.

    Fields are constant across the per-tile loop, so bundling them
    keeps :func:`_render_one_subplot_tile` ergonomic at the call site
    (it varies only ``renderer`` and ``ri`` between iterations).
    """

    tmp_dir: Path
    composite: PILImage.Image
    cache: SceneCache
    actor_to_renderer: dict[str, int]
    engine_params: _EngineParams
    width: int
    height: int
    n_cols: int


@dataclass(frozen=True, slots=True)
class _BakeChannels:
    """Per-channel selection for :func:`do_export_animation_blend`.

    Each field gates one independent bake (camera fcurves, mesh
    deformation, scalar fields, lights, actor transforms, materials,
    volumes, glyphs).
    """

    camera: bool = True
    deformation: bool | str = False
    scalars: bool = False
    lights: bool = False
    transforms: bool = False
    materials: bool = False
    volume: bool = False
    glyphs: bool = False


#: Light state captured at one frame: ``(world_position, world_focal_point,
#: intensity, diffuse_color)`` where each tuple-of-three holds linear-RGB
#: floats for colour and Cartesian coords for position / focal-point.
_LightSnapshot = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    tuple[float, float, float],
]

#: Principled BSDF input values captured at one frame:
#: ``(base_color RGB, metallic, roughness, alpha)``. Phong-shaded
#: properties get their ``specular_power`` converted to roughness via
#: the Walter et al. fit so the snapshot matches the static path.
_MaterialSnapshot = tuple[
    tuple[float, float, float],
    float,
    float,
    float,
]


class _AnimationSamples(NamedTuple):
    """Per-channel per-frame samples produced by ``_sample_animation``.

    Attribute access (``samples.cam``, ``samples.vertex``, ...) keeps
    the entry-point local-variable count down compared to unpacking a
    long tuple at the call site.
    """

    cam: list[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ]
    vertex: dict[str, list[tuple[int, np.ndarray]]]
    scalar: dict[str, list[tuple[int, np.ndarray]]]
    light: dict[int, list[tuple[int, _LightSnapshot]]]
    scalar_domains: dict[str, str]
    transform: dict[str, list[tuple[int, np.ndarray]]]
    material: dict[str, list[tuple[int, _MaterialSnapshot]]]
    volume: dict[str, list[tuple[int, np.ndarray]]]
    glyph: dict[int, dict[str, list[tuple[int, np.ndarray]]]]
