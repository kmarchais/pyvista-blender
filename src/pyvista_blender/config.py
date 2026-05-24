# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module-level configuration defaults for the Blender backend.

Resolution order for any render setting is:

    per-call kwarg  >  component attribute  >  module default

Examples
--------
Set process-wide defaults once at startup::

    import pyvista_blender as pvb

    pvb.config.engine = "cycles"
    pvb.config.device = "optix"
    pvb.config.samples = 128

A specific plotter can override the module default by setting an attribute on
its :class:`~pyvista_blender.BlenderComponent`::

    pl.blender.engine = "eevee"      # only this plotter
    pl.blender.samples = 32

A single call can override both::

    pl.blender.render("out.png", engine="cycles", samples=256)

"""

from __future__ import annotations

from typing import Final, Literal

#: Supported render engines.
#:
#: ``"cycles"`` is the cross-platform path-tracer with OptiX / CUDA / HIP /
#: Metal / oneAPI / CPU back-ends. ``"eevee"`` is the rasterizer-based
#: Eevee Next; headless mode is currently Linux-only.
Engine = Literal["cycles", "eevee"]

#: Compute device for Cycles.
#:
#: ``"auto"`` and ``"gpu"`` both pick the best available accelerator
#: (OptiX > CUDA > HIP > Metal > oneAPI > CPU); they're aliases.
#: Eevee Next ignores this field and uses whatever GPU context is
#: available.
Device = Literal["optix", "cuda", "hip", "metal", "oneapi", "cpu", "auto", "gpu"]

# ---------------------------------------------------------------------------
# Defaults. Mutate these once at startup, or leave them and pass per-call.
# ---------------------------------------------------------------------------

#: Default render engine. See :data:`Engine`.
engine: Engine = "cycles"

#: Default compute device. See :data:`Device`.
device: Device = "auto"

#: Default Cycles sample count for offline renders.
samples: int = 128

#: Whether to apply a denoiser (OptiX on NVIDIA, OIDN elsewhere).
denoise: bool = True

#: Render resolution percentage of the plotter's ``window_size``.
resolution_percentage: int = 100

#: Whether the rendered film is transparent (sets ``scene.render.film_transparent``).
transparent_bg: bool = False

#: Maximum number of subdivisions when tessellating high-order cells.
#:
#: Datasets containing VTK quadratic (cell types 21-37), Lagrange (68-74),
#: or Bezier (75-81) cells have curved faces that ``extract_surface()``
#: linearises into flat triangles, losing curvature. When the bridge sees
#: any such cell type, it runs ``dataset.tessellate(max_n_subdivide=N)``
#: first to refine the curved surface into denser linear triangles. ``N``
#: tunes the refinement budget: 0 disables (use the linearised surface
#: as-is), 1-2 gives coarse curvature, 3 (the VTK default) is balanced,
#: 4+ resolves cubic / quartic curvature at the cost of triangle count.
tessellation_subdivide: int = 3

#: Dihedral angle (degrees) above which a mesh edge is marked sharp.
#:
#: Matches VTK's default ``feature_angle`` of 30°. Smooth shading remains
#: the global default, but normals are split at edges whose adjacent faces
#: meet at a sharper angle than this, so geometry like cube corners reads
#: as hard while curved surfaces stay smooth. Set to ``180.0`` to disable
#: (everything fully smooth) or ``0.0`` for fully flat shading.
sharp_edge_angle: float = 30.0

# Interactive viewport defaults (see docs/architecture.md).

#: Cycles samples per pass during mouse interaction (50-100 ms target).
interactive_samples: int = 4

#: Cycles samples after EndInteractionEvent.
settled_samples: int = 32

#: Cycles samples for idle progressive refinement.
idle_samples: int = 128

# ---------------------------------------------------------------------------
# Read-only metadata
# ---------------------------------------------------------------------------

SUPPORTED_ENGINES: Final[tuple[str, ...]] = ("cycles", "eevee")
SUPPORTED_DEVICES: Final[tuple[str, ...]] = (
    "optix",
    "cuda",
    "hip",
    "metal",
    "oneapi",
    "cpu",
    "auto",
    "gpu",
)
