# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""HUD overlays composited on top of Cycles output.

PyVista's scalar bars, titles, text actors, corner axes, and bounding-
box annotations are 2D screen-space elements; in VTK they're drawn by
overlay actors after the main render. Cycles has no equivalent, so the
bridge generates each overlay as an RGBA image at render resolution and
composites them through Blender's compositor
(``CompositorNodeAlphaOver`` chain). Same orientation, same colormap,
same labels — produced via matplotlib / PIL so they look modern at any
resolution.

Modules:

* :mod:`compositor` — chain RenderLayers → AlphaOver* → Composite.
* :mod:`scalar_bar` — matplotlib ``Colorbar`` rendered to an RGBA PNG.
* :mod:`text` — PIL text rendering for ``add_text`` / ``add_title``.
* :mod:`axes` — corner XYZ triad + ``show_bounds`` snapshot via VTK.
"""

from __future__ import annotations

from pyvista_blender.hud.compositor import composite_hud_overlays

__all__ = ["composite_hud_overlays"]
