# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Trame-based browser viewport for ``pl.blender.show(backend="web")``.

A Trame app that shows the plotter through a
:class:`trame_vtk.widgets.vtk.VtkLocalView` (client-side VTK for the
camera drag preview) with the bridge's Cycles render overlaid on top
as an ``<img>``. On ``EndInteractionEvent`` the bridge re-renders
Cycles and updates the overlay; during drag the overlay hides and the
VTK widget handles the 60 fps real-time pass, same hybrid contract as
the desktop ``pl.blender.show()`` viewport, just with the browser as
the display surface. Per-tier sampling and an idle-promotion timer
layer on top so a longer pause yields a higher-quality settle.
"""

from __future__ import annotations

from pyvista_blender.web.app import (
    DEFAULT_IDLE_DELAY_MS,
    DEFAULT_IDLE_SAMPLES,
    DEFAULT_SETTLED_SAMPLES,
    BlenderWebApp,
    serve,
)

__all__ = [
    "DEFAULT_IDLE_DELAY_MS",
    "DEFAULT_IDLE_SAMPLES",
    "DEFAULT_SETTLED_SAMPLES",
    "BlenderWebApp",
    "serve",
]
