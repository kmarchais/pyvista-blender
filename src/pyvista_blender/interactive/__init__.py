# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive rendered viewport: ``pl.blender.show()``.

Architecture: VTK keeps owning the window and input; a second VTK renderer
on layer 1 with ``vtkActor2D`` + ``vtkImageMapper`` displays Cycles output
as a fullscreen overlay. VTK observers on ``InteractionEvent`` /
``EndInteractionEvent`` drive progressive Cycles re-renders.

Full design in ``docs/architecture.md`` ("Interactive viewports" section).

Modules:

* ``overlay`` — ``vtkActor2D`` overlay setup, pixel-buffer round-trip
* ``observers`` — VTK observer wiring + throttling + sample-tier dispatch
* ``camera_sync`` — ``pv.Camera`` → ``bpy.Object`` cam state synchronization
"""
