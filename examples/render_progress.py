# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a sphere while printing Blender status updates."""

import pyvista as pv

plotter = pv.Plotter()
plotter.add_mesh(pv.Sphere())
plotter.blender.render("sphere.png", on_progress=print)
plotter.close()
