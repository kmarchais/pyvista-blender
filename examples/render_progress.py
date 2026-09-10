# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a sphere while printing Blender status updates."""

import pyvista as pv

plotter = pv.Plotter(off_screen=True, window_size=[320, 240])
plotter.add_mesh(pv.Sphere(), color="coral")
plotter.blender.render("sphere.png", samples=32, on_progress=print)
plotter.close()
