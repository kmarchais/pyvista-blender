# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translators from PyVista plotter state into ``bpy`` scene state.

Modules:

* ``scene`` — top-level orchestrator walking ``plotter.renderers``
* ``mesh`` — ``DataSet`` → ``bpy.types.Mesh`` via ``foreach_set``
* ``material`` — ``pv.Property`` + mapper → ``bpy.types.Material``
* ``scalar`` — scalars + LUT → color attribute + shader graph
* ``camera`` — ``pv.Camera`` → ``bpy.types.Camera`` + ``matrix_world``
* ``light`` — ``pv.Light`` + light kit → ``bpy.types.Light`` objects
* ``background`` — solid / gradient / env texture → World shader
* ``volume`` — ``ImageData`` → packed atlas + Cycles Volume shader
* ``transform`` — ``GetMatrix()`` → ``mathutils.Matrix``
"""
