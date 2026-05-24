# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render PyVista plotter scenes through Blender (``bpy``) for photoreal output.

Importing this package is **not required** to use the bridge. The
``[project.entry-points."pyvista.plotter_components"]`` block in
``pyproject.toml`` registers the ``blender`` namespace on every
``pyvista.BasePlotter`` instance via PyVista 0.48's plotter-component
discovery, so ``pl.blender.render(...)`` works after a plain
``pip install pyvista-blender``.

This module re-exports :class:`BlenderComponent`, the :mod:`config`
namespace, and ``__version__`` for users who do want to import the
package directly (for type hints, custom configuration, or subclassing).
"""

from __future__ import annotations

from pyvista_blender import config
from pyvista_blender._compat import rna_get, rna_set
from pyvista_blender._component import BlenderComponent
from pyvista_blender._version import __version__
from pyvista_blender.animate import orbit_camera

__all__ = [
    "BlenderComponent",
    "__version__",
    "config",
    "orbit_camera",
    "rna_get",
    "rna_set",
]
