# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Package version, kept out of ``__init__.py`` to satisfy RUF067.

The ``__init__`` module is expected to contain only docstrings and
re-exports. The version probe sits here and is re-exported from the
package root.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyvista-blender")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"
