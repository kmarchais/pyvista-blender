# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared pytest fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def offscreen_plotter() -> Iterator[pv.Plotter]:
    """Yield a clean off-screen PyVista Plotter and close it on teardown.

    Yields
    ------
    pv.Plotter
        A freshly-constructed off-screen plotter at 640x480. The plotter
        is closed automatically when the test ends.

    """
    pl = pv.Plotter(off_screen=True, window_size=[640, 480])
    yield pl
    pl.close()
