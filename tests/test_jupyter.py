# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Jupyter inline backend (``pv.set_jupyter_backend("blender")``).

The handler renders the plotter through Cycles via the bridge and
returns an :class:`IPython.display.Image`. These tests exercise the
handler directly so they can run outside a notebook session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pyvista as pv
from IPython.display import Image

from pyvista_blender.jupyter import handler

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.bpy
def test_jupyter_handler_returns_ipython_image_with_png_bytes() -> None:
    """The handler returns an IPython Image with valid PNG bytes."""
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    try:
        pl.add_mesh(pv.Sphere(), color="red")
        result = handler(pl, samples=4)
    finally:
        pl.close()

    if not isinstance(result, Image):
        pytest.fail(f"handler returned {type(result).__name__}, expected IPython.Image")
    png_bytes = result.data
    if not isinstance(png_bytes, bytes) or len(png_bytes) == 0:
        pytest.fail("handler-produced image has no PNG bytes")
    png_magic = b"\x89PNG\r\n\x1a\n"
    if not png_bytes.startswith(png_magic):
        pytest.fail(f"handler bytes missing PNG magic; got {png_bytes[:8]!r}")


@pytest.mark.bpy
def test_jupyter_handler_writes_screenshot_when_path_given(tmp_path: Path) -> None:
    """When ``screenshot=path`` is passed, the same PNG also lands on disk.

    Mirrors pyvista's built-in backend contract: the inline image is
    returned regardless, but a path-argument additionally persists the
    render.
    """
    pl = pv.Plotter(off_screen=True, window_size=[320, 240])
    try:
        pl.add_mesh(pv.Sphere(), color="cyan")
        out = tmp_path / "jupyter.png"
        result = handler(pl, screenshot=out, samples=4)
    finally:
        pl.close()

    if not out.exists() or out.stat().st_size == 0:
        pytest.fail("screenshot path was not populated")
    if not isinstance(result, Image):
        pytest.fail("handler did not return an IPython Image even with screenshot=")


def test_jupyter_handler_filters_unknown_render_kwargs() -> None:
    """Unknown kwargs (eg. pyvista's ``return_img``) are dropped silently.

    The handler accepts ``**kwargs`` so pyvista's own keys flow through
    without breaking the bridge; only :func:`pl.blender.render`'s
    documented args are forwarded.
    """
    # White-box check via the private filter (it's the contract that
    # matters; the public handler exercises it on every call).
    from pyvista_blender.jupyter import _filter_render_kwargs  # noqa: PLC0415, PLC2701

    filtered = _filter_render_kwargs({
        "samples": 16,
        "engine": "cycles",
        "return_img": True,  # pyvista-side, must drop
        "window_size": (640, 480),  # pyvista-side, must drop
        "cpos": [(0, 0, 5), (0, 0, 0), (0, 1, 0)],  # pyvista-side, must drop
    })
    if filtered != {"samples": 16, "engine": "cycles"}:
        pytest.fail(
            f"filter dropped or kept the wrong keys; got {sorted(filtered)}, "
            f"expected ['engine', 'samples']"
        )


def test_jupyter_backend_registers_via_entry_point() -> None:
    """``pv.set_jupyter_backend("blender")`` loads the handler via entry-point.

    Pyvista's :func:`_ensure_entry_points` scans the
    ``pyvista.jupyter_backends`` group; the bridge ships the entry
    point in ``pyproject.toml``. Confirm the registration round-trips
    so future pyvista API churn surfaces here, not in production.
    """
    pv.set_jupyter_backend("blender")
    # Internal accessor: pyvista keeps registered handlers in
    # `_custom_backends`. Importing via the module path keeps the test
    # honest about which symbol we're inspecting.
    from pyvista.jupyter import _custom_backends  # noqa: PLC0415, PLC2701

    if "blender" not in _custom_backends:
        pytest.fail(
            f"'blender' did not land in pyvista._custom_backends; "
            f"registered: {sorted(_custom_backends)}"
        )
    if _custom_backends["blender"] is not handler:
        pytest.fail(
            "registered 'blender' handler is not pyvista_blender.jupyter.handler; "
            f"got {_custom_backends['blender']!r}"
        )
