# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the engine + device validation contracts.

These cover the ``SUPPORTED_*``-backed validation in
:mod:`pyvista_blender._component`. Both checks run before the lazy
``bpy`` import, so typos surface as :class:`ValueError` without paying
bpy's ~200 MB / ~3 s import cost. ``_device.select_cycles_device``
re-runs the same check as defense in depth for direct callers; that
path is covered indirectly through the public :meth:`render` API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyvista_blender.config import SUPPORTED_DEVICES, SUPPORTED_ENGINES

if TYPE_CHECKING:
    import pyvista as pv


def test_render_rejects_unknown_engine(offscreen_plotter: pv.Plotter) -> None:
    """A typo in ``engine`` raises :class:`ValueError` listing the allowlist."""
    pl = offscreen_plotter
    with pytest.raises(ValueError, match=r"supported engines:"):
        pl.blender.render("out.png", engine="cucles")  # type: ignore[arg-type]


def test_animate_rejects_unknown_engine(offscreen_plotter: pv.Plotter) -> None:
    """Same allowlist gate applies on the animation path."""
    pl = offscreen_plotter
    with pytest.raises(ValueError, match=r"supported engines:"):
        pl.blender.animate(
            "out.gif",
            updater=lambda _: None,
            frames=range(1),
            engine="cucles",  # type: ignore[arg-type]
        )


def test_render_rejects_unknown_device(offscreen_plotter: pv.Plotter) -> None:
    """A typo in the device name is a hard error, not a CPU fallback."""
    pl = offscreen_plotter
    with pytest.raises(ValueError, match=r"supported devices:"):
        pl.blender.render("out.png", device="cucla")  # type: ignore[arg-type]


def test_animate_rejects_unknown_device(offscreen_plotter: pv.Plotter) -> None:
    """Device validation also runs on the animation path."""
    pl = offscreen_plotter
    with pytest.raises(ValueError, match=r"supported devices:"):
        pl.blender.animate(
            "out.gif",
            updater=lambda _: None,
            frames=range(1),
            device="cucla",  # type: ignore[arg-type]
        )


def test_supported_engines_contains_cycles_and_eevee() -> None:
    """The allowlist documents the supported engines."""
    if "cycles" not in SUPPORTED_ENGINES:
        pytest.fail(f"'cycles' missing from SUPPORTED_ENGINES: {SUPPORTED_ENGINES!r}")
    if "eevee" not in SUPPORTED_ENGINES:
        pytest.fail(f"'eevee' missing from SUPPORTED_ENGINES: {SUPPORTED_ENGINES!r}")


def test_supported_devices_includes_gpu_alias() -> None:
    """``"gpu"`` is a public alias for ``"auto"`` in the allowlist.

    The feasibility report uses ``device="gpu"`` in published example
    code, so it must appear in :data:`SUPPORTED_DEVICES`.
    """
    if "gpu" not in SUPPORTED_DEVICES:
        pytest.fail(
            f"'gpu' missing from SUPPORTED_DEVICES, breaking the documented "
            f"contract: {SUPPORTED_DEVICES!r}"
        )
