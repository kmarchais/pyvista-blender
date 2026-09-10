# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scoped render progress, without importing Blender."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest
import pyvista as pv

from pyvista_blender._bpy_silence import (  # noqa: PLC2701 -- verify descriptor restoration
    silence_bpy_stderr,
)
from pyvista_blender._progress import (  # noqa: PLC2701 -- test bpy-free subscription lifetime
    report_render_progress,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_progress_handler_lifetime() -> None:
    """Forward status updates and preserve pre-existing handlers."""
    events: list[str] = []
    existing = events.append
    handlers: list[Callable[[str], None]] = [existing]
    with report_render_progress(events.append, handlers):
        handlers[-1]("Sample 1/4")
    if events != ["Rendering", "Sample 1/4", "Finished"]:
        pytest.fail(f"Unexpected progress events: {events}")
    if handlers != [existing]:
        pytest.fail("Progress handler leaked or removed another subscriber")


def test_progress_failure_cleanup() -> None:
    """A rendering failure removes only the temporary handler.

    Raises
    ------
    RuntimeError
        Simulated render failure, caught by pytest.

    """
    events: list[str] = []
    handlers: list[Callable[[str], None]] = []
    message = "render failed"
    with (
        pytest.raises(RuntimeError, match="render failed"),
        report_render_progress(events.append, handlers),
    ):
        raise RuntimeError(message)
    if handlers or "Finished" in events:
        pytest.fail("Failure reported completion or leaked its handler")


def test_callback_error_propagates_after_render() -> None:
    """Blender must not swallow a callback failure as a handler traceback."""
    handlers: list[Callable[[str], None]] = []

    def callback(status: str) -> None:
        if status.startswith("Sample"):
            message = "callback failed"
            raise ValueError(message)

    with (
        pytest.raises(ValueError, match="callback failed"),
        report_render_progress(callback, handlers),
    ):
        handlers[-1]("Sample 1/4")
    if handlers:
        pytest.fail("Failed callback leaked a handler")


def test_no_callback_leaves_handlers_unchanged() -> None:
    """Normal renders need no handler or output redirection machinery."""
    handlers: list[Callable[[str], None]] = []
    with report_render_progress(None, handlers):
        if handlers:
            pytest.fail("Registered a handler without a callback")


@pytest.mark.bpy
def test_render_progress_with_fresh_and_cached_scene(
    offscreen_plotter: pv.Plotter, tmp_path: Path
) -> None:
    """Fresh and cached scenes both report progress without leaking handlers."""
    plotter = offscreen_plotter
    plotter.window_size = [64, 64]
    plotter.add_mesh(pv.Sphere(), color="coral")
    for index in range(2):
        events: list[str] = []
        output = tmp_path / f"progress-{index}.png"
        plotter.blender.render(str(output), samples=4, on_progress=events.append)
        if events[:2] != ["Preparing scene", "Rendering"] or events[-1] != "Finished":
            pytest.fail(f"Missing lifecycle messages: {events}")
        if not any("Sample" in event for event in events):
            pytest.fail("No within-frame sampling progress received")
        if not output.is_file():
            pytest.fail("Completion was reported without an output image")
        count = len(events)
        plotter.blender.render(str(output), samples=4)
        if len(events) != count:
            pytest.fail("Callback received events from a later render")


def test_callback_output_during_silencing(capfd: pytest.CaptureFixture[str]) -> None:
    """Callback output remains visible while native renderer output stays suppressed."""
    handlers: list[Callable[[str], None]] = []

    def callback(status: str) -> None:
        sys.stdout.write(status + "\n")

    with report_render_progress(callback, handlers), silence_bpy_stderr():
        os.write(1, b"native noise\n")
        handlers[-1]("Sample 1/4")
        os.write(1, b"more native noise\n")
    output = capfd.readouterr().out
    if "Sample 1/4" not in output or "native noise" in output:
        pytest.fail(f"Incorrect output restoration: {output!r}")
