# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-call render statistics subscriptions and callback output restoration."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@contextmanager
def _restored_output(original: tuple[int, int]) -> Iterator[None]:
    """Temporarily restore the caller's stdout/stderr around a callback.

    Yields
    ------
    None
        Output is directed to the streams present before rendering.

    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(original[0], 1)
        os.dup2(original[1], 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


@contextmanager
def report_render_progress(
    callback: Callable[[str], None] | None,
    handlers: list[Callable[[str], None]],
) -> Iterator[None]:
    """Subscribe for one render and propagate callback errors after Blender returns.

    Parameters
    ----------
    callback
        Receives preparation, native statistics, and completion messages.
    handlers
        Blender's render_stats subscribers; injected to keep this module bpy-free.

    Yields
    ------
    None
        The temporary statistics subscriber is installed.

    """
    if callback is None:
        yield
        return
    callback("Rendering")
    errors: list[BaseException] = []
    original = (os.dup(1), os.dup(2))

    def notify(status: str) -> None:
        if errors:
            return
        try:
            with _restored_output(original):
                callback(status)
        # Blender catches handler exceptions. Preserve them for the Python caller,
        # including cancellation, instead of printing a traceback and losing it.
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    handlers.append(notify)
    try:
        yield
        if errors:
            raise errors[0]
        callback("Finished")
    finally:
        handlers.remove(notify)
        os.close(original[0])
        os.close(original[1])
