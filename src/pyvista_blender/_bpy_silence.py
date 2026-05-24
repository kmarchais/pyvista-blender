# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""File-descriptor-level silencer for Blender's render-time stderr chatter.

``bpy.ops.render.render(write_still=True)`` writes a timestamped
``render | Saved: ...`` line to stderr on every call, plus various
warnings during Cycles startup (``HIPEW initialization failed``, etc.).
Those come from Blender's C code, so Python-side ``sys.stderr``
redirection doesn't catch them — we have to dup2 the actual file
descriptor.

Use :func:`silence_bpy_stderr` as a context manager around any
``bpy.ops.render.render`` call where the noise isn't useful (i.e. all
production calls). Tests / benchmarks that want the chatter for
diagnostics can simply skip the wrapper.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["silence_bpy_stderr"]


@contextmanager
def silence_bpy_stderr() -> Iterator[None]:
    """Redirect stdout + stderr to ``/dev/null`` for the body of the block.

    Restores both descriptors on exit (including on exceptions). Cheap
    (~10 microseconds) and idempotent — nesting works. Blender's
    render-time chatter (``HIPEW initialization``, the per-frame
    ``Saved: ...`` line) sometimes lands on stdout and sometimes on
    stderr depending on the build, so we silence both.

    Yields
    ------
    None
        The context manager has no value; it's purely a side-effect on
        the process's standard streams.

    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(devnull)
        os.close(saved_stdout)
        os.close(saved_stderr)
