# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the animation frame muxer in :mod:`_render_impl`.

``_encode_animation_frames`` is exercised directly on synthetic PNG
frames — no plotter or Cycles render involved — so the tests only need
``bpy`` to be importable (the module imports it at top level).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.bpy

_RGBA_CHANNELS = 4
_OPAQUE = 255


def _square_frame(x0: int) -> np.ndarray:
    """Build an opaque red 32x32 square at column ``x0`` on a transparent field.

    Returns
    -------
    np.ndarray
        A 64x128 RGBA uint8 frame, fully transparent except the square.

    """
    frame = np.zeros((64, 128, 4), dtype=np.uint8)
    frame[16:48, x0 : x0 + 32] = (255, 0, 0, 255)
    return frame


def test_gif_transparent_frames_do_not_ghost(tmp_path: Path) -> None:
    """A moved square must vanish from its old spot in the next GIF frame.

    GIF frames default to compositing over the previous frame, so with a
    transparent background every frame's opaque content used to ghost
    through all later frames. The writer must request disposal mode 2
    (restore to background) to clear between frames.
    """
    from pyvista_blender._render_impl import (  # noqa: PLC0415, PLC2701
        _encode_animation_frames,
    )

    frame_paths = []
    for ordinal, x0 in enumerate((8, 80)):
        path = tmp_path / f"frame_{ordinal:06d}.png"
        iio.imwrite(path, _square_frame(x0))
        frame_paths.append(path)

    output = tmp_path / "anim.gif"
    _encode_animation_frames(frame_paths, str(output), suffix=".gif", fps=10)

    second = iio.imread(output, index=1)
    if second.shape[-1] != _RGBA_CHANNELS:
        pytest.fail(f"expected RGBA frames from the GIF, got shape {second.shape}")

    ghost_alpha = int(second[32, 24, 3])
    if ghost_alpha != 0:
        pytest.fail(
            "frame 1's square ghosts through frame 2: pixel (32, 24) has "
            f"alpha {ghost_alpha}, expected 0 (transparent)"
        )

    current = tuple(int(c) for c in second[32, 96])
    if current[3] != _OPAQUE:
        pytest.fail(f"frame 2's own square is missing: pixel (32, 96) = {current}")
