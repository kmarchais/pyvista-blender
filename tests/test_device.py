# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cycles device selection probes one backend and renders on the GPU alone."""

from __future__ import annotations

import subprocess  # noqa: S404 - a fresh interpreter: see _SCRIPT
import sys

import pytest

pytest.importorskip("bpy")

# Run in a fresh interpreter: Cycles prints its backend warnings (``HIPEW
# initialization failed`` where AMD's HIP runtime is missing) from C, once per
# process, so only a new process shows whether selecting a device probed them.
_SCRIPT = """
import bpy
from pyvista_blender.render._device import select_cycles_device
toggle = select_cycles_device("auto")
prefs = bpy.context.preferences.addons["cycles"].preferences
used = sorted({d.type for d in prefs.devices if d.use})
print("RESULT", toggle, prefs.compute_device_type, ",".join(used))
"""


def _select_in_subprocess() -> tuple[list[str], str]:
    out = subprocess.run(  # noqa: S603 - fixed argv: this interpreter and a literal script
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    text = out.stdout + out.stderr
    result = next(ln for ln in text.splitlines() if ln.startswith("RESULT"))
    return result.split()[1:], text


def test_auto_device_probes_only_the_chosen_backend() -> None:
    """Selecting a device asks Cycles for that backend's devices only."""
    _, text = _select_in_subprocess()
    for noise in ("HIPEW", "initialization failed"):
        if noise in text:
            pytest.fail(f"device selection probed another backend: {text!r}")


def test_gpu_render_leaves_the_cpu_off() -> None:
    """A GPU backend renders on its own devices, not alongside the CPU."""
    fields, _ = _select_in_subprocess()
    toggle, backend = fields[0], fields[1]
    used = fields[2].split(",") if len(fields) > 2 else []  # noqa: PLR2004 - 3 fields
    if toggle != "GPU":
        pytest.skip("no GPU backend on this machine")
    if used != [backend]:
        pytest.fail(f"{backend} selected but devices in use are {used}")
