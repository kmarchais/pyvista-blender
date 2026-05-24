# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resolve a user-facing device string to a Cycles ``scene.cycles.device``.

The picker maps the small alphabet exposed by
:data:`~pyvista_blender.config.Device` onto Blender's two-level device model:

* ``scene.cycles.device`` is the per-scene toggle (``"CPU"`` vs ``"GPU"``).
* ``bpy.context.preferences.addons["cycles"].preferences.compute_device_type``
  is the user-preference picking the GPU backend
  (``"CUDA"``, ``"OPTIX"``, ``"HIP"``, ``"METAL"``, ``"ONEAPI"``).

``"auto"`` and ``"gpu"`` walk OptiX > CUDA > HIP > Metal > oneAPI > CPU at
runtime; named backends force one. Unavailable GPUs fall back to CPU with
a :class:`UserWarning`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Protocol, cast

import bpy

from pyvista_blender.config import SUPPORTED_DEVICES

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["select_cycles_device"]


class _CyclesDevice(Protocol):
    """The duck-typed surface of a single Cycles ``devices[i]`` entry."""

    type: str
    use: bool


class _CyclesPrefs(Protocol):
    """The duck-typed surface of ``bpy.types.CyclesPreferences``.

    fake-bpy-module-5.0 only ships the base
    :class:`bpy.types.AddonPreferences`, which omits the cycles-specific
    fields. Declaring them via :class:`typing.Protocol` lets ty validate
    the call sites without dragging :data:`typing.Any` through the
    module surface.
    """

    compute_device_type: str

    @property
    def devices(self) -> Iterable[_CyclesDevice]: ...

    def refresh_devices(self) -> None: ...


_AUTO_ORDER = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")
_DEVICE_TO_BPY = {
    "cuda": "CUDA",
    "optix": "OPTIX",
    "hip": "HIP",
    "metal": "METAL",
    "oneapi": "ONEAPI",
}


def select_cycles_device(device: str) -> str:
    """Configure Cycles preferences and return the per-scene device toggle.

    Parameters
    ----------
    device
        One of :data:`~pyvista_blender.config.SUPPORTED_DEVICES`. Named
        backends ("cuda", "optix", ...) that are present in the runtime
        but unavailable fall back to CPU with a :class:`UserWarning`;
        names outside the allowlist raise :class:`ValueError`.

    Returns
    -------
    str
        ``"GPU"`` when a backend was successfully selected, ``"CPU"``
        otherwise. The caller assigns this to ``scene.cycles.device``.

    Raises
    ------
    ValueError
        When ``device`` is not in :data:`SUPPORTED_DEVICES`.

    """
    normalized = device.lower()
    if normalized not in SUPPORTED_DEVICES:
        msg = (
            f"unknown device {device!r}; supported devices: {sorted(SUPPORTED_DEVICES)}"
        )
        raise ValueError(msg)

    if normalized == "cpu":
        _enable_cpu_only()
        return "CPU"

    if normalized in {"auto", "gpu"}:
        backend_order = _AUTO_ORDER
    else:
        backend_order = (_DEVICE_TO_BPY[normalized],)

    for backend in backend_order:
        if _try_enable_backend(backend):
            return "GPU"

    if normalized not in {"auto", "gpu"}:
        warnings.warn(
            f"requested device {device!r} unavailable; falling back to CPU",
            UserWarning,
            stacklevel=3,
        )
    _enable_cpu_only()
    return "CPU"


def _cycles_prefs() -> _CyclesPrefs | None:
    """Return the Cycles addon preferences, or ``None`` if disabled.

    Routes the ``Preferences | None`` / ``AddonPreferences | None``
    unions that fake-bpy-module-5.0 advertises through a single cast
    site so the callers can read ``compute_device_type`` / ``devices``
    / ``refresh_devices`` without re-narrowing.

    Returns
    -------
    _CyclesPrefs or None
        The Cycles preferences singleton, or ``None`` when the cycles
        addon is not enabled in the current bpy build.

    """
    prefs = cast("Any", bpy.context.preferences)
    if prefs is None:
        return None
    addon = prefs.addons.get("cycles")
    if addon is None:
        return None
    return cast("_CyclesPrefs", addon.preferences)


def _try_enable_backend(backend: str) -> bool:
    """Attempt to enable a single Cycles GPU backend.

    Returns
    -------
    bool
        ``True`` if at least one matching non-CPU device was found and
        enabled, ``False`` otherwise.

    """
    cprefs = _cycles_prefs()
    if cprefs is None:
        return False

    try:
        cprefs.compute_device_type = backend
    except TypeError:
        # The bpy build doesn't support this backend (e.g. METAL on Linux).
        return False

    if hasattr(cprefs, "refresh_devices"):
        cprefs.refresh_devices()

    matching = [d for d in cprefs.devices if getattr(d, "type", "") == backend]
    if not matching:
        return False

    for dev in cprefs.devices:
        dev.use = getattr(dev, "type", "") in {backend, "CPU"}
    return True


def _enable_cpu_only() -> None:
    """Set the cycles preferences back to a CPU-only configuration."""
    cprefs = _cycles_prefs()
    if cprefs is None:
        return
    try:
        cprefs.compute_device_type = "NONE"
    except TypeError:
        return
    for dev in cprefs.devices:
        dev.use = getattr(dev, "type", "") == "CPU"
