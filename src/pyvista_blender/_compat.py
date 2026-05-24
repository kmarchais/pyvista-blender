# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility shim across the bpy 4.x and 5.x ABIs.

The notable breakage between the two majors is that RNA-owning data
blocks (``Scene``, ``World``, ``Object``, ...) no longer support
``__getitem__`` / ``__setitem__`` for RNA-defined sub-properties in 5.0:
``scene["cycles"]`` used to fall through to ``scene.cycles`` in 4.x; in
5.0 it raises ``KeyError`` because the subscript path is reserved for
custom ID properties only.

Use :func:`rna_get` / :func:`rna_set` whenever a sub-property could be
either a real RNA attribute or a user-set custom property. They prefer
attribute access (which works in both majors) and fall back to subscript
access so genuine custom properties still resolve.

Add new helpers here as the bridge encounters other 4 → 5 differences;
keep the module bpy-free at import time so the rest of the package can
type-check without ``fake-bpy-module`` reaching into the shim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Hashable

__all__ = ["rna_get", "rna_set"]

_MISSING = object()


def rna_get(owner: object, key: Hashable, default: object = None) -> object:
    """Read ``key`` from an RNA-owning data block in a 4.x/5.x-safe way.

    Resolution order:

    1. ``getattr(owner, key)`` when ``key`` is a string and the attribute
       exists. Covers every RNA-defined sub-property on both 4.x and 5.x.
    2. ``owner[key]`` when the owner supports subscripting and the key
       resolves. Covers custom ID properties (still subscript-only in 5.0)
       and 4.x fallthrough access for RNA properties.
    3. ``default`` otherwise.

    Parameters
    ----------
    owner
        Any bpy data block (``Scene``, ``World``, ``Object``, ...) or
        other object exposing RNA + custom-property storage.
    key
        Attribute / item name to look up.
    default
        Value to return when neither lookup succeeds.

    Returns
    -------
    object
        The resolved value, or ``default``.

    """
    if isinstance(key, str):
        value = getattr(owner, key, _MISSING)
        if value is not _MISSING:
            return value
    # Subscript access for custom ID properties. ``Any`` is justified
    # here: ``owner`` is genuinely any bpy data block at runtime, and
    # the whole point of this helper is to paper over both subscript
    # and attribute access on the same object.
    try:
        return cast("Any", owner)[key]
    except (KeyError, TypeError):
        return default


def rna_set(owner: object, key: Hashable, value: object) -> None:
    """Write ``key`` on an RNA-owning data block in a 4.x/5.x-safe way.

    Mirrors :func:`rna_get`: attribute assignment first (the only path
    that works for RNA-defined sub-properties on 5.0), subscript
    assignment as a fallback for custom ID properties.

    Parameters
    ----------
    owner
        The bpy data block to mutate.
    key
        Attribute / item name to set.
    value
        New value.

    Raises
    ------
    TypeError
        When neither attribute nor subscript assignment is supported by
        ``owner``.

    """
    if isinstance(key, str) and hasattr(owner, key):
        setattr(owner, key, value)
        return
    try:
        cast("Any", owner)[key] = value
    except TypeError as exc:
        msg = f"cannot set {key!r} on {type(owner).__name__}: {exc}"
        raise TypeError(msg) from exc
