# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the bpy 4↔5 RNA shim in :mod:`pyvista_blender._compat`.

The shim is bpy-free at import time and operates on duck-typed objects,
so it can be exercised with stand-in classes that mimic the three
shapes a real bpy data block can take: attribute-only (RNA), subscript-
only (custom ID property bag), and both.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pyvista_blender import rna_get, rna_set


@dataclass
class _AttrOnly:
    """Mimics an RNA-defined sub-property: attribute access only."""

    field: object


class _SubscriptOnly:
    """Mimics a custom ID-property bag: subscript access only."""

    def __init__(self, value: object) -> None:
        self._store = {"field": value}

    def __getitem__(self, key: str) -> object:
        return self._store[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._store[key] = value


class _BothAttrAndSubscript:
    """Mimics a 4.x RNA owner: attribute *and* subscript both work."""

    def __init__(self, attr_value: object, subscript_value: object) -> None:
        self.field = attr_value
        self._store = {"field": subscript_value}

    def __getitem__(self, key: str) -> object:
        return self._store[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._store[key] = value


def test_rna_get_reads_attribute_when_present() -> None:
    """Plain attribute access wins on a 5.x RNA owner."""
    owner = _AttrOnly("hello")
    if rna_get(owner, "field") != "hello":
        pytest.fail("rna_get did not read the attribute")


def test_rna_get_falls_back_to_subscript() -> None:
    """Subscript fallback fires when the attribute is absent."""
    owner = _SubscriptOnly("world")
    if rna_get(owner, "field") != "world":
        pytest.fail("rna_get did not fall back to subscript access")


def test_rna_get_prefers_attribute_over_subscript() -> None:
    """When both paths resolve, the attribute takes precedence.

    Reflects the 4.x → 5.x intent: RNA-defined sub-properties are the
    canonical surface; the subscript path exists for custom ID
    properties only.
    """
    owner = _BothAttrAndSubscript(attr_value="rna", subscript_value="custom")
    if rna_get(owner, "field") != "rna":
        pytest.fail("rna_get preferred subscript over attribute")


def test_rna_get_returns_default_when_nothing_resolves() -> None:
    """Missing on both paths → ``default``."""
    owner = _AttrOnly("present")
    if rna_get(owner, "missing", default="sentinel") != "sentinel":
        pytest.fail("rna_get did not surface the default for a missing key")


def test_rna_get_default_is_none() -> None:
    """The default default is ``None``."""
    owner = _AttrOnly("present")
    if rna_get(owner, "missing") is not None:
        pytest.fail("rna_get default should be None when not supplied")


def test_rna_set_writes_attribute_when_present() -> None:
    """Attribute assignment fires when the attribute already exists."""
    owner = _AttrOnly("initial")
    rna_set(owner, "field", "updated")
    if owner.field != "updated":
        pytest.fail(f"rna_set did not update the attribute, got {owner.field!r}")


def test_rna_set_falls_back_to_subscript() -> None:
    """Subscript assignment fires when the attribute is absent."""
    owner = _SubscriptOnly("initial")
    rna_set(owner, "field", "updated")
    if owner["field"] != "updated":
        pytest.fail(f"rna_set did not update via subscript, got {owner['field']!r}")


def test_rna_set_raises_when_neither_path_works() -> None:
    """An object without attribute or __setitem__ surfaces a TypeError."""

    class _Frozen:
        __slots__ = ()

    with pytest.raises(TypeError, match=r"cannot set"):
        rna_set(_Frozen(), "field", "anything")
