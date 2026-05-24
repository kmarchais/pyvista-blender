# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hide and restore PyVista layer-0 actors during ``pl.blender.show()``.

The Cycles overlay sits on layer 1 of the render window; PyVista's
normal 3D scene on layer 0 would render *behind* it. When the overlay
is opaque (the default) that's wasted work and a subtle source of
flicker. The two helpers here capture every layer-0 actor's
``visibility`` flag on install, then restore them on teardown.

Limited to ``pv.Actor`` instances and the renderer's ``actors`` dict —
2D HUD actors (scalar bars, axes widget) are owned by the renderer's
2D overlay infrastructure and don't pay the cost; leaving them visible
is fine and keeps the test snapshot a single-call API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["VisibilitySnapshot", "hide_underlying_actors", "restore_underlying_actors"]


VisibilitySnapshot = dict[str, bool]


def hide_underlying_actors(plotter: pv.BasePlotter) -> VisibilitySnapshot:
    """Hide every visible ``pv.Actor`` on layer 0 and return a snapshot to undo.

    Parameters
    ----------
    plotter
        Source plotter; ``plotter.renderer.actors`` is walked and
        mutated.

    Returns
    -------
    dict[str, bool]
        ``{actor_name: previous_visibility}`` for every actor whose
        flag was flipped. Pass back to
        :func:`restore_underlying_actors` to put them back. Empty when
        the plotter has no visible actors.

    """
    snapshot: VisibilitySnapshot = {}
    for name, actor in plotter.renderer.actors.items():
        if not hasattr(actor, "visibility"):
            continue
        if not actor.visibility:
            continue
        snapshot[name] = bool(actor.visibility)
        actor.visibility = False
    return snapshot


def restore_underlying_actors(
    plotter: pv.BasePlotter, snapshot: VisibilitySnapshot
) -> None:
    """Restore actor visibility from a prior :func:`hide_underlying_actors` snapshot.

    Actors that no longer exist on the plotter (removed via
    ``remove_actor`` while the overlay was up) are skipped silently —
    the snapshot is advisory, not authoritative.

    Parameters
    ----------
    plotter
        Source plotter to restore.
    snapshot
        The dict returned by :func:`hide_underlying_actors`.

    """
    for name, was_visible in snapshot.items():
        actor = plotter.renderer.actors.get(name)
        if actor is None or not hasattr(actor, "visibility"):
            continue
        actor.visibility = was_visible
