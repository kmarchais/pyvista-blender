# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Throttle helpers for VTK observer callbacks.

VTK's ``InteractionEvent`` fires per mouse-motion frame during a drag —
60 Hz on a smooth swipe, far faster than Cycles can keep up with at any
reasonable sample count. ``should_render_now`` decides whether enough
time has elapsed since the previous fire to justify another render.

Pure function, no state. The caller threads the "last render time" in
as a parameter.
"""

from __future__ import annotations

#: Throttle interval for ``InteractionEvent``-driven re-renders during
#: a mouse drag, in milliseconds. 80 ms ≈ 12.5 fires per second,
#: which matches the interactive Cycles pass latency budget of
#: 50-100 ms at low sample counts.
INTERACTION_THROTTLE_MS: float = 80.0


def should_render_now(now: float, last_render_at: float) -> bool:
    """Return whether enough time has elapsed since the last render.

    Parameters
    ----------
    now
        Current wall time, typically from ``time.monotonic()``.
    last_render_at
        Wall time of the previous render, in the same clock as ``now``.
        ``0.0`` is the canonical "never rendered" sentinel — the first
        call always returns ``True``.

    Returns
    -------
    bool
        ``True`` when the elapsed time is at or above
        :data:`INTERACTION_THROTTLE_MS`; ``False`` to drop the event.

    """
    elapsed_ms = (now - last_render_at) * 1000.0
    return elapsed_ms >= INTERACTION_THROTTLE_MS
