# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Three-tier sample-count resolution for the interactive viewport.

``pl.blender.show()`` exposes three sample tiers (interacting,
settling, idle) plus a legacy single-tier ``samples`` blanket. Each
tier resolves through the same three layers:

1. Explicit per-tier kwarg (e.g. ``samples_interacting=8``).
2. Legacy single-tier ``samples`` override (applies to every tier
   that didn't get its own value).
3. The matching :mod:`pyvista_blender.config` default
   (``interactive_samples`` / ``settled_samples`` / ``idle_samples``).

:func:`resolve_tier_samples` is a pure function so the resolution
logic can be unit-tested without spinning up an interactive plotter.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyvista_blender import config

__all__ = ["TierSamples", "resolve_tier_samples"]


@dataclass(frozen=True)
class TierSamples:
    """Per-tier sample counts after resolution."""

    interacting: int
    settling: int
    idle: int


def resolve_tier_samples(
    *,
    samples: int | None = None,
    samples_interacting: int | None = None,
    samples_settling: int | None = None,
    samples_idle: int | None = None,
) -> TierSamples:
    """Pick the per-tier sample counts from the layered ``show()`` kwargs.

    Parameters
    ----------
    samples
        Legacy single-tier blanket. Applies to every tier that wasn't
        set explicitly.
    samples_interacting, samples_settling, samples_idle
        Per-tier explicit overrides. ``None`` falls through to
        ``samples``, then to the matching ``config`` default.

    Returns
    -------
    TierSamples
        Resolved counts ready for ``render_and_blit``.

    """

    def _pick(per_tier: int | None, default: int) -> int:
        if per_tier is not None:
            return per_tier
        if samples is not None:
            return samples
        return default

    return TierSamples(
        interacting=_pick(samples_interacting, config.interactive_samples),
        settling=_pick(samples_settling, config.settled_samples),
        idle=_pick(samples_idle, config.idle_samples),
    )
