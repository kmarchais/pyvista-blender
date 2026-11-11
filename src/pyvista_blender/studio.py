# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""The studio lighting rig, without ``bpy``.

A key / fill / rim set of lights placed in the camera's frame and sized to
the scene's visible bounds. The Blender translation turns them into
shadow-casting area lights (``config.lighting = "studio"``);
``pl.blender.add_studio_lights()`` adds the same lights to the PyVista
preview so both renderers are lit from the same places.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

__all__ = [
    "BACKGROUND_FILL",
    "VTK_BACKGROUND_FILL",
    "VTK_KEY_INTENSITY",
    "StudioLight",
    "studio_rig",
    "uses_studio_rig",
]

#: The rig: name, direction in the camera's frame (towards the camera, its
#: right, its up; normalised on use), distance from the scene's centre as a
#: fraction of the bounds diagonal, and irradiance at that centre (W/m² in
#: Blender units: the power is this times the distance squared). Key above
#: and beside the camera, fill from the right, rim from behind on the left.
#: Tuned by eye on a dense packed-geometry scene and a primitives-on-a-plane
#: scene, together with :data:`BACKGROUND_FILL`: at this level 0.1 % and 3 %
#: of their pixels clip on a Standard view transform (1.4x the power: 2 % and
#: 18 %).
_RIG = (
    ("key", (0.85, 0.27, 0.45), 0.72, 6.1),
    ("fill", (-0.21, 0.92, 0.32), 0.56, 3.85),
    ("rim", (-0.32, -0.68, 0.67), 0.58, 3.65),
)
#: Side of each area light, as a fraction of the bounds diagonal.
_SIZE = 0.4
_MIN_EXTENT = 1e-12  # a scene with no visible extent has nothing to light

#: Strength of the white all-around fill in the Blender render (the world
#: for shading rays; camera rays still see the background colour).
BACKGROUND_FILL = 0.5

#: VTK preview of the rig (``pl.blender.add_studio_lights()``): the key's
#: intensity (fill and rim scaled by their irradiance, as VTK lights do not
#: fall off with distance) and a camera light standing in for the
#: background's fill. Matched to the Cycles render's mean luminance on the
#: two tuning scenes (within 2 %).
VTK_KEY_INTENSITY = 0.8
VTK_BACKGROUND_FILL = 0.2


@dataclass(frozen=True)
class StudioLight:
    """One light of the rig, in world coordinates."""

    name: str
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    up: tuple[float, float, float]
    size: float
    energy: float
    irradiance: float


def studio_rig(renderer: pv.Renderer) -> list[StudioLight]:
    """Place the rig about ``renderer``'s camera and visible bounds.

    Returns
    -------
    list of StudioLight
        Key, fill and rim, each aimed at the centre of the visible bounds;
        empty when nothing visible has an extent.

    """
    bounds = np.asarray(renderer.ComputeVisiblePropBounds(), dtype=float)
    lo, hi = bounds[0::2], bounds[1::2]
    size = float(np.linalg.norm(hi - lo)) if np.all(hi >= lo) else 0.0
    if size <= _MIN_EXTENT:
        return []
    centre = 0.5 * (lo + hi)
    cam = renderer.GetActiveCamera()
    back = np.asarray(cam.GetPosition(), dtype=float) - np.asarray(
        cam.GetFocalPoint(), dtype=float
    )
    back /= np.linalg.norm(back)
    up = np.asarray(cam.GetViewUp(), dtype=float)
    up -= back * float(up @ back)
    up /= np.linalg.norm(up)
    right = np.cross(up, back)
    rig = []
    for name, (a, b, c), dist, irradiance in _RIG:
        direction = a * back + b * right + c * up
        direction /= np.linalg.norm(direction)
        distance = dist * size
        position = centre + distance * direction
        rig.append(
            StudioLight(
                name=name,
                position=(float(position[0]), float(position[1]), float(position[2])),
                focal_point=(float(centre[0]), float(centre[1]), float(centre[2])),
                up=(float(up[0]), float(up[1]), float(up[2])),
                size=_SIZE * size,
                energy=irradiance * distance**2,
                irradiance=irradiance,
            )
        )
    return rig


#: Relative tolerance (of the bounds diagonal) when matching a scene light to
#: a rig position: the rig's positions are recomputed in float64 from the same
#: camera and bounds, so only round-off separates them.
_MATCH_RTOL = 1e-9


def uses_studio_rig(renderer: pv.Renderer) -> bool:
    """Whether studio lighting should replace ``renderer``'s lights.

    The rig stands in for the default lights (the light kit, ``"three
    lights"``: all attached to the camera) and for the preview rig of
    ``pl.blender.add_studio_lights()``. Any other light placed in the scene
    is the user's own setup and is kept.

    Returns
    -------
    bool
        ``True`` when every light is camera-attached or one of the rig's.

    """
    placed = [lt for lt in renderer.lights if lt.is_scene_light]
    if not placed:
        return True
    rig = studio_rig(renderer)
    if not rig:
        return False
    spots = np.array([spec.position for spec in rig])
    scale = float(np.linalg.norm(spots - np.asarray(rig[0].focal_point), axis=1).max())
    for lt in placed:
        gap = np.linalg.norm(spots - np.asarray(lt.position, dtype=float), axis=1)
        if float(gap.min()) > _MATCH_RTOL * scale:
            return False
    return True
