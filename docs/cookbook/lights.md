# Lights

PyVista's lighting model has two layers:

1. A **default light kit** (`vtkLightKit`, 5 lights) added to every
   plotter unless the user passes `lighting="none"`.
2. **Custom lights** added via `pl.add_light(pv.Light(...))`.

The bridge translates both. The light kit's headlight + key + fill +
back + camera-back become five PBR-tuned bpy lights on every render.
Custom `pv.Light` instances are translated according to their
`positional` + `cone_angle` combination — see the dispatch table
below — so a real CFD scene with directional and spot rigs renders
the same in Cycles as it does in VTK.

## SUN / POINT / SPOT dispatch

| PyVista                                       | Blender               | Notes                                                                          |
| --------------------------------------------- | --------------------- | ------------------------------------------------------------------------------ |
| `pv.Light(positional=False)`                  | `Light(type='SUN')`   | Directional, infinite-distance — direction comes from `position - focal_point` |
| `pv.Light(positional=True, cone_angle >= 90)` | `Light(type='POINT')` | Omnidirectional point source                                                   |
| `pv.Light(positional=True, cone_angle < 90)`  | `Light(type='SPOT')`  | Spot cone; `spot_size = 2 * cone_angle` (Blender uses full-cone angle)         |
| `pv.Light(light_type='headlight')`            | Camera-parented POINT | Follows the camera; behaves identically across `render` / `animate` / `show`   |
| `pv.Light(light_type='camera light')`         | Camera-parented POINT | Same as headlight — VTK's two names map to the same Blender setup              |

Intensity calibration is type-specific: SUN uses energy in W/m², POINT/SPOT use W. The bridge scales `pv.Light.intensity` by a per-type multiplier so a scene that reads correctly in VTK reads at the same brightness in Cycles without manual tuning.

## Default light kit

The default `vtkLightKit` decomposes into:

- **Key light** — bright frontal POINT, drives the principal shading
- **Fill light** — softer counter-angle POINT, lifts shadow detail
- **Back light** — rim from behind, opposite the key
- **Headlight** — camera-attached low-intensity POINT
- **Camera-back** — wide-aperture POINT on the camera's opposite side

The kit is camera-parented in the interactive viewport; the offline
`render()` path composes `camera × local-light` matrices directly so
the lighting is locked to the rendered camera pose.

If you want to start from scratch, pass `lighting="none"` to the
plotter:

```python
plotter = pv.Plotter(lighting="none")
plotter.add_mesh(...)
plotter.add_light(pv.Light(...))  # build your own rig
```

## Custom three-light stage rig

```python
import pyvista as pv

torus = pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.35)

plotter = pv.Plotter(off_screen=True, window_size=[800, 600], lighting="none")
plotter.add_mesh(
    torus, color="#e0c2a6", pbr=True, metallic=0.2, roughness=0.35
)
plotter.set_background("#0c0f14")
plotter.camera_position = [(3.2, -3.2, 2.6), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

# Soft directional fill from above-back-left → SUN in Blender.
sun = pv.Light(
    position=(-2.0, -2.5, 4.0),
    focal_point=(0.0, 0.0, 0.0),
    color="#a0c8ff",
    intensity=0.45,
    light_type="scene light",
)
sun.positional = False  # → Blender SUN
plotter.add_light(sun)

# Warm omnidirectional key from the right → POINT in Blender.
point = pv.Light(
    position=(3.0, 0.5, 1.5),
    focal_point=(0.0, 0.0, 0.0),
    color="#ffb070",
    intensity=0.9,
    light_type="scene light",
)
point.positional = True
point.cone_angle = 90.0  # → POINT
plotter.add_light(point)

# Tight focused beam from below-front → SPOT in Blender.
spot = pv.Light(
    position=(0.5, -2.0, -1.5),
    focal_point=(0.0, 0.0, 0.0),
    color="#ffffff",
    intensity=1.2,
    light_type="scene light",
)
spot.positional = True
spot.cone_angle = 25.0  # < 90 → SPOT, spot_size = 50°
plotter.add_light(spot)

plotter.blender.render("custom_lights.png", samples=96)
```

This is `examples/lights/custom_lights.py`; it exercises every branch of
`translate/light.py` in a single frame so the visual difference
between the three light types is obvious side-by-side.

## Animating lights (`bake_lights=True`)

For `export_animation_blend(..., bake_lights=True)`, each
`pv.Light`'s world position, focal point, intensity, and colour are
keyframed per frame. Light identity is the index in
`plotter.renderers[*].lights`, matching the `PVLight_{i}` naming
convention the translator uses.

```python
light = pv.Light(position=(5, 0, 0), light_type="scene light", intensity=1.0)
plotter.add_light(light)

def update(frame: int) -> None:
    # Orbit the light and pulse its intensity.
    light.position = (5 * np.cos(frame * 0.5), 5 * np.sin(frame * 0.5), 3.0)
    light.intensity = 0.5 + 0.5 * np.sin(frame * 0.3)

plotter.blender.export_animation_blend(
    "moving_lights.blend",
    update,
    frames=range(60),
    fps=24,
    bake_lights=True,
)
```

Static channels (e.g. colour, when the colour doesn't change across
frames) are skipped — the saved action only carries what actually
animates.

## Constraints and caveats

- **Energy units differ between SUN and POINT/SPOT.** SUN uses W/m²,
  POINT/SPOT use W. The bridge applies type-specific multipliers so
  `pv.Light.intensity` reads consistently across types; if you tune
  in Blender directly, expect 1000x scale differences.
- **`pl.show(backend="desktop")` parents lights to the camera** at
  installation time so the hybrid viewport's drag preview moves the
  lighting with the camera the same way the offline render does.
- **Eevee Next** has different light energy semantics than Cycles
  (it normalises intensity in a slightly different way). When
  switching engines, expect to retune intensity values ±20%.
