# Animated glyphs

`pl.blender.export_animation_blend(..., bake_glyphs=True)` extends the
GN-instanced glyph path with per-frame animation. The bridge
samples each registered `GlyphSpec`'s source dataset at every frame
of the timeline, packs the per-channel values into float-precision
images embedded in the `.blend`, and injects a small Geometry Nodes
sub-graph upstream of the existing instancer so the saved file plays
back natively — no Python script, no auto-execution prompt.

## A moving vector field

```python
import numpy as np
import pyvista as pv

# Static grid of points; the per-frame updater rotates the orient vectors.
xs = np.linspace(-1, 1, 6)
xx, yy, zz = np.meshgrid(xs, xs, [0.0], indexing="ij")
points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

cloud = pv.PolyData(points)
cloud["vec"] = np.tile([1.0, 0.0, 0.0], (cloud.n_points, 1)).astype(np.float32)
cloud["mag"] = np.ones(cloud.n_points, dtype=np.float32)

pl = pv.Plotter(off_screen=True, window_size=(960, 720))
pl.blender.add_glyph(
    cloud, geom=pv.Arrow(), orient="vec", scale="mag", factor=0.3
)
pl.camera_position = [(3.5, -3.5, 3.5), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

def update(frame: int) -> None:
    # Spin the orient vectors around +Z and pulse the scalar mag.
    theta = frame * (2.0 * np.pi / 24)
    cloud["vec"] = np.tile(
        [np.cos(theta), np.sin(theta), 0.0], (cloud.n_points, 1)
    ).astype(np.float32)
    cloud["mag"] = (0.5 + 0.5 * np.sin(theta)) * np.ones(
        cloud.n_points, dtype=np.float32
    )

pl.blender.export_animation_blend(
    "vector_field.blend",
    update,
    frames=range(24),
    fps=24,
    bake_camera=False,
    bake_glyphs=True,
)
```

Open `vector_field.blend` and press Space — the arrows rotate and
pulse without any Python evaluation. The animation rides on
keyframed image samples in the GN modifier.

## What gets baked

| Channel     | Source                           | Packed format                       | GN sink                                            |
| ----------- | -------------------------------- | ----------------------------------- | -------------------------------------------------- |
| `positions` | `spec.source.points`             | RGB × N_frames × N_points (alpha=1) | `Set Position` node upstream of the instancer      |
| `orient`    | `spec.source.point_data[orient]` | RGB × N_frames × N_points           | `Store Named Attribute("pv_orient", FLOAT_VECTOR)` |
| `scale`     | `spec.source.point_data[scale]`  | R × N_frames × N_points             | `Store Named Attribute("pv_scale", FLOAT)`         |

Each channel is baked independently. A channel whose per-frame values
are constant across the entire timeline is detected and **skipped** so
the saved tree stays clean (no Set Position node, no image, no
sampler) and the existing static value continues to drive that
channel.

## Sampler topology

Inside the GN tree, upstream of the existing instancer:

```
Index ─┐
       ├─ (i + 0.5) / N_points → u
       │
Scene Time ─ (frame - frame_start + 0.5) / N_frames → v
       │
Combine XYZ (u, v) → Image Texture (packed)
       │
       └─ → Set Position / Store Named Attribute
                 ↓
            Existing instancer
```

The Image Texture node uses `Closest` interpolation so each frame
reads the exact pixel for the current `Scene Time`, not a temporal
blend across rows.

## Constraints and caveats

- **Constant topology.** All frames must have the same `N_points`.
  Topology-changing updaters aren't supported (no warning yet — the
  bake will just produce a malformed image).
- **No cell-data scalars.** `orient` and `scale` must live on
  `point_data` (the `add_glyph` API requires this for the static
  path too).
- **Float-precision images.** Each channel is a 32-bit float OpenEXR
  packed inside the `.blend`. Per-channel size scales as
  `4 bytes × N_frames × N_points × 4 channels`. A 24-frame × 10,000-
  point bake is ~3.7 MB per channel.
- **Scene Time vs. frame remap.** Playback reads `Scene Time.Frame`
  directly, so users who shift `scene.frame_start` after export will
  see the animation scroll past the baked range; re-bake or adjust
  the GN `frame_start` subtract constant if you need to remap.
