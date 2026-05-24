# Export to a `.blend` file

`pl.blender.export_blend(path)` writes the live PyVista scene to a
Blender file. Open it in Blender's UI to tweak materials, animate
something the bridge doesn't expose, add props, render at higher
quality, or stash a snapshot for later.

```python
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=[800, 600])
pl.add_mesh(
    pv.examples.download_bunny(),
    color="#dec27c",
    pbr=True,
    metallic=0.3,
    roughness=0.35,
)
pl.add_light(pv.Light(position=(5, -5, 5), light_type="scene light"))
pl.camera_position = [(0.2, 0.18, 0.25), (-0.02, 0.1, 0.0), (0, 1, 0)]

pl.blender.export_blend("bunny.blend")
pl.close()
```

`bunny.blend` opens cleanly in Blender 4.5+ / 5.x. All the actors,
materials, lights, world shader, and camera that `pl.blender.render`
would draw are present in the file.

## What's preserved

| PyVista state                                             | Survives the export                                                             |
| --------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Meshes (PolyData, UnstructuredGrid via `extract_surface`) | yes — as `bpy.types.Mesh`                                                       |
| Materials (PBR, Phong, unlit, backface)                   | yes — full shader graphs                                                        |
| Light kits (SUN, POINT, SPOT, HEADLIGHT, CAMERA_LIGHT)    | yes                                                                             |
| World shader (solid / gradient / HDRI)                    | yes                                                                             |
| Glyphs registered via `pl.blender.add_glyph(...)`         | yes — as Geometry-Nodes-instanced objects                                       |
| Camera (pose + clipping range + ortho/perspective)        | yes                                                                             |
| Identity cache                                            | preserved — subsequent `render()` calls on this plotter still reuse data blocks |

## What's not preserved

- HUD overlays (scalar bars / text / axes / bounds) are composited
  _after_ the Cycles render, not stored in the `.blend`. To get them in
  Blender, set them up manually using Blender's compositor.
- Subplot layouts get flattened — `export_blend` writes a single bpy
  scene; subplot tile compositing is a render-time operation.

## In-session vs on-disk

`export_blend` uses `bpy.ops.wm.save_as_mainfile(copy=True)`, which
means the in-memory bpy session **doesn't switch over to the new
file**. You can keep calling `pl.blender.render(...)` afterwards and
it'll continue rendering the same in-memory scene. Open the `.blend`
in a separate Blender instance to make changes that won't conflict
with the live session.

## Animation: `export_animation_blend`

`pl.blender.export_animation_blend(path, updater, frames, fps=...)`
bakes a camera animation into the saved file. Open it in Blender and
press <kbd>Space</kbd> to play; render the timeline at higher quality
through the UI; tweak the f-curves to reshape the motion.

```python
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=[800, 600])
pl.add_mesh(pv.examples.load_random_hills(), cmap="viridis")
pl.camera_position = "iso"

n_frames = 60
updater = pl.blender.orbit_camera(n_frames=n_frames)

pl.blender.export_animation_blend(
    "hills_orbit.blend",
    updater,
    frames=range(n_frames),
    fps=30,
)
pl.close()
```

The saved file has:

- `scene.camera` carrying keyframes on `location` and
  `rotation_quaternion`, one per frame in `frames`.
- `scene.frame_start = frames[0]`, `scene.frame_end = frames[-1]`,
  `scene.render.fps = fps`.
- Quaternion rotation mode, so playback avoids Euler-interpolation
  surprises (gimbal flips, wrong-side rotations).

### Selective channels

`export_animation_blend` gates each animation channel behind its own
kwarg so callers can bake only what they want and finish the rest
by hand in Blender:

```python
pl.blender.export_animation_blend(
    "scene.blend", updater, frames=range(60), fps=30,
    bake_camera=True,        # default; set False to keep camera static
    bake_deformation="mdd",  # False | True | "mdd" | "shape_keys"
)
```

Common patterns:

| Goal                                     | kwargs                                            |
| ---------------------------------------- | ------------------------------------------------- |
| Camera-only orbit (default)              | `bake_camera=True, bake_deformation=False`        |
| Deformation-only, animate camera by hand | `bake_camera=False, bake_deformation=True`        |
| Both, MDD sidecar (default for `True`)   | `bake_camera=True, bake_deformation=True`         |
| Self-contained .blend (no sidecar)       | `bake_camera=True, bake_deformation="shape_keys"` |

### Baking mesh deformation

Two backends, picked by `bake_deformation`:

#### `"mdd"` — Mesh Cache sidecar (default for `True`)

The bridge writes a Lightwave MDD vertex-cache file next to the
`.blend` (e.g. `wave__PolyData.mdd`) and adds a Mesh Cache modifier
on the corresponding bpy mesh pointing at it. Blender's built-in
modifier replays the cache on file open — **no Python script in the
.blend, no auto-execution prompt.**

```python
import numpy as np
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=[800, 600])
plane = pv.Plane(i_resolution=80, j_resolution=80)
rest = plane.points.copy()
pl.add_mesh(plane, color="#dec27c", pbr=True)
pl.camera_position = [(5.0, -5.0, 5.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

n_frames = 60
orbit = pl.blender.orbit_camera(n_frames=n_frames)


def update(frame: int) -> None:
    orbit(frame)
    r = np.linalg.norm(rest[:, :2], axis=1)
    plane.points[:, 2] = rest[:, 2] + 0.3 * np.sin(2.0 * r - frame * 0.2)


pl.blender.export_animation_blend(
    "wave.blend", update, frames=range(n_frames), fps=30,
    bake_deformation=True,  # → "mdd"
)
pl.close()
```

Ship both `wave.blend` and `wave__PolyData.mdd` together — the
modifier stores the path relative to the .blend so the pair stays
portable as long as they sit in the same directory.

#### `"shape_keys"` — self-contained morph targets

When you want everything inside the `.blend` and don't mind the
size, `bake_deformation="shape_keys"` uses one Shape Key per frame,
value-keyframed with linear interpolation.

Choosing between the two:

| Trade-off                    | `"mdd"`                                                                          | `"shape_keys"`                                        |
| ---------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `.blend` size on its own     | Small (no per-frame data)                                                        | Large (vertex sets inside the file)                   |
| Sidecar to ship              | Yes (`.mdd`, uncompressed)                                                       | No                                                    |
| **Total on-disk size**       | Often **larger** — MDD is uncompressed; `.blend` zstd compresses Shape Keys well | Often smaller for dense per-frame data thanks to zstd |
| Plays without auto-exec      | Yes                                                                              | Yes                                                   |
| Outliner clutter             | None (one modifier)                                                              | One Shape Key per frame                               |
| Interpolation between frames | Linear (modifier setting)                                                        | Linear (linear keyframe points)                       |
| Constant topology required   | Yes                                                                              | Yes                                                   |
| Plays in Blender 4.x and 5.x | Yes                                                                              | Yes                                                   |

Pick `"mdd"` for cleaner scene structure, easy diff-/edit-ability of the sidecar, and when you'd rather ship one `.blend` + one cache file than one heavy `.blend`. Pick `"shape_keys"` when total on-disk size matters more than Outliner cleanliness or when you need a fully self-contained file.

#### Cost and constraints (both backends)

- **Constant topology required**. If `mesh.points.shape[0]` changes
  mid-animation, the bridge emits a `UserWarning` and skips the
  deformation bake for that mesh. The static state in the file
  still reflects the last frame.
- **Light and material animations** beyond scalars are not yet
  keyframed in the saved file. The static state reflects the last
  frame.

### Baking scalar field evolution

`bake_scalars=True` captures per-frame per-vertex scalar values for
every actor that has an active scalar field, applies the actor's
colormap, and builds a single PNG (rows = frames, columns = vertices,
RGBA = colormapped colour). The PNG is **packed into the .blend**
via Blender's standard image-pack mechanism, so the file stays
self-contained — only the `.blend` needs to ship. The bridge then
attaches a small **Geometry Nodes** modifier that samples the packed
image at `(U = vertex_index / N_verts, V = scene_frame / N_frames)`
on every frame and stores the result into the mesh's existing
`"scalars"` Color Attribute — which the material's scalar-aware
shader graph already reads. The morph plays back natively on file
open, **no Python script needed, no auto-execution prompt.**

```python
import numpy as np
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=[800, 600])
plane = pv.Plane(i_resolution=80, j_resolution=80)
plane["heat"] = np.zeros(plane.n_points, dtype=np.float32)
pl.add_mesh(
    plane,
    scalars="heat",
    cmap="viridis",
    show_scalar_bar=False,
    clim=[-1.0, 1.0],
)
pl.camera_position = [(3, -3, 3), (0, 0, 0), (0, 0, 1)]

n_frames = 60


def update(frame: int) -> None:
    # A travelling sine wave on the heat field.
    plane["heat"] = np.sin(
        plane.points[:, 0] * 2.0 + frame * 0.2
    ).astype(np.float32)


pl.blender.export_animation_blend(
    "heat.blend",
    update,
    frames=range(n_frames),
    fps=30,
    bake_scalars=True,
)
pl.close()
```

#### Cost and constraints

- `.blend` size grows by roughly `N_frames * N_verts * 4` bytes (PNG-
  compressed). A 60-frame animation of a 6561-vertex plane adds
  ~50-200 KB depending on how much the field varies. The packed image
  lives inside the `.blend` — no external sidecar to ship.
- Mixes freely with `bake_deformation` and `bake_camera` — the GN
  modifier sits in the stack alongside the Mesh Cache modifier, both
  evaluated per frame. Note that `bake_deformation="mdd"` still
  produces an external `.mdd` sidecar (the `MESH_CACHE` modifier
  can't read from packed data); use `bake_deformation="shape_keys"`
  if you need a fully self-contained `.blend`.
- **Constant topology required** (same `n_points` every frame).
  Drifts trigger a `UserWarning` and skip the bake.
- **Both point-data and cell-data scalars are supported.**
  Point-data scalars index the image by vertex; cell-data scalars
  index by polygon (the bridge re-runs
  `extract_surface().triangulate()` per frame so the per-cell array
  aligns with the bpy mesh's post-translation polygon count). The
  Store Named Attribute domain switches between POINT and FACE
  accordingly; the material's existing scalar-aware shader graph
  picks up either domain.
- **Colormap must resolve from the actor.** The actor must have been
  added with `scalars=...` + `cmap=...` so
  `actor.mapper.lookup_table.cmap` carries the matplotlib colormap.
  Pure RGB-coloured meshes (no scalar field) are skipped silently.

### Baking material properties

`bake_materials=True` keyframes each material's Principled BSDF
inputs per frame — Base Color (when no scalar field is driving it),
Metallic, Roughness, Alpha. Only inputs that actually vary across
frames are keyframed; static materials leave the saved action clean.

```python
import numpy as np
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=(800, 600))
actor = pl.add_mesh(
    pv.Sphere(),
    color="#dec27c",
    pbr=True,
    metallic=0.0,
    roughness=0.5,
)
pl.add_light(pv.Light(position=(5, -5, 5), light_type="scene light"))
pl.camera_position = [(3, 0, 0), (0, 0, 0), (0, 0, 1)]

n_frames = 60


def update(frame: int) -> None:
    # Pulse the sphere from dielectric to metal and back.
    actor.prop.metallic = 0.5 + 0.5 * np.sin(frame * (2 * np.pi / n_frames))
    actor.prop.roughness = 0.1 + 0.4 * abs(np.sin(frame * (2 * np.pi / n_frames)))


pl.blender.export_animation_blend(
    "pulse.blend",
    update,
    frames=range(n_frames),
    fps=30,
    bake_materials=True,
)
pl.close()
```

Open `pulse.blend` and the sphere's surface flickers between matte
dielectric and shiny metal as the timeline plays.

Notes:

- **Base Color is skipped when scalars drive it.** Materials with a
  visible scalar field already wire Base Color through a Color
  Attribute / Image Texture (the scalar-bake path owns that socket);
  the material bake leaves it alone.
- **Phong → GGX conversion.** Phong-shaded properties have their
  `specular_power` converted to roughness via the Walter et al.
  (2007) fit `α = √(2 / (n + 2))`, same as the static path.
- **Dual-sided materials** (front + backface BSDFs) get both nodes
  keyframed independently.

### Baking actor transforms

`bake_transforms=True` keyframes each actor's `user_matrix` per frame.
The bpy mesh's vertex positions stay in local coords; the per-frame
transform rides on `obj.matrix_world` (decomposed into `location` /
`rotation_quaternion` / `scale` fcurves). Pairs naturally with the
static-render path that already respects `user_matrix` for one-frame
output.

```python
import numpy as np
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=(800, 600))
sphere = pv.Sphere(radius=0.5)
actor = pl.add_mesh(sphere, color="#dec27c", pbr=True)
pl.add_mesh(pv.Cube(), color="gray", opacity=0.15)
pl.camera_position = [(6, 0, 3), (0, 0, 0), (0, 0, 1)]

n_frames = 60


def update(frame: int) -> None:
    # Orbit the sphere around the origin in the XY plane.
    angle = frame * (2.0 * np.pi / n_frames)
    m = np.eye(4)
    m[0, 3] = 2.0 * np.cos(angle)
    m[1, 3] = 2.0 * np.sin(angle)
    actor.user_matrix = m


pl.blender.export_animation_blend(
    "orbit.blend",
    update,
    frames=range(n_frames),
    fps=30,
    bake_transforms=True,
)
pl.close()
```

Open `orbit.blend` and the sphere orbits the cube. Only varying
actors are keyframed; the cube (static `user_matrix`) leaves the
saved action untouched.

### Baking light animation

`bake_lights=True` keyframes each `pv.Light`'s world pose, intensity,
and colour per frame. The bridge mirrors the camera path: the
PyVista light is sampled at every `frame_index`, and the resulting
state lands on the corresponding `PVLight_{i}` bpy object's location

- `rotation_quaternion` and on its light data-block's `energy` /
  `color`. Only the channels that actually vary across frames get
  keyframed, so static lights stay clean.

```python
import numpy as np
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=[800, 600])
pl.add_mesh(pv.Sphere(), color="#dec27c", pbr=True)
pl.camera_position = [(3, 0, 0), (0, 0, 0), (0, 0, 1)]

# A light that orbits the sphere and pulses in intensity.
light = pv.Light(position=(5, 0, 5), light_type="scene light", intensity=1.0)
pl.add_light(light)

n_frames = 60


def update(frame: int) -> None:
    angle = frame * (2.0 * np.pi / n_frames)
    light.position = (5.0 * np.cos(angle), 5.0 * np.sin(angle), 5.0)
    light.intensity = 0.5 + 0.5 * np.sin(angle * 2.0)


pl.blender.export_animation_blend(
    "lit.blend", update, frames=range(n_frames), fps=30,
    bake_lights=True,
)
pl.close()
```

The `vtkLightKit` default lights (`PVLight_0` through `PVLight_4`)
stay static; only the explicit `pv.Light` added above carries
keyframes (it lands at `PVLight_5`).

## Compression

`bpy.ops.wm.save_as_mainfile` accepts a `compress` kwarg, defaulting
to `False` (uncompressed). The bridge doesn't surface that knob —
the resulting file may end up zstd-compressed anyway on Blender 5.x
depending on the user-preferences override. Tests that need to be
robust should accept both the uncompressed `BLENDER` magic and the
zstd `\\x28\\xb5\\x2f\\xfd` magic.
