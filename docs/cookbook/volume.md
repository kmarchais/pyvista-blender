# Volume rendering

`pl.add_volume(image_data, scalars=..., cmap=..., opacity=...)` renders
a 3D scalar field as a translucent cloud through Blender's
Cycles Volume Principled shader. The bridge builds a closed cube mesh
spanning the volume's world-space bounds and drives the shader's
Emission Color + Density from a packed image-atlas lookup keyed by
world position.

**No OpenVDB / pyopenvdb dependency** — the bridge sidesteps the
upstream blocker (no pyopenvdb wheels for Python 3.11+) by doing the
3D scalar lookup in the shader graph rather than going through
Blender's native Volume data-block.

```python
import numpy as np
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=(800, 640))

# Build a 40-voxel-per-side cube with a sine-wave density field.
grid = pv.ImageData(dimensions=(40, 40, 40), spacing=(0.1, 0.1, 0.1))
x, y, z = grid.points.T
grid["density"] = (
    np.sin(3 * x) * np.cos(3 * y) * np.sin(3 * z) * 0.5 + 0.5
).astype(np.float32)

pl.add_volume(
    grid,
    scalars="density",
    cmap="inferno",
    opacity="linear",
    show_scalar_bar=False,
)
pl.camera_position = [(8, -6, 6), (2, 2, 2), (0, 0, 1)]
pl.renderer.set_background("#0a0e14")

pl.blender.render("volume.png", samples=64)
pl.close()
```

The rendered `volume.png` shows the inferno colormap with the
sine-wave structure clearly visible in 3D — purple low-density
regions fading into orange-yellow high-density peaks, lit by the
volume's own emission.

## What gets baked

- **Bounding box**: a closed cube mesh spanning the volume's world
  bounds (`dataset.bounds`).
- **Scalar atlas**: a single PNG packing the 3D field as a 2D image
  (`nz` horizontal slices, each `nx × ny`). The image lives inside
  the `.blend` (`image.pack()`).
- **Material**: Volume Principled BSDF with Emission Color driven by
  pyvista's colormap (sampled from `vol.mapper.lookup_table` — 256
  baked RGBA stops) and Density driven by pyvista's opacity transfer
  function (the alpha channel of the same LUT).
- **Shader-graph 3D lookup**: Position → normalised `(u, v, w)` →
  atlas UV = `((slice_index + u) / nz, v)` where
  `slice_index = floor(w * nz)`.

## Grid input types

The bridge accepts the four common pyvista grid types:

| Input                 | Path                                                                      |
| --------------------- | ------------------------------------------------------------------------- |
| `pv.ImageData`        | Zero-copy. Read scalars, bake atlas.                                      |
| `pv.RectilinearGrid`  | Resampled to ImageData via `vtkResampleToImage` (default `64 x 64 x 64`). |
| `pv.StructuredGrid`   | Resampled to ImageData.                                                   |
| `pv.UnstructuredGrid` | Resampled to ImageData.                                                   |

Resampling runs on a regular grid built over the source bounds. Points
that fall outside the source mesh's cells (e.g. concavities or the
empty regions around a tetrahedral mesh) are flagged via VTK's
`vtkValidPointMask` and the bridge collapses them to the low end of
the opacity transfer function — non-convex domains render with their
actual silhouette rather than as a filled bounding box.

For finer interior detail on a non-ImageData input, push more points
through the resampler before calling `add_volume` — either by
manually invoking `vtkResampleToImage` with larger
`SamplingDimensions` and feeding the result to `pl.add_volume`, or by
pre-converting your data to `pv.ImageData` at the resolution you want.

## Constraints and caveats

- **Point-data scalars.** The bridge reads
  `dataset.point_data[scalars]` and reshapes to `(nz, ny, nx)`. For
  cell-data scalars, call `mesh = mesh.cell_data_to_point_data()`
  before `add_volume`.
- **Resolution × memory.** The packed atlas is
  `nx*nz × ny × 1 byte`. A 128³ volume = ~2 MB packed, comfortable;
  256³ = ~16 MB, slower to render; beyond that you'll want to
  downsample.
- **Self-lit emission.** Volume Principled's `Color` socket is the
  _absorption_ colour — it tints transmitted light but reads as black
  without scene lighting. The bridge wires Emission Color to the same
  colormap so the volume is visible in any scene; if you add lights,
  the absorption tint layers on top.
- **Opacity transfer function = LUT alpha.** PyVista's
  `add_volume(..., opacity="linear" | "sigmoid" | ...)` bakes the
  transfer function into `vol.mapper.lookup_table`'s alpha channel,
  which the bridge reads directly. Custom Python callable transfer
  functions are picked up the same way (pyvista evaluates them into
  the LUT at `add_volume` time).
- **Density tuning.** Volume Principled's Density is in
  inverse-distance-along-ray units. The bridge defaults to a
  scale of `4.0 / opacity_unit_distance`, tuned for unit-sized
  volumes. Edit the material's `MULTIPLY` math node in Blender to
  brighten / dim the result.
- **Mix with mesh actors.** Mesh actors render alongside volumes in
  the same frame — every animation-export channel (camera, lights,
  transforms, materials, scalars, deformation, and now `bake_volume`)
  is independently togglable so you can bake the channels you want
  and leave the rest editable in Blender.

## Animating the volume itself

`pl.blender.export_animation_blend(..., bake_volume=True)` packs the
per-frame scalar field into a single multi-frame atlas image (frames
stacked vertically on top of the static slice atlas) and keyframes a
`ShaderNodeValue` so the volume material's shader graph scrolls the
atlas-V coordinate through the frame bands at playback. The file
stays self-contained — no sidecar, no Python in the `.blend`.

### Recommended: `pl.blender.add_volume`

```python
import numpy as np
import pyvista as pv

pl = pv.Plotter(off_screen=True)
grid = pv.ImageData(dimensions=(40, 40, 40), spacing=(0.1, 0.1, 0.1))
grid["density"] = np.zeros(grid.n_points, dtype=np.float32)

# Pin the original grid on the bridge — pl.add_volume internally copies,
# but the bridge knows to read from your grid.
pl.blender.add_volume(grid, scalars="density", cmap="inferno", opacity="linear")

def update(frame: int) -> None:
    # Mutate your own grid — no actor.mapper.dataset indirection needed.
    x, y, z = grid.points.T
    grid["density"] = (
        np.sin(2.0 * x + 0.3 * frame)
        * np.cos(2.0 * y)
        * np.sin(2.0 * z) * 0.5 + 0.5
    ).astype(np.float32)

pl.blender.export_animation_blend(
    "volume_anim.blend",
    update,
    frames=range(48),
    fps=24,
    bake_volume=True,
)
```

### Fallback for existing code: `pl.add_volume`

If you've already wired up `pl.add_volume(...)` somewhere upstream
and don't want to switch to the bridge wrapper, the bridge falls
back to reading from `actor.mapper.dataset` — pyvista's copy. Your
updater needs to mutate that dataset rather than the original
`grid`:

```python
vol = pl.add_volume(grid, scalars="density", cmap="inferno", opacity="linear")

def update(frame: int) -> None:
    ds = vol.mapper.dataset          # the *actor's* dataset, not yours
    x, y, z = ds.points.T
    ds["density"] = ...
```

The first form is preferred for new code — it keeps the updater
focused on the user-owned object instead of digging into VTK internals.

Open the resulting `.blend` and press Space — Blender plays the
animation by stepping the keyframed Value node through 48 atlas
bands, looking up the right slice band per shading point at each
frame. The atlas size scales as `nx*nz × ny*n_frames`: a 64-cube
volume × 48 frames is ~12 MB packed.

Volumes whose scalars don't actually change across frames keep their
single-frame material, `bake_volume=True` notices the constant field
and skips the bake to keep the saved action clean.
