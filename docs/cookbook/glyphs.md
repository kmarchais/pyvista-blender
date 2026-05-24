# Glyph instancing

Calling `pl.add_mesh(source.glyph(geom=geom))` bakes one copy of `geom`
per source point into a single polydata — render cost scales `N · V`.
`pl.blender.add_glyph(source, geom, ...)` routes through Blender's
`GeometryNodeInstanceOnPoints` instead: the glyph mesh data is stored
once and instances are generated at render time, so memory + render cost
scale `N + V`.

```python
import numpy as np
import pyvista as pv

n = 11
coords = np.linspace(-1.0, 1.0, n)
xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
positions = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
points = pv.PolyData(positions)

# Swirl: rotation around Z plus a small radial component.
points["vec"] = np.column_stack([
    -points.points[:, 1] - 0.2 * points.points[:, 0],
    points.points[:, 0] - 0.2 * points.points[:, 1],
    0.4 * np.cos(np.linalg.norm(points.points[:, :2], axis=1) * 2.5),
])
points["mag"] = np.linalg.norm(points["vec"], axis=1)

plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
plotter.blender.add_glyph(
    points,
    geom=pv.Arrow(),
    orient="vec",
    scale="mag",
    factor=0.25,
)
plotter.camera_position = [(3.0, -3.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

plotter.blender.render("glyph_vectors.png", samples=64)
plotter.close()
```

Full script: [`examples/glyphs/glyph_vectors.py`](https://github.com/kmarchais/pyvista-blender/blob/main/examples/glyphs/glyph_vectors.py).

## Signature

```python
pl.blender.add_glyph(
    source,                # pv.DataSet whose points become instance origins
    geom,                  # pv.DataSet — the glyph shape (pv.Arrow, pv.Cone, ...)
    *,
    orient=None,           # name of a point-data 3D vector field for orientation
    scale=None,            # name of a point-data scalar field for per-point scale
    factor=1.0,            # global scale multiplier on top of `scale`
    name=None,             # optional base name for the bpy data blocks
)
```

`orient=None` → identity rotation; `scale=None` → uniform scale.

## Inspecting and clearing the queue

`pl.blender.registered_glyphs` returns a tuple of `GlyphSpec`s in
registration order — read-only, snapshot-style. `pl.blender.clear_glyphs()`
empties the queue.

## Baked-glyph detection

If you pass a pre-baked `mesh.glyph(...)` polydata to `add_mesh(...)`,
the bridge detects the `GlyphVector` point-data signature that PyVista
leaves behind and emits a `UserWarning` directing you to `add_glyph()`.

The baked mesh still renders correctly — true auto-unbake is lossy
(original source points + glyph mesh aren't recoverable from the merged
polydata), so we render what you handed us and point at the lighter
path.
