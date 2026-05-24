# Point clouds

`pl.add_mesh(cloud, style="points")` and `style="points_gaussian"`
render each vertex of the actor's dataset as a discrete particle
instead of triangulating it into a surface. The bridge dispatches both
to a **native `bpy.types.PointCloud` data-block** rather than building
a sphere mesh per point. Cycles renders point clouds as per-point
spheres directly, so memory + render cost scale with `N_points`
instead of `N_points * N_geom_verts`.

## Opaque points (`style="points"`)

```python
import numpy as np
import pyvista as pv

rng = np.random.default_rng(seed=42)
points = rng.normal(0.0, 0.8, size=(2_000, 3)).astype(np.float32)
cloud = pv.PolyData(points)
cloud["height"] = points[:, 2]

pl = pv.Plotter(off_screen=True, window_size=(800, 600))
pl.add_mesh(
    cloud,
    style="points",
    scalars="height",
    cmap="viridis",
    point_size=8,
    show_scalar_bar=False,
)
pl.camera_position = [(3.0, 3.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
pl.set_background("#0c1018")

pl.blender.render("points.png", samples=32)
pl.close()
```

The bridge builds a Principled BSDF coloured either by the active
scalar field (`scalars=` was set) or by `prop.color` (no scalars).
Per-point colours land on a `"scalars"` POINT-domain Color attribute
on the point cloud, so the material's `ShaderNodeAttribute` reads them
exactly like the mesh path does.

## Gaussian splats (`style="points_gaussian"`)

```python
pl = pv.Plotter(off_screen=True, window_size=(800, 600))
pl.add_mesh(
    cloud,
    style="points_gaussian",
    scalars="height",
    cmap="inferno",
    point_size=20,
    show_scalar_bar=False,
)
pl.camera_position = [(3.0, 3.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
pl.set_background("#000000")

pl.blender.render("gaussian.png", samples=32)
pl.close()
```

For `style="points_gaussian"` the bridge swaps the Principled BSDF for
a **Transparent + Emission mix** driven by a camera-facing falloff:
`alpha = max(0, N . V) ** 4`, where `N` is the per-point sphere's
surface normal and `V` is the view direction. The result is bright at
the centre of each splat (where the sphere faces the camera) and
fades smoothly to transparent at the silhouette — the conventional
soft-splat look used by particle systems and Gaussian-Splatting
datasets, without a per-point centre accessor or geometry-nodes
instancing.

## How the bridge picks the path

| Trigger                                               | Mode       | Material                                    |
| ----------------------------------------------------- | ---------- | ------------------------------------------- |
| `style="points"` (DataSetMapper, prop.style="Points") | `points`   | Principled BSDF                             |
| `style="points_gaussian"` (vtkPointGaussianMapper)    | `gaussian` | Transparent + Emission, camera-facing alpha |
| anything else                                         | mesh       | (handled by the surface translator)         |

The decision happens in `scene.py:_detect_point_cloud_mode`, which
inspects the actor's mapper class first (`pv.PointGaussianMapper`)
and falls back to `prop.style == "Points"`. The mesh path stays
unchanged for surface / wireframe rendering.

## Constraints and caveats

- **Point size is in pixels (PyVista) vs. world units (Cycles).**
  PyVista's `point_size` is screen pixels; Cycles' PointCloud uses
  world-space radii. The bridge converts via `POINT_SIZE_TO_WORLD_RADIUS
= 0.005`, tuned for unit-bound data. For very small or very large
  coordinate extents, override by mutating the `radius` attribute on
  the returned PointCloud or by scaling the actor before
  `pl.add_mesh`.
- **Point-data scalars only.** PointCloud has no "cell" concept, so
  cell-data scalars on a points actor are not surfaced. Convert with
  `mesh = mesh.cell_data_to_point_data()` first.
- **No interactive viewport handling yet.** `pl.blender.show()`
  doesn't draw the cloud in the VTK overlay; the Cycles overlay does
  render it correctly. Drag preview falls back to whatever VTK chooses
  to draw for point styles.
