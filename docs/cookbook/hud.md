# HUD overlays

PyVista's scalar bars, titles, text actors, corner axes, and bounding-box
annotations are 2D screen-space elements. Cycles has no equivalent, so
the bridge generates each one as an RGBA image at render resolution and
alpha-composites them over the Cycles output via a PIL post-pass.

## All overlays at once

```python
import pyvista as pv

hills = pv.examples.load_random_hills()

plotter = pv.Plotter(off_screen=True, window_size=[960, 540])
plotter.add_mesh(
    hills,
    cmap="viridis",
    show_scalar_bar=True,
    scalar_bar_args={"title": "Elevation (m)"},
)
plotter.add_title("PyVista to Blender HUD", font_size=18, color="white")
plotter.add_text("Random Hills", position="upper_left", font_size=14, color="white")
plotter.show_axes()
plotter.show_bounds(color="#bbbbbb")
plotter.camera_position = "iso"

plotter.blender.render("hud_demo.png", samples=64)
plotter.close()
```

Full script: [`examples/hud/hud_demo.py`](https://github.com/kmarchais/pyvista-blender/blob/main/examples/hud/hud_demo.py).

The overlays auto-discover from the matching PyVista actors — no
`hud=...` kwarg, no per-overlay opt-in. If `pl.add_scalar_bar(...)` is
on the plotter, the scalar bar overlay fires; same for titles, text,
axes, and bounds.

## What composites

| PyVista source                                 | Overlay                                                     |
| ---------------------------------------------- | ----------------------------------------------------------- |
| `pl.add_mesh(..., show_scalar_bar=True)`       | Scalar bar (matplotlib `ColorbarBase`)                      |
| `pl.add_title(...)`                            | Title text (PIL)                                            |
| `pl.add_text(..., position="upper_left", ...)` | Corner text (PIL) at the matching anchor                    |
| `pl.show_axes()`                               | XYZ axes triad (PIL arrows, projected through camera basis) |
| `pl.show_bounds()` / `pl.show_grid()`          | Bounding box (PIL polyline through 12 edges)                |

## Debugging a single overlay

`pl.blender.render_hud_overlay(kind, *, width, height)` returns one overlay
as an RGBA numpy array without running Cycles. Useful when iterating on a
scalar bar's font size, the axes triad's position, or the bounds box's
projection without paying the full render cost.

```python
import pyvista as pv

sphere = pv.Sphere()
sphere["z"] = sphere.points[:, 2]

pl = pv.Plotter(off_screen=True, window_size=[640, 480])
pl.add_mesh(sphere, scalars="z", cmap="viridis", show_scalar_bar=True)

rgba = pl.blender.render_hud_overlay("scalar_bar", width=640, height=480)
# rgba is (480, 640, 4) float32 in [0, 1], or None if no scalar bars
```

Valid `kind` values: `"scalar_bar"`, `"text"`, `"axes"`, `"bounds"`.
