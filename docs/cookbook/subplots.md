# Subplots

`pv.Plotter(shape=(rows, cols))` lays multiple viewports out in one
window — each with its own camera, light kit, and background. The
bridge translates each renderer to a tile, runs Cycles per-tile, and
composites the tiles into a single output PNG.

## Side-by-side comparison

```python
import pyvista as pv

pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1200, 600])

pl.subplot(0, 0)
pl.add_mesh(pv.examples.load_random_hills(), cmap="terrain")
pl.add_text("Terrain", position="upper_edge", font_size=14, color="white")
pl.renderer.set_background("#0d1a2f")

pl.subplot(0, 1)
pl.add_mesh(
    pv.examples.load_random_hills(),
    color="#dec27c",
    pbr=True,
    metallic=0.4,
    roughness=0.3,
)
pl.add_text("Metallic", position="upper_edge", font_size=14, color="white")
pl.renderer.set_background("#1a1d22")

pl.blender.render("hills_comparison.png", samples=64)
pl.close()
```

Each subplot's camera, lights, and background are translated
independently. If you don't set a camera explicitly, the bridge calls
`renderer.reset_camera()` so the content fits the tile — same as
pyvista's own `screenshot()`.

## Per-renderer state

Each `pl.subplot(r, c)` makes that renderer active. Calls afterwards
target it:

| Per-renderer API                         | Effect on the tile                              |
| ---------------------------------------- | ----------------------------------------------- |
| `pl.add_mesh(...)` / `pl.add_glyph(...)` | Actors that appear in the tile                  |
| `pl.renderer.set_background(...)`        | Tile background colour or gradient              |
| `pl.add_light(pv.Light(...))`            | Light kit for that tile only                    |
| `pl.camera_position = [...]` / `"iso"`   | Tile camera pose                                |
| `pl.show_axes()`, `pl.show_bounds()`     | HUD elements composited **per tile**; see below |

## Layout shapes

Any `shape=(rows, cols)` works. The output PNG keeps the configured
`window_size`; tiles get equal-sized portions of that.

```python
# Two rows, three columns — six tiles.
pl = pv.Plotter(shape=(2, 3), off_screen=True, window_size=[1500, 800])
for r in range(2):
    for c in range(3):
        pl.subplot(r, c)
        pl.add_mesh(pv.Cube(), color=f"#{r * 80 + c * 40:02x}aacc")
pl.blender.render("six_tiles.png", samples=16)
pl.close()
```

## HUD overlays on subplots

Per-tile HUD compositing is wired into the subplot tile path. After
each tile's Cycles render, the bridge switches the plotter's active
renderer to that tile (via `plotter.subplot(row, col)`) and re-runs
the HUD compositor at the tile's resolution, so:

- `pl.add_text(...)` / `pl.add_title(...)` annotations sit inside
  the right tile, not stretched across the full image.
- `pl.show_axes()` triads use the tile's camera basis.
- `pl.show_bounds()` boxes track the tile's actors and camera.

```python
import pyvista as pv

pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1200, 600])

pl.subplot(0, 0)
pl.add_mesh(pv.examples.load_random_hills(), cmap="terrain")
pl.add_text("Terrain", position="upper_edge", font_size=14, color="white")
pl.show_axes()

pl.subplot(0, 1)
pl.add_mesh(
    pv.examples.load_random_hills(),
    color="#dec27c",
    pbr=True,
    metallic=0.4,
    roughness=0.3,
)
pl.add_text("Metallic", position="upper_edge", font_size=14, color="white")
pl.show_bounds()

pl.blender.render("hills_comparison.png", samples=64)
pl.close()
```

### Per-tile scalar bars

Each tile only draws the scalar bars added under its own `pl.subplot(r, c)`.
Under the hood the bridge walks each renderer's `GetActors2D()`
collection to find the `vtkScalarBarActor` instances that belong to
it, then filters `pl.scalar_bars` accordingly.

```python
import pyvista as pv

pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1200, 600])

pl.subplot(0, 0)
mesh = pv.Sphere()
mesh["height"] = mesh.points[:, 2]
pl.add_mesh(
    mesh,
    scalars="height",
    cmap="viridis",
    show_scalar_bar=True,
    scalar_bar_args={"title": "Sphere height", "color": "white"},
)

pl.subplot(0, 1)
cube = pv.Cube()
cube.cell_data["heat"] = [1, 2, 3, 4, 5, 6]
pl.add_mesh(
    cube,
    scalars="heat",
    cmap="inferno",
    show_scalar_bar=True,
    scalar_bar_args={"title": "Cube heat", "color": "white"},
)

pl.blender.render("two_bars.png", samples=16)
pl.close()
```

Left tile shows only "Sphere height" (viridis); right tile shows only
"Cube heat" (inferno). No cross-tile leakage.

## Interactive subplots

`pl.blender.show()` still uses the single-renderer fast path. The
hybrid drag-VTK / settle-Cycles flow doesn't account for per-tile
cameras yet; `show()` on a subplot plotter renders the active
renderer only.
