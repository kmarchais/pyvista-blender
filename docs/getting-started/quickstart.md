# Quick start

```python
import pyvista as pv
from pyvista import examples

pl = pv.Plotter(off_screen=True, window_size=(1920, 1080))
pl.add_mesh(
    examples.download_bunny(),
    color="lightgrey",
    pbr=True, metallic=0.1, roughness=0.4,
)
pl.add_light(pv.Light(position=(5, -5, 5), light_type="scene light"))
pl.camera_position = "iso"

pl.blender.render("bunny.png", samples=128)
```

Render quality scales with `samples`. 32 for previews, 128 – 512 for
publication-quality output. With OptiX denoising (default), 64 samples
is usually visually clean.

## Animation

```python
import numpy as np

trajectory = np.load("frames.npz")

def updater(t: int) -> None:
    mesh.points = trajectory[t]

pl.blender.animate("out.mp4", updater=updater, frames=range(120), fps=30)
```

## Interactive rendered viewport

```python
pl.blender.show()   # one window, mouse rotates / zooms, Cycles renders in real-time
```

See [Architecture](../architecture.md) for the design of the single-window
overlay and the three-tier sample regime.

## Render status callbacks

Pass `on_progress` to receive `"Preparing scene"`, `"Rendering"`, Blender's native render
statistics, and `"Finished"` after the image and overlays are written:

```python
import pyvista as pv

plotter = pv.Plotter()
plotter.add_mesh(pv.Sphere())
plotter.blender.render("sphere.png", on_progress=print)
plotter.close()
```

Statistics vary by Blender version and engine; they are status strings, not
normalized percentages. Subplot renders may restart sample counts per tile.
Callbacks run synchronously inside the render operation. Keep them brief and
do not mutate the scene, start another render, or write from background threads.
Callback output is visible even while native Blender output is silenced.
Exceptions from native-statistics callbacks are re-raised after Blender returns;
no `"Finished"` message is emitted if rendering or a callback fails. The temporary
handler and output descriptors are restored on exit.
