# Animation

`pl.blender.animate(output, updater, frames, *, fps, ...)` drives a Python
per-frame loop. The `updater(frame_index)` callback mutates the PyVista
scene in place, the bridge reconciles against the identity cache (mesh
data blocks are _refreshed_, not rebuilt), Cycles renders the frame, and
the resulting PNGs are muxed into the requested container.

## Deformation

```python
import numpy as np
import pyvista as pv

xs = np.arange(-10.0, 10.0, 0.25)
ys = np.arange(-10.0, 10.0, 0.25)
x, y = np.meshgrid(xs, ys)
radius = np.sqrt(x**2 + y**2)

grid = pv.StructuredGrid(x, y, np.sin(radius))
grid["height"] = grid.points[:, 2].copy()

plotter = pv.Plotter(off_screen=True, window_size=[640, 480])
plotter.add_mesh(grid, scalars="height", cmap="viridis", clim=(-1.0, 1.0))
plotter.camera_position = [(20.0, -20.0, 18.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

phases = np.linspace(0.0, 2.0 * np.pi, 30, endpoint=False)

def update(frame: int) -> None:
    z = np.sin(radius + phases[frame]).ravel()
    grid.points[:, 2] = z
    grid["height"] = z

plotter.blender.animate(
    "wave.mp4",
    updater=update,
    frames=range(30),
    fps=30,
    samples=32,
)
plotter.close()
```

The vertex positions are written in place via `foreach_set` on the
existing bpy mesh — no per-frame allocation, no leaked data blocks.
Topology must stay constant across frames; if `mesh.points.shape` changes
the cache rebuilds from scratch and warns.

Full script: [`examples/animation/wave_animation.py`](https://github.com/kmarchais/pyvista-blender/blob/main/examples/animation/wave_animation.py).

## Camera orbits

`pl.blender.orbit_camera(...)` returns a per-frame updater that rotates
the camera around a focal point. The orbit radius and elevation are
derived from the plotter's _current_ camera, so frame 0 reproduces the
starting pose and a full loop returns to it.

```python
import pyvista as pv

airplane = pv.examples.load_airplane()
center = tuple(float(c) for c in airplane.center)

plotter = pv.Plotter(off_screen=True, window_size=[480, 360])
plotter.add_mesh(airplane, color="#9ec5e8", pbr=True, metallic=0.7, roughness=0.3)
plotter.camera_position = [
    (center[0] + 2500.0, center[1] - 2500.0, center[2] + 1200.0),
    center,
    (0.0, 0.0, 1.0),
]

orbit = plotter.blender.orbit_camera(focal_point=center, n_frames=48)
plotter.blender.animate(
    "orbit.gif",
    updater=orbit,
    frames=range(48),
    fps=24,
    samples=32,
)
plotter.close()
```

Full script: [`examples/animation/orbit_animation.py`](https://github.com/kmarchais/pyvista-blender/blob/main/examples/animation/orbit_animation.py).

## Output format

The extension picks the muxer:

| Extension      | Backend              | Codec        |
| -------------- | -------------------- | ------------ |
| `.gif`         | imageio's gif plugin | (built-in)   |
| `.mp4`         | imageio-ffmpeg       | `libx264`    |
| `.mov`, `.mkv` | imageio-ffmpeg       | `libx264`    |
| `.webm`        | imageio-ffmpeg       | `libvpx-vp9` |

Anything else raises `ValueError` with the supported set.
