# Jupyter inline rendering

`pv.set_jupyter_backend("blender")` routes pyvista's notebook display
through the bridge. Calling `pl.show()` in a cell triggers one offline
Cycles render and returns an `IPython.display.Image` so the
path-traced PNG renders inline.

## Install

The Jupyter backend needs `ipython`. The bridge reuses pyvista's
notebook stack rather than declaring its own extra, so install
pyvista's `jupyter` extra alongside `pyvista-blender`:

```bash
pip install "pyvista[jupyter]" pyvista-blender
```

(`uv add 'pyvista[jupyter]' pyvista-blender` works too.)

## Use

```python
import pyvista as pv

pv.set_jupyter_backend("blender")

pl = pv.Plotter(window_size=(640, 480))
pl.add_mesh(pv.examples.load_random_hills(), cmap="terrain")
pl.show(samples=64)   # → path-traced PNG inline
```

That's the whole API. Anything you'd pass to `pl.blender.render`
(`engine`, `device`, `samples`, `denoise`, `transparent_bg`) is
accepted by `pl.show(...)` and forwarded to the render call.
Pyvista-side kwargs (`window_size`, `cpos`, `return_img`) flow
through unchanged: the bridge filters them out before calling the
render so they don't crash the path.

## Compared to the in-process viewport

For notebooks you have two options now:

| Surface                                                  | When                                                                                                                                                                                          |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Inline backend** (`pv.set_jupyter_backend("blender")`) | A static, settled-quality frame in the cell. No popup window. Best for static plots, embedded reports, JupyterLab cells.                                                                      |
| **In-process viewport** (`pl.blender.show()`)            | The hybrid VTK + Cycles interactive viewport (drag preview in VTK, settled Cycles render on release). Requires a real desktop session, works from notebooks but pops out into its own window. |

Same translator + cache underneath; only the display surface differs.

## How registration works

The bridge ships an entry point in `pyproject.toml`:

```toml
[project.entry-points."pyvista.jupyter_backends"]
blender = "pyvista_blender.jupyter:handler"
```

PyVista's `_ensure_entry_points()` scans that group on first
`set_jupyter_backend(...)` call, so the user never needs an explicit
`import pyvista_blender` — the same auto-discovery contract as the
`pl.blender` accessor itself.

## Notes

- **First call cost.** The handler runs the same translation + Cycles
  render as `pl.blender.render`, so the first call pays the bpy
  import (~3 s) and any scene-build cost. Subsequent calls on the
  same plotter reuse the bridge's identity cache.
- **No interactivity.** Each cell returns a still image. To navigate
  the camera, mutate the plotter (or use `pl.blender.orbit_camera`)
  and re-display.
- **`screenshot=` parameter.** Passing a path keeps the PNG on disk
  in addition to returning the inline image — matches pyvista's
  built-in `static` backend contract.
