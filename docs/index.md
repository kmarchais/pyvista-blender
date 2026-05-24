---
icon: lucide/rocket
---

# pyvista-blender

Render [PyVista](https://docs.pyvista.org/) plotter scenes through [Blender](https://www.blender.org/) (`bpy`) for photoreal output.

Build the scene once in PyVista. Choose VTK's preview or Blender's Cycles / Eevee Next for the final render.

## The shape of the API

```python
import pyvista as pv

pl = pv.Plotter(off_screen=True, window_size=(1920, 1080))
pl.add_mesh(pv.read("data.vtu"), scalars="von_mises", cmap="viridis", pbr=True)
pl.add_light(pv.Light(position=(5, -5, 5), light_type="scene light"))
pl.camera_position = "iso"

# No `import pyvista_blender` needed, installing the package registers
# the `bpy` namespace via PyVista 0.48's plotter-component registry.
pl.blender.render("frame.png", samples=128)
```

For an interactive rendered viewport, `pl.blender.show()` replaces
PyVista's VTK preview with Cycles output in the same window:

```python
pl.blender.show()   # single window, mouse-driven, real-time Cycles
```

## Why does this exist?

VTK's renderer is fast and interactive. Blender's Cycles produces images
suitable for paper figures, presentations, and explainer videos. This
library is the bridge: same scene in Python, two render quality tiers.

## Features

- **Materials**: PBR (Principled BSDF), Phong with Walter et al. roughness fit, unlit, double-sided. ([recipe](cookbook/pbr.md))
- **Styles**: surface, wireframe, edge overlay, points, Gaussian splats as native `bpy.types.PointCloud`. ([recipe](cookbook/styles.md))
- **Lights**: SUN / POINT / SPOT / HEADLIGHT / CAMERA_LIGHT, the default `vtkLightKit`, custom rigs. ([recipe](cookbook/lights.md))
- **Backgrounds**: solid, gradient, HDRI / image-based lighting. ([recipe](cookbook/backgrounds.md))
- **Datasets**: PolyData, UnstructuredGrid, MultiBlock, high-order cells. Point and cell scalars with colormaps and `clim`.
- **Volume rendering**: closed-cube + Cycles Volume Principled atlas, no OpenVDB. ([recipe](cookbook/volume.md))
- **Glyphs**: Geometry-Nodes instancer (`N + V` verts, not `N * V`). ([recipe](cookbook/glyphs.md))
- **HUD**: scalar bars, text, axes triad, bounds box composited over Cycles. ([recipe](cookbook/hud.md))
- **Subplots**: per-tile camera / lights / HUD, PIL composited. ([recipe](cookbook/subplots.md))
- **Cameras**: perspective + orthographic, `user_matrix`, orbit updater.
- **Animation**: `pl.blender.animate(...)` to gif / mp4 / webm; `pl.blender.export_animation_blend(...)` bakes camera, deformation (MDD or shape keys), scalars, lights, transforms, materials, volumes, and glyphs into a `.blend` that plays natively on open. ([recipe](cookbook/animation.md))
- **Interactive**: `pl.blender.show()` for desktop, `pl.blender.show(backend="web")` for browser, with three sample tiers and idle-promotion. ([recipe](cookbook/web.md))
- **Jupyter**: `pv.set_jupyter_backend("blender")` returns an inline `IPython.display.Image`. ([recipe](cookbook/jupyter.md))
- **Authoring**: `pl.blender.export_blend(path)` to finish the scene in Blender. ([recipe](cookbook/export_blend.md))

GPU auto-detection walks OptiX, CUDA, HIP, Metal, oneAPI, CPU. Engine and device strings resolve in three tiers: per-call kwarg, component attribute (`pl.blender.engine = ...`), module default (`pyvista_blender.config.*`). The identity-keyed Level-1 cache reuses `bpy.types.Mesh` and material data-blocks across renders on the same plotter.

## Where next

- **[Getting started](getting-started/installation.md)** for install and a first render.
- **[Cookbook](cookbook/pbr.md)** for one recipe per feature branch.
- **[Gallery](gallery/index.md)** for the runnable examples under `examples/`.
- **[Architecture](architecture.md)** for the translation pipeline and cache strategy.
