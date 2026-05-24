# PBR materials

PyVista's `pbr=True` enables the metallic / roughness path on the Principled
BSDF in Blender. The bridge maps `metallic` and `roughness` straight onto
the matching BSDF inputs; Cycles then gives true Fresnel reflection and
view-dependent shading that the VTK Phong shader can't reproduce.

```python
import pyvista as pv

bunny = pv.examples.download_bunny()

plotter = pv.Plotter(off_screen=True, window_size=[1280, 960])
plotter.add_mesh(
    bunny,
    color="#dec27c",
    pbr=True,
    metallic=0.3,
    roughness=0.35,
)
plotter.set_background("#1a1d22")
plotter.camera_position = [(0.2, 0.18, 0.25), (-0.02, 0.1, 0.0), (0, 1, 0)]

plotter.blender.render("bunny_blender.png", samples=64)
plotter.close()
```

Full script: [`examples/pbr/bunny_pbr.py`](https://github.com/kmarchais/pyvista-blender/blob/main/examples/pbr/bunny_pbr.py).

## How the mapping works

| PyVista         | bpy / Cycles            |
| --------------- | ----------------------- |
| `pbr=True`      | Principled BSDF         |
| `metallic=...`  | BSDF `Metallic` input   |
| `roughness=...` | BSDF `Roughness` input  |
| `color=...`     | BSDF `Base Color` input |

## Phong → PBR (when `pbr=False`)

When `pbr=False` the bridge still goes through the Principled BSDF but
converts the Phong specular exponent using the Walter et al. roughness fit
`r = sqrt(2 / (n + 2))` from `prop.specular_power`, with `prop.specular`
driving the BSDF specular weight. Glossy plastic look without setting
`pbr=True`.

## Unlit (when `lighting=False`)

Setting `lighting=False` on the actor swaps the Principled BSDF for an
Emission shader so the surface ignores scene lights — matches VTK's flat
path. Useful for label meshes, overlays, or stylised renders.
