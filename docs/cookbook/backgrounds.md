# Backgrounds and environments

The bridge maps PyVista's three background modes onto Blender's
World shader:

| PyVista call                                  | Blender World setup                                                                               |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `pl.set_background("#1d2236")`                | Background node with constant colour                                                              |
| `pl.set_background(bottom, top=...)`          | Camera-Y ColorRamp tuned to the camera FOV                                                        |
| `pl.set_environment_texture(tex)`             | `ShaderNodeTexEnvironment` in-memory bpy image, used for both background and image-based lighting |
| `pl.blender.render(..., transparent_bg=True)` | `scene.render.film_transparent = True` (alpha preserved in the PNG)                               |

## Solid colour

A plain background colour, quick, no IBL, no environment reflections.

```python
plotter.set_background("#1d2236")
plotter.blender.render("solid.png", samples=96)
```

## Vertical gradient

`set_background(bottom, top=...)` produces a Camera-Y ColorRamp. The
bridge reads the plotter's camera FOV and pins the gradient stops to
the framed-view edges so the same gradient reads correctly across
camera angles. The world is hidden from glossy rays in this branch
(only the visible background is gradient-shaded; reflections still
read the solid base).

```python
plotter.set_background("#0a0a30", top="#ffaa44")
plotter.blender.render("gradient.png", samples=96)
```

## Environment texture (HDRI / IBL)

`set_environment_texture(tex)` loads any `vtkTexture` (or a path you
hand to `pv.examples.load_globe_texture()` and friends) as an
in-memory bpy image. Cycles uses it for both the camera-visible
background **and** image-based lighting on glossy / metallic
materials — a glossy sphere will pick up colours from the environment
naturally.

```python
plotter.set_environment_texture(pv.examples.load_globe_texture())
plotter.blender.render("hdri.png", samples=128)
```

For real photogrametric HDRI use a `.hdr` or `.exr` image:

```python
from vtkmodules.vtkIOImage import vtkHDRReader

reader = vtkHDRReader()
reader.SetFileName("studio.hdr")
reader.Update()
tex = pv.Texture(reader.GetOutput())
plotter.set_environment_texture(tex)
```

The bridge passes the image through unchanged so the float dynamic
range is preserved end-to-end.

## Transparent background

`transparent_bg=True` flips `scene.render.film_transparent` to True
and the PNG output keeps its alpha channel. Useful for compositing
the render over a custom background in PIL or downstream tools.

```python
plotter.blender.render("alpha.png", samples=96, transparent_bg=True)
```

## All three in one example

`examples/backgrounds/environment.py` renders the same metallic sphere under
solid, gradient, and HDRI backgrounds in sequence. The bridge's
identity cache reuses the mesh + material across all three calls so
the only thing that changes between renders is the World shader:

```python
plotter.add_mesh(
    pv.Sphere(theta_resolution=64, phi_resolution=64),
    color="#cccccc", pbr=True, metallic=0.95, roughness=0.18,
)

plotter.set_background("#1d2236")
plotter.blender.render("solid.png", samples=96)

plotter.set_background("#0a0a30", top="#ffaa44")
plotter.blender.render("gradient.png", samples=96)

plotter.set_environment_texture(pv.examples.load_globe_texture())
plotter.blender.render("hdri.png", samples=128)
```

The HDRI render needs higher samples (128 vs 96) because the
environment lighting introduces softer noise that the OpenImageDenoise
pass cleans up more aggressively at higher counts.

## Constraints and caveats

- **HDRI rotation is fixed.** The bridge uses VTK's texture
  orientation directly; if you need to rotate the environment, edit
  the World shader in the saved `.blend` (a `ShaderNodeMapping` node
  inserted between the texture and the background output gives you
  rotation control).
- **Gradient + glossy reflections.** Gradient backgrounds intentionally
  hide the world from glossy rays so the reflection map stays neutral.
  If you want a gradient _to_ reflect, use a generated HDRI image
  instead of `set_background(bottom, top=...)`.
- **`transparent_bg` + HDRI.** Setting both works: Cycles renders
  the scene with HDRI lighting but writes alpha-zero pixels wherever
  no geometry is hit. Useful for compositing path-traced subjects
  into other backgrounds while keeping the IBL.
