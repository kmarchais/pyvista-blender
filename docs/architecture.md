# Architecture

This page summarises the parts of `pyvista-blender` that affect day-to-day
use: the entry-point shape, the translation pipeline, the identity cache,
the volumetric dispatch, and the interactive viewports.

## One sentence

The bridge walks `plotter.renderers`, dispatches each `pv.Actor` through
mesh + material + transform translators, builds the `bpy` scene from
NumPy arrays via `foreach_set`, and runs Cycles or Eevee Next.

## The `pl.blender` namespace

PyVista 0.48 ships a plotter-component registry modelled on pandas'
accessor pattern. `pyvista-blender` declares an entry point under
`[project.entry-points."pyvista.plotter_components"]` named `blender`,
so every `pv.BasePlotter` gets a `blender` attribute on first access:

```toml
[project.entry-points."pyvista.plotter_components"]
blender = "pyvista_blender._component"
```

Users **never `import pyvista_blender`** in their code; installing the
package is enough. The accessor acceptance test
(`tests/test_accessor.py`) guards that contract.

Three-tier config resolution applies everywhere:

1. Per-call kwarg (`pl.blender.render("frame.png", samples=256)`)
2. Component attribute (`pl.blender.engine = "EEVEE_NEXT"`)
3. Module default (`pyvista_blender.config.samples = 64`)

`pl.blender.resolve_config(attr, call_value)` is the public way to
inspect what would resolve.

## Translation pipeline

Each `pv.Actor` walks through specialised translators under
`pyvista_blender/translate/`:

| Translator      | What it produces                                                  |
| --------------- | ----------------------------------------------------------------- |
| `mesh.py`       | `bpy.types.Mesh` from PolyData, UnstructuredGrid, MultiBlock      |
| `material.py`   | Principled BSDF graph for PBR; Phong-fit BSDF graph for non-PBR   |
| `light.py`      | SUN / POINT / SPOT / HEADLIGHT / CAMERA_LIGHT, plus `vtkLightKit` |
| `camera.py`     | Perspective or orthographic, with `user_matrix` propagation       |
| `background.py` | Solid / gradient / HDRI on the world shader                       |
| `volume.py`     | Closed-cube + Cycles Volume Principled atlas (no OpenVDB)         |
| `glyph.py`      | Geometry-Nodes instancer driven by a `GlyphSpec` dataclass        |

High-order cells (quadratic / Lagrange / Bezier) are tessellated before
the surface extract, controlled by `config.tessellation_subdivide`.
Point and cell scalars share the same colormap codepath; cell scalars
write flat per-face attributes via Geometry Nodes' face-domain reads.

## Identity-keyed caching

For animations and interactive renders, re-extracting surfaces and
rebuilding materials every frame is wasteful. The bridge caches by
`id(pv_dataset)` with an `MTime` ledger:

- Unchanged geometry reuses its `bpy.types.Mesh`; only mutated arrays
  re-upload via `foreach_set`.
- Materials are content-hashed and shared across actors that resolve to
  identical PBR / Phong inputs.
- When topology changes (vertex count differs) the mesh is rebuilt; when
  only positions / colors change, the existing buffers are refreshed in
  place.

The cache is **per-plotter**; closing the plotter clears it.

## Volumetric dispatch

`pv.UniformGrid` actors with a `volume` mapper route through
`translate/volume.py`. The bridge builds a unit-cube proxy mesh and a
Cycles `Volume Principled` shader whose density / color attributes read
from a packed 3D texture atlas. No `.vdb` round-trip; the NumPy array
travels directly to GPU via `bpy.data.images`.

## Interactive viewports

### Desktop (`pl.blender.show()`)

Single window. VTK keeps owning input and the window; a second VTK
renderer on layer 1 with `vtkActor2D` + `vtkImageMapper` displays
Cycles output as a fullscreen overlay. From the user's perspective the
experience is identical to Blender's "Rendered" viewport mode, but it
lives inside PyVista's window with all the existing widgets, picking,
and trackball math intact.

Three sample tiers degrade during interaction and promote on idle:

| Tier          | When                              | Samples kwarg         |
| ------------- | --------------------------------- | --------------------- |
| `interactive` | Mouse held / dragging             | `samples_interactive` |
| `settled`     | Mouse released                    | `samples_settled`     |
| `idle`        | `idle_delay_ms` of no interaction | `samples_idle`        |

### Browser (`pl.blender.show(backend="web")`)

Serves a Trame app with a `VtkLocalView` for the interactive base plus a
Cycles overlay refreshed on settle / idle. Reuses pyvista's Trame stack
(`pyvista[jupyter]`), so the only client-side requirement is a modern
browser.

### Jupyter (`pv.set_jupyter_backend("blender")`)

Inline `IPython.display.Image` rendered through Cycles. Routes via
`[project.entry-points."pyvista.jupyter_backends"]`, picked up by
pyvista's `_ensure_entry_points()` on first call.

## Animation export

`pl.blender.export_animation_blend(...)` bakes per-channel data into a
`.blend` that plays natively on file open — no Python in the `.blend`,
no auto-execution prompt:

| Channel     | Mechanism                                                             |
| ----------- | --------------------------------------------------------------------- |
| Camera      | Keyframes on `location` + `rotation_quaternion`                       |
| Deformation | Shape Keys (one per frame) **or** Lightwave MDD + `MESH_CACHE` mod    |
| Scalars     | Packed PNG (rows = frames, cols = vertices / cells) + Geometry Nodes  |
| Lights      | Per-light keyframes on `location`, `rotation`, `energy`, `color`      |
| Transforms  | Object-level location / rotation / scale keyframes                    |
| Glyphs      | Float-image baked positions / orient / scale + Geometry Nodes sampler |

All channels are opt-in via dedicated kwargs (`bake_camera`,
`bake_deformation`, `bake_scalars`, `bake_lights`, `bake_transforms`,
`bake_glyphs`).

## bpy 4.x vs 5.x

`_compat.py` is the shim. The notable API breakage is RNA dict-access:
`scene["cycles"]` stopped working in 5.0. Use `rna_get(owner, key)` /
`rna_set(owner, key, value)` (both provided) rather than subscripting RNA
owners. Add new shim helpers as the bridge encounters other 4 → 5
differences.

## GPU device dispatch

`config.device` resolves to a real Cycles device via a deterministic
walk: OptiX → CUDA → HIP → Metal → oneAPI → CPU. The auto walk only
considers devices that Cycles actually reports as available on the
current machine; falling all the way through to CPU is a normal outcome
on CI / WSL.

## Where the code lives

```
src/pyvista_blender/
├── __init__.py        Re-exports (BlenderComponent, config, orbit_camera, __version__)
├── _component.py      @register_plotter_component("blender")
├── _compat.py         bpy 4.x ↔ 5.x shim
├── _glyph.py          GlyphSpec dataclass (bpy-free)
├── _options.py        Internal kwarg-bundle dataclasses (_EngineParams, ...)
├── _render_impl.py    do_render / do_animate; owns the bpy import (lazy)
├── _version.py        Version probe
├── animate.py         Pure-numpy frame-update helpers (orbit_camera, ...)
├── config.py          Module-level defaults
├── jupyter.py         pv.set_jupyter_backend("blender") entry point
├── translate/         PyVista → bpy translators (one per concept)
├── render/            Cycles / Eevee engine + device dispatch
├── interactive/       pl.blender.show() overlay viewport
├── web/               pl.blender.show(backend="web") Trame app
└── hud/               Compositor-based 2D overlays
```
