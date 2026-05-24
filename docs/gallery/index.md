# Gallery

Every runnable example lives under `examples/`, grouped by feature so
the file layout mirrors the [Cookbook](../cookbook/pbr.md). Each
script renders the same scene twice, once with `pl.screenshot(...)`
(PyVista's GL output) and once with `pl.blender.render(...)` (Cycles
path-traced through the bridge), so the visual difference between
engines is immediate.

Rendered outputs land under `docs/assets/examples/<group>/` and are
embedded inline on each group page below.

Run any example with:

```bash
uv run python examples/<group>/<name>.py
```

## Groups

- [PBR materials](pbr.md), Stanford Bunny, material-mode comparison.
- [Rendering styles](styles.md), wire overlay, wireframe, plain surface.
- [Lights](lights.md), custom SUN + POINT + SPOT kit.
- [Backgrounds](backgrounds.md), solid / gradient / HDRI worlds, transparent renders.
- [Scalars & datasets](scalars.md), random hills, St. Helens, FEA bracket, cell scalars.
- [Glyph instancing](glyphs.md), Geometry-Nodes-instanced arrow glyphs.
- [Point clouds](point_clouds.md), Gaussian splats sized by a scalar field.
- [Cameras & layout](cameras.md), multi-actor identity cache, orthographic multi-view.
- [HUD overlays](hud.md), scalar bar, text, axes triad, bounds box in one render.
- [Animation](animation.md), turntable orbit and wave deformation.

## Related cookbook recipes

Each example backs a narrative recipe in the cookbook, so the gallery
is the integration test surface for the cookbook content:

- [PBR materials](../cookbook/pbr.md)
- [Lights](../cookbook/lights.md)
- [Backgrounds](../cookbook/backgrounds.md)
- [Rendering styles](../cookbook/styles.md)
- [Animation](../cookbook/animation.md)
- [Animated glyphs](../cookbook/animated_glyphs.md)
- [HUD overlays](../cookbook/hud.md)
- [Glyph instancing](../cookbook/glyphs.md)
- [Subplots](../cookbook/subplots.md)
- [Export to .blend](../cookbook/export_blend.md)
- [Volume rendering](../cookbook/volume.md)
- [Point clouds](../cookbook/point_clouds.md)
- [Jupyter notebooks](../cookbook/jupyter.md)
- [Browser viewport (Trame)](../cookbook/web.md)
