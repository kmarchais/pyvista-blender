# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `pl.blender.render(..., on_progress=callback)` for render status updates.
- Add studio lighting, the new default (`pyvista_blender.config.lighting =
"studio"`): a key / fill / rim rig of soft, shadow-casting area lights placed
  about the camera and sized to the scene, with a soft white fill from all
  around. It gives stacked or packed geometry its contact shadows. It replaces
  the default light kit only: lights placed in the scene with `add_light` are
  translated as before. Set `config.lighting = "vtk"` to translate the default
  kit too.
- Add `pyvista_blender.studio_lights(pl)`: the same rig as `pv.Light`
  objects to add with `pl.add_light`, so the PyVista preview and the Blender
  render are lit alike.

### Changed

- Cycles renders on the selected GPU backend alone, without the CPU, and asks
  Cycles for that backend's devices only (no more `HIPEW initialization
failed` on machines without AMD's HIP runtime).
- `export_animation_blend(..., bake_lights=True)` warns and skips the lights
  under studio lighting, which replaces the plotter's lights.
- Gallery baselines refreshed for studio lighting and linear flat colours
  (the scalar-bar baseline also lacked the bar's title, which `main` already
  renders).

### Fixed

- Direct RGB(A) scalars (`add_mesh(..., rgb=True)`, or `uint8` colour arrays)
  keep their colours instead of failing in the colour attribute.
- Flat colours (`color=`, edge colours, point clouds, light colours) are
  converted from sRGB to scene-linear, as scalar and background colours
  already were; they rendered too light before.

- Line-only meshes, including outlines and polylines, now render with the actor's
  color and `line_width`, including cached position and width updates.

- GIF animations rendered with `transparent_bg=True` no longer ghost previous
  frames through transparent pixels: the writer now requests disposal mode 2
  (restore to background) so each frame is cleared before the next is drawn.

## [0.1.0] - 2026-07-06

Initial release of `pyvista-blender`. See the [features list](docs/index.md#features)
for the full surface this version covers.

- Improved the documentation landing page with fullscreen demo-video previews,
  example-code buttons, a visible GitHub link, and automatic light/dark theme
  selection.

[Unreleased]: https://github.com/kmarchais/pyvista-blender/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kmarchais/pyvista-blender/releases/tag/v0.1.0
