# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
