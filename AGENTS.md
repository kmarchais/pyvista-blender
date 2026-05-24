# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, Aider, ...) working on
`pyvista-blender`. This file is the project's **single source of truth** for
conventions; `CLAUDE.md` points here.

## What this project is

A Python library that translates a live `pyvista.Plotter` scene into a
Blender (`bpy`) scene and renders it through Cycles or Eevee Next. Users
build scenes with PyVista's API; rendering becomes a backend choice. No
file round-trip between the two — translation is in-memory with an
identity-keyed cache.

The architectural surface is documented in [`docs/architecture.md`](./docs/architecture.md):

- Translation pipeline (PyVista actor → bpy mesh / material / light).
- Identity-keyed Level-1 cache (mesh data-blocks and materials reused
  across renders on the same plotter).
- Volumetric dispatch (closed-cube + Cycles Volume Principled atlas).
- Interactive viewport architecture (`pl.blender.show()` overlay +
  Trame backend).
- Animation export channels (camera, deformation, scalars, lights,
  glyphs).

## API shape

The bridge attaches to PyVista's `BasePlotter` via
`@pv.register_plotter_component("blender")` plus an entry point in
`pyproject.toml`:

```toml
[project.entry-points."pyvista.plotter_components"]
blender = "pyvista_blender._component"
```

Users **never `import pyvista_blender`** — installing the package makes
`pl.blender.render(...)` work via PyVista 0.48's plotter-component
auto-discovery. The accessor acceptance test
(`tests/test_accessor.py::test_blender_accessor_resolves_without_explicit_import`)
guards that contract.

Three-tier config resolution: **per-call kwarg → component attribute →
module default**. `pl.blender.resolve_config(attr, call_value)` is the public
way to inspect what would resolve.

## Version policy

| Python             | bpy wheel  | Blender                  |
| ------------------ | ---------- | ------------------------ |
| 3.11               | `>=4.5,<5` | 4.5 LTS                  |
| 3.13               | `>=5.0,<6` | 5.0 / 5.1+               |
| 3.10 / 3.12 / 3.14 | none       | no matching wheel exists |

`requires-python = ">=3.11,!=3.12.*,<3.14"`. The split is dispatched via
PEP-508 markers in `[project] dependencies`. `fake-bpy-module` mirrors the
same split in the `dev` group (5.0 stubs cover the 5.x line since 5.1
stubs aren't on PyPI yet).

## License: GPL-3.0-or-later

Forced by `bpy`'s GPL license. Every Python file under `src/` and
`tests/` starts with an SPDX header:

```python
# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
```

Ruff recognises SPDX headers via `[tool.ruff.lint.flake8-copyright]
notice-rgx`. Do not switch to "Copyright (C) ..." — SPDX is the modern
convention.

## Linting philosophy: fix code, not ignore rules

`select = ["ALL"]` with `preview = true`. The top-level `ignore` list
contains exactly three entries, each a **structural conflict in ruff
itself** that cannot be satisfied:

- `D203` ↔ `D211` (mutually exclusive class-docstring blank-line rules)
- `D213` ↔ `D212` (mutually exclusive multi-line docstring summary rules)
- `COM812` (ruff's formatter docs require disabling)

**No project-specific ignores. No per-file ignores.** When a rule fires:

1. Try to fix the code first (rename, restructure, add docstring section, change type).
2. If the rule literally cannot be satisfied (a framework contract or
   external convention), use an **inline `# noqa: RULE`** with a comment
   explaining why.

Load-bearing inline `noqa`s fall into a small number of families:

- **Framework contracts** (`PLW3201`, `ANN401` on
  `pyvista_blender.jupyter.handler`): PyVista's component registry
  requires the exact dunder name `__plotter_close__`, and PyVista's
  jupyter-backend protocol calls the handler with arbitrary kwargs
  (the user's `pl.show(**user_kwargs)` flows through pyvista's own
  `window_size` / `return_img` / `cpos` / ... wrapper). The handler's
  `**kwargs: Any` is the canonical way to accept that open-ended
  contract — TypedDict / Unpack would lock us to whatever subset of
  pyvista's evolving signature we chose to enumerate.
- **Lazy bpy imports** (`PLC0415`): each public entry point on
  `BlenderComponent` (`render`, `animate`, `show`, `export_blend`,
  `export_animation_blend`) and the Jupyter / web handlers
  lazy-import the bpy-touching submodule so PyVista's entry-point
  discovery (which imports `_component` the moment a user touches
  `pl.blender`) doesn't pay bpy's ~200 MB / ~3 s startup cost
  upfront.
- **Deliberately flat public API** (`PLR0913`): `show()` carries 15
  user-facing kwargs spanning the desktop + web viewport + sample
  tiers + render config + HUD toggles. That's the bridge's surface
  area for the interactive viewport, matching PyVista's house style
  (cf. `pl.add_mesh` with ~30 kwargs). The kwargs _are_ the API.
- **Typed-bypass for under-typed stubs** (`B010` for
  `setattr(node, "operation", ...)`): fake-bpy-module's stubs miss
  dynamic `operation` / `domain` / ... attributes on Math / Mix /
  Store-Attribute nodes that exist at runtime. `setattr` is the
  smallest escape.
- **Genuine physical constants in test asserts** (`PLR2004`): a
  handful of magic numbers (`< 2` for "at least two frames",
  `1e-12` for epsilon) where extracting a named constant adds noise.
- **Test access to internal API** (`PLC2701`): tests in
  `test_interactive.py` and `test_jupyter.py` import
  `_EngineParams` / `_PlotterSources` / `_filter_render_kwargs` to
  exercise internals directly. A legitimate signal that the test
  reaches past the public surface.
- **VTK positional-bool calls** (`FBT003`): `setUseInputBounds(True)`
  matches VTK's C++ signature; spelling it `True` is the only option.

`Any` and `object` are both undesirable for parameter annotations.
Prefer enumerated explicit kwargs; reach for `Any` (with a
`# noqa: ANN401` and framework-contract justification) only when
accepting an open-ended forwarding callable from an external library
(today: only `jupyter.handler`).

New `# noqa` comments need similar load-bearing justification: a
framework contract, a lazy bpy import on a user-facing call, a
deliberate flat-kwarg public API, or a typed-bypass for an
under-typed stub.

Configurable knobs (e.g. `max-args = 12`, `max-positional-args = 5`)
are not the same as ignores. The current settings reflect a real
architectural choice:

- **`max-positional-args = 5`**: anything past 5 positional args must
  be keyword-only (use a `*` separator in the signature). Call sites
  end up self-documenting (`func(a, b, c, width=W, height=H)` instead
  of `func(a, b, c, W, H)`), and the public-API methods all stay
  under 5 positional even before keyword-only kwargs.
- **`max-args = 12`**: the bridge's internal helpers fit naturally
  after the option-bundle refactor (`_EngineParams`, `_BakeChannels`,
  `_PlotterSources`, `_SubplotTileContext`, etc. in `_options.py`),
  while the few public-API methods that legitimately exceed it carry
  an inline `# noqa: PLR0913` at the signature.

Don't raise the knobs to absorb new violations. For too-many-args:
either bundle related kwargs into a dataclass alongside the existing
ones in `_options.py`, or `# noqa: PLR0913` the function with a
one-line architectural justification. For too-many-positional-args:
add a `*` separator at the natural split between required positionals
and the trailing config-like kwargs.

## Tooling

| Tool              | Purpose                          | Canonical command                                     |
| ----------------- | -------------------------------- | ----------------------------------------------------- |
| `uv`              | Sync, run, build                 | `uv sync --group dev --no-install-package bpy`        |
| `ruff`            | Lint + format                    | `uv run ruff check src/ tests/`, `uv run ruff format` |
| `ty`              | Type checker (Astral)            | `uv run ty check`                                     |
| `pytest`          | Test runner                      | `uv run pytest`                                       |
| `pytest-pyvista`  | Visual regression fixture        | `verify_image_cache` fixture                          |
| `prek`            | Pre-commit (Rust-native)         | `uvx prek install`, `uv run prek run --all-files`     |
| `zensical`        | Documentation (MkDocs successor) | `uv run zensical serve` / `build`                     |
| `fake-bpy-module` | Gives `ty` a resolvable `bpy`    | dev-group only                                        |

**Always `uv run ty`, never `uvx ty`.** `uvx` runs in an isolated env
without the project venv; ty then can't see `fake-bpy-module` and
`unresolved-import` / `unresolved-attribute` light up. The pre-commit
hook is configured for `uv run` for the same reason.

**Always `uv sync --no-install-package bpy`** in CI and for fast local
sync. Real `bpy` is ~200 MB and not needed for the no-bpy smoke tests;
`fake-bpy-module` covers ty's needs.

## Test conventions

- **No bare `assert`** (S101). Use `pytest.fail(msg)` for explicit
  failures, `pytest.raises(...)` for exception paths.
- **No private member access from tests** (SLF001). If a test needs
  internal state, promote it to a public API on the class.
- **Type-annotate fixtures and tests.** `Iterator[pv.Plotter]` etc. go
  behind `if TYPE_CHECKING:` because of `from __future__ import annotations`.
- **`tests/__init__.py` exists** so tests aren't an implicit namespace
  package (INP001).
- **`window_size=[640, 480]`** not `(640, 480)`. PyVista's stub types it
  as `list[int] | None`.

## Source conventions

- **`__init__.py` is for re-exports only** (RUF067). Version probes,
  side-effecting imports, and computation live in dedicated submodules
  (`_version.py`).
- **Restrict signatures to current need.** Don't pre-add kwargs for
  future features. When the implementation lands, add them then. The
  `pl.blender.add_volume` wrapper deliberately exposes only the
  kwargs the bridge actively uses (`scalars`, `cmap`, `opacity`, ...)
  rather than re-exporting pyvista's full ~30-kwarg surface — users
  who need other pyvista kwargs call `pl.add_volume` directly.
- **Docstrings**: Returns section only when the function returns
  something. Raises section when it raises. NumPy style.

## bpy 4.x vs 5.x

`_compat.py` is the shim. The notable API breakage is RNA dict-access:
`scene["cycles"]` stopped working in 5.0. Use `rna_get(owner, key)` /
`rna_set(owner, key, value)` (both already provided) rather than
subscripting RNA owners. Add new shim helpers as the bridge encounters
other 4 → 5 differences.

## Workflow

1. Read the relevant section of [`docs/architecture.md`](./docs/architecture.md).
2. Add or update tests first.
3. Implement.
4. Run the full gate:
   ```bash
   uv run pytest
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run ty check
   ```
5. `uv run prek run --all-files` before pushing.
6. Add a `CHANGELOG.md` entry under `## [Unreleased]` for user-visible
   changes.

## Project layout

```
src/pyvista_blender/
├── __init__.py        Re-exports BlenderComponent, config, orbit_camera, __version__
├── _component.py      @register_plotter_component("blender")
├── _compat.py         bpy 4.x ↔ 5.x shim (rna_get / rna_set)
├── _glyph.py          GlyphSpec dataclass (bpy-free, importable on accessor wiring)
├── _render_impl.py    do_render / do_animate; owns the bpy import (lazy)
├── _version.py        Version probe (kept out of __init__ for RUF067)
├── animate.py         Pure-numpy frame-update helpers (orbit_camera, ...)
├── config.py          Module-level defaults (engine, device, samples, ...)
├── translate/         PyVista → bpy translators
├── render/            Cycles/Eevee engine + device dispatch
├── interactive/       pl.blender.show() overlay viewport
└── hud/               Compositor-based 2D overlays
```

The architecture lives in [`docs/architecture.md`](./docs/architecture.md);
the directory structure is the skeleton it fills.

## Commit and PR conventions

- **Single-line commit messages**, imperative mood, no body. Pass with
  `-m` directly.
- **No AI co-author, no mention of AI** in commits, PRs, or code
  comments.
- **`git add -u`** or named files only. Never `git add .` or `git add -A`.
- **Feature branches**; never push to `main`.
- PRs use **short, structured descriptions** (`## Summary`, `## Changes`).
  No `## Test plan` section.

## Where to look first

| Question                                   | File                             |
| ------------------------------------------ | -------------------------------- |
| Why is the API shaped like `pl.blender.X`? | `docs/architecture.md`           |
| What's the accessor acceptance criterion?  | `tests/test_accessor.py`         |
| How does an incoming change get validated? | This file's **Workflow** section |
| What did the last release ship?            | `CHANGELOG.md`                   |
| What's the post-push checklist?            | `PUBLISH.md`                     |
| How do I run docs locally?                 | `uv run zensical serve`          |
