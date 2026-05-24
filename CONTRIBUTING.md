# Contributing

Thanks for your interest. This project is GPL-3.0-or-later because it links against `bpy`. Contributions are accepted under the same license.

## Dev setup

Requires Python 3.11 or 3.13 (not 3.12 — no matching `bpy` wheel exists).

```bash
git clone https://github.com/kmarchais/pyvista-blender.git
cd pyvista-blender
uv sync --all-groups        # installs dev
uv run prek install         # set up the git pre-commit hooks
```

## Daily commands

| Task                     | Command                                                             |
| ------------------------ | ------------------------------------------------------------------- |
| Run tests                | `uv run pytest`                                                     |
| Run tests w/ coverage    | `uv run pytest --cov=src/pyvista_blender --cov-report=term-missing` |
| Lint + format            | `uv run ruff check --fix && uv run ruff format`                     |
| Type-check               | `uv run ty check`                                                   |
| Run all pre-commit hooks | `uv run prek run --all-files`                                       |
| Build docs locally       | `uv run zensical serve`                                             |
| Run the bridge benchmark | `uv run python benchmarks/large_mesh.py`                            |

All daily commands use `uv run` (not `uvx`) so the lockfile-pinned dev tools
run, and so `ty` can see `fake-bpy-module` in the project venv.

## Pull requests

- Branch from `main`; never push directly to `main`.
- `prek` runs on every commit; CI re-runs the same hooks. Both must pass.
- Pytest matrix covers Python 3.11 and 3.13 on Linux / Windows / macOS. Keep them green.
- Conventional commit messages preferred (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
- One feature per PR; reference issues in the body, not the title.
- Add a CHANGELOG entry under `## [Unreleased]` for any user-visible change.

## Project layout

```
src/pyvista_blender/
├── _component.py    Registered @pv.register_plotter_component("blender")
├── _compat.py       bpy 4.x vs 5.x API shim
├── config.py        Module-level engine/device defaults
├── translate/       PyVista → bpy translators (mesh, material, camera, light, ...)
├── render/          Cycles/Eevee engine dispatch, output, animation
├── interactive/     pl.blender.show() overlay viewport
└── hud/             2D overlays (scalar bar, text, axes triad)
```

The architectural rationale, identity cache, volumetric dispatch, and
interactive viewport are documented in
[`docs/architecture.md`](./docs/architecture.md). Read that before
making non-trivial changes.

## License

By contributing you agree to license your contributions under GPL-3.0-or-later.
