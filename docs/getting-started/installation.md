# Installation

## Requirements

| What                        | Why                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Python **3.11** or **3.13** | Each `bpy` wheel is pinned to one Python minor. Python 3.12 has no matching wheel and is rejected at install time. |
| A GPU is recommended        | Cycles + denoiser at interactive frame rates needs OPTIX / CUDA / HIP / Metal. CPU works but is slow.              |

## Install

=== "uv (recommended)"

    ```bash
    uv add pyvista-blender
    ```

=== "pip"

    ```bash
    pip install pyvista-blender
    ```

This installs `pyvista>=0.48`, `numpy>=1.26`, and the matching `bpy`
wheel (4.5 LTS on Python 3.11, 5.x on Python 3.13). The `bpy` wheel is
**168 – 319 MB** depending on platform — that's Blender's full runtime.

## Verify

```python
import pyvista as pv

pl = pv.Plotter(off_screen=True)
if not hasattr(pl, "blender"):
    raise SystemExit("entry-point registration failed — reinstall pyvista-blender")

pl.add_mesh(pv.Sphere(), color="red")
pl.blender.render("verify.png", samples=8)
pl.close()
```

If `pl.blender` is missing, PyVista's entry-point discovery didn't pick up
the install. Reinstall the package and check `pip show pyvista-blender`.

## License note

`pyvista-blender` is **GPL-3.0-or-later** because it imports `bpy`,
which is itself GPL. Importing this package into your own code does
not relicense your code; only redistributing `pyvista-blender` itself
is constrained.
