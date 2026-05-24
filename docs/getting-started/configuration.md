# Configuration

Engine, device, samples, and denoise can be set at three layers, with
later layers overriding earlier ones:

| Layer                                            | When                                |
| ------------------------------------------------ | ----------------------------------- |
| Module default (`pyvista_blender.config`)        | Process-wide; set once at startup.  |
| Component attribute (`pl.blender.engine = ...`)  | Per-plotter; survives across calls. |
| Per-call kwarg (`pl.blender.render(engine=...)`) | One-off override.                   |

## Engines

`"cycles"` is the path-tracer (default). `"eevee"` is Eevee Next, the
rasterizer-based engine. Eevee Next headless is **Linux-only**;
Windows / macOS users must use Cycles.

```python
import pyvista_blender as pvb
pvb.config.engine = "cycles"      # or "eevee"
```

## Devices

```python
pvb.config.device = "auto"        # picks the best available
```

Valid values:

- `"optix"` — NVIDIA RTX (best for Cycles + denoise)
- `"cuda"` — NVIDIA (older or non-RTX)
- `"hip"` — AMD
- `"metal"` — Apple Silicon / AMD on macOS
- `"oneapi"` — Intel
- `"cpu"` — fallback, slow
- `"auto"` — try OPTIX > CUDA > HIP > METAL > oneAPI > CPU

## Samples

```python
pvb.config.samples = 128          # default offline render
pvb.config.interactive_samples = 4    # during mouse drag
pvb.config.settled_samples = 32       # on EndInteractionEvent
pvb.config.idle_samples = 128         # progressive refinement when idle
```

## Examples

Process-wide setup at startup:

```python
import pyvista_blender as pvb

pvb.config.engine = "cycles"
pvb.config.device = "optix"
pvb.config.samples = 256
pvb.config.denoise = True
```

Per-plotter override:

```python
pl.blender.engine = "eevee"            # this plotter only
pl.blender.samples = 32                # cheap previews on this plotter
```

One-off override on a single call:

```python
pl.blender.render("hero.png", samples=512)
```
