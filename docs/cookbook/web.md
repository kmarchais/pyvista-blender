# Browser viewport (Trame)

`pl.blender.show(backend="web")` opens a browser tab with a Cycles
overlay on top of a live VTK widget. Drag the mouse to rotate
(VTK's 60 fps real-time path handles it); on release, the bridge
runs one Cycles pass and the path-traced result lands as the
overlay. Same hybrid contract as the desktop viewport, just with the
browser as the display surface.

## Install

The web backend reuses pyvista's Trame stack. Install pyvista's
`jupyter` extra alongside the bridge:

```bash
pip install "pyvista[jupyter]" pyvista-blender
```

That pulls in `trame`, `trame-vtk`, and `trame-vuetify`. Users who
don't need the web (or notebook) viewport can install
`pyvista-blender` on its own and skip the Trame cost entirely.

## Use

```python
import pyvista as pv

pl = pv.Plotter(window_size=(960, 720))
pl.add_mesh(pv.examples.load_random_hills(), cmap="terrain")
pl.show(backend="web")  # blocks until the browser tab closes
```

The browser opens automatically. The toolbar carries:

- **Samples slider** — Cycles sample count for the on-release
  render (default 32, range 4-512).
- **Camera-iris button** — force a re-render with the current
  sample count without moving the camera.
- **Reset-camera button** — frame the scene + re-render.

## How the drag handoff works

The `<img>` overlay is z-stacked above a `VtkLocalView` widget
showing the same `plotter.ren_win`. Two Trame state variables
drive the swap:

| State                   | Purpose                                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `cycles_visible` (bool) | `v_show` flag on the overlay. False during drag, True after the on-release Cycles render lands.           |
| `cycles_data_url` (str) | The `data:image/png;base64,...` URL bound to the overlay's `src`. Updated when a Cycles render completes. |

Wiring (server-side):

```
VtkLocalView.StartInteractionEvent → cycles_visible = False
VtkLocalView.EndInteractionEvent   → render Cycles → cycles_data_url
                                                  → cycles_visible = True
```

The Cycles render runs on a background thread so the Trame event
loop stays responsive. A debounce flag (`state.rendering`) drops
overlapping requests if the user clicks the re-render button while
a previous pass is still in flight.

## Headless / remote use

```python
pl.show(backend="web", host="0.0.0.0", port=8765, open_browser=False)
```

Binds to all interfaces and skips the local browser launch. There's
**no authentication** — only do this on a trusted network.

Print the resolved URL from the calling process (Trame logs it on
startup) and open it manually from another machine.

## Per-tier sampling + idle promotion

The web viewport mirrors the desktop's three-tier story, simplified
to **two tiers** because the web backend doesn't run Cycles during
drag (only on release):

| Tier    | When it fires                                  | Samples                         | Tuned by             |
| ------- | ---------------------------------------------- | ------------------------------- | -------------------- |
| Settled | `EndInteractionEvent` (mouse release)          | `samples_settling` (default 32) | Toolbar slider       |
| Idle    | `idle_delay_ms` after the settled render lands | `samples_idle` (default 128)    | `samples_idle` kwarg |

The idle timer is a `threading.Timer` scheduled on the server. Any
new interaction cancels it, so dragging in succession only ever pays
the settled-tier cost; the publication-quality settle only happens
when the user genuinely stops touching the scene.

```python
pl.blender.show(
    backend="web",
    samples_settling=16,    # interactive feedback
    samples_idle=256,       # final settle when the user pauses
    idle_delay_ms=1500.0,   # wait 1.5 s after release before promoting
)
```

Disable idle promotion entirely with `idle_delay_ms=0` — the settled
render becomes the final state.

Programmatic control:

```python
from pyvista_blender.web import BlenderWebApp, serve

app = BlenderWebApp(pl, samples_idle=512)
# ... later, from another callback or thread:
app.cancel_idle_promotion()    # drop any pending higher-quality settle
app.schedule_idle_promotion()  # arm one explicitly (e.g. after editing the scene)
```

## Comparison with the desktop viewport

| Capability                         | Desktop (`backend="desktop"`)           | Web (`backend="web"`)                           |
| ---------------------------------- | --------------------------------------- | ----------------------------------------------- |
| Drag preview                       | VTK in the window (60 fps)              | VTK in the browser (60 fps via VtkLocalView)    |
| Settled render                     | Cycles overlay on `EndInteractionEvent` | Cycles `<img>` overlay on `EndInteractionEvent` |
| Per-tier sampling (settled / idle) | Yes                                     | Yes                                             |
| Idle-promotion timer               | Yes (1 s default)                       | Yes (1 s default)                               |
| Remote display                     | No (X11 / GL local)                     | Yes (any browser)                               |
| Auth                               | OS process                              | None — only bind `127.0.0.1`                    |

Every desktop-viewport kwarg now has a web-viewport counterpart;
HUD overlays, sample counts, denoising, and transparent BG work
end-to-end because the web backend forwards `render_kwargs` straight
through to `pl.blender.render`.

## Constraints and caveats

- **One client at a time.** Trame supports multi-client but the
  bridge's bpy session is single-threaded; concurrent renders
  serialize on the same lock. Multiple browser tabs work but
  share the same per-render latency budget.
- **No live mesh re-translation.** Editing the plotter from Python
  after `show(backend="web")` blocks the main thread — there's no
  Python REPL while the Trame server runs. Use the desktop viewport
  for live editing, or pre-build the scene before calling
  `show(backend="web")`.
- **`pl.off_screen=True` is fine** for the web backend (unlike
  desktop, which needs a real VTK window). The VtkLocalView still
  has something to draw because it reads the in-process render
  window's scene graph, not the window's pixels.
