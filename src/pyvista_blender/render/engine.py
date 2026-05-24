# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Configure Cycles or Eevee Next for a headless render."""

from __future__ import annotations

import bpy

from pyvista_blender.render._device import select_cycles_device

__all__ = ["configure_cycles", "configure_eevee", "configure_engine"]


def configure_engine(
    *,
    engine: str,
    device: str = "auto",
    samples: int = 64,
    denoise: bool = True,
    transparent_bg: bool = False,
) -> None:
    """Dispatch ``configure_cycles`` / ``configure_eevee`` based on ``engine``.

    Parameters
    ----------
    engine
        ``"cycles"`` or ``"eevee"``. Validated against
        :data:`~pyvista_blender.config.SUPPORTED_ENGINES` upstream in
        ``BlenderComponent._resolve_engine``.
    device, samples, denoise, transparent_bg
        Forwarded to the engine-specific configurator. ``device`` and
        ``denoise`` are Cycles-only and ignored by Eevee Next.

    Raises
    ------
    ValueError
        When ``engine`` is neither ``"cycles"`` nor ``"eevee"``.

    """
    if engine == "cycles":
        configure_cycles(
            device=device,
            samples=samples,
            denoise=denoise,
            transparent_bg=transparent_bg,
        )
    elif engine == "eevee":
        configure_eevee(samples=samples, transparent_bg=transparent_bg)
    else:
        msg = f"unknown engine {engine!r}; expected 'cycles' or 'eevee'"
        raise ValueError(msg)


def configure_cycles(
    *,
    device: str = "auto",
    samples: int = 64,
    denoise: bool = True,
    transparent_bg: bool = False,
) -> None:
    """Apply Cycles render settings to the active bpy scene.

    Parameters
    ----------
    device
        Compute device. ``"auto"`` / ``"gpu"`` walk OptiX > CUDA > HIP >
        Metal > oneAPI > CPU at runtime. Named backends force one. ``"cpu"``
        stays on CPU. Unavailable GPU backends fall back to CPU with a warning.
    samples
        Path-tracing samples per pixel. 32 - 64 is a good preview range; 128+
        for publication output.
    denoise
        Enable post-render denoising via OpenImageDenoise (CPU-safe).
    transparent_bg
        Render with a transparent film (alpha channel in the PNG).

    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = bool(denoise)
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.device = select_cycles_device(device)
    # Keep the ray-acceleration structure (BVH) and intermediate render
    # data alive between successive renders. The first render still pays
    # the full build cost; subsequent renders on the same scene skip it,
    # which is worth ~300-500 ms per frame on a typical scivis mesh.
    # Camera-only changes (the common interactive-viewport case) get the
    # full speedup. Topology mutations invalidate the cache transparently.
    scene.render.use_persistent_data = True
    _apply_common_render_settings(scene, transparent_bg=transparent_bg)


def configure_eevee(*, samples: int = 64, transparent_bg: bool = False) -> None:
    """Apply Eevee Next settings to the active bpy scene.

    Eevee Next is the rasterizer-based engine (no path tracing). Per-frame
    cost is roughly an order of magnitude lower than Cycles at comparable
    quality — typical bpy headless render is ~50-150 ms regardless of
    sample count. The trade-off: no true global illumination, weaker
    PBR fidelity, no built-in denoiser.

    On Blender 5.x headless, Eevee Next renders through EGL on Linux. On
    Windows / macOS the bpy build typically lacks a working GL context
    in headless mode; ``bpy.ops.render.render`` raises in that case.

    Eevee Next is incompatible with the interactive viewport
    (``pl.blender.show()``) on shared X11 displays: VTK already owns the
    GL context and Eevee's ``X_GLXMakeCurrent`` call crashes with
    ``BadAccess``. Use Eevee for off-screen renders (``render`` /
    ``animate`` on plotters with ``off_screen=True``); stay on Cycles
    for the interactive viewport.

    Parameters
    ----------
    samples
        TAA samples per pixel for the final render
        (``scene.eevee.taa_render_samples``). 16-64 covers preview to
        publication.
    transparent_bg
        Preserve the film alpha so PNG output carries transparency.

    """
    scene = bpy.context.scene
    scene.render.engine = _eevee_engine_id()
    scene.eevee.taa_render_samples = int(samples)
    _apply_common_render_settings(scene, transparent_bg=transparent_bg)


def _eevee_engine_id() -> str:
    """Return the engine-enum name for Eevee Next on the running bpy version.

    Blender 4.2 LTS introduced Eevee Next under the enum name
    ``"BLENDER_EEVEE_NEXT"`` (the original Eevee stayed as
    ``"BLENDER_EEVEE"``). Blender 4.3+ replaced the original Eevee with
    Eevee Next and reverted the enum to ``"BLENDER_EEVEE"``. We pick
    by sniffing the available enum entries.

    Returns
    -------
    str
        ``"BLENDER_EEVEE_NEXT"`` on bpy 4.2 LTS where both engines
        coexist; ``"BLENDER_EEVEE"`` on bpy 4.3+ / 5.x where Eevee
        Next replaced the original.

    """
    enum_items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    available = {item.identifier for item in enum_items}
    if "BLENDER_EEVEE_NEXT" in available:
        return "BLENDER_EEVEE_NEXT"
    return "BLENDER_EEVEE"


def _apply_common_render_settings(scene: object, *, transparent_bg: bool) -> None:
    """Shared render-time settings: film, output format, colour management."""
    scene.render.film_transparent = bool(transparent_bg)  # type: ignore[attr-defined]
    scene.render.image_settings.file_format = "PNG"  # type: ignore[attr-defined]
    scene.render.image_settings.color_mode = (  # type: ignore[attr-defined]
        "RGBA" if transparent_bg else "RGB"
    )
    # "Standard" view transform = linear->sRGB with no tone mapping. For
    # scivis we want the colormap to reproduce exactly; Blender's default
    # "AgX" / "Filmic" curves desaturate and crush brights to look photo-real.
    scene.view_settings.view_transform = "Standard"  # type: ignore[attr-defined]
