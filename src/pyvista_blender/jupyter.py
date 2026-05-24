# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Jupyter inline backend for ``pl.show(jupyter_backend="blender")``.

Registers via the ``pyvista.jupyter_backends`` entry point so calling
``pv.set_jupyter_backend("blender")`` (or letting pyvista pick it up
through auto-detection) routes notebook display through the bridge.
The handler runs one offline Cycles render via the existing
:meth:`pl.blender.render` path and returns an ``IPython.display.Image``
holding the resulting PNG bytes.

This is the lightest possible Jupyter integration: no live viewport,
no per-cell interactivity, no Trame plumbing. The user gets a
path-traced still frame inline; if they want interactivity, the
in-process ``pl.blender.show()`` viewport remains the right tool. The
two surfaces share the same translator + cache, so quality and
behaviour stay consistent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pyvista as pv

    from pyvista_blender._options import _RenderKwargs

__all__ = ["handler"]


def handler(
    plotter: pv.BasePlotter,
    *,
    screenshot: str | Path | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> object:
    """Render ``plotter`` through Cycles and return an inline IPython Image.

    Parameters
    ----------
    plotter
        The :class:`pyvista.BasePlotter` whose state should be
        rendered. The bridge translates exactly what
        :meth:`pl.blender.render` would.
    screenshot
        Optional path to also save the PNG to disk (mirrors pyvista's
        contract for built-in backends). ``None`` (the default) keeps
        the render in a temp file that's deleted after the bytes are
        captured.
    **kwargs
        Forwarded to :meth:`pl.blender.render`. Common ones:
        ``engine`` (``"cycles"`` / ``"eevee"``), ``samples`` (int),
        ``denoise`` (bool), ``device`` (``"auto"`` / ``"optix"`` / …),
        ``transparent_bg`` (bool). Unknown keys are dropped before
        the forward so pyvista's own kwargs (e.g. ``window_size``)
        don't crash the render path.

    Returns
    -------
    object
        ``IPython.display.Image`` carrying the PNG bytes for inline
        display. The PNG is also saved to ``screenshot`` when that
        path is provided.

    Raises
    ------
    ImportError
        When IPython is not installed. The bridge reuses pyvista's
        notebook stack, so the error guides the user to
        ``pip install 'pyvista[jupyter]'``.

    """
    try:
        from IPython.display import Image  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - import guard
        msg = (
            "the 'blender' jupyter backend needs IPython. Install "
            "pyvista's jupyter extra: pip install 'pyvista[jupyter]'"
        )
        raise ImportError(msg) from err

    render_kwargs = _filter_render_kwargs(kwargs)
    target = Path(screenshot) if screenshot is not None else None
    if target is None:
        with tempfile.NamedTemporaryFile(
            prefix="pvblender_jupyter_", suffix=".png", delete=False
        ) as fh:
            target = Path(fh.name)
        cleanup = True
    else:
        cleanup = False

    try:
        plotter.blender.render(str(target), **render_kwargs)
        png_bytes = target.read_bytes()
    finally:
        if cleanup:
            target.unlink(missing_ok=True)

    return Image(data=png_bytes, format="png")


#: Render kwargs the handler forwards to ``pl.blender.render``. Anything
#: not in this allowlist is dropped before the forward so pyvista-side
#: keys (``window_size``, ``return_img``, ``screenshot``, ``cpos`` and
#: similar) can't crash the bridge.
_FORWARD_KEYS: frozenset[str] = frozenset({
    "engine",
    "device",
    "samples",
    "denoise",
    "transparent_bg",
})


def _filter_render_kwargs(kwargs: dict[str, Any]) -> _RenderKwargs:
    """Drop unknown keys so only ``pl.blender.render`` kwargs survive.

    The input is typed as ``dict[str, Any]`` because pyvista's jupyter
    backend protocol passes through whatever the user typed at the
    notebook level; the output is the precisely-typed subset that
    ``pl.blender.render`` actually accepts.

    Returns
    -------
    _RenderKwargs
        Subset of ``kwargs`` whose keys appear in :data:`_FORWARD_KEYS`.

    """
    return cast(
        "_RenderKwargs", {k: v for k, v in kwargs.items() if k in _FORWARD_KEYS}
    )
