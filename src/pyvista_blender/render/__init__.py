# SPDX-FileCopyrightText: 2026 Kevin Marchais
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render engine dispatch, output configuration, animation.

Modules:

* ``engine`` — Cycles vs Eevee dispatch + GPU device discovery
  (CUDA / OPTIX / HIP / Metal / oneAPI / CPU)
* ``output`` — resolution, file format, transparent film, codec
* ``animate`` — ``frame_change_pre`` handler driving the user ``updater(frame)``
"""
