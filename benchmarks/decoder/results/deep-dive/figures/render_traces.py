"""Render retained SVG timelines to PNG using system librsvg and Cairo.

Run: python benchmarks/decoder/results/deep-dive/figures/render_traces.py
No Python packages are required. These are existing trace figures, not new captures.
"""
import ctypes as C
from ctypes.util import find_library
from pathlib import Path
import xml.etree.ElementTree as ET


def function(library, name, result, *arguments):
    fn = getattr(library, name)
    fn.restype, fn.argtypes = result, arguments
    return fn


rsvg = C.CDLL(find_library('rsvg-2'))
cairo = C.CDLL(find_library('cairo'))
gobject = C.CDLL(find_library('gobject-2.0'))
new_svg = function(rsvg, 'rsvg_handle_new_from_file', C.c_void_p, C.c_char_p, C.c_void_p)
render = function(rsvg, 'rsvg_handle_render_cairo', C.c_int, C.c_void_p, C.c_void_p)
unref = function(gobject, 'g_object_unref', None, C.c_void_p)
new_surface = function(cairo, 'cairo_image_surface_create', C.c_void_p, C.c_int, C.c_int, C.c_int)
new_context = function(cairo, 'cairo_create', C.c_void_p, C.c_void_p)
scale = function(cairo, 'cairo_scale', None, C.c_void_p, C.c_double, C.c_double)
write_png = function(cairo, 'cairo_surface_write_to_png', C.c_int, C.c_void_p, C.c_char_p)
destroy_context = function(cairo, 'cairo_destroy', None, C.c_void_p)
destroy_surface = function(cairo, 'cairo_surface_destroy', None, C.c_void_p)

HERE = Path(__file__).resolve().parent
for name in ('small', 'wide', 'long', 'vocab'):
    source = HERE.parent / 'traces' / (name + '.svg')
    root = ET.parse(source).getroot()
    width, height = (round(float(root.attrib[key]) * 2) for key in ('width', 'height'))
    handle = new_svg(str(source).encode(), None)
    if not handle:
        raise ValueError(f'Cannot load {source}')
    surface = new_surface(0, width, height)  # CAIRO_FORMAT_ARGB32
    context = new_context(surface)
    try:
        scale(context, 2, 2)
        if not render(handle, context):
            raise ValueError(f'Cannot render {source}')
        if write_png(surface, str(HERE / (name + '-timeline.png')).encode()):
            raise OSError(f'Cannot write PNG: {name}')
    finally:
        destroy_context(context)
        destroy_surface(surface)
        unref(handle)
