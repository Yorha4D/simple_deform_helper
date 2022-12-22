# SPDX-License-Identifier: GPL-2.0-or-later
from . import operator, preferences, timers, translate, gizmo

bl_info = {
    "name": "SimpleDeformHelper",
    "author": "AIGODLIKE Community(BlenderCN辣椒,小萌新)",
    "version": (0, 1, 1),
    "blender": (3, 0, 0),
    "location": "3D View -> Select an object and the active modifier is simple deformation",
    "description": "Simple Deform visualization adjustment tool",
    "doc_url": "https://github.com/Yorha4D/simple_deform_helper/blob/main/README_CN.md",
    "wiki_url": "",
    "category": "3D View"
}

module_tuple = (
    gizmo,
    timers,
    operator,
    translate,
    preferences,
)


def register():
    for item in module_tuple:
        item.register()


def unregister():
    for item in module_tuple:
        item.unregister()
