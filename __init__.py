# SPDX-License-Identifier: GPL-2.0-or-later


bl_info = {
    "name": "SimpleDeformHelper",
    "author": "AIGODLIKE Community(BlenderCN辣椒,小萌新)",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "3D View -> Select an object and the active modifier is simple deformation",
    "description": "Simple Deform visualization adjustment tool",
    "doc_url": "{BLENDER_MANUAL_URL}/addons/simple_deform_helper/index.html",
    "wiki_url": "",
    "category": "3D View"
}

from . import (
    gizmo,
    operators,
    preferences
)

module_tuple = (
    gizmo,
    operators,
    preferences,
)


def register():
    for item in module_tuple:
        item.register()


def unregister():
    for item in module_tuple:
        item.unregister()
