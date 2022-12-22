import bpy.utils

from .select_deform_axis import SimpleDeformGizmoGroupDisplayBendAxiSwitchGizmo
from .gizmo_group import CustomGizmo, SimpleDeformGizmoGroup
from .limits_point_gizmo import ViewSimpleDeformGizmo

class_tuple = (
    CustomGizmo,
    ViewSimpleDeformGizmo,
    SimpleDeformGizmoGroup,
    SimpleDeformGizmoGroupDisplayBendAxiSwitchGizmo,
)

register_class, unregister_class = bpy.utils.register_classes_factory(
    class_tuple)


def register():
    register_class()


def unregister():
    unregister_class()
