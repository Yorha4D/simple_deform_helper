import bpy
from bpy.props import (FloatProperty,
                       PointerProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       BoolProperty)
from bpy.types import (
    AddonPreferences,
    PropertyGroup,
)

from .data import G_ORIGIN_MODE_ITEMS, G_ADDON_NAME
from .utils import Pref, GizmoUtils, get_active_simple_modifier


class SimpleDeformGizmoAddonPreferences(AddonPreferences, Pref):
    bl_idname = G_ADDON_NAME

    default_color_data = {"soft_max": 1,
                          "soft_min": 0,
                          "size": 4,
                          "subtype": "COLOR",
                          }

    deform_wireframe_color: FloatVectorProperty(
        name='Deform Wireframe',
        description='Draw Deform Wireframe Color',
        default=(1, 1, 1, 0.3),
        **default_color_data)
    bound_box_color: FloatVectorProperty(
        name='Bound Box',
        description='Draw Bound Box Color',
        default=(1, 0, 0, 0.1),
        **default_color_data)
    limits_bound_box_color: FloatVectorProperty(
        name='Upper and lower limit Bound Box Color',
        description='Draw Upper and lower limit Bound Box Color',
        default=(0.3, 1, 0.2, 0.5),
        **default_color_data)
    modifiers_limits_tolerance: FloatProperty(
        name='Upper and lower limit tolerance',
        description='Minimum value between upper and lower limits',
        default=0.05,
        max=1,
        min=0.0001
    )
    display_bend_axis_switch_gizmo: BoolProperty(
        name='Show Toggle Axis Gizmo',
        default=False,
        options={'SKIP_SAVE'})

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'deform_wireframe_color')
        layout.prop(self, 'bound_box_color')
        layout.prop(self, 'limits_bound_box_color')
        layout.prop(self, 'modifiers_limits_tolerance')
        layout.prop(self, 'display_bend_axis_switch_gizmo')

    def draw_header_tool_settings(self, context) -> None:
        """绘制工具栏,在显示Gizmo时显示在顶栏
        :param context:
        :return: None
        """
        layout = self.layout
        if GizmoUtils.simple_deform_poll(context):
            layout.separator(factor=5)
            active_mod = context.object.modifiers.active
            prop = context.object.SimpleDeformGizmo_PropertyGroup
            pref = Pref.pref_()

            if active_mod.origin:
                layout.prop(active_mod.origin.SimpleDeformGizmo_PropertyGroup,
                            'origin_mode',
                            text='')
            else:
                layout.prop(prop,
                            'origin_mode',
                            text='')
            layout.prop(pref,
                        'modifiers_limits_tolerance',
                        text='')
            if active_mod.deform_method == 'BEND':
                layout.prop(pref,
                            'display_bend_axis_switch_gizmo',
                            toggle=1)
            layout.separator(factor=0.5)
            layout.prop(active_mod,
                        'deform_method',
                        expand=True)
            layout.prop(active_mod,
                        'deform_axis',
                        expand=True)
            layout.prop(active_mod,
                        'angle')
            layout.prop(active_mod,
                        'factor')


def get_limits(index) -> float:
    """输入索引反回限制值

    :param index:
    :return:
    """
    mod = get_active_simple_modifier()
    if mod:
        return mod.limits[index]
    return 114


def set_limits(value, index) -> None:
    """设置限制值
    :param value:
    :param index:
    :return:
    """
    mod = get_active_simple_modifier()
    if mod:
        mod.limits[index] = value


class SimpleDeformGizmoObjectPropertyGroup(PropertyGroup):
    def update(self, context):
        ...

    up_limits: FloatProperty(name='up',
                             description='UP Limits(Red)',
                             get=lambda _: get_limits(1),
                             set=lambda _, value: set_limits(value, 1),
                             update=update,
                             default=1,
                             max=1,
                             min=0)

    down_limits: FloatProperty(name='down',
                               description='Lower limit(Green)',
                               get=lambda _: get_limits(0),
                               set=lambda _, value: set_limits(value, 0),
                               default=0,
                               update=update,
                               max=1,
                               min=0)

    origin_mode: EnumProperty(name='Origin control mode',
                              default='NOT',
                              items=G_ORIGIN_MODE_ITEMS)


class_list = (
    SimpleDeformGizmoAddonPreferences,
    SimpleDeformGizmoObjectPropertyGroup,
)

register_class, unregister_class = bpy.utils.register_classes_factory(class_list)


def register():
    register_class()

    Pref.pref_().display_bend_axis_switch_gizmo = False
    bpy.types.Object.SimpleDeformGizmo_PropertyGroup = PointerProperty(
        type=SimpleDeformGizmoObjectPropertyGroup,
        name='SimpleDeformGizmo_PropertyGroup')
    bpy.types.VIEW3D_MT_editor_menus.append(
        SimpleDeformGizmoAddonPreferences.draw_header_tool_settings)


def unregister():
    unregister_class()
    bpy.types.VIEW3D_MT_editor_menus.remove(
        SimpleDeformGizmoAddonPreferences.draw_header_tool_settings)
