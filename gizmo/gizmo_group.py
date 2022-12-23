import bpy
from bpy.types import (
    Gizmo,
    GizmoGroup,
)
from mathutils import Vector

from ..utils import GizmoUtils, Pref


class CustomGizmo(Gizmo, GizmoUtils):
    """绘制自定义Gizmo"""
    bl_idname = "Draw_Custom_Gizmo"

    draw_type = "None_GizmoGroup_"

    bl_target_properties = (
        {'id': 'value', 'type': 'FLOAT', 'array_length': 1},
    )

    def setup(self):
        self.load_custom_shape_gizmo()
        self.add_handler()

    def draw(self, context):
        self.draw_custom_shape(self.custom_shape[self.draw_type])

    def draw_select(self, context, select_id):
        self.draw_custom_shape(
            self.custom_shape[self.draw_type], select_id=select_id)

    def invoke(self, context, event):
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        self.add_handler()
        return {'RUNNING_MODAL'}


class SimpleDeformGizmoGroup(GizmoGroup, GizmoUtils, Pref):
    """显示Gizmo
    此类管理 上下限及角度gizmo还有显示切换轴的2d按钮
    """
    bl_idname = 'OBJECT_GGT_SimpleDeformGizmoGroup'
    bl_label = 'SimpleDeformGizmoGroup'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}
    gizmo_angle: "bpy.types.Gizmo"

    def set_simple_control_gizmo(self):
        """生成gizmo的上限下限及角度的gizmo
        """

        general_data = {
            'alpha_highlight': 0.3,
            'use_draw_modal': True,
            'scale_basis': 0.1,
            'use_draw_value': True
        }

        gizmo_data = (('up_limits',
                       {'control_mode': 'up_limits',
                        'draw_type': 'Sphere_GizmoGroup_',
                        'mouse_dpi': 1000,
                        'color': (1.0, 0, 0),
                        'alpha': 0.5,
                        'color_highlight': (1.0, 1.0, 1.0),
                        'target_set_prop': ('offset', self.simple_modifier, 'angle'),
                        **general_data
                        }),
                      ('down_limits',
                       {'control_mode': 'down_limits',
                        'draw_type': 'Sphere_GizmoGroup_',
                        'mouse_dpi': 1000,
                        'color': (0, 1.0, 0),
                        'alpha': 0.5,
                        'color_highlight': (1.0, 1.0, 1.0),
                        'target_set_prop': ('offset', self.object_property, 'down_limits'),
                        **general_data
                        }),

                      )

        from .limits_point_gizmo import ViewSimpleDeformGizmo
        # ViewSimpleDeformGizmo.bl_idname
        for gizmo_id, gizmo_info in gizmo_data:
            gizmo_name = "gizmo_" + gizmo_id

            setattr(self, gizmo_name, self.gizmos.new("GIZMO_GT_arrow_3d"))
            gizmo = getattr(self, gizmo_name)
            for key in gizmo_info:
                if key == 'target_set_operator':  # 操作符
                    gizmo.target_set_operator(gizmo_info[key])
                elif key == 'target_set_prop':  # 设置属性
                    gizmo.target_set_prop(*gizmo_info[key])
                elif getattr(gizmo, key, False):
                    setattr(gizmo, key, gizmo_info[key])

    def set_angle_gizmo(self):
        from .angle_gizmo import AngleGizmo
        gizmo = self.gizmos.new(AngleGizmo.bl_idname)
        gizmo.target_set_prop('value', self.simple_modifier, 'angle')
        self.gizmo_angle = gizmo

    def set_axis_switch_gizmo(self):
        """生成切换轴的2D 按钮

        :return:
        """
        data_path = 'object.modifiers.active.deform_axis'
        set_enum = 'wm.context_set_enum'

        for axis in ('X', 'Y', 'Z'):
            # show toggle axis button
            gizmo = self.gizmos.new('GIZMO_GT_button_2d')
            gizmo.icon = f'EVENT_{axis.upper()}'
            gizmo.draw_options = {'BACKDROP', 'HELPLINE'}
            ops = gizmo.target_set_operator(set_enum)
            ops.data_path = data_path
            ops.value = axis
            gizmo.color = (0, 0, 0)
            gizmo.alpha = 0.3
            gizmo.color_highlight = 1.0, 1.0, 1.0
            gizmo.alpha_highlight = 0.3
            gizmo.use_draw_modal = True
            gizmo.use_draw_value = True
            gizmo.scale_basis = 0.1
            setattr(self, f'gizmo_deform_axis_{axis.lower()}', gizmo)

    def update_2d_button_translation(self):
        obj = bpy.context.object
        mat = obj.matrix_world
        if self.object_max_min_co:
            def _mat(f):
                co = self.object_max_min_co[0]
                co = (co[0] + (max(obj.dimensions) * f), co[1],
                      co[2] - (min(obj.dimensions) * 0.3))
                return mat @ Vector(co)

            self.gizmo_deform_axis_x.matrix_basis.translation = _mat(0)
            self.gizmo_deform_axis_y.matrix_basis.translation = _mat(0.3)
            self.gizmo_deform_axis_z.matrix_basis.translation = _mat(0.6)

    @classmethod
    def poll(cls, context):
        pol = cls.simple_deform_poll(context)
        pref = cls.pref_()
        deform_method = (
                pol and (context.object.modifiers.active.deform_method != 'BEND'))
        display_gizmo = pref.display_bend_axis_switch_gizmo
        switch = (not display_gizmo)
        return pol and (deform_method or switch)

    def setup(self, context):
        self.set_simple_control_gizmo()
        self.set_axis_switch_gizmo()
        self.set_angle_gizmo()
        self.add_handler()

    def refresh(self, context):
        self.add_handler()
        ob = context.object

        matrix = ob.matrix_world.normalized()
        for index, key in enumerate(('up_limits',
                                     'down_limits',
                                     )):
            gizmo = getattr(self, "gizmo_" + key)
            mat = matrix.copy()
            mat.translation.x += index
            gizmo.matrix_basis = mat
        self.gizmo_angle.matrix_basis.translation = Vector()

    def draw_prepare(self, context):
        """TODO 更新2d切换按钮位置

        :param context:
        :return:
        """
        # self.update_2d_button_translation()
        self.add_handler()
