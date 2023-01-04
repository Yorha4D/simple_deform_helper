from math import radians

from bpy.types import GizmoGroup
from mathutils import Euler, Vector

from .gizmo_group import CustomGizmo
from ..operators import DeformAxisOperator
from ..utils import GizmoUtils, Pref


class SimpleDeformGizmoGroupDisplayBendAxiSwitchGizmo(GizmoGroup, GizmoUtils, Pref):
    """绘制切换变型轴的
    变换方向
    仅显示在简易形变修改器为弯曲时
    """
    bl_idname = 'OBJECT_GGT_SimpleDeformGizmoGroup_display_bend_axis_switch_gizmo'
    bl_label = 'SimpleDeformGizmoGroup_display_bend_axis_switch_gizmo'

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {
        '3D',
        'PERSISTENT',
    }

    @classmethod
    def poll(cls, context):
        pref = cls.pref_()
        simple = cls.simple_deform_poll(context)
        bend = simple and (
                context.object.modifiers.active.deform_method == 'BEND')
        switch_axis = pref.display_bend_axis_switch_gizmo is True
        return switch_axis and bend

    def setup(self, context):
        draw_type = 'SimpleDeform_Bend_Direction_'
        color_a = 1, 0, 0
        color_b = 0, 1, 0
        self.add_handler()
        r = radians

        for direction_name, axis, rot, positive in (
                ('top_a', 'X', (r(90), 0, r(90)), True),
                ('top_b', 'X', (r(90), 0, 0), True),

                ('bottom_a', 'X', (r(90), 0, r(90)), False),
                ('bottom_b', 'X', (r(90), 0, 0), False),

                ('left_a', 'Y', (r(90), 0, 0), False),
                ('left_b', 'Y', (0, 0, 0), False),

                ('right_a', 'Y', (r(90), 0, 0), True),
                ('right_b', 'Y', (0, 0, 0), True),

                ('front_a', 'Z', (0, 0, 0), False),
                ('front_b', 'X', (0, 0, 0), False),
                ('back_a', 'Z', (0, 0, 0), True),
                ('back_b', 'X', (0, 0, 0), True),):
            is_a = (direction_name.split('_')[1] == 'a')
            setattr(self, direction_name, self.gizmos.new(CustomGizmo.bl_idname))
            gizmo = getattr(self, direction_name)
            gizmo.mode = direction_name
            gizmo.draw_type = draw_type
            gizmo.color = color_a if is_a else color_b
            gizmo.alpha = 0.3
            gizmo.color_highlight = 1.0, 1.0, 1.0
            gizmo.alpha_highlight = 1
            gizmo.use_draw_modal = True
            gizmo.scale_basis = 0.2
            gizmo.use_draw_value = True

            # set operator property
            ops = gizmo.target_set_operator(DeformAxisOperator.bl_idname)
            ops.Deform_Axis = axis
            ops.X_Value = rot[0]
            ops.Y_Value = rot[1]
            ops.Z_Value = rot[2]
            ops.Is_Positive = positive

    def draw_prepare(self, context):
        ob = context.object
        matrix = ob.matrix_world
        top, bottom, left, right, front, back = self.each_face_pos(matrix)

        rad = radians
        for_list = (
            ('top_a', top, (0, 0, 0),),
            ('top_b', top, (0, 0, rad(90)),),

            ('bottom_a', bottom, (0, rad(180), 0),),
            ('bottom_b', bottom, (0, rad(180), rad(90)),),

            ('left_a', left, (rad(-90), 0, rad(90)),),
            ('left_b', left, (0, rad(-90), 0),),

            ('right_a', right, (rad(90), 0, rad(90)),),
            ('right_b', right, (0, rad(90), 0),),

            ('front_a', front, (rad(90), 0, 0),),
            ('front_b', front, (rad(90), rad(90), 0),),

            ('back_a', back, (rad(-90), 0, 0),),
            ('back_b', back, (rad(-90), rad(-90), 0),),
        )
        for i, j, w, in for_list:
            gizmo = getattr(self, i, False)
            rot = Euler(w, 'XYZ').to_matrix().to_4x4()

            gizmo.matrix_basis = matrix.to_euler().to_matrix().to_4x4() @ rot
            gizmo.matrix_basis.translation = Vector(j)
