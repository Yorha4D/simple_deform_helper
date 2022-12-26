from bpy.types import Gizmo
from mathutils import Vector, Euler, Matrix
from math import pi

from ..utils import GizmoUtils


class AngleGizmo(Gizmo, GizmoUtils):
    """绘制自定义Gizmo
        ('angle',
         {'control_mode': 'angle',
          'draw_type': 'SimpleDeform_GizmoGroup_',
          'color': (1.0, 0.5, 1.0),
          'alpha': 0.3,
          'color_highlight': (1.0, 1.0, 1.0),
          'target_set_prop': ,
          **general_data
          }),
    """
    bl_idname = "Draw_Angle_Gizmo"

    draw_type = "SimpleDeform_GizmoGroup_"

    bl_target_properties = (
        {'id': 'angle', 'type': 'FLOAT', 'array_length': 1},
    )
    scale_basis: float
    float_angle_value: float
    alpha: float
    color: "tuple[float]"
    mouse_dpi = 10
    matrix_basis: "Matrix"

    def update_angle(self):
        """ 更新修改器角度

        :return:
        """
        delta = self.delta
        self.target_set_value('angle', self.float_angle_value - delta)

    def update_matrix(self):
        """更新gizmo变换坐标

        :return:
        """
        translation = Vector((self.object_max_min_co[1]))
        matrix = Matrix()
        matrix.translation = translation
        self.matrix_basis = self.object.matrix_world @ matrix

    def update(self):
        self.update_deform_wireframe(self.object)
        self.update_matrix()

    def setup(self):
        self.load_custom_shape_gizmo()
        self.add_handler()
        self.scale_basis = .1
        self.color = (1, .5, 1)
        self.alpha = .5
        self.update_bound_box(self.object)
        self.update_matrix()

    def draw(self, context):
        self.draw_custom_shape(self.custom_shape[self.draw_type])
        self.update_matrix()

    def draw_select(self, context, select_id):
        self.draw_custom_shape(
            self.custom_shape[self.draw_type], select_id=select_id)
        self.update_matrix()

    def invoke(self, context, event):
        self.init_invoke(context, event)
        self.float_angle_value = self.target_get_value('angle')
        self.init_event(event)
        self.update_bound_box(self.object)
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        self.init_modal_data(context, event, tweak)

        self.update_angle()
        self.add_handler()
        return {'RUNNING_MODAL'}
