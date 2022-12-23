from bpy.types import Gizmo

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

    draw_type = "None_GizmoGroup_"

    bl_target_properties = (
        {'id': 'angle', 'type': 'FLOAT', 'array_length': 1},
    )

    def update_angle(self):
        ...

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
