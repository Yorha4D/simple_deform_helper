
import bpy
from bpy.types import Gizmo
from math import pi, sqrt, degrees

from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from ..utils import Pref, GizmoUtils


class UpdateGizmo(Gizmo, GizmoUtils, Pref):
    """TODO 更改参数时更新
    切换物体时更新
    手动更改值时更新
    """
    draw_type = "None_GizmoGroup_"
    matrix_basis: "Matrix"
    rotate_follow_modifier = True
    float_value = 0  # 控制值的数据

    @property
    def mouse_value(self):
        event = self.event
        context = self.context

        matrix = self.object.matrix_world

        x, y = view3d_utils.location_3d_to_region_2d(
            context.region, context.space_data.region_3d, matrix @ self.up_point)
        x2, y2 = view3d_utils.location_3d_to_region_2d(
            context.region, context.space_data.region_3d, matrix @ self.down_point)

        mouse_line_distance = sqrt(((event.mouse_region_x - x2) ** 2) +
                                   ((event.mouse_region_y - y2) ** 2))
        straight_line_distance = sqrt(((x2 - x) ** 2) +
                                      ((y2 - y) ** 2))
        delta = mouse_line_distance / straight_line_distance

        v_up = Vector((x, y))
        v_down = Vector((x2, y2))
        limits_angle = v_up - v_down

        mouse_v = Vector((event.mouse_region_x, event.mouse_region_y))

        mouse_angle = mouse_v - v_down
        angle_ = mouse_angle.angle(limits_angle)
        if angle_ > (pi / 2):
            delta = 0
        return delta

    def set_down_value(self):
        mu = self.limits_middle
        limit_value = mu - self.limits_scope if self.is_middle_mode else self.limits_max_value

        value = self.value_limit(self.mouse_value,
                                 max_value=limit_value,
                                 )
        set_value: float
        if self.event.ctrl:
            set_value = value + self.limits_difference
        elif self.is_middle_mode:
            if self.origin_mode == 'LIMITS_MIDDLE':
                set_value = mu - (value - mu)
            elif self.origin_mode == 'MIDDLE':
                set_value = 1 - value
            else:
                set_value = self.simple_modifier_up_limits_value
        else:
            set_value = self.simple_modifier_up_limits_value

        self.target_set_value('down_limits_value', value)
        self.target_set_value('up_limits_value', set_value)

    def set_up_value(self):
        limits_scope = self.limits_scope
        mu = self.limits_middle
        limit_value = mu + limits_scope if self.is_middle_mode else self.limits_min_value
        value = self.value_limit(self.mouse_value,
                                 min_value=limit_value)
        set_value: float

        if self.event.ctrl:
            set_value = value - self.limits_difference
        elif self.is_middle_mode:
            if self.origin_mode == 'LIMITS_MIDDLE':
                set_value = mu - (value - mu)
            elif self.origin_mode == 'MIDDLE':
                set_value = 1 - value
            else:
                set_value = self.simple_modifier_down_limits_value
        else:
            set_value = self.simple_modifier_down_limits_value

        self.target_set_value('down_limits_value', set_value)
        self.target_set_value('up_limits_value', value)

    def update_prop_value(self):
        if 'up_limits' == self.control_mode:
            self.set_up_value()
        elif 'down_limits' == self.control_mode:
            self.set_down_value()

    def update_gizmo_translation(self):
        """更新gizmo矩阵

        :return:
        """
        matrix = self.object.matrix_world
        if 'up_limits' == self.control_mode:
            self.matrix_basis.translation = matrix @ self.up_limits
        elif 'down_limits' == self.control_mode:
            self.matrix_basis.translation = matrix @ self.down_limits

    def update_header_text(self, context):
        def tr(te):
            return bpy.app.translations.pgettext(te)

        if (self.simple_modifier.deform_method in ('TWIST', 'BEND')) and (self.control_mode in ('angle',)):
            text = tr("Angle") + ':{}'.format(degrees(self.simple_modifier.angle))
        elif 'up_limits' == self.control_mode:
            text = tr("Upper limit") + ':{}'.format(self.simple_modifier_up_limits_value)
        elif 'down_limits' == self.control_mode:
            text = tr("Down limit") + ':{}'.format(self.simple_modifier_down_limits_value)
        else:
            text = tr("Coefficient") + ':{}'.format(self.simple_modifier.factor)
        text += '       '
        text += tr(self.object_property.bl_rna.properties[
                       'origin_mode'].enum_items[self.object_property.origin_mode].name)
        context.area.header_text_set(text)


class ViewSimpleDeformGizmo(UpdateGizmo):
    """显示轴向切换拖动点Gizmo(两个点)和角度控制
    """
    bl_idname = "ViewSimpleDeformGizmo"

    bl_target_properties = (
        {'id': 'up_limits_value', 'type': 'FLOAT', 'array_length': 1},
        {'id': 'down_limits_value', 'type': 'FLOAT', 'array_length': 1},
    )

    def setup(self):
        self.load_custom_shape_gizmo()
        self.update_bound_box(self.object)
        self.update_empty()

        self.add_handler()

    def draw(self, context):
        self.draw_custom_shape(self.custom_shape[self.draw_type])
        self.update_gizmo_translation()

    def draw_select(self, context, select_id):
        self.draw_custom_shape(
            self.custom_shape[self.draw_type], select_id=select_id)
        self.update_gizmo_translation()

    def invoke(self, context, event):
        """TODO 简化参数
        :param context:
        :param event:
        :return:
        """
        self.init_invoke(context, event)
        self.init_event(event)

        down, up = self._limits

        if self.control_mode == "up_limits":
            self.target_set_value('up_limits_value', up)
        elif self.control_mode == "down_limits":
            self.target_set_value('down_limits_value', down)

        self.update_bound_box(self.object)
        self.update_deform_wireframe()
        self.update_empty()
        return {'RUNNING_MODAL'}

    def modal(self, context, event, tweak):
        self.init_modal_data(context, event, tweak)
        self.update_bound_box(self.object)
        self.update_prop_value()
        self.update_limits_and_bound()
        self.update_gizmo_translation()

        self.update_empty()
        self.update_deform_wireframe()
        self.add_handler()
        return self.event_ops()

    def refresh(self, context):
        self.update_empty()
