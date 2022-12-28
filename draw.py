import bgl
import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .data import G_NAME, G_INDICES, G_MODIFIERS_PROPERTY


class Handler:
    handler = None
    handle_text = None

    @classmethod
    def add_handler(cls):
        """向3d视图添加绘制handler
        并将其存储下来
        """
        if not cls.handler:
            cls.handler = bpy.types.SpaceView3D.draw_handler_add(
                Draw3D().draw_bound_box, (), 'WINDOW', 'POST_VIEW')

    @classmethod
    def draw_scale_text(cls, ob):
        if (ob.scale != Vector((1, 1, 1))) and not cls.handle_text:
            cls.handle_text = bpy.types.SpaceView3D.draw_handler_add(cls.draw_str, (), 'WINDOW', 'POST_PIXEL')

    @classmethod
    def del_handler_text(cls):
        if cls.handle_text:
            bpy.types.SpaceView3D.draw_handler_remove(cls.handle_text, 'WINDOW')
            cls.handle_text = None

    @classmethod
    def del_handler(cls):
        data = bpy.data
        if data.meshes.get(G_NAME):
            data.meshes.remove(data.meshes.get(G_NAME))

        if data.objects.get(G_NAME):
            data.objects.remove(data.objects.get(G_NAME))

        cls.del_handler_text()

        if cls.handler:
            bpy.types.SpaceView3D.draw_handler_remove(cls.handler, 'WINDOW')
            cls.handler = None


class Draw3D(Handler):
    simple_modifier_limits_co = [Vector(), Vector()]  # 存上下限移动的点,根据系数来设置位置
    simple_modifier_point_co = [Vector(), Vector()]
    object_max_min_co = [Vector(), Vector()]  # 存储网格的最大最小坐标,使用物体的名称作为key:value是最大最小坐标   "co":None co储存活动项的最大最小坐标
    simple_modifier_limits_bound = []
    deform_bound_draw_data = []  # 绘制形变变界框的数据,就是白色的边框,数据从复制出来的一个物体里面拿到 indices: "(list[Vector],(int,int))"
    numpy_data = {}  # 存放变形网格绘制信息

    @classmethod
    def draw_3d_shader(cls, pos, indices, color=None, *, shader_name='3D_UNIFORM_COLOR', draw_type='LINES'):
        """
        :param draw_type:
        :param shader_name:
        :param color:
        :param indices:
        :param pos:
        :type pos:list ((0,0,0),(1,1,1))
        2D_FLAT_COLOR - 2D_IMAGE - 2D_SMOOTH_COLOR - 2D_UNIFORM_COLOR - 3D_FLAT_COLOR - 3D_SMOOTH_COLOR - 3D_UNIFORM_COLOR - 3D_POLYLINE_FLAT_COLOR - 3D_POLYLINE_SMOOTH_COLOR - 3D_POLYLINE_UNIFORM_COLOR
        ('POINTS', 'LINES', 'TRIS', 'LINE_STRIP', 'LINE_LOOP','TRI_STRIP',
        'TRI_FAN', 'LINES_ADJ', 'TRIS_ADJ', 'LINE_STRIP_ADJ')
        `NONE`, `ALWAYS`, `LESS`, `LESS_EQUAL`, `EQUAL`, `GREATER` and `GREATER_EQUAL`
        """

        shader = gpu.shader.from_builtin(shader_name)
        if draw_type == 'POINTS':
            batch = batch_for_shader(shader, draw_type, {'pos': pos})
        else:
            batch = batch_for_shader(
                shader, draw_type, {'pos': pos}, indices=indices)

        shader.bind()
        if color:
            shader.uniform_float('color', color)

        batch.draw(shader)

    @classmethod
    def draw_text(cls, x, y, text='Hello Word', font_id=0, size=10, *, color=(0.5, 0.5, 0.5, 1), dpi=72, column=0):
        blf.position(font_id, x, y - (size * (column + 1)), 0)
        blf.size(font_id, size, dpi)
        blf.draw(font_id, text)
        blf.color(font_id, *color)

    font_info = {
        'font_id': 0,
        'handler': None,
    }

    @property
    def pref(self=None):
        from .utils import Pref
        return Pref.pref_()

    @classmethod
    def co_to_bound(cls, data):
        """将两个最大最小坐标转换为8个坐标用于绘制
        :param data:
        :return:list[list[Vector]]
        """
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) = data
        return (
            (max_x, min_y, min_z),
            (min_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (max_x, min_y, max_z),
            (min_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z))

    @classmethod
    def c_matrix(cls, mat, data):
        from .utils import GizmoUtils
        return GizmoUtils.matrix_calculation(mat, data)

    @classmethod
    def draw_str(cls):
        obj = bpy.context.object
        font_id = cls.font_info['font_id']

        blf.position(font_id, 200, 80, 0)
        blf.size(font_id, 15, 72)
        blf.color(font_id, 1, 1, 1, 1)
        blf.draw(
            font_id,
            f'The scaling value of the object {obj.name_full} is not 1,'
            f' which will cause the deformation of the simple deformation modifier.'
            f' Please apply the scaling before deformation')
        if obj.scale == Vector((1, 1, 1)):
            cls.del_handler_text()

    @classmethod
    def draw_box(cls, matrix):
        coords = cls.c_matrix(matrix, cls.co_to_bound(cls.object_max_min_co))
        cls.draw_3d_shader(coords, G_INDICES, cls.pref.fget().bound_box_color)

    @classmethod
    def draw_limits_bound_box(cls, matrix):
        """TODO绘制限制边界框

        :return:
        """
        if cls.simple_modifier_limits_bound:
            # draw limits_bound_box
            bgl.glEnable(bgl.GL_DEPTH_TEST)
            cls.draw_3d_shader(
                cls.c_matrix(matrix, cls.co_to_bound(cls.simple_modifier_limits_bound)),
                G_INDICES,
                cls.pref.fget().limits_bound_box_color
            )

    def draw_limits_line(self, matrix):
        """绘制上下限的线
        :return:
        """
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        # if :
        #     # draw  line
        #     cls.draw_3d_shader(cls.simple_modifier_point_co, ((1, 0),), (1, 1, 0, 0.3))

        co = self.__class__.simple_modifier_limits_co

        if co:
            # draw limits line
            self.draw_3d_shader(self.c_matrix(matrix, co), ((1, 0),), (1, 1, 0, 0.5))

            # TODO draw pos
            # cls.draw_3d_shader([line_pos[1]], (), (0, 1, 0, 0.5),
            #                    shader_name='3D_UNIFORM_COLOR', draw_type='POINTS')

    @classmethod
    def draw_deform_bound(cls, obj, modifier):
        """TODO 绘制形变框
        添加一个物体并把当前简易形变修改器的参数复制到这个物体
        再绘制出这个物体的线,就形成了形变边界框

        :return:
        """
        if cls.deform_bound_draw_data:
            matrix = obj.matrix_world
            ver, indices, limits, modifier_property, mat = cls.deform_bound_draw_data
            mod_data = ([getattr(modifier, i) for i in G_MODIFIERS_PROPERTY] == modifier_property)
            if mod_data and mat == matrix:
                bgl.glEnable(bgl.GL_DEPTH_TEST)
                cls.draw_3d_shader(
                    ver,
                    indices,
                    cls.pref.fget().deform_wireframe_color,
                )

    def is_draw_box(self, context):
        """绘制框"""

        from .utils import GizmoUtils
        obj = context.object  # 活动物体
        matrix = obj.matrix_world  # 活动物体矩阵
        modifier = context.object.modifiers.active  # 活动修改器

        simple_poll = GizmoUtils.simple_deform_poll(context)
        is_bend = modifier and (modifier.deform_method == 'BEND')
        display_switch_axis = not self.pref.display_bend_axis_switch_gizmo

        self.draw_scale_text(obj)

        if simple_poll and ((not is_bend) or display_switch_axis):
            # draw bound box
            self.draw_box(matrix)
            self.draw_limits_line(matrix)
            self.draw_limits_bound_box(matrix)

            self.draw_deform_bound(obj, modifier)
        elif simple_poll and (is_bend and not display_switch_axis):
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            self.draw_box(matrix)
            # GizmoUtils.empty_new(obj, modifier)

    def draw_bound_box(self):
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1)
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glEnable(bgl.GL_ALPHA)
        bgl.glDisable(bgl.GL_DEPTH_TEST)

        context = bpy.context
        from .utils import GizmoUtils
        if GizmoUtils.simple_deform_poll(context):
            self.is_draw_box(context)
        else:
            self.del_handler()
