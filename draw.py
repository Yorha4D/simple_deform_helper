import bgl
import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .data import G_NAME, G_INDICES

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
                Draw3D.draw_bound_box, (), 'WINDOW', 'POST_VIEW')

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
    object_bound_max_min_co = Vector(), Vector()  # 物体边界框最大最小坐标
    object_limits_co = Vector(), Vector()  # 物体限制坐标 (上限,下限)

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

    font_info = {
        'font_id': 0,
        'handler': None,
    }

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
            Handler.del_handler_text()

    @classmethod
    def draw_text(cls, x, y, text='Hello Word', font_id=0, size=10, *, color=(0.5, 0.5, 0.5, 1), dpi=72, column=0):
        blf.position(font_id, x, y - (size * (column + 1)), 0)
        blf.size(font_id, size, dpi)
        blf.draw(font_id, text)
        blf.color(font_id, *color)

    @classmethod
    def draw_box(cls, data, mat):
        pref = cls.pref_()
        from .utils import Utils
        coords = Utils.matrix_calculation(mat,
                                          cls.data_to_calculation(data))
        cls.draw_3d_shader(coords, G_INDICES, pref.bound_box_color)

    @classmethod
    def data_to_calculation(cls, data):
        """将两个最大最小坐标转换为8个坐梂用于绘制
        :param data:
        :return:
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
    def draw_limits_bound_box(cls):
        """绘制限制边界框

        :return:
        """
        pref = cls.pref_()
        if cls.limits_bound_box:
            # draw limits_bound_box
            mat, data = cls.limits_bound_box
            bgl.glEnable(bgl.GL_DEPTH_TEST)
            from .utils import Utils
            coords = Utils.matrix_calculation(mat, cls.data_to_calculation(data))
            cls.draw_3d_shader(coords,
                               G_INDICES,
                               pref.limits_bound_box_color)

    @classmethod
    def draw_limits_line(cls):
        """绘制上下限的线
        :return:
        """
        if cls.limits_line:
            line_pos, limits_pos, = cls.limits_line
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            # draw limits line
            cls.draw_3d_shader(limits_pos, ((1, 0),), (1, 1, 0, 0.5))
            # draw  line
            cls.draw_3d_shader(line_pos, ((1, 0),), (1, 1, 0, 0.3))
            # draw pos
            cls.draw_3d_shader([line_pos[1]], (), (0, 1, 0, 0.5),
                               shader_name='3D_UNIFORM_COLOR', draw_type='POINTS')

    @classmethod
    def draw_deform_mesh(cls, ob, context):
        """绘制形变框
        添加一个物体并把当前简易形变修改器的参数复制到这个物体
        再绘制出这个物体的线,就形成了形变边界框

        :param ob:
        :param context:
        :return:
        """
        pref = cls.pref_()
        handler_dit = cls.G_SimpleDeformGizmoHandlerDit
        active = context.object.modifiers.active
        # draw deform mesh
        # TODO draw deform
        # if 'draw' in handler_dit:
        #     pos, indices, mat, mod_data, limits = handler_dit['draw']
        #     if ([getattr(active, i) for i in G_MODIFIERS_PROPERTY] == mod_data) and (
        #             ob.matrix_world == mat) and limits == active.limits[:]:
        #         bgl.glEnable(bgl.GL_DEPTH_TEST)
        #         cls.draw_3d_shader(
        #             pos, indices, pref.deform_wireframe_color)

    @classmethod
    def is_draw_box(cls, context):
        """绘制框"""
        from .utils import Utils
        obj = context.object  # 活动物体
        matrix = obj.matrix_world  # 活动物体矩阵
        modifier = context.object.modifiers.active  # 活动修改器

        pref = cls.pref_()
        simple_poll = Utils.simple_deform_poll(context)
        bend = modifier and (modifier.deform_method == 'BEND')
        display_switch_axis = not pref.display_bend_axis_switch_gizmo

        cls.draw_scale_text(obj)
        Utils.update_co_data(obj, modifier)

        co_data = Utils.generate_co_data()

        if simple_poll and ((not bend) or display_switch_axis):
            # draw bound box
            cls.draw_box(co_data, matrix)
            cls.draw_deform_mesh(obj, context)
            cls.draw_limits_line()
            cls.draw_limits_bound_box()
        elif simple_poll and (bend and not display_switch_axis):
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            cls.draw_box(co_data, matrix)
            Utils.new_empty(obj, modifier)

    @classmethod
    def draw_bound_box(cls):

        # return 
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1)
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glEnable(bgl.GL_ALPHA)
        bgl.glDisable(bgl.GL_DEPTH_TEST)

        context = bpy.context
        from .utils import Utils
        if Utils.simple_deform_poll(context):
            cls.is_draw_box(context)
        else:
            Handler.del_handler()
