import bgl
import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .gizmo import Handler

class Draw3D:

    @classmethod
    def draw_3d_shader(cls, pos, indices, color=None, *, shader_name='3D_POLYLINE_UNIFORM_COLOR', draw_type='LINES',
                       enable_depth=None):
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
            font_id, f'物体{obj.name_full}缩放值不为1,将会导致简易形变修改器变形,请应用缩放后再进行形变操作')
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
        pref = Data.pref.fget(None)
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) = data
        coords = Utils.matrix_calculation(mat,
                                          (
                                              (max_x, min_y, min_z),
                                              (min_x, min_y, min_z),
                                              (max_x, max_y, min_z),
                                              (min_x, max_y, min_z),
                                              (max_x, min_y, max_z),
                                              (min_x, min_y, max_z),
                                              (max_x, max_y, max_z),
                                              (min_x, max_y, max_z))
                                          )
        cls.draw_3d_shader(coords, cls.G_Indices, pref.bound_box_color)

    @classmethod
    def draw_limits_bound_box(cls):
        pref = Data.pref.fget()
        handler_dit = Data.G_SimpleDeformGizmoHandlerDit
        if 'draw_limits_bound_box' in handler_dit:
            # draw limits_bound_box
            mat, ((x, y, z), (_x, _y, _z)
                  ) = handler_dit['draw_limits_bound_box']
            bgl.glEnable(bgl.GL_DEPTH_TEST)
            coords = Utils.matrix_calculation(mat, ((x, _y, _z),
                                                    (_x, _y, _z),
                                                    (x, y, _z),
                                                    (_x, y, _z),
                                                    (x, _y, z),
                                                    (_x, _y, z),
                                                    (x, y, z),
                                                    (_x, y, z),))
            cls.draw_3d_shader(coords, cls.G_Indices,
                               pref.limits_bound_box_color)

    @classmethod
    def draw_limits_line(cls):
        handler_dit = Data.G_SimpleDeformGizmoHandlerDit
        if 'draw_line' in handler_dit:
            line_pos, limits_pos, = handler_dit['draw_line']
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
        prefs = Data.prefs.fget()
        handler_dit = Data.G_SimpleDeformGizmoHandlerDit
        active = context.object.modifiers.active
        # draw deform mesh
        if ('draw', ob) in handler_dit:
            pos, indices, mat, mod_data, limits = handler_dit[(
                'draw', ob)]
            if ([getattr(active, i) for i in modifiers_data] == mod_data) and (
                    ob.matrix_world == mat) and limits == active.limits[:]:
                bgl.glEnable(bgl.GL_DEPTH_TEST)
                cls.draw_3d_shader(
                    pos, indices, prefs.deform_wireframe_color)

    @classmethod
    def draw_scale_text(cls, ob):
        handler_dit = Data.G_SimpleDeformGizmoHandlerDit
        if (ob.scale != Vector((1, 1, 1))) and ('handler_text' not in handler_dit):
            handler_dit['handler_text'] = bpy.types.SpaceView3D.draw_handler_add(
                cls.draw_str, (), 'WINDOW', 'POST_PIXEL')

    @classmethod
    def is_draw_box(cls, context):
        obj = context.object  # 活动物体
        matrix = obj.matrix_world  # 活动物体矩阵
        modifier = context.object.modifiers.active  # 活动修改器

        prefs = Data.prefs.fget()
        simple_poll = Utils.simple_deform_poll(context)
        bend = modifier and (modifier.deform_method == 'BEND')
        display_switch_axis: bool = (prefs.display_bend_axis_switch_gizmo == False)

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
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1)
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glEnable(bgl.GL_ALPHA)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        context = bpy.context

        if Utils.simple_deform_poll(context):
            cls.is_draw_box(context)
        else:
            Handler.del_handler()

