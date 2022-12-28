import math
import uuid

import bpy
import numpy as np
from bpy.types import AddonPreferences
from mathutils import Vector, Matrix

from .data import G_ADDON_NAME, G_NAME, G_INDICES, G_MODIFIERS_PROPERTY, G_CON_LIMIT_NAME, G_GizmoCustomShapeDict
from .draw import Draw3D


class NotUse:

    @classmethod
    def properties_is_modifier(cls) -> bool:
        """
        反回活动窗口内是否有修改器属性面板被打开,如果打开则反回True else False
        """
        for area in bpy.context.screen.areas:
            if area.type == 'PROPERTIES':
                for space in area.spaces:
                    if space.type == 'PROPERTIES' and space.context == 'MODIFIER':
                        return True
        return False


class Pref:
    @staticmethod
    def pref_() -> "AddonPreferences":
        return bpy.context.preferences.addons[G_ADDON_NAME].preferences

    @property
    def pref(self=None) -> 'AddonPreferences':
        """
        :return: AddonPreferences
        """
        return Pref.pref_()


class CustomShape(Pref, Draw3D):
    custom_shape = {}

    @classmethod
    def load_custom_shape_gizmo(cls):
        from bpy.types import Gizmo
        for key, value in G_GizmoCustomShapeDict.items():
            if key not in cls.custom_shape:
                cls.custom_shape[key] = Gizmo.new_custom_shape('TRIS', value)


class Property(CustomShape):
    mouse_dpi = 10
    init_mouse_x: float
    init_mouse_y: float
    event: "bpy.types.Event"
    tweak = None
    context: "bpy.types.Context"
    control_mode: "str"  # "up_limits"  # up_limits , down_limits,angle

    def init_event(self, event: "bpy.types.Event"):
        self.init_mouse_y = event.mouse_y
        self.init_mouse_x = event.mouse_x

    def init_modal_data(self, context, event, tweak):
        self.tweak = tweak
        self.init_invoke(context, event)

    def init_invoke(self, context, event):
        self.event = event
        self.context = context

    @classmethod
    def get_origin_property_group(cls, mod, ob):
        if mod.origin:
            return mod.origin.SimpleDeformGizmo_PropertyGroup
        else:
            return ob.SimpleDeformGizmo_PropertyGroup

    @classmethod
    def get_vector_axis(cls, mod: "bpy.types.Modifer"):
        """获取矢量轴

        :param mod:
        :return:
        """
        axis = mod.deform_axis
        x = Vector((1, 0, 0))
        y = Vector((0, 1, 0))
        z = Vector((0, 0, 1))
        if 'BEND' == mod.deform_method:
            vector_axis = z if axis in ('Y', 'X') else x
        else:
            vector = x if (axis == 'X') else y
            vector_axis = z if (axis == 'Z') else vector
        return vector_axis

    # PROPERTY
    @property
    def simple_modifier(self) -> "bpy.types.Modifier":
        """反回活动物体的简易形变修改器"""
        return get_active_simple_modifier()

    @property
    def simple_modifier_down_limits_value(self) -> "float":
        return self._limits[0]

    @property
    def simple_modifier_up_limits_value(self) -> "float":
        return self._limits[1]

    @property
    def simple_modifier_angle_value(self) -> "float":
        return get_active_simple_modifier().angle

    @property
    def _limits(self) -> "list[float]":
        return self.simple_modifier.limits

    @property
    def limits_scope(self) -> "float":
        return Pref.pref_().modifiers_limits_tolerance

    @property
    def is_middle_mode(self) -> bool:
        """反回控制模式是中间的布尔值
        :return: bool
        """
        return self.origin_mode in ("MIDDLE", "LIMITS_MIDDLE")

    @property
    def limits_difference(self) -> "float":
        return self.simple_modifier_up_limits_value - self.simple_modifier_down_limits_value

    @property
    def limits_max_value(self) -> "float":
        return self.simple_modifier_up_limits_value - self.limits_scope

    @property
    def limits_min_value(self) -> "float":
        return self.simple_modifier_down_limits_value + self.limits_scope

    @property
    def limits_middle(self) -> "float":
        return (self.simple_modifier_up_limits_value + self.simple_modifier_down_limits_value) / 2

    @property
    def object_property(self) -> "preferences.SimpleDeformGizmoObjectPropertyGroup":
        """反回物体的插件属性
        :return:
        """
        return self.get_origin_property_group(self.simple_modifier, self.object)

    @property
    def origin_mode(self) -> "str":
        """反回物体的原点模式,
        :return:
        """
        return self.object_property.origin_mode

    @property
    def simple_modifier_deform_axis(self) -> "str":
        """反回形变修改器的轴
        :return:
        """
        return self.simple_modifier.deform_axis

    @property
    def object(self) -> "bpy.types.Object":
        return bpy.context.object

    @property
    def up_point(self) -> "Vector":
        return self.simple_modifier_point_co[1]

    @property
    def down_point(self) -> "Vector":
        return self.simple_modifier_point_co[0]

    @property
    def up_limits(self) -> "Vector":
        return self.simple_modifier_limits_co[1]

    @property
    def down_limits(self) -> "Vector":
        return self.simple_modifier_limits_co[0]


class Calculation(Property):

    @classmethod
    def set_reduce(cls, list_a, list_b, operation_type='-') -> list:
        """
        :param list_a: 列表a
        :type list_a: list or tuple
        :param list_b: 列表b
        :type list_b:list or tuple
        :param operation_type :运算方法Enumerator in ['+','-','*','/'].
        :type operation_type :str
        :return list: 反回运算后的列表
        """
        if operation_type == '-':
            return [list_a[i] - list_b[i] for i in range(0, len(list_a))]
        elif operation_type == '+':
            return [list_a[i] + list_b[i] for i in range(0, len(list_a))]
        elif operation_type == '/':
            return [list_a[i] / list_b[i] for i in range(0, len(list_a))]
        elif operation_type == '*':
            return [list_a[i] * list_b[i] for i in range(0, len(list_a))]

    @classmethod
    def bound_box_to_list(cls, obj: 'bpy.context.object') -> tuple:
        """
        :param obj:输入一个物体,反回物体的边界框列表
        :type obj:bpy.types.Object
        :return tuple:
        """
        return tuple(i[:] for i in obj.bound_box)

    @classmethod
    def point_to_angle(cls, a, b, c, axis):
        """仨个点转换为角度

        :param a:
        :param b:
        :param c:
        :param axis:
        :return:
        """
        if a == b:
            if c == 0:
                a[0] += 0.1
                b[0] -= 0.1
            elif c == 1:
                a[1] -= 0.1
                b[1] += 0.1
            else:
                a[2] -= 0.1
                b[2] += 0.1
        vector_value = a - b
        angle = (180 * vector_value.angle(axis) / math.pi)
        return angle

    @classmethod
    def co_to_direction(cls, mat, data):
        """坐标到方向

        :param mat:
        :param data:
        :return:
        """
        (min_x, min_y, min_z), (max_x, max_y,
                                max_z) = data
        a = mat @ Vector((max_x, max_y, max_z))
        b = mat @ Vector((max_x, min_y, min_z))
        c = mat @ Vector((min_x, max_y, min_z))
        d = mat @ Vector((min_x, min_y, max_z))

        def pos_get(a, b):
            return cls.set_reduce(cls.set_reduce(a, b, '+'), [2, 2, 2], '/')

        top = Vector(pos_get(a, d))
        bottom = Vector(pos_get(c, b))
        left = Vector(pos_get(c, d))
        right = Vector(pos_get(a, b))
        front = Vector(pos_get(d, b))
        back = Vector(pos_get(c, a))
        return top, bottom, left, right, front, back

    @classmethod
    def each_face_pos(cls, mat: 'Matrix' = None):
        """获取每个面的点,用作选择轴的点

        :param mat:
        :return:
        """
        if mat is None:
            mat = Matrix()
        return cls.co_to_direction(mat, cls.object_max_min_co)

    @classmethod
    def get_up_down_return_list(cls, mod, axis, up_, down_, data):
        top, bottom, left, right, front, back = data
        if 'BEND' == mod.deform_method:
            if axis in ('X', 'Y'):
                top = up_
                bottom = down_
            elif axis == 'Z':
                right = up_
                left = down_
        else:
            if axis == 'X':
                right = up_
                left = down_
            elif axis == 'Y':
                back = up_
                front = down_
            elif axis == 'Z':
                top = up_
                bottom = down_
        return top, bottom, left, right, front, back


class UpdateAndGetData(Calculation):
    tmp_save_data = None

    @property
    def need_update(self) -> bool:
        """存储物体的信息,如果更改了物体或者更改了参数就更新
        反回是否需要更新的布尔值
        :return:
        """

        modifier_property = [getattr(self.simple_modifier, i)
                             for i in G_MODIFIERS_PROPERTY]
        tk = (self.object, self.simple_modifier, modifier_property)
        if self.tmp_save_data != tk:
            self.tmp_save_data = tk
            return True
        return False

    @classmethod
    def get_origin_bounds(cls, obj: 'bpy.context.object') -> list:
        modifiers_list = {}
        for mod in obj.modifiers:
            if (mod == obj.modifiers.active) or (modifiers_list != {}):
                modifiers_list[mod] = (mod.show_render, mod.show_viewport)
                mod.show_viewport = False
                mod.show_render = False
        matrix_obj = obj.matrix_world.copy()
        obj.matrix_world.zero()
        obj.scale = (1, 1, 1)
        bound = cls.bound_box_to_list(obj)
        obj.matrix_world = matrix_obj
        for mod in modifiers_list:
            show_render, show_viewport = modifiers_list[mod]
            mod.show_render = show_render
            mod.show_viewport = show_viewport
        return list(bound)

    @classmethod
    def get_mesh_max_min_co(cls, obj: 'bpy.context.object') -> tuple:
        """获取网格的最大最小坐标

        :param obj:
        :return:
        """
        obj
        if obj.type == 'MESH':
            ver_len = obj.data.vertices.__len__()
            list_vertices = np.zeros(ver_len * 3, dtype=np.float32)
            obj.data.vertices.foreach_get('co', list_vertices)
            list_vertices = list_vertices.reshape(ver_len, 3)
        elif obj.type == 'LATTICE':
            ver_len = obj.data.points.__len__()
            list_vertices = np.zeros(ver_len * 3, dtype=np.float32)
            obj.data.points.foreach_get('co', list_vertices)
            list_vertices = list_vertices.reshape(ver_len, 3)
        return [Vector(list_vertices.min(axis=0)), Vector(list_vertices.max(axis=0))]

    @classmethod
    def get_up_down(cls, mod, axis, top, bottom, left, right, front, back):
        """获取向上轴和向下轴

        :param mod:
        :param axis:
        :param top:
        :param bottom:
        :param left:
        :param right:
        :param front:
        :param back:
        :return:
        """
        if 'BEND' == mod.deform_method:
            if axis in ('X', 'Y'):
                return top, bottom
            elif axis == 'Z':
                return right, left
        else:
            if axis == 'X':
                return right, left
            elif axis == 'Y':
                return back, front
            elif axis == 'Z':
                return top, bottom

    @classmethod
    def from_simple_modifiers_get_limits_pos(cls, mod, data):
        """从简易形变修改器获取限制点

        :param mod:
        :param data:
        :return:
        """
        top, bottom, left, right, front, back = data
        up_limits = mod.limits[1]
        down_limits = mod.limits[0]
        axis = mod.deform_axis

        up: "Vector"
        down: "Vector"

        if mod.origin:
            vector_axis = cls.get_vector_axis(mod)
            origin_mat = mod.origin.matrix_world.to_3x3()
            axis_ = origin_mat @ vector_axis
            point_lit = [[top, bottom], [left, right], [front, back]]

            for f in range(point_lit.__len__()):
                i = point_lit[f][0]
                j = point_lit[f][1]
                angle = cls.point_to_angle(i, j, f, axis_)
                if abs(angle - 180) < 0.1:
                    up, down = j, i
                elif abs(angle) < 0.1:
                    up, down = i, j
        else:
            up, down = cls.get_up_down(mod, axis, top, bottom,
                                       left, right, front, back)

        print("get\t\t", up, down)
        e = lambda a: Vector((cls.set_reduce(down, cls.set_reduce(cls.set_reduce(
            up, down, '-'), [a, a, a], '*'), '+')))

        up_ = e(up_limits)
        down_ = e(down_limits)
        return (up, down), (up_, down_)

    @classmethod
    def update_bound_box(cls, obj):
        """更新边界框和形变框

        :param obj:
        :return:
        """
        # print("更新前", cls.object_max_min_co, cls)
        cls.object_max_min_co[:] = cls.get_mesh_max_min_co(obj)
        # print("更新后", cls.object_max_min_co, cls)

    def update_deform_wireframe(self, ):
        """更新形变框
        只能在模态内更新
        """
        context = bpy.context
        data = bpy.data
        matrix = self.object.matrix_world.copy()  # 物体矩阵
        vertexes = self.co_to_bound(self.object_max_min_co)
        if data.objects.get(G_NAME):
            data.objects.remove(data.objects.get(G_NAME))

        if data.meshes.get(G_NAME):
            data.meshes.remove(data.meshes.get(G_NAME))
        mesh = data.meshes.new(G_NAME)
        mesh.from_pydata(vertexes, G_INDICES, [])
        mesh.update()

        new_object = data.objects.new(G_NAME, mesh)

        self.link_active_collection(new_object)

        if new_object.parent != self.object:
            new_object.parent = self.object

        new_object.modifiers.clear()
        subdivision = new_object.modifiers.new('1', 'SUBSURF')
        subdivision.levels = 7
        for mo in context.object.modifiers:
            if mo.type == 'SIMPLE_DEFORM':
                simple_deform = new_object.modifiers.new(
                    mo.name, 'SIMPLE_DEFORM')
                simple_deform.deform_method = mo.deform_method
                simple_deform.deform_axis = mo.deform_axis
                simple_deform.lock_x = mo.lock_x
                simple_deform.lock_y = mo.lock_y
                simple_deform.lock_z = mo.lock_z
                simple_deform.origin = mo.origin
                simple_deform.limits[1] = mo.limits[1]
                simple_deform.limits[0] = mo.limits[0]
                simple_deform.angle = mo.angle
                simple_deform.show_viewport = mo.show_viewport
                obj = self.get_depsgraph(new_object)

        new_object.hide_set(True)
        new_object.hide_viewport = False
        new_object.hide_select = True
        new_object.hide_render = True
        new_object.hide_viewport = True
        new_object.hide_set(True)
        ver_len = obj.data.vertices.__len__()
        edge_len = obj.data.edges.__len__()

        key = (ver_len, edge_len)
        list_edges = np.zeros(edge_len * 2, dtype=np.int32)
        list_vertices = np.zeros(ver_len * 3, dtype=np.float32)
        if key in self.numpy_data:
            list_edges, list_vertices = self.numpy_data[key]
        else:
            self.numpy_data[key] = (list_edges, list_vertices)

        obj.data.vertices.foreach_get('co', list_vertices)
        ver = np.insert(list_vertices.reshape((ver_len, 3)), 3, 1, axis=1).T
        ver[:] = np.dot(matrix, ver)

        ver /= ver[3, :]
        ver = ver.T[:, :3]
        obj.data.edges.foreach_get('vertices', list_edges)
        indices = list_edges.reshape((edge_len, 2))

        limits = obj.modifiers.active.limits[:]
        modifier_property = [getattr(context.object.modifiers.active, i)
                             for i in G_MODIFIERS_PROPERTY]
        self.deform_bound_draw_data[:] = ver, indices, limits, modifier_property, matrix

    def update_limits_and_bound(self):
        """更新上下限边界框
        """
        modifier = self.simple_modifier
        axis = self.simple_modifier_deform_axis

        top, bottom, left, right, front, back = self.each_face_pos()
        print(self.each_face_pos())

        # (up_point, down_point), (up_limits, down_limits) = self.from_simple_modifiers_get_limits_pos(modifier, (
        #     top, bottom, left, right, front, back))

        # self.simple_modifier_limits_co[:] = down_limits, up_limits
        # self.simple_modifier_point_co[:] = down_point, up_point

        if modifier.origin:
            vector_axis = self.get_vector_axis(modifier)
            origin_mat = modifier.origin.matrix_world.to_3x3()
            axis_ = origin_mat @ vector_axis
        #     point_list = [[top, bottom], [left, right], [front, back]]
        #     for f in range(point_list.__len__()):
        #         i = point_list[f][0]
        #         j = point_list[f][1]
        #         angle = self.point_to_angle(i, j, f, axis_)
        #         if abs(angle - 180) < 0.00001:
        #             point_list[f][1], point_list[f][0] = up_limits, down_limits
        #         elif abs(angle) < 0.00001:
        #             point_list[f][0], point_list[f][1] = up_limits, down_limits
        #     [[top, bottom], [left, right], [front, back]] = point_list
        # else:
        #     top, bottom, left, right, front, back = self.get_up_down_return_list(
        #         modifier, axis, up_limits, down_limits, data)
        #
        # self.simple_modifier_limits_bound[:] = (right[0], back[1], top[2]), (left[0], front[1], bottom[2],)


class Empty(UpdateAndGetData):
    empty_object: "str" = ""

    def update_empty(self):
        if self.origin_mode != 'NOT':
            obj, _ = self.empty_new(self.object, self.simple_modifier)
            self.empty_object = obj.name
            self.set_empty_obj_matrix()
        elif self.simple_modifier.origin:  # 不对原点进行操作但是还但在原点物体
            ...

    def set_empty_obj_matrix(self):
        tow = [2] * 3

        empty_object = bpy.context.scene.objects.get(self.empty_object, None)
        if not empty_object:
            return

        matrix = self.object.matrix_world
        down_point, up_point = self.down_point, self.up_point
        down_limits, up_limits = self.simple_modifier_limits_co

        origin_mode = self.origin_mode
        if origin_mode == 'UP_LIMITS':
            empty_object.matrix_world.translation = matrix @ up_limits
        elif origin_mode == 'DOWN_LIMITS':
            empty_object.matrix_world.translation = matrix @ down_limits
        elif origin_mode == 'LIMITS_MIDDLE':
            empty_object.matrix_world.translation = self.set_reduce(
                self.set_reduce(matrix @ up_limits, matrix @ down_limits, '+'), tow, '/')
        elif origin_mode == 'MIDDLE':
            empty_object.matrix_world.translation = self.set_reduce(
                self.set_reduce(matrix @ up_point, matrix @ down_point, '+'), tow, '/')

    @classmethod
    def empty_new(cls, obj, mod):
        """新建空物体作为轴来使用
        :param obj:
        :param mod:
        :return:
        """
        origin = mod.origin
        if origin is None:
            new_name = G_NAME + '_Empty_' + str(uuid.uuid4())
            origin_object = bpy.data.objects.new(new_name, None)
            cls.link_active_collection(origin_object)
            origin_object.hide_set(True)
            origin_object.empty_display_size = min(obj.dimensions)
            mod.origin = origin_object
        else:
            origin_object = mod.origin
            origin_object.hide_viewport = False

        if origin_object.parent != obj:
            origin_object.parent = obj

        # add constraints
        if G_CON_LIMIT_NAME in origin_object.constraints.keys():
            limit_constraints = origin.constraints.get(G_CON_LIMIT_NAME)
        else:
            limit_constraints = origin_object.constraints.new(
                'LIMIT_ROTATION')
            limit_constraints.name = G_CON_LIMIT_NAME
            limit_constraints.owner_space = 'WORLD'
            limit_constraints.space_object = obj
        limit_constraints.use_transform_limit = True
        limit_constraints.use_limit_x = True
        limit_constraints.use_limit_y = True
        limit_constraints.use_limit_z = True
        con_copy_name = G_NAME + 'constraints_copy_rotation'
        if con_copy_name in origin_object.constraints.keys():
            copy_constraints = origin.constraints.get(con_copy_name)
        else:
            copy_constraints = origin_object.constraints.new(
                'COPY_ROTATION')
            copy_constraints.name = con_copy_name
        copy_constraints.target = obj
        copy_constraints.mix_mode = 'BEFORE'
        copy_constraints.target_space = 'WORLD'
        copy_constraints.owner_space = 'WORLD'
        origin_object.rotation_euler.zero()
        origin_object.scale = 1, 1, 1
        return origin_object, G_CON_LIMIT_NAME

    def empty_remove(self):
        ...


class GizmoUtils(Empty):

    @classmethod
    def value_limit(cls, value, max_value=1, min_value=0) -> float:
        """
        :param value: 输入值
        :type value: float
        :param max_value: 允许的最大值
        :type max_value: float
        :param min_value: 允许的最小值
        :type min_value: float
        :return float: 反回小于最大值及大于最小值的浮点数
        """
        if value > max_value:
            return max_value
        elif value < min_value:
            return min_value
        else:
            return value

    @classmethod
    def is_positive(cls, number: 'int') -> bool:
        """return bool value
        if number is positive return True else return False
        """
        return number == abs(number)

    @classmethod
    def get_depsgraph(cls, obj: 'bpy.context.object'):
        """
        :param obj: 要被评估的物体
        :type obj: bpy.types.Object
        :return bpy.types.Object: 反回评估后的物体,计算应用修改器和实例化的数据
        如果未输入物休将会评估活动物体
        """
        context = bpy.context
        if obj is None:
            obj = context.object
        depsgraph = context.evaluated_depsgraph_get()
        return obj.evaluated_get(depsgraph)

    @classmethod
    def link_active_collection(cls,
                               obj: 'bpy.context.object') -> \
            'bpy.context.view_layer.active_layer_collection.collection.objects':
        context = bpy.context
        if obj.name not in context.view_layer.active_layer_collection.collection.objects:
            context.view_layer.active_layer_collection.collection.objects.link(
                obj)
        return context.view_layer.active_layer_collection.collection.objects

    @classmethod
    def simple_deform_poll(cls, context: 'bpy.context') -> bool:
        """
        :param context:输入一个上下文
        :type context:bpy.context
        :return bool:反回布尔值,如果活动物体为网格或晶格并且活动修改器为简易形变反回 True else False
        """
        obj = context.object
        mesh = (obj.type in ('MESH', 'LATTICE')) if obj else False
        modifiers_type = (obj.modifiers.active.type ==
                          'SIMPLE_DEFORM') if (obj and (obj.modifiers.active is not None)) else False
        obj_ok = context and obj and modifiers_type and mesh
        module_ok = (context.mode == 'OBJECT')
        view = context.space_data
        show_gizmo = view.show_gizmo
        return obj_ok and module_ok and show_gizmo

    @classmethod
    def matrix_calculation(cls, mat: 'Matrix', calculation_list: 'iter') -> list:
        return [mat @ Vector(i) for i in calculation_list]

    def event_ops(self):
        """通过输入键位来更改属性"""
        # event ctrl
        data_path = ('object.SimpleDeformGizmo_PropertyGroup.origin_mode',
                     'object.modifiers.active.origin.SimpleDeformGizmo_PropertyGroup.origin_mode')

        event = self.event

        if event.type in ('WHEELUPMOUSE', 'WHEELDOWNMOUSE'):
            reverse = (event.type == 'WHEELUPMOUSE')
            for path in data_path:
                bpy.ops.wm.context_cycle_enum(
                    data_path=path, reverse=reverse, wrap=True)
        elif event.type in ('X', 'Y', 'Z'):
            self.simple_modifier.deform_axis = event.type
        elif event.type == 'A':
            self.pref.display_bend_axis_switch_gizmo = True
            return {'FINISHED'}
        self.add_handler()

        return {'RUNNING_MODAL'}


def get_active_simple_modifier() -> "bpy.types.Modifier":
    """获取活动简易形变修改器
    :return:
    """
    obj = bpy.context.object
    if obj.modifiers and obj.modifiers.active.type == 'SIMPLE_DEFORM':
        return obj.modifiers.active


def register():
    ...


def unregister():
    ...
