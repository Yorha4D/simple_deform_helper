import bpy
import numpy as np
from bpy.types import Operator
from bpy.props import FloatProperty, StringProperty, BoolProperty

from .data import G_MODIFIERS_PROPERTY, G_MODIFIERS_COPY_PROPERTY, G_NAME, G_INDICES
from .utils import GizmoUtils


class DeformAxisOperator(Operator, GizmoUtils):
    bl_idname = 'simple_deform_gizmo.deform_axis'
    bl_label = 'deform_axis'
    bl_description = 'deform_axis operator'
    bl_options = {'REGISTER'}

    Deform_Axis: StringProperty(default='X', options={'SKIP_SAVE'})

    X_Value: FloatProperty(default=-0, options={'SKIP_SAVE'})
    Y_Value: FloatProperty(default=-0, options={'SKIP_SAVE'})
    Z_Value: FloatProperty(default=-0, options={'SKIP_SAVE'})

    Is_Positive: BoolProperty(default=True, options={'SKIP_SAVE'})

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        from .utils import GizmoUtils

        mod = context.object.modifiers.active
        mod.deform_axis = self.Deform_Axis
        empty, con_limit_name = GizmoUtils.empty_new(context.object, mod)
        is_positive = GizmoUtils.is_positive(mod.angle)

        for limit, value in (('max_x', self.X_Value),
                             ('min_x', self.X_Value),
                             ('max_y', self.Y_Value),
                             ('min_y', self.Y_Value),
                             ('max_z', self.Z_Value),
                             ('min_z', self.Z_Value),
                             ):
            setattr(empty.constraints[con_limit_name], limit, value)

        if ((not is_positive) and self.Is_Positive) or (is_positive and (not self.Is_Positive)):
            mod.angle = mod.angle * -1

        if not event.ctrl:
            self.pref.display_bend_axis_switch_gizmo = False
        return {'FINISHED'}


class UpdateDeformWireframe(Operator, GizmoUtils):
    bl_idname = 'simple_deform_gizmo.update_deform_wireframe'
    bl_label = 'deform_axis'
    bl_description = 'deform_axis operator'
    bl_options = {'REGISTER'}

    change_co: BoolProperty()

    def execute(self, context: 'Context'):
        data = bpy.data
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

        modifiers = self.simple_modifiers  # 此物体的所有简易形变修改器
        mod_len = len(modifiers)  # 所有简易形变修改器长度
        active_index = modifiers.index(self.simple_modifier)  # 活动简易修改器索引

        def add_mod(mod):
            simple_deform = new_object.modifiers.new(mod.name, 'SIMPLE_DEFORM')

            for prop_name in G_MODIFIERS_COPY_PROPERTY:
                setattr(simple_deform, prop_name, getattr(mod, prop_name))

            simple_deform.limits[1] = mod.limits[1]
            simple_deform.limits[0] = mod.limits[0]

        for index, mo in enumerate(modifiers):
            if self.change_co:
                add_mod(mo)
                if index + 1 == active_index:
                    obj = self.get_depsgraph(new_object)
                    self.object_max_min_co[:] = self.get_mesh_max_min_co(self.get_depsgraph(obj))
                elif index + 1 == mod_len:
                    obj = self.get_depsgraph(new_object)

            elif index == active_index:
                add_mod(mo)
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
        matrix = obj.matrix_world.copy()  # 物体矩阵
        ver[:] = np.dot(matrix, ver)

        ver /= ver[3, :]
        ver = ver.T[:, :3]
        obj.data.edges.foreach_get('vertices', list_edges)
        indices = list_edges.reshape((edge_len, 2))

        limits = obj.modifiers.active.limits[:]
        modifier_property = [getattr(context.object.modifiers.active, i)
                             for i in G_MODIFIERS_PROPERTY]
        self.__class__.deform_bound_draw_data[:] = ver, indices, limits, modifier_property, matrix
        return {"FINISHED"}


class_list = (
    DeformAxisOperator,
    UpdateDeformWireframe,
)

register_class, unregister_class = bpy.utils.register_classes_factory(class_list)


def register():
    register_class()


def unregister():
    unregister_class()
