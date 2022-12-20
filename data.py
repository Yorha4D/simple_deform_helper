from os.path import dirname, basename, realpath

G_MODIFIERS_PROPERTY = [  # copy modifier data
    'angle',
    'deform_axis',
    'deform_method',
    'factor',
    'invert_vertex_group',
    'limits',
    'lock_x',
    'lock_y',
    'lock_z',
    'origin',
    'show_expanded',
    'show_in_editmode',
    'vertex_group',
]

G_INDICES = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7))

G_NAME = 'ViewSimpleDeformGizmo_'  # Temporary use files
G_CON_LIMIT_NAME = G_NAME + 'constraints_limit_rotation'  # 约束名称
G_ADDON_NAME = basename(dirname(realpath(__file__)))  # "simple_deform_helper"

G_ORIGIN_MODE_ITEMS = (
    ('UP_LIMITS',
     'Follow Upper Limit(Red)',
     'Add an empty object origin as the rotation axis (if there is an origin, do not add it), and set the origin '
     'position as the upper limit during operation'),
    ('DOWN_LIMITS',
     'Follow Lower Limit(Green)',
     'Add an empty object origin as the rotation axis (if there is an origin, do not add it), and set the origin '
     'position as the lower limit during operation'),
    ('LIMITS_MIDDLE',
     'Middle',
     'Add an empty object origin as the rotation axis (if there is an origin, do not add it), and set the '
     'origin position between the upper and lower limits during operation'),
    ('MIDDLE',
     'Bound Middle',
     'Add an empty object origin as the rotation axis (if there is an origin, do not add it), and set the origin '
     'position as the position between the bounding boxes during operation'),
    ('NOT', 'No origin operation', ''),
)


def load_gizmo_data() -> dict:
    """加载gizmo数据从.json里面
    :return:
    """
    import json
    import os
    json_path = os.path.join(os.path.dirname(__file__), "gizmo/shape.json")
    with open(json_path, "r") as file:
        return json.load(file)


def from_bmesh_get_triangle_face_co(mesh: 'bpy.types.Mesh') -> list:
    """
    :param mesh: 输入一个网格数据
    :type mesh: bpy.data.meshes
    :return list: 反回顶点列表[[co1,co2,co3],[co1,co2,co3]...]
    """
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    bmesh.ops.triangulate(bm, faces=bm.faces)
    co_list = [list(float(format(j, ".4f")) for j in vert.co) for face in bm.faces for vert in face.verts]
    bm.free()
    return co_list


G_GizmoCustomShapeDict = load_gizmo_data()
