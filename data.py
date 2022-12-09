modifiers_data = [  # 修改器复制数据
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

G_Indices = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7))

G_GizmoCustomShapeDict = {}
G_SimpleDeformGizmoHandlerDit = {}
addon_name = "simple_deform_helper"

G_NAME = 'ViewSimpleDeformGizmo_'  # Temporary use files
CON_LIMIT_NAME = G_NAME + 'constraints_limit_rotation'  # 约束名称
