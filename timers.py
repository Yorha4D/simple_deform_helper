import bpy
from bpy.app import timers
from bpy.app.handlers import undo_post, persistent, undo_pre


def remove_not_use_empty(remove_name: str = "ViewSimpleDeformGizmo__Empty_"):
    """循环场景内的所有物体,找出没用的空物体并删掉
    """
    context = bpy.context
    for obj in context.scene.objects:
        is_empty = obj.type == "EMPTY"
        not_parent = not obj.parent
        name_ok = obj.name.count(remove_name)
        if name_ok and not_parent and is_empty:
            bpy.data.objects.remove(obj)  # remove object


@persistent
def clear_data_pre(scene):
    print('clear_data_pre', scene)


@persistent
def clear_data_post(scene):
    print('clear_data_post', scene)


def update_timers() -> float:
    remove_not_use_empty()
    return 3.0


def register():
    timers.register(update_timers, persistent=True)
    # undo_pre.append(clear_data_pre)
    # undo_post.append(clear_data_post)


def unregister():
    if timers.is_registered(update_timers):
        timers.unregister(update_timers)
    # undo_pre.remove(clear_data_pre)
    # undo_post.remove(clear_data_post)
