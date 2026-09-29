import bpy
import numpy as np
import os

character = "michelle"
static_dir = rf"E:\didaktoriko\diffusion_solution\dataset\{character}\static_data"

armature = bpy.context.active_object
if armature is None or armature.type != 'ARMATURE':
    raise RuntimeError("Select the character's ARMATURE as the active object first.")

os.makedirs(static_dir, exist_ok=True)
print(f"Exporting '{armature.name}' -> {static_dir}")


def _is_identity(matrix, tol=1e-4):
    identity = matrix.__class__.Identity(4)
    diff = matrix - identity
    return max(abs(x) for row in diff for x in row) < tol


def get_character_meshes(armature):
    meshes = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE' and mod.object == armature:
                    meshes.append(obj)
                    break
    return meshes


def get_bone_map(armature):
    bones = [b.name for b in armature.data.bones]
    bone_map = {name: i for i, name in enumerate(bones)}
    return bones, bone_map


def get_rest_pose_mesh(meshes, armature):
    original_pose_position = armature.data.pose_position
    armature.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    try:
        verts_all = []
        depsgraph = bpy.context.evaluated_depsgraph_get()

        for mesh_obj in meshes:
            assert _is_identity(mesh_obj.matrix_world), (
                f"'{mesh_obj.name}' has a non-identity matrix_world -- this export assumes "
                f"local-space vertices already equal world-space ones. Apply this object's "
                f"transform first (Object -> Apply -> All Transforms), or this rest_pose.npy "
                f"will not match the mesh's real world-space position."
            )
            eval_obj = mesh_obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            verts = np.array([v.co[:] for v in eval_mesh.vertices], dtype=np.float32)
            verts_all.append(verts)
            eval_obj.to_mesh_clear()

        return np.concatenate(verts_all, axis=0)
    finally:
        armature.data.pose_position = original_pose_position
        bpy.context.view_layer.update()


def export_skinning_weights(meshes, bone_map):
    weights_all = []

    for mesh_obj in meshes:
        V = len(mesh_obj.data.vertices)
        B = len(bone_map)
        W = np.zeros((V, B), dtype=np.float32)

        for v in mesh_obj.data.vertices:
            for g in v.groups:
                bone_name = mesh_obj.vertex_groups[g.group].name
                if bone_name in bone_map:
                    W[v.index, bone_map[bone_name]] = g.weight

        row_sums = W.sum(axis=1, keepdims=True) + 1e-8
        W /= row_sums
        weights_all.append(W)

    return np.concatenate(weights_all, axis=0)


def get_bone_rest(armature):
    assert _is_identity(armature.matrix_world), (
        f"Armature '{armature.name}' has a non-identity matrix_world -- bone_rest.npy would be in "
        f"world space while rest_pose.npy is in local space, an inconsistent pair. Apply this "
        f"object's transform first (Object -> Apply -> All Transforms)."
    )
    B = len(armature.data.bones)
    bones_rest = np.zeros((B, 3, 4), dtype=np.float32)
    for i, b in enumerate(armature.data.bones):
        mat = armature.matrix_world @ b.matrix_local
        bones_rest[i] = np.array(mat)[:3, :]
    return bones_rest


meshes = get_character_meshes(armature)
if not meshes:
    raise RuntimeError(f"No mesh found with an Armature modifier pointing at '{armature.name}'.")
print(f"Found {len(meshes)} mesh object(s) deforming by this armature.")

rest_pose = get_rest_pose_mesh(meshes, armature)
np.save(os.path.join(static_dir, "rest_pose.npy"), rest_pose)
print(f"Saved rest_pose.npy: {rest_pose.shape}")

bone_names, bone_map = get_bone_map(armature)
np.save(os.path.join(static_dir, "bone_rest.npy"), get_bone_rest(armature))
with open(os.path.join(static_dir, "bone_names.txt"), "w") as f:
    f.write("\n".join(bone_names))
print(f"Saved bone_rest.npy ({len(bone_names)} bones) & bone_names.txt")

skinning_weights = export_skinning_weights(meshes, bone_map)
np.save(os.path.join(static_dir, "skinning_weights.npy"), skinning_weights)
print(f"Saved skinning_weights.npy: {skinning_weights.shape}")

print(f"\nDone -- all files written directly to {static_dir}")
