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
    return bones, {name: i for i, name in enumerate(bones)}


def get_rest_pose_mesh(meshes, armature):
    original_pose_position = armature.data.pose_position
    armature.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    try:
        verts_all = []
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for mesh_obj in meshes:
            assert _is_identity(mesh_obj.matrix_world), (
                f"'{mesh_obj.name}' has a non-identity matrix_world -- apply "
                f"its transform first (Object -> Apply -> All Transforms)."
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
        W /= (W.sum(axis=1, keepdims=True) + 1e-8)
        weights_all.append(W)
    return np.concatenate(weights_all, axis=0)


def get_bone_rest(armature):
    assert _is_identity(armature.matrix_world), (
        f"Armature '{armature.name}' has a non-identity matrix_world -- apply "
        f"its transform first (Object -> Apply -> All Transforms)."
    )
    B = len(armature.data.bones)
    bones_rest = np.zeros((B, 3, 4), dtype=np.float32)
    for i, b in enumerate(armature.data.bones):
        mat = armature.matrix_world @ b.matrix_local
        bones_rest[i] = np.array(mat)[:3, :]
    return bones_rest


def verify_consistency(bone_names, W, bone_rest, rest_pose, gap_threshold=15.0):
    dom = W.argmax(1)
    print("\n--- consistency check: bone_rest position vs dominant-vertex centroid ---")
    bad = []
    for bidx, name in enumerate(bone_names):
        verts = np.where(dom == bidx)[0]
        if len(verts) < 5:
            continue
        centroid = rest_pose[verts].mean(0)
        bpos = bone_rest[bidx, :, 3]
        gap = float(np.linalg.norm(centroid - bpos))
        if gap > gap_threshold:
            bad.append((name, gap))
    if bad:
        print(f"WARNING: {len(bad)} bone(s) still show a gap > {gap_threshold}:")
        for name, gap in bad:
            print(f"   {name}: gap={gap:.2f}")
        print("This means bone_rest.npy and rest_pose.npy are STILL inconsistent "
              "for these bones -- do not overwrite production files with this "
              "output yet; investigate the .blend rig itself.")
    else:
        print("OK -- no bone exceeds the gap threshold. bone_rest.npy and "
              "rest_pose.npy agree on the same bind pose.")


meshes = get_character_meshes(armature)
if not meshes:
    raise RuntimeError(f"No mesh found with an Armature modifier pointing at '{armature.name}'.")
print(f"Found {len(meshes)} mesh object(s) deforming by this armature.")

rest_pose = get_rest_pose_mesh(meshes, armature)
bone_names, bone_map = get_bone_map(armature)
bone_rest = get_bone_rest(armature)
skinning_weights = export_skinning_weights(meshes, bone_map)

verify_consistency(bone_names, skinning_weights, bone_rest, rest_pose)

np.save(os.path.join(static_dir, "rest_pose.npy"), rest_pose)
print(f"Saved rest_pose.npy: {rest_pose.shape}")

np.save(os.path.join(static_dir, "bone_rest.npy"), bone_rest)
np.save(os.path.join(static_dir, "bone_names.npy"), np.array(bone_names, dtype=object))
with open(os.path.join(static_dir, "bone_names.txt"), "w") as f:
    f.write("\n".join(bone_names))
print(f"Saved bone_rest.npy & bone_names.npy/.txt ({len(bone_names)} bones)")

np.save(os.path.join(static_dir, "skinning_weights.npy"), skinning_weights)
print(f"Saved skinning_weights.npy: {skinning_weights.shape}")

print(f"\nDone -- all files written directly to {static_dir}")
print("Check the consistency-check output above before trusting this export.")
