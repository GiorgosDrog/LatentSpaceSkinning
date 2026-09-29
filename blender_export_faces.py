import bpy
import bmesh
import numpy as np
import os

character = "michelle"
static_dir = rf"E:\didaktoriko\diffusion_solution\dataset\{character}\static_data"

obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    raise RuntimeError("Select the character's mesh as the active object first.")

rest_pose_path = os.path.join(static_dir, "rest_pose.npy")
rest_pose = np.load(rest_pose_path).astype(np.float32)

mesh_verts = np.array([v.co for v in obj.data.vertices], dtype=np.float32)

if mesh_verts.shape != rest_pose.shape:
    raise RuntimeError(
        f"Vertex count mismatch: this mesh has {mesh_verts.shape[0]} vertices, "
        f"rest_pose.npy has {rest_pose.shape[0]} -- not the same mesh, or wrong object selected."
    )

max_diff = np.abs(mesh_verts - rest_pose).max()
if max_diff > 1e-2:
    raise RuntimeError(
        f"This mesh's vertex positions don't match rest_pose.npy closely enough "
        f"(max difference {max_diff:.4f}) -- vertex ORDER may not correspond. "
        f"Do not trust faces exported from a mismatched mesh; check you have the "
        f"correct object/pose selected (should be the T-pose, not an animated frame)."
    )
print(f"Vertex match OK (max diff {max_diff:.6f}) -- safe to export faces on this vertex order.")

bm = bmesh.new()
bm.from_mesh(obj.data)
bmesh.ops.triangulate(bm, faces=bm.faces[:])

faces = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int64)
bm.free()

print(f"Triangulated: {faces.shape[0]} triangles (from {len(obj.data.polygons)} original polygons).")
assert faces.shape[1] == 3, "Triangulation produced non-triangle faces -- unexpected."
assert faces.max() < rest_pose.shape[0], "Face indices exceed vertex count -- something is wrong."

np.save(os.path.join(static_dir, "faces.npy"), faces)
print(f"Saved faces.npy {faces.shape} -> {static_dir}")

bone_names_txt = os.path.join(static_dir, "bone_names.txt")
if os.path.exists(bone_names_txt):
    with open(bone_names_txt, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    np.save(os.path.join(static_dir, "bone_names.npy"), np.array(names, dtype=object))
    print(f"Saved bone_names.npy ({len(names)} names) -> {static_dir}")

print("\nDone. Re-run the character audit to confirm faces.npy now shows OK.")
