import os
import numpy as np
from scipy.spatial.transform import Rotation

from mixamo_hierarchy import build_parent_indices

ROOT_BONE_INDEX = 0


def to4(M):
    out = np.zeros(M.shape[:-2] + (4, 4), dtype=np.float64)
    out[..., :3, :] = M
    out[..., 3, 3] = 1.0
    return out


def rest_offsets(bone_rest_world, parents):
    T = to4(bone_rest_world)
    J = len(parents)
    off = np.zeros((J, 3), dtype=np.float64)
    for i in range(J):
        p = parents[i]
        if p < 0:
            off[i] = T[i, :3, 3]
        else:
            local = np.linalg.inv(T[p]) @ T[i]
            off[i] = local[:3, 3]
    return off


def local_quats(bones_world, parents):
    F, J = bones_world.shape[:2]
    T = to4(bones_world)
    local_R = np.zeros((F, J, 3, 3), dtype=np.float64)
    for i in range(J):
        p = parents[i]
        if p < 0:
            local_R[:, i] = T[:, i, :3, :3]
        else:
            Rp = T[:, p, :3, :3]
            Rc = T[:, i, :3, :3]
            local_R[:, i] = np.einsum("fji,fjk->fik", Rp, Rc)
    q_xyzw = Rotation.from_matrix(local_R.reshape(-1, 3, 3)).as_quat().reshape(F, J, 4)
    return q_xyzw[..., [3, 0, 1, 2]]


def fk_positions(local_q, offsets, parents, root_p):
    F, J = local_q.shape[:2]
    local_R = Rotation.from_quat(local_q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(F, J, 3, 3)
    global_R = np.zeros_like(local_R)
    global_P = np.zeros((F, J, 3), dtype=local_q.dtype)
    for i in range(J):
        p = parents[i]
        if p < 0:
            global_R[:, i] = local_R[:, i]
            global_P[:, i] = root_p
        else:
            global_R[:, i] = np.einsum("fij,fjk->fik", global_R[:, p], local_R[:, i])
            global_P[:, i] = global_P[:, p] + np.einsum("fij,j->fi", global_R[:, p], offsets[i])
    return global_P, global_R


def root_velocity(bones_world):
    root_p = bones_world[:, ROOT_BONE_INDEX, :, 3]
    return root_p[1:] - root_p[:-1]


def foot_contact(bones_world, foot_bone_indices, floor_z=0.0, z_eps_frac=0.008, scale=None):
    z_eps = z_eps_frac * (scale if scale is not None else 1.0)
    translations = bones_world[..., 3]
    z = translations[:, foot_bone_indices, :][..., 2]
    return (z < floor_z + z_eps).astype(np.float32)


def active_bone_names(root_dir, character):
    static_dir = os.path.join(root_dir, character, "static_data")
    names = [str(n) for n in np.load(os.path.join(static_dir, "bone_names.npy"), allow_pickle=True)]
    W = np.load(os.path.join(static_dir, "skinning_weights.npy"))
    act = np.where(W.sum(0) > 0)[0]
    return [names[i] for i in act], act


if __name__ == "__main__":
    root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
    character = "michelle"
    static_dir = os.path.join(root_dir, character, "static_data")
    anim_dir = os.path.join(root_dir, character, "animation_data")

    names, act = active_bone_names(root_dir, character)
    bone_rest = np.load(os.path.join(static_dir, "bone_rest.npy")).astype(np.float64)[act]
    rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
    scale = float(np.linalg.norm(rest_pose.max(0) - rest_pose.min(0)))
    parents = build_parent_indices(names)

    bones_raw = np.load(os.path.join(anim_dir, "Armature.016_Bones.npy")).astype(np.float32)
    F = bones_raw.shape[0]
    max_bones = bones_raw.shape[1] // 12
    bones_world = bones_raw.reshape(F, max_bones, 3, 4).astype(np.float64)
    print(f"active bones: {len(names)}, bones_world: {bones_world.shape}, parents: {parents.shape}")
    assert bones_world.shape[1] == len(parents), "active-bone count must match parents length"

    off = rest_offsets(bone_rest, parents)
    li = names.index("mixamorig:LeftUpLeg")
    print(f"rest offsets: {off.shape}, sample (LeftUpLeg): {off[li]}")

    q = local_quats(bones_world, parents)
    print(f"local quats: {q.shape}, unit-norm check (should be ~1.0): {np.linalg.norm(q, axis=-1).mean():.6f}")

    rv = root_velocity(bones_world)
    print(f"root velocity: {rv.shape}, mean speed: {np.linalg.norm(rv, axis=-1).mean():.4f}")

    foot_names = ["mixamorig:LeftFoot", "mixamorig:LeftToeBase", "mixamorig:RightFoot", "mixamorig:RightToeBase"]
    foot_idx = [names.index(n) for n in foot_names]
    contact = foot_contact(bones_world, foot_idx, scale=scale)
    print(f"contact: {contact.shape}, mean contact fraction per column: {contact.mean(0)}")

    real_pos = bones_world[..., 3]
    root_p = real_pos[:, ROOT_BONE_INDEX]
    fk_pos, _ = fk_positions(q, off, parents, root_p)
    err = np.linalg.norm(fk_pos - real_pos, axis=-1)
    print(f"\nFK round-trip error: mean={err.mean():.6f}, max={err.max():.6f} "
          f"(world units, scale={scale:.1f}) -- should be ~0")
