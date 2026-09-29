import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from blend_utils import (
    load_bones, resample_frames, blend_weight_curve, blend_root_transform,
    resampled_root_transform, ROOT_BONE_INDEX,
)
from mixamo_hierarchy import build_parent_indices

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = sys.argv[1] if len(sys.argv) > 1 else "michelle"
sequence_a = sys.argv[2] if len(sys.argv) > 2 else "Armature.016"
sequence_b = sys.argv[3] if len(sys.argv) > 3 else "Armature.010"

NUM_OUTPUT_FRAMES = 150
BLEND_START_FRAC = 0.25
BLEND_END_FRAC = 0.75

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
device = "cuda" if torch.cuda.is_available() else "cpu"


def to4_np(M):
    out = np.zeros(M.shape[:-2] + (4, 4), dtype=np.float64)
    out[..., :3, :] = M
    out[..., 3, 3] = 1.0
    return out


def _rotmat_to_quat(R):
    xyzw = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat()
    q = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=-1)
    return q.reshape(R.shape[:-2] + (4,))


def _quat_mult_np(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def _quat_mult_torch(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_conj_torch(q):
    w, xyz = q[..., :1], q[..., 1:]
    return torch.cat([w, -xyz], dim=-1)


def _quat_to_rotmat_torch(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


class Skin:
    def __init__(self, character):
        char_static_dir = os.path.join(root_dir, character, "static_data")
        W = np.load(os.path.join(char_static_dir, "skinning_weights.npy")).astype(np.float64)
        act = np.where(W.sum(0) > 0)[0]
        self.act = act
        names_full = [str(n) for n in np.load(os.path.join(char_static_dir, "bone_names.npy"), allow_pickle=True)]
        self.names = [names_full[i] for i in act]
        self.parents = build_parent_indices(self.names)
        order, done, remaining = [], np.zeros(len(self.names), dtype=bool), list(range(len(self.names)))
        while remaining:
            progressed = False
            for i in list(remaining):
                if self.parents[i] < 0 or done[self.parents[i]]:
                    order.append(i)
                    done[i] = True
                    remaining.remove(i)
                    progressed = True
            if not progressed:
                order.extend(remaining)
                break
        self.order = order

        self.W = torch.from_numpy(W[:, act].astype(np.float32)).to(device)
        Rr = to4_np(np.load(os.path.join(char_static_dir, "bone_rest.npy")).astype(np.float64)[act])
        self.Rinv = np.linalg.inv(Rr)
        rest = np.load(os.path.join(char_static_dir, "rest_pose.npy")).astype(np.float32)
        self.rest = torch.from_numpy(rest).to(device)
        self.scale = float(np.linalg.norm(rest.max(0) - rest.min(0)))
        self.V, self.B = self.W.shape

    def world(self, bones):
        S = to4_np(bones) @ self.Rinv[None]
        R, t = S[..., :3, :3], S[..., :3, 3]

        q0 = _rotmat_to_quat(R)
        for i in self.order:
            p = self.parents[i]
            if p < 0:
                continue
            flip = np.sum(q0[:, i] * q0[:, p], axis=-1) < 0
            q0[flip, i] *= -1

        t_pure = np.concatenate([np.zeros(t.shape[:-1] + (1,)), t], axis=-1)
        qe = 0.5 * _quat_mult_np(t_pure, q0)

        q0_t = torch.from_numpy(q0.astype(np.float32)).to(device)
        qe_t = torch.from_numpy(qe.astype(np.float32)).to(device)

        out = []
        for s in range(0, len(q0_t), 16):
            q0f, qef = q0_t[s:s + 16], qe_t[s:s + 16]
            bq0 = torch.einsum("vb,fbi->fvi", self.W, q0f)
            bqe = torch.einsum("vb,fbi->fvi", self.W, qef)
            norm = bq0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            bq0, bqe = bq0 / norm, bqe / norm
            Rf = _quat_to_rotmat_torch(bq0)
            tf = 2 * _quat_mult_torch(bqe, _quat_conj_torch(bq0))[..., 1:]
            out.append(torch.einsum("fvij,vj->fvi", Rf, self.rest) + tf)
        return torch.cat(out, 0).cpu().numpy()


def skeleton_blend(bones_a, bones_b, w, R_blend, t_blend):
    def rel(b):
        T = to4_np(b)
        return np.einsum("fij,fbjk->fbik", np.linalg.inv(T[:, 0]), T)
    A, Bm = rel(bones_a), rel(bones_b)
    F, B = A.shape[:2]
    rA = Rotation.from_matrix(A[..., :3, :3].reshape(-1, 3, 3))
    rB = Rotation.from_matrix(Bm[..., :3, :3].reshape(-1, 3, 3))
    ww = np.repeat(w, B)
    R = (rA * Rotation.from_rotvec((rA.inv() * rB).as_rotvec() * ww[:, None])).as_matrix().reshape(F, B, 3, 3)
    t = (1 - w)[:, None, None] * A[..., :3, 3] + w[:, None, None] * Bm[..., :3, 3]
    rel_b = np.zeros((F, B, 4, 4))
    rel_b[..., :3, :3], rel_b[..., :3, 3], rel_b[..., 3, 3] = R, t, 1.0
    M = np.zeros((F, 4, 4))
    M[:, :3, :3], M[:, :3, 3], M[:, 3, 3] = R_blend, t_blend, 1.0
    return np.einsum("fij,fbjk->fbik", M, rel_b)[..., :3, :]


if __name__ == "__main__":
    bones_a = load_bones(animation_dir, sequence_a)
    bones_b = load_bones(animation_dir, sequence_b)
    assert bones_a.shape[1] == bones_b.shape[1], "sequences must match the character's rig"

    skin = Skin(character)

    Ra, ta = resampled_root_transform(bones_a, ROOT_BONE_INDEX, NUM_OUTPUT_FRAMES)
    Rb, tb = resampled_root_transform(bones_b, ROOT_BONE_INDEX, NUM_OUTPUT_FRAMES)
    ba_rs = resample_frames(bones_a, NUM_OUTPUT_FRAMES)
    bb_rs = resample_frames(bones_b, NUM_OUTPUT_FRAMES)

    w = blend_weight_curve(NUM_OUTPUT_FRAMES, BLEND_START_FRAC, BLEND_END_FRAC)
    R_blend, t_blend = blend_root_transform(Ra, ta, Rb, tb, w)

    bones_blend = skeleton_blend(ba_rs, bb_rs, w, R_blend, t_blend)
    mesh_vertices = skin.world(bones_blend).astype(np.float32)

    out_path = os.path.join(root_dir, character,
                             f"SkeletonBlendLBS_{sequence_a}_to_{sequence_b}_FullMesh_{character}")
    np.save(out_path, mesh_vertices)
    print(f"Saved classical skeleton-blend+LBS motion: {mesh_vertices.shape} -> {out_path}")
