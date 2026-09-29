import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT_BONE_INDEX = 0


def load_bones(animation_dir, seq_name):
    bones_raw = np.load(os.path.join(animation_dir, f"{seq_name}_Bones.npy")).astype(np.float32)
    F = bones_raw.shape[0]
    max_bones = bones_raw.shape[1] // 12
    return bones_raw.reshape(F, max_bones, 3, 4)


def resample_frames(x, target_len):
    src_len = x.shape[0]
    if src_len == target_len:
        return x.copy()
    src_t = np.linspace(0.0, 1.0, src_len)
    tgt_t = np.linspace(0.0, 1.0, target_len)
    flat = x.reshape(src_len, -1)
    out = np.empty((target_len, flat.shape[1]), dtype=x.dtype)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(tgt_t, src_t, flat[:, d])
    return out.reshape((target_len,) + x.shape[1:])


def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def blend_weight_curve(n, start_frac, end_frac):
    t = np.linspace(0.0, 1.0, n)
    w = np.clip((t - start_frac) / max(end_frac - start_frac, 1e-6), 0.0, 1.0)
    return (w * w * (3 - 2 * w)).astype(np.float32)


def _slerp_quat_batch(qa, qb, weights):
    dot = np.clip(np.sum(qa * qb, axis=-1), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    out = np.empty_like(qa)
    near_parallel = sin_theta < 1e-6
    far = ~near_parallel
    if np.any(far):
        s0 = (np.sin((1 - weights[far]) * theta[far]) / sin_theta[far])[:, None]
        s1 = (np.sin(weights[far] * theta[far]) / sin_theta[far])[:, None]
        out[far] = s0 * qa[far] + s1 * qb[far]
    if np.any(near_parallel):
        w = weights[near_parallel][:, None]
        out[near_parallel] = (1 - w) * qa[near_parallel] + w * qb[near_parallel]

    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def blend_root_transform(R_a, t_a, R_b, t_b, weights):
    n = len(weights)

    dR_a = np.matmul(np.transpose(R_a[:-1], (0, 2, 1)), R_a[1:])
    dR_b = np.matmul(np.transpose(R_b[:-1], (0, 2, 1)), R_b[1:])
    v_a = np.einsum('nji,nj->ni', R_a[:-1], t_a[1:] - t_a[:-1])
    v_b = np.einsum('nji,nj->ni', R_b[:-1], t_b[1:] - t_b[:-1])

    q_a = Rotation.from_matrix(dR_a).as_quat()
    q_b = Rotation.from_matrix(dR_b).as_quat()
    q_a[q_a[:, 3] < 0] *= -1
    q_b[q_b[:, 3] < 0] *= -1

    w = weights[1:]
    dR = Rotation.from_quat(_slerp_quat_batch(q_a, q_b, w)).as_matrix()
    v_body = (1 - w)[:, None] * v_a + w[:, None] * v_b

    R_blend = np.empty((n, 3, 3), dtype=np.float32)
    t_blend = np.empty((n, 3), dtype=np.float32)
    R_blend[0] = R_a[0]
    t_blend[0] = t_a[0]
    for i in range(1, n):
        t_blend[i] = t_blend[i - 1] + R_blend[i - 1] @ v_body[i - 1]
        R_blend[i] = orthonormalize(R_blend[i - 1] @ dR[i - 1])
    return R_blend, t_blend


def resampled_root_transform(bones, root_bone_index, target_len):
    R_rs = np.stack([orthonormalize(r) for r in
                      resample_frames(bones[:, root_bone_index, :, :3], target_len)])
    t_rs = resample_frames(bones[:, root_bone_index, :, 3], target_len)
    return R_rs.astype(np.float32), t_rs.astype(np.float32)


def encode_and_decode(model, bones, device):
    bones_t = torch.from_numpy(bones).unsqueeze(0).to(device)
    B, F, B_max, _, _ = bones_t.shape
    bones_flat = bones_t.contiguous().view(B, F, B_max * 12)
    with torch.no_grad():
        latent = model.bone_encoder(bones_flat)
        latent = model.attention(latent)
        coeffs = model.decoder(latent)
    return coeffs.squeeze(0).cpu().numpy()


def foot_vertex_mask(root_dir, character):
    static_dir = os.path.join(root_dir, character, "static_data")
    W = np.load(os.path.join(static_dir, "skinning_weights.npy"))
    names = np.load(os.path.join(static_dir, "bone_names.npy"), allow_pickle=True)
    foot_bones = [i for i, n in enumerate(names) if "foot" in str(n).lower() or "toe" in str(n).lower()]
    if not foot_bones:
        return None
    dominant = W.argmax(1)
    return np.isin(dominant, foot_bones)


def encode_latent(model, bones, device):
    bones_t = torch.from_numpy(bones).unsqueeze(0).to(device)
    B, F, B_max, _, _ = bones_t.shape
    bones_flat = bones_t.contiguous().view(B, F, B_max * 12)
    with torch.no_grad():
        latent = model.bone_encoder(bones_flat)
        latent = model.attention(latent)
    return latent.squeeze(0)
