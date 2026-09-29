import itertools
import numpy as np
import torch
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("models", "preprocessing", "inference", "blendings"):
    sys.path.insert(0, os.path.join(_ROOT, _d))
from scipy.spatial.transform import Rotation

from model import VolumetricModelPCA

ROOT_BONE_INDEX = 0

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = "alpha"
SEQUENCES = [f"Armature.{i:03d}" for i in range(1, 5)]

HIDDEN_SIZE = 64
NUM_LAYERS = 2

NUM_OUTPUT_FRAMES = 150
BLEND_START_FRAC = 0.25
BLEND_END_FRAC = 0.75

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
output_dir = os.path.join(root_dir, character, "blends")
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_bones(seq_name):
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


def _unroll_quaternions(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


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
    qa = _unroll_quaternions(Rotation.from_matrix(R_a).as_quat())
    qb = _unroll_quaternions(Rotation.from_matrix(R_b).as_quat())

    q_blend = _slerp_quat_batch(qa, qb, weights)
    R_blend = Rotation.from_quat(q_blend).as_matrix().astype(np.float32)

    t_blend = (1 - weights)[:, None] * t_a + weights[:, None] * t_b
    return R_blend, t_blend.astype(np.float32)


def encode_and_decode(model, bones):
    bones_t = torch.from_numpy(bones).unsqueeze(0).to(device)
    B, F, B_max, _, _ = bones_t.shape
    bones_flat = bones_t.contiguous().view(B, F, B_max * 12)
    with torch.no_grad():
        latent = model.bone_encoder(bones_flat)
        latent = model.attention(latent)
        coeffs = model.decoder(latent)
    return coeffs.squeeze(0).cpu().numpy()


ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}.pth")
state_dict = torch.load(ckpt_path, map_location=device)

pca_mean = state_dict["pca_mean"].cpu().numpy()
pca_components = state_dict["pca_components"].cpu().numpy()
hidden_size = state_dict["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
num_layers = sum(1 for k in state_dict if k.startswith("bone_encoder.lstm.weight_ih_l"))

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
sample_bones = load_bones(SEQUENCES[0])
max_bones = sample_bones.shape[1]

model = VolumetricModelPCA(
    max_bones=max_bones, hidden_size=hidden_size, num_layers=num_layers,
    pca_mean=pca_mean, pca_components=pca_components,
).to(device)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {ckpt_path} (hidden_size={hidden_size}, num_layers={num_layers}, K={pca_components.shape[0]})")

cache = {}
for seq in SEQUENCES:
    bones = load_bones(seq)
    assert bones.shape[1] == max_bones, f"{seq} has a different bone count -- not the same rig"

    coeffs = encode_and_decode(model, bones)
    coeffs_rs = resample_frames(coeffs, NUM_OUTPUT_FRAMES)

    R_root_rs = np.stack([orthonormalize(r) for r in
                           resample_frames(bones[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
    t_root_rs = resample_frames(bones[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)

    cache[seq] = {"coeffs_rs": coeffs_rs, "R_root_rs": R_root_rs, "t_root_rs": t_root_rs}
    print(f"Cached {seq}: {bones.shape[0]} frames -> resampled to {NUM_OUTPUT_FRAMES}")

pairs = list(itertools.combinations(SEQUENCES, 2))
weights = blend_weight_curve(NUM_OUTPUT_FRAMES, BLEND_START_FRAC, BLEND_END_FRAC)
rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)

os.makedirs(output_dir, exist_ok=True)
print(f"\nGenerating {len(pairs)} pairwise blends from {len(SEQUENCES)} sequences -> {output_dir}\n")

for i, (seq_a, seq_b) in enumerate(pairs, 1):
    a, b = cache[seq_a], cache[seq_b]

    coeffs_blend = (1 - weights)[:, None] * a["coeffs_rs"] + weights[:, None] * b["coeffs_rs"]
    R_root_blend, t_root_blend = blend_root_transform(
        a["R_root_rs"], a["t_root_rs"], b["R_root_rs"], b["t_root_rs"], weights
    )

    coeffs_blend_t = torch.from_numpy(coeffs_blend.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_disp = model.reconstruct(coeffs_blend_t)

    pred_local = rest_pose_t.unsqueeze(1) + pred_disp

    R_blend_t = torch.from_numpy(R_root_blend).unsqueeze(0).to(device)
    t_blend_t = torch.from_numpy(t_root_blend).unsqueeze(0).to(device)
    world_pos = torch.matmul(pred_local, R_blend_t.transpose(-1, -2)) + t_blend_t.unsqueeze(-2)

    mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)

    out_path = os.path.join(output_dir, f"Blend_{seq_a}_to_{seq_b}_FullMesh_{character}.npy")
    np.save(out_path, mesh_vertices)
    print(f"[{i:3d}/{len(pairs)}] {seq_a} + {seq_b} -> {os.path.basename(out_path)}  {mesh_vertices.shape}")

print(f"\nDone. {len(pairs)} blended .npy files saved to {output_dir}")
