import numpy as np
import torch

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("models", "preprocessing", "inference", "blendings"):
    sys.path.insert(0, os.path.join(_ROOT, _d))
from model import VolumetricModelPCA
from blend_utils import (
    load_bones, resample_frames, orthonormalize, blend_weight_curve,
    blend_root_transform, encode_latent, ROOT_BONE_INDEX, foot_vertex_mask,
)
from loss_functions import (
    contact_noslide_loss, curvature_consistency_loss, build_knn_laplacian,
)

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = sys.argv[1] if len(sys.argv) > 1 else "michelle"
sequence_a = sys.argv[2] if len(sys.argv) > 2 else "Armature.016"
sequence_b = sys.argv[3] if len(sys.argv) > 3 else "Armature.010"

NUM_OUTPUT_FRAMES = 150
BLEND_START_FRAC = 0.25
BLEND_END_FRAC = 0.75
NUM_CONTROL_POINTS = 4

OPT_STEPS = 1500
OPT_LR = 0.03
W_VELOCITY = 1.0
W_ACCELERATION = 0.1
W_FLOOR = 100000
W_CONTACT = 0.5
W_CURVATURE = 20
FLOOR_Z = 0.0

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"{ckpt_path} not found -- run synthesis_finetune.py first."
    )
state_dict = torch.load(ckpt_path, map_location=device)
pca_mean = state_dict["pca_mean"].cpu().numpy()
pca_components = state_dict["pca_components"].cpu().numpy()
hidden_size = state_dict["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
num_layers = sum(1 for k in state_dict if k.startswith("bone_encoder.lstm.weight_ih_l"))
max_bones = state_dict["bone_encoder.input_norm.weight"].shape[0] // 12

model = VolumetricModelPCA(
    max_bones=max_bones, hidden_size=hidden_size, num_layers=num_layers,
    pca_mean=pca_mean, pca_components=pca_components,
).to(device)
model.load_state_dict(state_dict)
for p in model.parameters():
    p.requires_grad_(False)
model.decoder.train()
print(f"Loaded fine-tuned synthesis checkpoint: {ckpt_path}")

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
bones_a = load_bones(animation_dir, sequence_a)
bones_b = load_bones(animation_dir, sequence_b)
assert bones_a.shape[1] == bones_b.shape[1] == max_bones, "sequences must match the character's rig"

z_a_native = encode_latent(model, bones_a, device).cpu().numpy()
z_b_native = encode_latent(model, bones_b, device).cpu().numpy()
z_a_rs = torch.from_numpy(resample_frames(z_a_native, NUM_OUTPUT_FRAMES)).to(device)
z_b_rs = torch.from_numpy(resample_frames(z_b_native, NUM_OUTPUT_FRAMES)).to(device)
hidden_dim = z_a_rs.shape[-1]
print(f"Encoded {sequence_a}: latent {z_a_native.shape} | {sequence_b}: latent {z_b_native.shape}")

R_root_a = np.stack([orthonormalize(r) for r in
                      resample_frames(bones_a[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
t_root_a = resample_frames(bones_a[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)
R_root_b = np.stack([orthonormalize(r) for r in
                      resample_frames(bones_b[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
t_root_b = resample_frames(bones_b[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)

rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)
scale_t = torch.tensor([float(np.linalg.norm(rest_pose.max(0) - rest_pose.min(0)))], device=device)

foot_mask_np = foot_vertex_mask(root_dir, character)
foot_mask_t = torch.from_numpy(foot_mask_np).to(device) if foot_mask_np is not None else None
if foot_mask_t is not None:
    print(f"Foot vertices: {int(foot_mask_np.sum())} / {len(foot_mask_np)}")
else:
    print("[warn] no foot/toe bones found by name -- contact loss will use ALL vertices")

with torch.no_grad():
    c_a_ref = model.decoder(z_a_rs.unsqueeze(0))
    c_b_ref = model.decoder(z_b_rs.unsqueeze(0))
    mesh_a_local = model.reconstruct(c_a_ref)
    mesh_b_local = model.reconstruct(c_b_ref)

laplacian = build_knn_laplacian(rest_pose, k=8).to(device)

def curve_to_theta(w):
    d = np.diff(w).clip(min=1e-6)
    d = d / d.sum()
    return np.log(d)


w_init = blend_weight_curve(NUM_OUTPUT_FRAMES, BLEND_START_FRAC, BLEND_END_FRAC)
theta = torch.tensor(curve_to_theta(w_init), dtype=torch.float32, device=device, requires_grad=True)


def theta_to_w(theta):
    d = torch.softmax(theta, dim=0)
    return torch.cat([torch.zeros(1, device=theta.device), torch.cumsum(d, dim=0)])


control_points = torch.zeros(NUM_CONTROL_POINTS, hidden_dim, device=device, requires_grad=True)


def catmull_rom_path(nodes, num_frames):
    n_pts = nodes.shape[0]
    n_seg = n_pts - 1
    u = torch.linspace(0, n_seg, num_frames, device=nodes.device)
    seg = torch.clamp(u.floor().long(), 0, n_seg - 1)
    s = (u - seg.float()).unsqueeze(-1)

    idx0 = torch.clamp(seg - 1, 0, n_pts - 1)
    idx1 = torch.clamp(seg, 0, n_pts - 1)
    idx2 = torch.clamp(seg + 1, 0, n_pts - 1)
    idx3 = torch.clamp(seg + 2, 0, n_pts - 1)

    P0, P1, P2, P3 = nodes[idx0], nodes[idx1], nodes[idx2], nodes[idx3]
    s2, s3 = s * s, s * s * s
    return 0.5 * (2 * P1 + (-P0 + P2) * s + (2 * P0 - 5 * P1 + 4 * P2 - P3) * s2
                  + (-P0 + 3 * P1 - 3 * P2 + P3) * s3)


def correction_path(control_points, num_frames):
    zero = torch.zeros(1, control_points.shape[1], device=control_points.device)
    nodes = torch.cat([zero, control_points, zero], dim=0)
    return catmull_rom_path(nodes, num_frames)


def decode_and_assemble(theta, control_points):
    w = theta_to_w(theta)
    correction = correction_path(control_points, NUM_OUTPUT_FRAMES)
    z_blend = (1 - w).unsqueeze(-1) * z_a_rs + w.unsqueeze(-1) * z_b_rs + correction
    coeffs = model.decoder(z_blend.unsqueeze(0))
    mesh_local = model.reconstruct(coeffs)

    w_np = w.detach().cpu().numpy()
    R_blend, t_blend = blend_root_transform(R_root_a, t_root_a, R_root_b, t_root_b, w_np)
    R_blend_t = torch.from_numpy(R_blend).unsqueeze(0).to(device)
    t_blend_t = torch.from_numpy(t_blend).unsqueeze(0).to(device)

    pred_local = rest_pose_t.unsqueeze(1) + mesh_local
    world_pos = torch.matmul(pred_local, R_blend_t.transpose(-1, -2)) + t_blend_t.unsqueeze(-2)
    return coeffs.squeeze(0), mesh_local, world_pos, w


def roughness_loss(c):
    vel = (c[1:] - c[:-1]).pow(2).sum(dim=-1).mean()
    acc = (c[2:] - 2 * c[1:-1] + c[:-2]).pow(2).sum(dim=-1).mean()
    return W_VELOCITY * vel + W_ACCELERATION * acc


def floor_penalty_sharp(world_pos, scale, floor_z=0.0):
    world_z = world_pos[..., 2]
    depth = torch.clamp(floor_z - world_z, min=0.0)
    scale_r = scale.view(-1, 1, 1).clamp_min(1e-6)
    worst_per_frame = (depth / scale_r).amax(dim=-1)
    return worst_per_frame.pow(2).mean()


def loss_terms(theta, control_points):
    coeffs, mesh_local, world_pos, w = decode_and_assemble(theta, control_points)
    terms = {
        "roughness": roughness_loss(coeffs),
        "floor": W_FLOOR * floor_penalty_sharp(world_pos, scale_t, FLOOR_Z),
        "contact": W_CONTACT * contact_noslide_loss(world_pos, scale_t, FLOOR_Z, foot_mask=foot_mask_t),
        "curvature": W_CURVATURE * curvature_consistency_loss(
            mesh_local, mesh_a_local, mesh_b_local, w, laplacian),
    }
    total = sum(terms.values())
    return total, terms


def total_loss(theta, control_points):
    return loss_terms(theta, control_points)[0]


with torch.no_grad():
    baseline_loss, baseline_terms = loss_terms(theta, control_points)
    baseline_loss = baseline_loss.item()
    baseline_terms = {k: v.item() for k, v in baseline_terms.items()}
print(f"Baseline terms: " + " | ".join(f"{k}={v:.5f}" for k, v in baseline_terms.items()))

optimizer = torch.optim.Adam([theta, control_points], lr=OPT_LR)
for step in range(1, OPT_STEPS + 1):
    optimizer.zero_grad()
    loss, terms = loss_terms(theta, control_points)
    loss.backward()
    optimizer.step()
    if step % 100 == 0 or step == OPT_STEPS:
        term_str = " | ".join(f"{k}={v.item():.5f}" for k, v in terms.items())
        print(f"[{step:4d}/{OPT_STEPS}] loss = {loss.item():.6f} || {term_str}")

with torch.no_grad():
    final_loss, final_terms = loss_terms(theta, control_points)
    final_loss = final_loss.item()
    final_terms = {k: v.item() for k, v in final_terms.items()}
improvement = 100.0 * (1.0 - final_loss / baseline_loss)
print(f"\nBaseline (initial curve, zero correction) loss: {baseline_loss:.6f}")
print(f"Optimized (4-point free path) loss:              {final_loss:.6f}  ({improvement:+.1f}%)")
print("Per-term baseline -> optimized:")
for k in baseline_terms:
    term_improvement = 100.0 * (1.0 - final_terms[k] / baseline_terms[k]) if baseline_terms[k] != 0 else 0.0
    print(f"  {k:10s}: {baseline_terms[k]:.5f} -> {final_terms[k]:.5f}  ({term_improvement:+.1f}%)")
print("If 'contact' barely moved despite the other terms improving, the frozen "
      "decoder likely can't produce correct foot-lift anywhere near this path "
      "-- path search alone can't fix that; it would need decoder fine-tuning "
      "(e.g. more/targeted synthesis_finetune.py steps for this pair) instead.")

with torch.no_grad():
    _, _, world_pos, w_final = decode_and_assemble(theta, control_points)

mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)

raw_out_path = os.path.join(root_dir, character,
                             f"GeodesicBlend4pt_{sequence_a}_to_{sequence_b}_FullMesh_{character}_raw")
np.save(raw_out_path, mesh_vertices)

min_z_per_frame = mesh_vertices[..., 2].min(axis=1)
violation_per_frame = np.clip(FLOOR_Z - min_z_per_frame, a_min=0.0, a_max=None)
frames_with_violations = int((violation_per_frame > 0).sum())
if frames_with_violations > 0:
    print(f"\nSoft floor penalty left {frames_with_violations}/{mesh_vertices.shape[0]} frames with "
          f"at least one vertex below floor (worst penetration: {violation_per_frame.max():.4f} units) "
          f"-- rigidly lifting each such frame upward by its own worst violation so every vertex ends "
          f"up >= floor WITHOUT flattening/distorting local shape.")
    mesh_vertices[..., 2] += violation_per_frame[:, None]
else:
    print("\nNo floor violations survived optimization -- soft penalty alone was sufficient here; "
          "no correction needed.")

out_path = os.path.join(root_dir, character,
                         f"GeodesicBlend4pt_{sequence_a}_to_{sequence_b}_FullMesh_{character}")
np.save(out_path, mesh_vertices)
print(f"\nSaved 4-control-point geodesic-blended motion: {mesh_vertices.shape} -> {out_path}")
print(f"Compare against: geodesic_blend_inference.py's GeodesicBlend_{sequence_a}_to_{sequence_b}_"
      f"FullMesh_{character}.npy and motion_blend_latent.py's LatentBlend_{sequence_a}_to_{sequence_b}_"
      f"FullMesh_{character}.npy")
