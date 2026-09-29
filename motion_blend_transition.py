import os
import sys
import numpy as np
import torch

from model import VolumetricModelPCA
from transition_net import TransitionNet, blended_latent
from blend_utils import (
    load_bones, resample_frames, orthonormalize, blend_weight_curve,
    blend_root_transform, encode_latent, ROOT_BONE_INDEX,
)

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

ckpt_path = os.path.join(root_dir, f"best_transition_{character}.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"{ckpt_path} not found -- run train_transition.py first.")
bundle = torch.load(ckpt_path, map_location=device)
model_state = bundle["model"]
hidden_size = bundle["hidden_size"]

pca_mean = model_state["pca_mean"].cpu().numpy()
pca_components = model_state["pca_components"].cpu().numpy()
num_layers = sum(1 for k in model_state if k.startswith("bone_encoder.lstm.weight_ih_l"))
max_bones = model_state["bone_encoder.input_norm.weight"].shape[0] // 12

model = VolumetricModelPCA(
    max_bones=max_bones, hidden_size=hidden_size, num_layers=num_layers,
    pca_mean=pca_mean, pca_components=pca_components,
).to(device)
model.load_state_dict(model_state)
model.eval()

transition_net = TransitionNet(hidden_size=hidden_size).to(device)
transition_net.load_state_dict(bundle["transition_net"])
transition_net.eval()
print(f"Loaded: {ckpt_path}")

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
bones_a = load_bones(animation_dir, sequence_a)
bones_b = load_bones(animation_dir, sequence_b)
assert bones_a.shape[1] == bones_b.shape[1] == max_bones, "sequences must match the character's rig"

z_a_native = encode_latent(model, bones_a, device).cpu().numpy()
z_b_native = encode_latent(model, bones_b, device).cpu().numpy()
z_a_rs = torch.from_numpy(resample_frames(z_a_native, NUM_OUTPUT_FRAMES)).to(device)
z_b_rs = torch.from_numpy(resample_frames(z_b_native, NUM_OUTPUT_FRAMES)).to(device)
print(f"Encoded {sequence_a}: latent {z_a_native.shape} | {sequence_b}: latent {z_b_native.shape}")

R_root_a = np.stack([orthonormalize(r) for r in
                      resample_frames(bones_a[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
t_root_a = resample_frames(bones_a[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)
R_root_b = np.stack([orthonormalize(r) for r in
                      resample_frames(bones_b[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
t_root_b = resample_frames(bones_b[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)

weights_np = blend_weight_curve(NUM_OUTPUT_FRAMES, BLEND_START_FRAC, BLEND_END_FRAC)
weights_t = torch.from_numpy(weights_np).to(device)

with torch.no_grad():
    z_blend = blended_latent(z_a_rs.unsqueeze(0), z_b_rs.unsqueeze(0), weights_t.unsqueeze(0),
                              transition_net)
    coeffs = model.decoder(z_blend)
    pred_disp = model.reconstruct(coeffs)

rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)
pred_local = rest_pose_t.unsqueeze(1) + pred_disp

R_blend, t_blend = blend_root_transform(R_root_a, t_root_a, R_root_b, t_root_b, weights_np)
R_blend_t = torch.from_numpy(R_blend).unsqueeze(0).to(device)
t_blend_t = torch.from_numpy(t_blend).unsqueeze(0).to(device)
world_pos = torch.matmul(pred_local, R_blend_t.transpose(-1, -2)) + t_blend_t.unsqueeze(-2)

mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)

raw_out_path = os.path.join(root_dir, character,
                             f"TransitionBlend_{sequence_a}_to_{sequence_b}_FullMesh_{character}_raw")
np.save(raw_out_path, mesh_vertices)

FLOOR_Z = 0.0
min_z_per_frame = mesh_vertices[..., 2].min(axis=1)
violation_per_frame = np.clip(FLOOR_Z - min_z_per_frame, a_min=0.0, a_max=None)
frames_with_violations = int((violation_per_frame > 0).sum())
if frames_with_violations > 0:
    print(f"Soft floor penalty left {frames_with_violations}/{mesh_vertices.shape[0]} frames with "
          f"at least one vertex below floor (worst penetration: {violation_per_frame.max():.4f} units) "
          f"-- rigidly lifting each such frame upward by its own worst violation so every vertex ends "
          f"up >= floor WITHOUT flattening/distorting local shape.")
    mesh_vertices[..., 2] += violation_per_frame[:, None]
else:
    print("No floor violations survived -- soft penalty alone was sufficient here; no correction needed.")

out_path = os.path.join(root_dir, character,
                         f"TransitionBlend_{sequence_a}_to_{sequence_b}_FullMesh_{character}")
np.save(out_path, mesh_vertices)
print(f"Saved transition-net-blended motion: {mesh_vertices.shape} -> {out_path}")
print(f"Compare against geodesic_blend_4point.py's GeodesicBlend4pt_... for the same pair.")
