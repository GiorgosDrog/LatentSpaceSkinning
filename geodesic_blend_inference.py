import os
import sys
import numpy as np
import torch

from model import VolumetricModelPCA
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

OPT_STEPS = 1000
OPT_LR = 0.05
W_VELOCITY = 1.0
W_ACCELERATION = 0.1

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}_synthesis.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"{ckpt_path} not found -- run synthesis_finetune.py first. "
        f"The base reconstruction checkpoint was never trained to decode "
        f"blended latents and will distort badly here too."
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
print(f"Encoded {sequence_a}: latent {z_a_native.shape} | {sequence_b}: latent {z_b_native.shape}")

R_root_a = np.stack([orthonormalize(r) for r in
                      resample_frames(bones_a[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
t_root_a = resample_frames(bones_a[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)
R_root_b = np.stack([orthonormalize(r) for r in
                      resample_frames(bones_b[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
t_root_b = resample_frames(bones_b[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)

def curve_to_theta(w):
    d = np.diff(w).clip(min=1e-6)
    d = d / d.sum()
    return np.log(d)


w_init = blend_weight_curve(NUM_OUTPUT_FRAMES, BLEND_START_FRAC, BLEND_END_FRAC)
theta = torch.tensor(curve_to_theta(w_init), dtype=torch.float32, device=device, requires_grad=True)


def theta_to_w(theta):
    d = torch.softmax(theta, dim=0)
    return torch.cat([torch.zeros(1, device=theta.device), torch.cumsum(d, dim=0)])


def decoded_coeffs(w):
    z_blend = (1 - w).unsqueeze(-1) * z_a_rs + w.unsqueeze(-1) * z_b_rs
    return model.decoder(z_blend.unsqueeze(0)).squeeze(0)


def roughness_loss(c):
    vel = (c[1:] - c[:-1]).pow(2).sum(dim=-1).mean()
    acc = (c[2:] - 2 * c[1:-1] + c[:-2]).pow(2).sum(dim=-1).mean()
    return W_VELOCITY * vel + W_ACCELERATION * acc


with torch.no_grad():
    baseline_loss = roughness_loss(decoded_coeffs(theta_to_w(theta))).item()

optimizer = torch.optim.Adam([theta], lr=OPT_LR)
for step in range(1, OPT_STEPS + 1):
    optimizer.zero_grad()
    loss = roughness_loss(decoded_coeffs(theta_to_w(theta)))
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == OPT_STEPS:
        print(f"[{step:4d}/{OPT_STEPS}] roughness = {loss.item():.6f}")

with torch.no_grad():
    w_final = theta_to_w(theta)
    final_loss = roughness_loss(decoded_coeffs(w_final)).item()
improvement = 100.0 * (1.0 - final_loss / baseline_loss)
print(f"\nBaseline (smoothstep) roughness:      {baseline_loss:.6f}")
print(f"Optimized (geodesic-search) roughness: {final_loss:.6f}  ({improvement:+.1f}%)")

with torch.no_grad():
    coeffs_blend = decoded_coeffs(w_final).unsqueeze(0)
    pred_disp = model.reconstruct(coeffs_blend)

w_final_np = w_final.cpu().numpy()
R_root_blend, t_root_blend = blend_root_transform(R_root_a, t_root_a, R_root_b, t_root_b, w_final_np)

rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)
pred_local = rest_pose_t.unsqueeze(1) + pred_disp

R_blend_t = torch.from_numpy(R_root_blend).unsqueeze(0).to(device)
t_blend_t = torch.from_numpy(t_root_blend).unsqueeze(0).to(device)
world_pos = torch.matmul(pred_local, R_blend_t.transpose(-1, -2)) + t_blend_t.unsqueeze(-2)

mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)

out_path = os.path.join(root_dir, character, f"GeodesicBlend_{sequence_a}_to_{sequence_b}_FullMesh_{character}")
np.save(out_path, mesh_vertices)
print(f"\nSaved geodesic-blended motion: {mesh_vertices.shape} -> {out_path}")
print(f"Compare against the fixed-curve version: motion_blend_latent.py's "
      f"LatentBlend_{sequence_a}_to_{sequence_b}_FullMesh_{character}.npy")
