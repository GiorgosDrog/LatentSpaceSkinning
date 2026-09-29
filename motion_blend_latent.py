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

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}_synthesis.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"{ckpt_path} not found -- run synthesis_finetune.py first. "
        f"The base reconstruction checkpoint (best_volumetric_model_pca_{character}.pth) "
        f"was never trained to decode blended latents and will distort badly if used here."
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
model.eval()
print(f"Loaded fine-tuned synthesis checkpoint: {ckpt_path}")

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
bones_a = load_bones(animation_dir, sequence_a)
bones_b = load_bones(animation_dir, sequence_b)
assert bones_a.shape[1] == bones_b.shape[1] == max_bones, "sequences must match the character's rig"

z_a_native = encode_latent(model, bones_a, device).cpu().numpy()
z_b_native = encode_latent(model, bones_b, device).cpu().numpy()
print(f"Encoded {sequence_a}: latent {z_a_native.shape} | {sequence_b}: latent {z_b_native.shape}")

z_a_rs = resample_frames(z_a_native, NUM_OUTPUT_FRAMES)
z_b_rs = resample_frames(z_b_native, NUM_OUTPUT_FRAMES)

R_root_a = np.stack([orthonormalize(r) for r in
                      resample_frames(bones_a[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
t_root_a = resample_frames(bones_a[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)
R_root_b = np.stack([orthonormalize(r) for r in
                      resample_frames(bones_b[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
t_root_b = resample_frames(bones_b[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)

weights = blend_weight_curve(NUM_OUTPUT_FRAMES, BLEND_START_FRAC, BLEND_END_FRAC)
z_blend = (1 - weights)[:, None] * z_a_rs + weights[:, None] * z_b_rs

z_blend_t = torch.from_numpy(z_blend.astype(np.float32)).unsqueeze(0).to(device)
with torch.no_grad():
    coeffs_blend = model.decoder(z_blend_t)
    pred_disp = model.reconstruct(coeffs_blend)

R_root_blend, t_root_blend = blend_root_transform(R_root_a, t_root_a, R_root_b, t_root_b, weights)

rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)
pred_local = rest_pose_t.unsqueeze(1) + pred_disp

R_blend_t = torch.from_numpy(R_root_blend).unsqueeze(0).to(device)
t_blend_t = torch.from_numpy(t_root_blend).unsqueeze(0).to(device)
world_pos = torch.matmul(pred_local, R_blend_t.transpose(-1, -2)) + t_blend_t.unsqueeze(-2)

mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)

out_path = os.path.join(root_dir, character, f"LatentBlend_{sequence_a}_to_{sequence_b}_FullMesh_{character}")
np.save(out_path, mesh_vertices)
print(f"Saved latent-space blended motion: {mesh_vertices.shape} -> {out_path}")
print(f"Blend curve: pure {sequence_a} until {BLEND_START_FRAC*100:.0f}%, "
      f"pure {sequence_b} after {BLEND_END_FRAC*100:.0f}% of the clip.")
