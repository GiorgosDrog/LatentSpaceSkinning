import numpy as np
import torch

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("models", "preprocessing", "inference", "blendings"):
    sys.path.insert(0, os.path.join(_ROOT, _d))
from model import VolumetricModelPCA

ROOT_BONE_INDEX = 0

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = "alpha"
sequence = "Armature.002"

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
device = "cuda" if torch.cuda.is_available() else "cpu"

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
bones_raw = np.load(os.path.join(animation_dir, f"{sequence}_Bones.npy")).astype(np.float32)

F = bones_raw.shape[0]
max_bones = bones_raw.shape[1] // 12
bones = bones_raw.reshape(F, max_bones, 3, 4)

ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}_synthesis.pth")
state_dict = torch.load(ckpt_path, map_location=device)

pca_mean = state_dict["pca_mean"].cpu().numpy()
pca_components = state_dict["pca_components"].cpu().numpy()

hidden_size = state_dict["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
num_layers = sum(1 for k in state_dict if k.startswith("bone_encoder.lstm.weight_ih_l"))
print(f"Inferred from checkpoint: hidden_size={hidden_size}, num_layers={num_layers}, "
      f"K={pca_components.shape[0]}, V={pca_components.shape[1] // 3}")

model = VolumetricModelPCA(
    max_bones=max_bones,
    hidden_size=hidden_size,
    num_layers=num_layers,
    pca_mean=pca_mean,
    pca_components=pca_components,
).to(device)

model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

bones_t = torch.from_numpy(bones).unsqueeze(0).to(device)
rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)

with torch.no_grad():
    pred_disp = model(bones_t)

pred_local = rest_pose_t.unsqueeze(1) + pred_disp

R_root = torch.from_numpy(bones[:, ROOT_BONE_INDEX, :, :3]).unsqueeze(0).to(device)
t_root = torch.from_numpy(bones[:, ROOT_BONE_INDEX, :, 3]).unsqueeze(0).to(device)

world_pos = torch.matmul(pred_local, R_root.transpose(-1, -2)) + t_root.unsqueeze(-2)

mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)

out_path = os.path.join(root_dir, character, f"{sequence}_Predicted_FullMesh.npy")
np.save(out_path, mesh_vertices)
print(f"Saved predicted mesh: {mesh_vertices.shape} -> {out_path}")
print(f"Compare in Blender against ground truth: "
      f"{os.path.join(animation_dir, f'{sequence}_FullMesh.npy')}")
