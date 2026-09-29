import itertools
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
    blend_root_transform, encode_latent, ROOT_BONE_INDEX,
)

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = "alpha"

EXCLUDED_SEQUENCES = {7, 15}
TRAIN_SEQUENCES = [f"Armature.{i:03d}" for i in range(1, 18) if i not in EXCLUDED_SEQUENCES]

NUM_OUTPUT_FRAMES = 150
BLEND_START_FRAC = 0.25
BLEND_END_FRAC = 0.75
FLOOR_Z = 0.0
FOOTSKATE_Z_THRESH = 2.0

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}_synthesis.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"{ckpt_path} not found -- run synthesis_finetune.py first.")
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
print(f"Loaded: {ckpt_path}")
print(f"Screening {len(TRAIN_SEQUENCES)} sequences -> "
      f"{len(TRAIN_SEQUENCES) * (len(TRAIN_SEQUENCES) - 1) // 2} pairs.\n")

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)
bbox_diag = float(np.linalg.norm(rest_pose.max(0) - rest_pose.min(0)))

cache = {}
for seq in TRAIN_SEQUENCES:
    bones = load_bones(animation_dir, seq)
    z_native = encode_latent(model, bones, device).cpu().numpy()
    z_rs = resample_frames(z_native, NUM_OUTPUT_FRAMES)
    R_rs = np.stack([orthonormalize(r) for r in
                      resample_frames(bones[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
    t_rs = resample_frames(bones[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)
    cache[seq] = {"z_rs": torch.from_numpy(z_rs).to(device), "R_rs": R_rs, "t_rs": t_rs}
print(f"Cached {len(cache)} sequences.\n")

w_np = blend_weight_curve(NUM_OUTPUT_FRAMES, BLEND_START_FRAC, BLEND_END_FRAC)
w_t = torch.from_numpy(w_np).to(device)


@torch.no_grad()
def score_pair(seq_a, seq_b):
    z_a, z_b = cache[seq_a]["z_rs"], cache[seq_b]["z_rs"]
    z_blend = (1 - w_t).unsqueeze(-1) * z_a + w_t.unsqueeze(-1) * z_b
    coeffs = model.decoder(z_blend.unsqueeze(0))
    mesh_local = model.reconstruct(coeffs)

    c = coeffs.squeeze(0)
    vel = (c[1:] - c[:-1]).pow(2).sum(dim=-1).mean().item()
    acc = (c[2:] - 2 * c[1:-1] + c[:-2]).pow(2).sum(dim=-1).mean().item()
    roughness = vel + 0.1 * acc

    R_blend, t_blend = blend_root_transform(cache[seq_a]["R_rs"], cache[seq_a]["t_rs"],
                                             cache[seq_b]["R_rs"], cache[seq_b]["t_rs"], w_np)
    R_blend_t = torch.from_numpy(R_blend).unsqueeze(0).to(device)
    t_blend_t = torch.from_numpy(t_blend).unsqueeze(0).to(device)
    pred_local = rest_pose_t.unsqueeze(1) + mesh_local
    world_pos = torch.matmul(pred_local, R_blend_t.transpose(-1, -2)) + t_blend_t.unsqueeze(-2)
    world_np = world_pos.squeeze(0).cpu().numpy()

    z = world_np[:, :, 2]
    floor_violation = float(np.clip(FLOOR_Z - z, a_min=0.0, a_max=None).max())

    contact = (z[:-1] < FOOTSKATE_Z_THRESH) & (z[1:] < FOOTSKATE_Z_THRESH)
    dxy = np.linalg.norm(world_np[1:, :, :2] - world_np[:-1, :, :2], axis=-1)
    contact_count = contact.sum(axis=1)
    per_frame_drift = np.divide(
        (dxy * contact).sum(axis=1), contact_count,
        out=np.zeros(contact_count.shape), where=contact_count > 0,
    )
    footskate_mean = float(dxy[contact].mean()) if contact.any() else 0.0
    footskate_max = float(per_frame_drift.max()) if contact.any() else 0.0

    return roughness, footskate_mean, footskate_max, floor_violation


results = []
pairs = list(itertools.combinations(TRAIN_SEQUENCES, 2))
for i, (seq_a, seq_b) in enumerate(pairs, 1):
    roughness, footskate_mean, footskate_max, floor_violation = score_pair(seq_a, seq_b)
    results.append((seq_a, seq_b, roughness, footskate_mean, footskate_max, floor_violation))
    if i % 20 == 0 or i == len(pairs):
        print(f"  scored {i}/{len(pairs)} pairs...")

results.sort(key=lambda r: (r[4], r[2], r[5]))

print(f"\n{'='*100}")
print(f"BEST (least foot-sliding, worst-frame) 10 pairs for {character} -- good candidates for the paper:")
print(f"{'='*100}")
print(f"{'seq_a':>14} {'seq_b':>14} {'roughness':>12} {'skate_mean':>12} {'skate_MAX':>12} {'floor_viol':>12}")
for seq_a, seq_b, r, sm, sx, f in results[:10]:
    print(f"{seq_a:>14} {seq_b:>14} {r:12.4f} {sm:12.4f} {sx:12.4f} {f:12.4f}")

print(f"\n{'='*100}")
print(f"WORST (most foot-sliding, worst-frame) 10 pairs -- avoid, or verify with geodesic_blend_4point.py first:")
print(f"{'='*100}")
for seq_a, seq_b, r, sm, sx, f in results[-10:]:
    print(f"{seq_a:>14} {seq_b:>14} {r:12.4f} {sm:12.4f} {sx:12.4f} {f:12.4f}")

all_roughness = np.array([r[2] for r in results])
all_skate_mean = np.array([r[3] for r in results])
all_skate_max = np.array([r[4] for r in results])
all_floor = np.array([r[5] for r in results])
print(f"\n{'='*100}")
print(f"Summary over all {len(results)} pairs:")
print(f"  roughness:  mean={all_roughness.mean():.4f}  median={np.median(all_roughness):.4f}  "
      f"min={all_roughness.min():.4f}  max={all_roughness.max():.4f}")
print(f"  skate_mean: mean={all_skate_mean.mean():.4f}  median={np.median(all_skate_mean):.4f}  "
      f"min={all_skate_mean.min():.4f}  max={all_skate_mean.max():.4f}")
print(f"  skate_MAX:  mean={all_skate_max.mean():.4f}  median={np.median(all_skate_max):.4f}  "
      f"min={all_skate_max.min():.4f}  max={all_skate_max.max():.4f}")
print(f"  floor_viol: mean={all_floor.mean():.4f}  median={np.median(all_floor):.4f}  "
      f"min={all_floor.min():.4f}  max={all_floor.max():.4f}")
print(f"{'='*100}")
