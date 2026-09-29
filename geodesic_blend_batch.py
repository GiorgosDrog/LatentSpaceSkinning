import os
import numpy as np
import torch

from model import VolumetricModelPCA
from blend_utils import (
    load_bones, resample_frames, orthonormalize, blend_weight_curve,
    blend_root_transform, encode_latent, ROOT_BONE_INDEX,
)

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = "alpha"

PAIR_NUMBERS = [(1, 2), (3, 6), (8, 9), (2, 3),(4, 8), (6, 11)]

NUM_OUTPUT_FRAMES = 150
BLEND_START_FRAC = 0.25
BLEND_END_FRAC = 0.75

OPT_STEPS = 250
OPT_LR = 0.05
W_VELOCITY = 1.0
W_ACCELERATION = 0.1

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
output_dir = os.path.join(root_dir, character, "blends")
device = "cuda" if torch.cuda.is_available() else "cpu"

SEQUENCE_PAIRS = [(f"Armature.{a:03d}", f"Armature.{b:03d}") for a, b in PAIR_NUMBERS]

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
rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)

sequences_needed = sorted({seq for pair in SEQUENCE_PAIRS for seq in pair})
cache = {}
for seq in sequences_needed:
    bones = load_bones(animation_dir, seq)
    assert bones.shape[1] == max_bones, f"{seq} has a different bone count -- not the same rig"
    z_native = encode_latent(model, bones, device).cpu().numpy()
    z_rs = torch.from_numpy(resample_frames(z_native, NUM_OUTPUT_FRAMES)).to(device)
    R_rs = np.stack([orthonormalize(r) for r in
                      resample_frames(bones[:, ROOT_BONE_INDEX, :, :3], NUM_OUTPUT_FRAMES)])
    t_rs = resample_frames(bones[:, ROOT_BONE_INDEX, :, 3], NUM_OUTPUT_FRAMES)
    cache[seq] = {"z_rs": z_rs, "R_rs": R_rs, "t_rs": t_rs}
    print(f"Cached {seq}: latent {z_native.shape} -> resampled to {NUM_OUTPUT_FRAMES}")

def curve_to_theta(w):
    d = np.diff(w).clip(min=1e-6)
    d = d / d.sum()
    return np.log(d)


w_init_np = blend_weight_curve(NUM_OUTPUT_FRAMES, BLEND_START_FRAC, BLEND_END_FRAC)
theta_init_np = curve_to_theta(w_init_np)


def theta_to_w(theta):
    d = torch.softmax(theta, dim=0)
    return torch.cat([torch.zeros(1, device=theta.device), torch.cumsum(d, dim=0)])


def decoded_coeffs(w, z_a_rs, z_b_rs):
    z_blend = (1 - w).unsqueeze(-1) * z_a_rs + w.unsqueeze(-1) * z_b_rs
    return model.decoder(z_blend.unsqueeze(0)).squeeze(0)


def roughness_loss(c):
    vel = (c[1:] - c[:-1]).pow(2).sum(dim=-1).mean()
    acc = (c[2:] - 2 * c[1:-1] + c[:-2]).pow(2).sum(dim=-1).mean()
    return W_VELOCITY * vel + W_ACCELERATION * acc


def search_geodesic_curve(z_a_rs, z_b_rs):
    theta = torch.tensor(theta_init_np, dtype=torch.float32, device=device, requires_grad=True)
    with torch.no_grad():
        baseline = roughness_loss(decoded_coeffs(theta_to_w(theta), z_a_rs, z_b_rs)).item()

    optimizer = torch.optim.Adam([theta], lr=OPT_LR)
    for _ in range(OPT_STEPS):
        optimizer.zero_grad()
        loss = roughness_loss(decoded_coeffs(theta_to_w(theta), z_a_rs, z_b_rs))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        w_final = theta_to_w(theta)
        final = roughness_loss(decoded_coeffs(w_final, z_a_rs, z_b_rs)).item()
    improvement = 100.0 * (1.0 - final / baseline)
    return w_final, baseline, final, improvement


os.makedirs(output_dir, exist_ok=True)
print(f"\nRunning geodesic search for {len(SEQUENCE_PAIRS)} pairs -> {output_dir}\n")

for i, (seq_a, seq_b) in enumerate(SEQUENCE_PAIRS, 1):
    a, b = cache[seq_a], cache[seq_b]

    w_final, baseline, final, improvement = search_geodesic_curve(a["z_rs"], b["z_rs"])
    print(f"[{i}/{len(SEQUENCE_PAIRS)}] {seq_a} -> {seq_b}: "
          f"roughness {baseline:.1f} -> {final:.1f} ({improvement:+.1f}%)")

    with torch.no_grad():
        coeffs_blend = decoded_coeffs(w_final, a["z_rs"], b["z_rs"]).unsqueeze(0)
        pred_disp = model.reconstruct(coeffs_blend)

    w_final_np = w_final.cpu().numpy()
    R_blend, t_blend = blend_root_transform(a["R_rs"], a["t_rs"], b["R_rs"], b["t_rs"], w_final_np)

    pred_local = rest_pose_t.unsqueeze(1) + pred_disp
    R_blend_t = torch.from_numpy(R_blend).unsqueeze(0).to(device)
    t_blend_t = torch.from_numpy(t_blend).unsqueeze(0).to(device)
    world_pos = torch.matmul(pred_local, R_blend_t.transpose(-1, -2)) + t_blend_t.unsqueeze(-2)

    mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)
    out_path = os.path.join(output_dir, f"GeodesicBlend_{seq_a}_to_{seq_b}_FullMesh_{character}.npy")
    np.save(out_path, mesh_vertices)
    print(f"    saved {mesh_vertices.shape} -> {os.path.basename(out_path)}")

print(f"\nDone. {len(SEQUENCE_PAIRS)} geodesic-blended clips saved to {output_dir}")
