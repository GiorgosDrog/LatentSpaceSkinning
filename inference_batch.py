import os
import time
import numpy as np
import torch

from model import VolumetricModelPCA

ROOT_BONE_INDEX = 0

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = "x_bot"

animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
device = "cuda" if torch.cuda.is_available() else "cpu"

sequences = sorted(
    f[:-len("_Bones.npy")] for f in os.listdir(animation_dir)
    if f.endswith("_Bones.npy")
)
print(f"Found {len(sequences)} sequences: {sequences}")

ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}.pth")
state_dict = torch.load(ckpt_path, map_location=device)

pca_mean = state_dict["pca_mean"].cpu().numpy()
pca_components = state_dict["pca_components"].cpu().numpy()
hidden_size = state_dict["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
num_layers = sum(1 for k in state_dict if k.startswith("bone_encoder.lstm.weight_ih_l"))
max_bones = state_dict["bone_encoder.input_norm.weight"].shape[0] // 12
print(f"Inferred from checkpoint: hidden_size={hidden_size}, num_layers={num_layers}, "
      f"K={pca_components.shape[0]}, V={pca_components.shape[1] // 3}, max_bones={max_bones}")

model = VolumetricModelPCA(
    max_bones=max_bones, hidden_size=hidden_size, num_layers=num_layers,
    pca_mean=pca_mean, pca_components=pca_components,
).to(device)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {ckpt_path}\n")

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)


def timed_forward(bones_t):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        pred_disp = model(bones_t)
    if device == "cuda":
        torch.cuda.synchronize()
    return pred_disp, time.perf_counter() - start


if device == "cuda" and sequences:
    warmup_bones_raw = np.load(os.path.join(animation_dir, f"{sequences[0]}_Bones.npy")).astype(np.float32)
    F0 = warmup_bones_raw.shape[0]
    warmup_bones = torch.from_numpy(warmup_bones_raw.reshape(F0, max_bones, 3, 4)).unsqueeze(0).to(device)
    _, _ = timed_forward(warmup_bones)
    print("GPU warmup done.\n")

timings = {}
for i, sequence in enumerate(sequences, 1):
    bones_raw = np.load(os.path.join(animation_dir, f"{sequence}_Bones.npy")).astype(np.float32)
    F = bones_raw.shape[0]
    bones = bones_raw.reshape(F, max_bones, 3, 4)
    bones_t = torch.from_numpy(bones).unsqueeze(0).to(device)

    pred_disp, elapsed = timed_forward(bones_t)
    timings[sequence] = elapsed

    pred_local = rest_pose_t.unsqueeze(1) + pred_disp

    R_root = torch.from_numpy(bones[:, ROOT_BONE_INDEX, :, :3]).unsqueeze(0).to(device)
    t_root = torch.from_numpy(bones[:, ROOT_BONE_INDEX, :, 3]).unsqueeze(0).to(device)
    world_pos = torch.matmul(pred_local, R_root.transpose(-1, -2)) + t_root.unsqueeze(-2)

    mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)
    out_path = os.path.join(root_dir, character, f"{sequence}_Predicted_FullMesh.npy")
    np.save(out_path, mesh_vertices)

    ms_per_frame = 1000.0 * elapsed / F
    print(f"[{i:2d}/{len(sequences)}] {sequence}: {F:4d} frames -> "
          f"{elapsed*1000:7.2f} ms total ({ms_per_frame:.3f} ms/frame) -> {os.path.basename(out_path)}")

values = list(timings.values())
print(f"\n{'='*60}")
print(f"Timed {len(values)} sequences on {device}")
print(f"  total:   {sum(values)*1000:8.2f} ms")
print(f"  mean:    {sum(values)/len(values)*1000:8.2f} ms")
print(f"  min:     {min(values)*1000:8.2f} ms  ({[s for s,v in timings.items() if v==min(values)][0]})")
print(f"  max:     {max(values)*1000:8.2f} ms  ({[s for s,v in timings.items() if v==max(values)][0]})")
print(f"{'='*60}")
