import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from model import VolumetricModelPCA
from dataloader import PCAMeshSequenceDataset, root_relative_mesh
from loss_functions import make_total_loss, make_synthesis_loss, build_knn_laplacian, evaluate_metrics
from blend_utils import (
    load_bones, resample_frames, blend_weight_curve, blend_root_transform,
    resampled_root_transform, ROOT_BONE_INDEX,
)

root_dir = r"e:/didaktoriko/diffusion_solution/dataset"
character = "x_bot"
animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")


TRAIN_SEQUENCES = [f"Armature.{i:03d}" for i in range(2, 6)]
HELD_OUT_SEQUENCE = "Armature.001"

NUM_OUTPUT_FRAMES = 150
EPOCHS = 40
STEPS_PER_EPOCH = 20
EVAL_EVERY = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


base_ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}.pth")
state_dict = torch.load(base_ckpt_path, map_location=device)
pca_mean = state_dict["pca_mean"].cpu().numpy()
pca_components = state_dict["pca_components"].cpu().numpy()
hidden_size = state_dict["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
num_layers = sum(1 for k in state_dict if k.startswith("bone_encoder.lstm.weight_ih_l"))
max_bones = state_dict["bone_encoder.input_norm.weight"].shape[0] // 12

model = VolumetricModelPCA(max_bones=max_bones, hidden_size=hidden_size, num_layers=num_layers,
                            pca_mean=pca_mean, pca_components=pca_components).to(device)
model.load_state_dict(state_dict)
print(f"Loaded base reconstruction checkpoint: {base_ckpt_path}")

for p in model.bone_encoder.parameters():
    p.requires_grad_(False)
for p in model.attention.parameters():
    p.requires_grad_(False)
print("Frozen: bone_encoder, attention. Trainable: decoder only.")

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
bbox_diag = float(np.linalg.norm(rest_pose.max(axis=0) - rest_pose.min(axis=0)))
rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)
scale_t = torch.tensor([bbox_diag], device=device)

print("Building KNN proximity Laplacian for curvature-consistency (k=8)...")
laplacian = build_knn_laplacian(rest_pose, k=8).to(device)
print(f"bbox_diag={bbox_diag:.2f}")


cache = {}
for seq in TRAIN_SEQUENCES:
    bones = load_bones(animation_dir, seq)
    R_rs, t_rs = resampled_root_transform(bones, ROOT_BONE_INDEX, NUM_OUTPUT_FRAMES)
    bones_rs = resample_frames(bones, NUM_OUTPUT_FRAMES)
    cache[seq] = {"bones_rs": bones_rs, "R_rs": R_rs, "t_rs": t_rs}
print(f"Cached {len(cache)} training sequences: {TRAIN_SEQUENCES}")

def unpadded_collate_fn(batch):
    out = {}
    for key in batch[0]:
        out[key] = default_collate([b[key] for b in batch])
    return out

replay_dataset = PCAMeshSequenceDataset([os.path.join(animation_dir, s) for s in TRAIN_SEQUENCES])
replay_loader = DataLoader(replay_dataset, batch_size=1, shuffle=True, collate_fn=unpadded_collate_fn)
replay_iter = iter(replay_loader)


def next_replay_batch():
    global replay_iter
    try:
        return next(replay_iter)
    except StopIteration:
        replay_iter = iter(replay_loader)
        return next(replay_iter)


replay_loss_fn = make_total_loss(w_vertex=1.0, w_smooth=0.02, w_l1=1.0, w_floor=0.05)
synthesis_loss_fn = make_synthesis_loss(
    w_floor=0.05, w_contact=0.5, w_smooth=0.02, w_curvature=0.05, w_anchor=0.25
)


optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4, weight_decay=1e-5)


def encode_native(seq):
    bones_t = torch.from_numpy(cache[seq]["bones_rs"]).unsqueeze(0).to(device)
    B, F, Bm, _, _ = bones_t.shape
    bones_flat = bones_t.contiguous().view(B, F, Bm * 12)
    with torch.no_grad():
        latent = model.attention(model.bone_encoder(bones_flat))
        coeffs = model.decoder(latent)
    return latent.squeeze(0), coeffs


def blend_world_pos(seq_a, seq_b, weights_np, trainable=True):
    weights = torch.from_numpy(weights_np).to(device)

    z_a, c_a_native = encode_native(seq_a)
    z_b, c_b_native = encode_native(seq_b)

    z_blend = (1 - weights).unsqueeze(-1) * z_a + weights.unsqueeze(-1) * z_b
    if trainable:
        c_blend = model.decoder(z_blend.unsqueeze(0))
    else:
        with torch.no_grad():
            c_blend = model.decoder(z_blend.unsqueeze(0))

    mesh_blend_local = model.reconstruct(c_blend)
    mesh_a_local = model.reconstruct(c_a_native)
    mesh_b_local = model.reconstruct(c_b_native)
    c_anchor = (1 - weights).view(1, -1, 1) * c_a_native + weights.view(1, -1, 1) * c_b_native

    R_a, t_a = cache[seq_a]["R_rs"], cache[seq_a]["t_rs"]
    R_b, t_b = cache[seq_b]["R_rs"], cache[seq_b]["t_rs"]
    R_blend, t_blend = blend_root_transform(R_a, t_a, R_b, t_b, weights_np)
    R_blend_t = torch.from_numpy(R_blend).unsqueeze(0).to(device)
    t_blend_t = torch.from_numpy(t_blend).unsqueeze(0).to(device)

    pred_local = rest_pose_t.unsqueeze(1) + mesh_blend_local
    world_pos = torch.matmul(pred_local, R_blend_t.transpose(-1, -2)) + t_blend_t.unsqueeze(-2)

    return {
        "world_pos": world_pos, "c_blend": c_blend, "c_anchor": c_anchor,
        "mesh_blend_local": mesh_blend_local, "mesh_a_local": mesh_a_local,
        "mesh_b_local": mesh_b_local, "weights": weights,
    }


def synthesis_step():
    seq_a, seq_b = random.sample(TRAIN_SEQUENCES, 2)
    start_frac = random.uniform(0.15, 0.35)
    end_frac = random.uniform(0.65, 0.85)
    weights_np = blend_weight_curve(NUM_OUTPUT_FRAMES, start_frac, end_frac)

    out = blend_world_pos(seq_a, seq_b, weights_np, trainable=True)
    return synthesis_loss_fn(
        out["world_pos"], scale_t, out["c_blend"], out["c_anchor"],
        out["mesh_blend_local"], out["mesh_a_local"], out["mesh_b_local"],
        out["weights"], laplacian,
    )


def replay_step():
    batch = next_replay_batch()
    bones = batch["bone_matrices"].to(device)
    mesh_vertices = batch["mesh_vertices"].to(device)
    rp = batch["rest_pose"].to(device)
    sc = batch["scale"].to(device)
    root_R = bones[:, :, ROOT_BONE_INDEX, :, :3]
    root_t = bones[:, :, ROOT_BONE_INDEX, :, 3]
    target = mesh_vertices - rp.unsqueeze(1)
    pred = model(bones)
    return replay_loss_fn(pred, target, sc, rp, root_R, root_t)


@torch.no_grad()
def eval_held_out_reconstruction():
    bones = load_bones(animation_dir, HELD_OUT_SEQUENCE)
    bones_t = torch.from_numpy(bones).unsqueeze(0).to(device)
    mesh = np.load(os.path.join(animation_dir, f"{HELD_OUT_SEQUENCE}_FullMesh.npy")).astype(np.float32)
    mesh_local = root_relative_mesh(mesh, bones)
    target = torch.from_numpy(mesh_local - rest_pose[None]).unsqueeze(0).to(device)
    pred = model(bones_t)
    return evaluate_metrics(pred, target, scale_t)["DisPer"]


@torch.no_grad()
def eval_synthetic_footskate(n_samples=3, z_thresh=2.0):
    vals = []
    for _ in range(n_samples):
        seq_a, seq_b = random.sample(TRAIN_SEQUENCES, 2)
        weights_np = blend_weight_curve(NUM_OUTPUT_FRAMES, 0.25, 0.75)
        out = blend_world_pos(seq_a, seq_b, weights_np, trainable=False)
        world_np = out["world_pos"].squeeze(0).cpu().numpy()
        z = world_np[:, :, 2]
        contact = (z[:-1] < z_thresh) & (z[1:] < z_thresh)
        dxy = np.linalg.norm(world_np[1:, :, :2] - world_np[:-1, :, :2], axis=-1)
        sample = dxy[contact]
        if sample.size:
            vals.append(float(sample.mean()))
    return float(np.mean(vals)) if vals else 0.0


ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}_synthesis.pth")
best_skate = eval_synthetic_footskate()
base_dis_per = eval_held_out_reconstruction()
print(f"Baseline (before fine-tuning): held-out DisPer={base_dis_per:.3f}% | synth foot-skate={best_skate:.3f}")

saved_any = False
for epoch in range(1, EPOCHS + 1):
    model.decoder.train()
    total_loss = 0.0
    for step in range(STEPS_PER_EPOCH):
        is_synthesis_step = (step % 2 == 0)
        optimizer.zero_grad()
        loss = synthesis_step() if is_synthesis_step else replay_step()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / STEPS_PER_EPOCH

    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
        model.decoder.eval()
        dis_per = eval_held_out_reconstruction()
        skate = eval_synthetic_footskate()
        print(f"[{epoch:04d}/{EPOCHS}] loss={avg_loss:.5f} | "
              f"held-out DisPer={dis_per:.3f}% (baseline {base_dis_per:.3f}%) | "
              f"synth foot-skate={skate:.3f} (baseline {best_skate:.3f})")
        if skate < best_skate and dis_per < base_dis_per * 1.5:
            best_skate = skate
            torch.save(model.state_dict(), ckpt_path)
            saved_any = True
            print(f"  --> Saved new best synthesis checkpoint to: {ckpt_path}")

print()
if saved_any:
    print(f"Done. Best checkpoint: {ckpt_path}")
    print("Use motion_blend_latent.py with this checkpoint for genuine decoder-driven synthesis.")
else:
    print("Done, but NO checkpoint met both acceptance criteria (foot-skate improvement AND "
          "reconstruction within 1.5x baseline) -- nothing was saved. Do not point "
          "motion_blend_latent.py at a checkpoint from this run; it doesn't exist. "
          "Consider more epochs, a lower learning rate, or a higher w_anchor.")
