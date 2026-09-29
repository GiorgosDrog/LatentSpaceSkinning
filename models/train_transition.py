import random
import numpy as np
import torch
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("models", "preprocessing", "inference", "blendings"):
    sys.path.insert(0, os.path.join(_ROOT, _d))
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from model import VolumetricModelPCA
from transition_net import TransitionNet, blended_latent
from dataloader import PCAMeshSequenceDataset, root_relative_mesh
from loss_functions import (
    make_total_loss, build_knn_laplacian, evaluate_metrics,
    contact_reference_velocity_loss, foot_velocity_supervision_loss,
    floor_penalty_from_world_pos, smoothness_loss_world, curvature_consistency_loss,
)
from blend_utils import (
    load_bones, resample_frames, blend_weight_curve, blend_root_transform,
    resampled_root_transform, ROOT_BONE_INDEX, foot_vertex_mask,
)
from find_transition_point import find_best_center

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = "alpha"
animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")
if(character == "monster"):
    EXCLUDED_SEQUENCES = {}
    train_dataset = [
        os.path.join(animation_dir, f"Armature.{str(i).zfill(3)}")
        for i in range(1, 10)
    ]

    val_dataset = [
        os.path.join(animation_dir, f"Armature.{i:03d}") for i in [2,5]
    ]

    TRAIN_SEQUENCES = [f"Armature.{i:03d}" for i in range(1, 10)]
    MONITOR_SEQUENCES = ["Armature.002","Armature.004", "Armature.009"]
else:
    EXCLUDED_SEQUENCES = {7,15}
    train_dataset = [
        os.path.join(animation_dir, f"Armature.{str(i).zfill(3)}")
        for i in range(1, 18) if i not in EXCLUDED_SEQUENCES
    ]

    val_dataset = [
        os.path.join(animation_dir, f"Armature.{i:03d}") for i in [2,10,16] if i not in EXCLUDED_SEQUENCES
    ]

    TRAIN_SEQUENCES = [f"Armature.{i:03d}" for i in range(1, 18) if i not in EXCLUDED_SEQUENCES]
    MONITOR_SEQUENCES = ["Armature.002","Armature.004", "Armature.009", "Armature.013"]

NUM_OUTPUT_FRAMES = 150
EPOCHS = 2000
STEPS_PER_EPOCH = 20
EVAL_EVERY = 10
W_DELTA_MAGNITUDE = 0.05
W_CONTACT_REF = 50
W_FOOT_REPLAY = 0.5

_eval_rng = random.Random(1234)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

synthesis_ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}_synthesis.pth")
base_ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}.pth")
start_ckpt_path = synthesis_ckpt_path if os.path.exists(synthesis_ckpt_path) else base_ckpt_path

state_dict = torch.load(start_ckpt_path, map_location=device)
pca_mean = state_dict["pca_mean"].cpu().numpy()
pca_components = state_dict["pca_components"].cpu().numpy()
hidden_size = state_dict["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
num_layers = sum(1 for k in state_dict if k.startswith("bone_encoder.lstm.weight_ih_l"))
max_bones = state_dict["bone_encoder.input_norm.weight"].shape[0] // 12

model = VolumetricModelPCA(max_bones=max_bones, hidden_size=hidden_size, num_layers=num_layers,
                            pca_mean=pca_mean, pca_components=pca_components).to(device)
model.load_state_dict(state_dict)
print(f"Loaded starting checkpoint: {start_ckpt_path}")

transition_net = TransitionNet(hidden_size=hidden_size).to(device)
print("TransitionNet created (zero-init head -- starts as an exact no-op).")

for p in model.bone_encoder.parameters():
    p.requires_grad_(False)
for p in model.attention.parameters():
    p.requires_grad_(False)
print("Frozen: bone_encoder, attention. Trainable: decoder + transition_net.")

rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
bbox_diag = float(np.linalg.norm(rest_pose.max(axis=0) - rest_pose.min(axis=0)))
rest_pose_t = torch.from_numpy(rest_pose).unsqueeze(0).to(device)
scale_t = torch.tensor([bbox_diag], device=device)

foot_mask_np = foot_vertex_mask(root_dir, character)
foot_mask_t = torch.from_numpy(foot_mask_np).to(device) if foot_mask_np is not None else None
if foot_mask_t is not None:
    print(f"Foot vertices: {int(foot_mask_np.sum())} / {len(foot_mask_np)}")
else:
    print("[warn] no foot/toe bones found by name -- contact losses will use ALL vertices")

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

FOOTSKATE_EVAL_PAIRS = [tuple(_eval_rng.sample(TRAIN_SEQUENCES, 2)) for _ in range(6)]
print(f"Fixed foot-skate evaluation pairs: {FOOTSKATE_EVAL_PAIRS}")

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
SYNTH_W_FLOOR = 100
SYNTH_W_SMOOTH = 0.02
SYNTH_W_CURVATURE = 0.05
SYNTH_W_ANCHOR = 0.05
SYNTH_W_ACCEL = 2

optimizer = torch.optim.AdamW(
    list(model.decoder.parameters()) + list(transition_net.parameters()),
    lr=1e-4, weight_decay=1e-5,
)


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

    if trainable:
        z_blend = blended_latent(z_a.unsqueeze(0), z_b.unsqueeze(0), weights.unsqueeze(0),
                                  transition_net).squeeze(0)
        c_blend = model.decoder(z_blend.unsqueeze(0))
    else:
        with torch.no_grad():
            z_blend = blended_latent(z_a.unsqueeze(0), z_b.unsqueeze(0), weights.unsqueeze(0),
                                      transition_net).squeeze(0)
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

    R_a_t = torch.from_numpy(R_a).unsqueeze(0).to(device)
    t_a_t = torch.from_numpy(t_a).unsqueeze(0).to(device)
    R_b_t = torch.from_numpy(R_b).unsqueeze(0).to(device)
    t_b_t = torch.from_numpy(t_b).unsqueeze(0).to(device)
    pred_a_local = rest_pose_t.unsqueeze(1) + mesh_a_local
    pred_b_local = rest_pose_t.unsqueeze(1) + mesh_b_local
    world_a = torch.matmul(pred_a_local, R_a_t.transpose(-1, -2)) + t_a_t.unsqueeze(-2)
    world_b = torch.matmul(pred_b_local, R_b_t.transpose(-1, -2)) + t_b_t.unsqueeze(-2)

    with torch.no_grad() if not trainable else torch.enable_grad():
        raw_delta = transition_net(z_a.unsqueeze(0), z_b.unsqueeze(0), weights.unsqueeze(0))

    return {
        "world_pos": world_pos, "c_blend": c_blend, "c_anchor": c_anchor,
        "mesh_blend_local": mesh_blend_local, "mesh_a_local": mesh_a_local,
        "mesh_b_local": mesh_b_local, "weights": weights, "raw_delta": raw_delta,
        "world_a": world_a, "world_b": world_b,
    }


def synthesis_step(verbose=False):
    seq_a, seq_b = random.sample(TRAIN_SEQUENCES, 2)
    best_frac, _ = find_best_center(cache[seq_a]["bones_rs"], cache[seq_b]["bones_rs"], lo=0.1, hi=0.9)
    width = random.uniform(0.3, 0.6)
    start_frac = max(0.02, best_frac - width / 2)
    end_frac = min(0.98, best_frac + width / 2)
    weights_np = blend_weight_curve(NUM_OUTPUT_FRAMES, start_frac, end_frac)

    out = blend_world_pos(seq_a, seq_b, weights_np, trainable=True)

    scale_r = scale_t.view(-1, 1, 1).clamp_min(1e-6)
    floor = floor_penalty_from_world_pos(out["world_pos"], scale_t, 0.0)
    smooth = smoothness_loss_world(out["world_pos"], scale_t)
    anchor = (((out["c_blend"] - out["c_anchor"]) / scale_r) ** 2).mean()
    curvature = curvature_consistency_loss(
        out["mesh_blend_local"], out["mesh_a_local"], out["mesh_b_local"], out["weights"], laplacian)
    delta_mag = (out["raw_delta"] / scale_t.view(-1, 1, 1)).pow(2).mean()
    contact_ref = contact_reference_velocity_loss(
        out["world_pos"], out["world_a"], out["world_b"], out["weights"], scale_t,
        foot_mask=foot_mask_t)
    c = out["c_blend"] / scale_r
    accel = (c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]).pow(2).mean()

    terms = {
        "floor": SYNTH_W_FLOOR * floor, "smooth": SYNTH_W_SMOOTH * smooth,
        "anchor": SYNTH_W_ANCHOR * anchor, "curvature": SYNTH_W_CURVATURE * curvature,
        "delta_mag": W_DELTA_MAGNITUDE * delta_mag, "contact_ref": W_CONTACT_REF * contact_ref,
        "accel": SYNTH_W_ACCEL * accel,
    }
    if verbose:
        print("    synth terms (weighted): " + " | ".join(f"{k}={v.item():.4e}" for k, v in terms.items()))
        print("    synth terms (raw, unweighted): "
              f"floor={floor.item():.4e} | smooth={smooth.item():.4e} | anchor={anchor.item():.4e} | "
              f"curvature={curvature.item():.4e} | delta_mag={delta_mag.item():.4e} | "
              f"contact_ref={contact_ref.item():.4e} | accel={accel.item():.4e}")
    return sum(terms.values())


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
    loss = replay_loss_fn(pred, target, sc, rp, root_R, root_t)

    pred_local = rp.unsqueeze(1) + pred
    world_pred = torch.matmul(pred_local, root_R.transpose(-1, -2)) + root_t.unsqueeze(-2)
    world_gt = torch.matmul(mesh_vertices, root_R.transpose(-1, -2)) + root_t.unsqueeze(-2)
    foot_loss = foot_velocity_supervision_loss(world_pred, world_gt, sc, foot_mask=foot_mask_t)

    return loss + W_FOOT_REPLAY * foot_loss


@torch.no_grad()
def eval_held_out_reconstruction():
    total = 0.0
    for seq in MONITOR_SEQUENCES:
        bones = load_bones(animation_dir, seq)
        bones_t = torch.from_numpy(bones).unsqueeze(0).to(device)
        mesh = np.load(os.path.join(animation_dir, f"{seq}_FullMesh.npy")).astype(np.float32)
        mesh_local = root_relative_mesh(mesh, bones)
        target = torch.from_numpy(mesh_local - rest_pose[None]).unsqueeze(0).to(device)
        pred = model(bones_t)
        total += evaluate_metrics(pred, target, scale_t)["DisPer"]
    return total / len(MONITOR_SEQUENCES)


@torch.no_grad()
def eval_synthetic_quality(z_thresh=2.0):
    skate_vals, rough_vals = [], []
    for seq_a, seq_b in FOOTSKATE_EVAL_PAIRS:
        best_frac, _ = find_best_center(cache[seq_a]["bones_rs"], cache[seq_b]["bones_rs"], lo=0.1, hi=0.9)
        s0, e0 = max(0.02, best_frac - 0.225), min(0.98, best_frac + 0.225)
        weights_np = blend_weight_curve(NUM_OUTPUT_FRAMES, s0, e0)
        out = blend_world_pos(seq_a, seq_b, weights_np, trainable=False)
        world_np = out["world_pos"].squeeze(0).cpu().numpy()
        if foot_mask_np is not None:
            world_np_skate = world_np[:, foot_mask_np]
        else:
            world_np_skate = world_np
        z = world_np_skate[:, :, 2]
        contact = (z[:-1] < z_thresh) & (z[1:] < z_thresh)
        dxy = np.linalg.norm(world_np_skate[1:, :, :2] - world_np_skate[:-1, :, :2], axis=-1)
        contact_count = contact.sum(axis=1)
        per_frame_drift = np.divide(
            (dxy * contact).sum(axis=1), contact_count,
            out=np.zeros(contact_count.shape), where=contact_count > 0,
        )
        if contact.any():
            skate_vals.append(float(per_frame_drift.max()))

        c = out["c_blend"].squeeze(0).cpu().numpy()
        vel = ((c[1:] - c[:-1]) ** 2).sum(axis=-1).mean()
        acc = ((c[2:] - 2 * c[1:-1] + c[:-2]) ** 2).sum(axis=-1).mean()
        rough_vals.append(float(vel + 0.1 * acc))

    skate = float(np.mean(skate_vals)) if skate_vals else 0.0
    rough = float(np.mean(rough_vals))
    return skate, rough


ckpt_path = os.path.join(root_dir, f"best_transition_{character}.pth")
best_skate, base_roughness = eval_synthetic_quality()
base_dis_per = eval_held_out_reconstruction()
print(f"Baseline (before transition training): held-out DisPer={base_dis_per:.3f}% | "
      f"synth foot-skate(worst-frame)={best_skate:.3f} | roughness={base_roughness:.1f}")

saved_any = False
for epoch in range(1, EPOCHS + 1):
    model.decoder.train()
    transition_net.train()
    total_loss = 0.0
    for step in range(STEPS_PER_EPOCH):
        is_synthesis_step = (step % 2 == 0)
        optimizer.zero_grad()
        verbose = is_synthesis_step and step == 0 and (epoch % EVAL_EVERY == 0 or epoch == 1)
        loss = synthesis_step(verbose=verbose) if is_synthesis_step else replay_step()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.decoder.parameters()) + list(transition_net.parameters()), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / STEPS_PER_EPOCH

    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
        model.decoder.eval()
        transition_net.eval()
        dis_per = eval_held_out_reconstruction()
        skate, rough = eval_synthetic_quality()
        print(f"[{epoch:04d}/{EPOCHS}] loss={avg_loss:.5f} | "
              f"held-out DisPer={dis_per:.3f}% (baseline {base_dis_per:.3f}%) | "
              f"synth foot-skate(worst-frame)={skate:.3f} (baseline {best_skate:.3f}) | "
              f"roughness={rough:.1f} (baseline {base_roughness:.1f})")

        if skate < best_skate and dis_per < base_dis_per * 1.5 and rough < base_roughness * 1.5:
            best_skate = skate
            torch.save({
                "model": model.state_dict(),
                "transition_net": transition_net.state_dict(),
                "hidden_size": hidden_size,
            }, ckpt_path)
            saved_any = True
            print(f"  --> Saved new best checkpoint to: {ckpt_path}")

print()
if saved_any:
    print(f"Done. Best checkpoint: {ckpt_path}")
    print("Use motion_blend_transition.py with this checkpoint for T-driven synthesis.")
else:
    print("Done, but NO checkpoint met both acceptance criteria (foot-skate improvement AND "
          "reconstruction within 1.5x baseline) -- nothing was saved. Consider more epochs, "
          "a lower learning rate, or a higher w_anchor/W_DELTA_MAGNITUDE.")
