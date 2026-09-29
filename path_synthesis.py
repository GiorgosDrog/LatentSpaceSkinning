import argparse
import os
import struct

import numpy as np
import torch

from model import VolumetricModelPCA
from blend_utils import load_bones, resample_frames

ROOT = r"E:\didaktoriko\diffusion_solution"
DATASET = os.path.join(ROOT, "dataset")
N = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def write_pc2(path, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        f.write(b"POINTCACHE2" + bytes([0]) + b"")
        f.write(struct.pack("<iiffi", 1, arr.shape[1], 0.0, 1.0, arr.shape[0]))
        f.write(arr.tobytes())


def world_of_local(local, R, t):
    return np.einsum("fij,fvj->fvi", R, local) + t[:, None, :]


def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros(theta.shape + (3, 3), dtype=np.float64)
    R[..., 0, 0], R[..., 0, 1] = c, -s
    R[..., 1, 0], R[..., 1, 1] = s, c
    R[..., 2, 2] = 1.0
    return R


def new_path(kind, n, radius, total_turn_deg):
    t = np.linspace(0.0, 1.0, n)
    total_turn = np.radians(total_turn_deg)
    if kind == "circle":
        angle = t * total_turn
        pos = radius * np.stack([np.cos(angle) - 1.0, np.sin(angle), np.zeros(n)], axis=-1)
        heading = angle + np.pi / 2
    elif kind == "uturn":
        angle = t * total_turn
        pos = radius * np.stack([np.sin(angle), 1 - np.cos(angle), np.zeros(n)], axis=-1)
        heading = angle
    else:
        raise ValueError(f"unknown path kind: {kind}")
    return pos, heading


@torch.no_grad()
def encode_decode(model, bones_world):
    b = torch.from_numpy(bones_world.astype(np.float32)).unsqueeze(0).to(DEVICE)
    B, F, Bm, _, _ = b.shape
    z = model.attention(model.bone_encoder(b.contiguous().view(B, F, Bm * 12)))
    disp = model.reconstruct(model.decoder(z))[0].cpu().numpy()
    return disp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("character")
    ap.add_argument("locomotion", type=int, help="clip number to take the gait from")
    ap.add_argument("--path", choices=["circle", "uturn"], default="circle")
    ap.add_argument("--radius", type=float, default=150.0, help="path radius, in the same units as the rig (cm-ish)")
    ap.add_argument("--turn", type=float, default=360.0, help="total heading change over the clip, in degrees")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    character = a.character
    ckpt_path = os.path.join(DATASET, a.ckpt) if a.ckpt else os.path.join(DATASET, f"best_volumetric_model_pca_{character}.pth")
    print(f"{character}: gait from clip {a.locomotion}, new path = {a.path} (radius={a.radius}, turn={a.turn} deg)")
    print(f"checkpoint: {ckpt_path}")

    anim_dir = os.path.join(DATASET, character, "animation_data")
    bones_raw = load_bones(anim_dir, f"Armature.{a.locomotion:03d}")
    bones_rs = resample_frames(bones_raw, N)
    R_orig, t_orig = bones_rs[:, 0, :, :3].astype(np.float64), bones_rs[:, 0, :, 3].astype(np.float64)

    net_disp = t_orig[-1, :2] - t_orig[0, :2]
    clip_heading = float(np.arctan2(net_disp[1], net_disp[0])) if np.linalg.norm(net_disp) > 1e-3 else 0.0
    print(f"  clip's own recorded heading: {np.degrees(clip_heading):.1f} deg, net XY displacement {np.linalg.norm(net_disp):.1f}")

    new_pos_xy, new_heading = new_path(a.path, N, a.radius, a.turn)
    new_pos = new_pos_xy + np.array([t_orig[0, 0], t_orig[0, 1], 0.0])
    new_pos[:, 2] = t_orig[:, 2]

    delta_heading = new_heading - clip_heading
    t_new = new_pos

    R_orig_all = bones_rs[..., :3].astype(np.float64)
    t_orig_all = bones_rs[..., 3].astype(np.float64)
    Rz_delta = rotz(delta_heading)
    new_R = np.einsum("fij,fbjk->fbik", Rz_delta, R_orig_all)
    new_t = np.einsum("fij,fbj->fbi", Rz_delta, t_orig_all - t_orig[:, None, :]) + t_new[:, None, :]

    bones_world = bones_rs.copy()
    bones_world[..., :3] = new_R.astype(np.float32)
    bones_world[..., 3] = new_t.astype(np.float32)

    rest = np.load(os.path.join(DATASET, character, "static_data", "rest_pose.npy")).astype(np.float32)
    sd = torch.load(ckpt_path, map_location=DEVICE)
    hidden = sd["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
    layers = sum(1 for k in sd if k.startswith("bone_encoder.lstm.weight_ih_l"))
    bones_n = sd["bone_encoder.input_norm.weight"].shape[0] // 12
    model = VolumetricModelPCA(max_bones=bones_n, hidden_size=hidden, num_layers=layers,
                               pca_mean=sd["pca_mean"].cpu().numpy(), pca_components=sd["pca_components"].cpu().numpy()).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    disp = encode_decode(model, bones_world)
    R_root, t_root = bones_world[:, 0, :, :3], bones_world[:, 0, :, 3]
    result = world_of_local(rest[None] + disp, R_root, t_root)

    out_dir = os.path.join(ROOT, "paper_results", "blend_view", f"{character}_path_{a.path}_{a.locomotion:03d}")
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, "FINAL_synthesis_result.npy")
    np.save(npy_path, result)
    write_pc2(os.path.join(out_dir, "FINAL_synthesis_result.pc2"), result)
    print(f"\nFINAL RESULT: {npy_path}   shape {result.shape}  (frames, vertices, xyz)")
