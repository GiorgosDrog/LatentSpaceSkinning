import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, r"E:\didaktoriko\diffusion_solution\linear_lss")
from model import VolumetricModelPCA
from blend_utils import load_bones, resample_frames, resampled_root_transform
from lss_linear import relative_bones, DEVICE

ROOT = r"E:\didaktoriko\diffusion_solution"
DATASET = os.path.join(ROOT, "dataset")
N = 150

UPPER_PATTERNS = ("Spine", "Neck", "Head", "Shoulder", "Arm", "Hand")
LOWER_PATTERNS = ("Hips", "UpLeg", "Leg", "Foot", "Toe")


def write_pc2(path, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        f.write(b"POINTCACHE2" + bytes([0]) + b"")
        f.write(struct.pack("<iiffi", 1, arr.shape[1], 0.0, 1.0, arr.shape[0]))
        f.write(arr.tobytes())


def world_of_local(local, R, t):
    return np.einsum("fij,fvj->fvi", R, local) + t[:, None, :]


def bone_groups(character):
    static = os.path.join(DATASET, character, "static_data")
    names = np.load(os.path.join(static, "bone_names.npy"), allow_pickle=True)
    W = np.load(os.path.join(static, "skinning_weights.npy"))
    active = np.where(W.sum(0) > 0)[0]
    upper = np.zeros(len(active), dtype=bool)
    for pos, orig_idx in enumerate(active):
        name = str(names[orig_idx])
        if any(p in name for p in LOWER_PATTERNS):
            upper[pos] = False
        elif any(p in name for p in UPPER_PATTERNS):
            upper[pos] = True
        else:
            print(f"  [warn] unclassified bone '{name}' at active position {pos} -> defaulting to lower body")
            upper[pos] = False
    assert not upper[0], "position 0 (root) ended up in the upper-body group -- check bone_names.npy for this character"
    return upper


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
    ap.add_argument("gesture", type=int, help="clip number: upper body (e.g. waving)")
    ap.add_argument("locomotion", type=int, help="clip number: lower body + root (e.g. walking, turning)")
    ap.add_argument("--ckpt", default=None, help="checkpoint filename inside dataset/ (default: best_volumetric_model_pca_<character>.pth)")
    a = ap.parse_args()
    character, gesture_id, locomotion_id = a.character, a.gesture, a.locomotion
    ckpt_path = os.path.join(DATASET, a.ckpt) if a.ckpt else os.path.join(DATASET, f"best_volumetric_model_pca_{character}.pth")
    print(f"{character}: gesture (upper body) <- clip {gesture_id}   |   locomotion (lower body + root) <- clip {locomotion_id}")
    print(f"checkpoint: {ckpt_path}")

    rest = np.load(os.path.join(DATASET, character, "static_data", "rest_pose.npy")).astype(np.float32)
    upper = bone_groups(character)
    print(f"  {upper.sum()} upper-body bones (gesture), {len(upper) - upper.sum()} lower-body/root bones (locomotion)")

    anim_dir = os.path.join(DATASET, character, "animation_data")
    bg = resample_frames(load_bones(anim_dir, f"Armature.{gesture_id:03d}"), N)
    bl = resample_frames(load_bones(anim_dir, f"Armature.{locomotion_id:03d}"), N)

    rel_g = relative_bones(bg)
    rel_l = relative_bones(bl)
    rel_composite = np.where(upper[None, :, None, None], rel_g, rel_l).astype(np.float32)
    Rl, tl = resampled_root_transform(bl, 0, N)

    Rl4 = np.zeros((N, 4, 4)); Rl4[:, :3, :3], Rl4[:, :3, 3], Rl4[:, 3, 3] = Rl, tl, 1.0
    rel4 = np.zeros((N, rel_composite.shape[1], 4, 4)); rel4[..., :3, :] = rel_composite; rel4[..., 3, 3] = 1.0
    bones_composite_world = np.einsum("fij,fbjk->fbik", Rl4, rel4)[..., :3, :].astype(np.float32)

    sd = torch.load(ckpt_path, map_location=DEVICE)
    hidden = sd["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
    layers = sum(1 for k in sd if k.startswith("bone_encoder.lstm.weight_ih_l"))
    bones_n = sd["bone_encoder.input_norm.weight"].shape[0] // 12
    model = VolumetricModelPCA(max_bones=bones_n, hidden_size=hidden, num_layers=layers,
                               pca_mean=sd["pca_mean"].cpu().numpy(), pca_components=sd["pca_components"].cpu().numpy()).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    disp = encode_decode(model, bones_composite_world)
    result = world_of_local(rest[None] + disp, Rl, tl)

    out_dir = os.path.join(ROOT, "paper_results", "blend_view", f"{character}_composite_g{gesture_id:03d}_l{locomotion_id:03d}")
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, "FINAL_synthesis_result.npy")
    np.save(npy_path, result)
    write_pc2(os.path.join(out_dir, "FINAL_synthesis_result.pc2"), result)
    print(f"\nFINAL RESULT: {npy_path}   shape {result.shape}  (frames, vertices, xyz)")
