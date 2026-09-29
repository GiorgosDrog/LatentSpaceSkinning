
import numpy as np
import torch
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("models", "preprocessing", "inference", "blendings"):
    sys.path.insert(0, os.path.join(_ROOT, _d))
from scipy.spatial.transform import Rotation

sys.path.insert(0, r"E:\didaktoriko\diffusion_solution\linear_lss")
from lss_linear import relative_bones, DEVICE
from model import VolumetricModelPCA
from transition_net import TransitionNet, blended_latent
from blend_utils import load_bones, resample_frames, blend_weight_curve, blend_root_transform, resampled_root_transform
from find_transition_point import find_best_center

ROOT = r"E:\didaktoriko\diffusion_solution"
DATASET = os.path.join(ROOT, "dataset")
N = 150


def to4(m):
    out = np.zeros(m.shape[:-2] + (4, 4), dtype=np.float64)
    out[..., :3, :] = m
    out[..., 3, 3] = 1.0
    return out


class Skin:
    def __init__(self, character):
        st = os.path.join(DATASET, character, "static_data")
        W = np.load(os.path.join(st, "skinning_weights.npy")).astype(np.float32)
        act = np.where(W.sum(0) > 0)[0]
        self.W = torch.from_numpy(W[:, act]).to(DEVICE)
        Rr = to4(np.load(os.path.join(st, "bone_rest.npy")).astype(np.float64)[act])
        self.Rinv = torch.from_numpy(np.linalg.inv(Rr).astype(np.float32)).to(DEVICE)
        rest = np.load(os.path.join(st, "rest_pose.npy")).astype(np.float32)
        self.rest_h = torch.from_numpy(np.concatenate([rest, np.ones((len(rest), 1), np.float32)], 1)).to(DEVICE)
        self.scale = float(np.linalg.norm(rest.max(0) - rest.min(0)))
        self.V, self.B = self.W.shape

    def world(self, bones):
        T = torch.from_numpy(to4(bones).astype(np.float32)).to(DEVICE)
        S = T @ self.Rinv[None]
        out = []
        for s in range(0, len(T), 16):
            Sf = S[s:s + 16]
            M = torch.matmul(self.W, Sf.reshape(len(Sf), self.B, 16)).reshape(len(Sf), self.V, 4, 4)
            out.append(torch.einsum("fvij,vj->fvi", M, self.rest_h)[..., :3])
        return torch.cat(out, 0).cpu().numpy()


def skeleton_blend(bones_a, bones_b, w, R_blend, t_blend):
    def rel(b):
        T = to4(b)
        return np.einsum("fij,fbjk->fbik", np.linalg.inv(T[:, 0]), T)
    A, Bm = rel(bones_a), rel(bones_b)
    F, B = A.shape[:2]
    rA = Rotation.from_matrix(A[..., :3, :3].reshape(-1, 3, 3))
    rB = Rotation.from_matrix(Bm[..., :3, :3].reshape(-1, 3, 3))
    ww = np.repeat(w, B)
    R = (rA * Rotation.from_rotvec((rA.inv() * rB).as_rotvec() * ww[:, None])).as_matrix().reshape(F, B, 3, 3)
    t = (1 - w)[:, None, None] * A[..., :3, 3] + w[:, None, None] * Bm[..., :3, 3]
    rel_b = np.zeros((F, B, 4, 4))
    rel_b[..., :3, :3], rel_b[..., :3, 3], rel_b[..., 3, 3] = R, t, 1.0
    M = np.zeros((F, 4, 4))
    M[:, :3, :3], M[:, :3, 3], M[:, 3, 3] = R_blend, t_blend, 1.0
    return np.einsum("fij,fbjk->fbik", M, rel_b)[..., :3, :]


def world_of_local(local, R, t):
    return np.einsum("fij,fvj->fvi", R, local) + t[:, None, :]


def local_of_world(world, R, t):
    return np.einsum("fji,fvj->fvi", R, world - t[:, None, :])


def build_from_state(sd):
    hidden = sd["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4
    layers = sum(1 for k in sd if k.startswith("bone_encoder.lstm.weight_ih_l"))
    bones = sd["bone_encoder.input_norm.weight"].shape[0] // 12
    m = VolumetricModelPCA(max_bones=bones, hidden_size=hidden, num_layers=layers,
                           pca_mean=sd["pca_mean"].cpu().numpy(), pca_components=sd["pca_components"].cpu().numpy()).to(DEVICE)
    m.load_state_dict(sd)
    return m.eval(), hidden


@torch.no_grad()
def encode(model, bones):
    b = torch.from_numpy(bones.astype(np.float32)).unsqueeze(0).to(DEVICE)
    B, F, Bm, _, _ = b.shape
    return model.attention(model.bone_encoder(b.contiguous().view(B, F, Bm * 12)))


@torch.no_grad()
def decode_local(model, z, rest):
    disp = model.reconstruct(model.decoder(z))[0].cpu().numpy()
    return rest[None] + disp


def metrics(world, scale, w, ref_loc, R, t):
    z = world[..., 2]
    thr = 0.008 * scale
    contact = (z[:-1] < thr) & (z[1:] < thr)
    dxy = np.linalg.norm(world[1:, :, :2] - world[:-1, :, :2], axis=-1)
    cnt = contact.sum(1)
    per_frame = np.divide((dxy * contact).sum(1), cnt, out=np.zeros(len(cnt)), where=cnt > 0)
    skate_worst = per_frame.max() / scale * 100
    loc = local_of_world(world, R, t)
    dist = 100 * np.linalg.norm(loc - ref_loc, axis=-1).mean() / scale
    return skate_worst, dist


def run_blend(character, ia, ib, base, tmodel, tnet, rest_t, skin, start_frac, end_frac):
    anim_dir = os.path.join(DATASET, character, "animation_data")
    ba = resample_frames(load_bones(anim_dir, f"Armature.{ia:03d}"), N)
    bb = resample_frames(load_bones(anim_dir, f"Armature.{ib:03d}"), N)
    Ra, ta = resampled_root_transform(ba, 0, N)
    Rb, tb = resampled_root_transform(bb, 0, N)
    w = blend_weight_curve(N, start_frac, end_frac)
    Rbl, tbl = blend_root_transform(Ra, ta, Rb, tb, w)

    za, zb = encode(tmodel, ba), encode(tmodel, bb)
    wt = torch.from_numpy(w).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_blend = blended_latent(za, zb, wt, tnet)
    world = world_of_local(decode_local(tmodel, z_blend, rest_t), Rbl, tbl)

    bones_bl = skeleton_blend(ba, bb, w, Rbl, tbl)
    ref_world = skin.world(bones_bl)
    ref_loc = local_of_world(ref_world, Rbl, tbl)

    return metrics(world, skin.scale, w, ref_loc, Rbl, tbl)


if __name__ == "__main__":
    character = sys.argv[1] if len(sys.argv) > 1 else "amy"
    ia = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    ib = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    ck = os.path.join(DATASET, f"best_transition_{character}.pth")
    bundle = torch.load(ck, map_location=DEVICE)
    tmodel, hidden = build_from_state(bundle["model"])
    tnet = TransitionNet(hidden_size=hidden).to(DEVICE)
    tnet.load_state_dict(bundle["transition_net"])
    tnet.eval()
    rest_t = np.load(os.path.join(DATASET, character, "static_data", "rest_pose.npy")).astype(np.float32)
    skin = Skin(character)

    anim_dir = os.path.join(DATASET, character, "animation_data")
    ba = resample_frames(load_bones(anim_dir, f"Armature.{ia:03d}"), N)
    bb = resample_frames(load_bones(anim_dir, f"Armature.{ib:03d}"), N)
    best_frac, dist = find_best_center(ba, bb)
    print(f"{character}: clip {ia} vs {ib} -- fixed center t=0.50 ({dist[75]:.1f} deg) vs best center t={best_frac:.2f} ({dist[int(best_frac * N)]:.1f} deg)")

    fixed = run_blend(character, ia, ib, None, tmodel, tnet, rest_t, skin, 0.25, 0.75)
    s0, e0 = max(0.0, best_frac - 0.25), min(1.0, best_frac + 0.25)
    smart = run_blend(character, ia, ib, None, tmodel, tnet, rest_t, skin, s0, e0)

    print(f"\n{'':30s} {'skate_worst':>12s} {'dist_from_skel_blend':>22s}")
    print(f"{'fixed window (0.25-0.75)':30s} {fixed[0]:12.3f} {fixed[1]:22.3f}")
    print(f"{'smart window (' + f'{s0:.2f}-{e0:.2f}' + ')':30s} {smart[0]:12.3f} {smart[1]:22.3f}")
