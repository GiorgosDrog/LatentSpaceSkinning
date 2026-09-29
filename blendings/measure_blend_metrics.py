import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp

from blend_utils import load_bones, resample_frames, blend_weight_curve

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"

METHODS = [
    ("native blend",     "LatentBlend"),
    ("geodesic",         "GeodesicBlend"),
    ("geodesic 4-point", "GeodesicBlend4pt"),
    ("TransitionNet",    "TransitionBlend"),
    ("skeleton blend+LBS (classical)", "SkeletonBlendLBS"),
    ("RMI (Robust Motion In-betweening)", "RMIBlend"),
]


def foot_vertex_mask(character):
    W = np.load(os.path.join(root_dir, character, "static_data", "skinning_weights.npy"))
    names = np.load(os.path.join(root_dir, character, "static_data", "bone_names.npy"), allow_pickle=True)
    foot_bones = [i for i, n in enumerate(names) if "foot" in str(n).lower() or "toe" in str(n).lower()]
    if not foot_bones:
        return None
    dominant = W.argmax(1)
    return np.isin(dominant, foot_bones)


def build_laplacian(character, rest):
    nv = len(rest)
    fp = os.path.join(root_dir, character, "static_data", "faces.npy")
    e = None
    if os.path.exists(fp):
        f = np.load(fp).astype(np.int64)
        if f.max() < nv:
            e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    if e is None:
        _, nn = cKDTree(rest).query(rest, k=7)
        e = np.stack([np.repeat(np.arange(nv), 6), nn[:, 1:].reshape(-1)], 1)
    A = sp.coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    return (sp.diags(1.0 / deg) @ A).tocsr().astype(np.float32)


def curvature(world, P, scale):
    F, V, _ = world.shape
    X = world.transpose(1, 0, 2).reshape(V, F * 3)
    LV = X - P @ X
    return np.linalg.norm(LV.reshape(V, F, 3), axis=-1).mean(0) / scale * 100


def reference_curvature(character, seq_a, seq_b, P, scale, num_frames):
    anim_dir = os.path.join(root_dir, character, "animation_data")
    mesh_a = np.load(os.path.join(anim_dir, f"{seq_a}_FullMesh.npy")).astype(np.float32)
    mesh_b = np.load(os.path.join(anim_dir, f"{seq_b}_FullMesh.npy")).astype(np.float32)
    mesh_a_rs = resample_frames(mesh_a, num_frames)
    mesh_b_rs = resample_frames(mesh_b, num_frames)
    cur_a = curvature(mesh_a_rs, P, scale)
    cur_b = curvature(mesh_b_rs, P, scale)
    w = blend_weight_curve(num_frames, 0.25, 0.75)
    return (1 - w) * cur_a + w * cur_b


def reference_mesh_interp(character, seq_a, seq_b, num_frames):
    anim_dir = os.path.join(root_dir, character, "animation_data")
    mesh_a = np.load(os.path.join(anim_dir, f"{seq_a}_FullMesh.npy")).astype(np.float32)
    mesh_b = np.load(os.path.join(anim_dir, f"{seq_b}_FullMesh.npy")).astype(np.float32)
    mesh_a_rs = resample_frames(mesh_a, num_frames)
    mesh_b_rs = resample_frames(mesh_b, num_frames)
    mesh_a_rel = mesh_a_rs - mesh_a_rs.mean(1, keepdims=True)
    mesh_b_rel = mesh_b_rs - mesh_b_rs.mean(1, keepdims=True)
    w = blend_weight_curve(num_frames, 0.25, 0.75)
    return (1 - w)[:, None, None] * mesh_a_rel + w[:, None, None] * mesh_b_rel


def shape_dev(world, scale, mesh_exp):
    world_rel = world - world.mean(1, keepdims=True)
    dev = np.linalg.norm(world_rel - mesh_exp, axis=-1)
    return float(dev.mean() / scale * 100)


def root_motion_pct(world, scale):
    centroid = world.mean(1)
    path_len = np.linalg.norm(centroid[1:] - centroid[:-1], axis=-1).sum()
    return float(100 * path_len / scale)


def _sliding(world, scale, mask=None):
    z = world[..., 2] if mask is None else world[..., mask, 2]
    thr = 0.008 * scale
    contact = (z[:-1] < thr) & (z[1:] < thr)
    w_xy = world[..., :2] if mask is None else world[..., mask, :2]
    dxy = np.linalg.norm(w_xy[1:] - w_xy[:-1], axis=-1)
    cnt = contact.sum(1)
    per_frame = np.divide((dxy * contact).sum(1), cnt, out=np.zeros(len(cnt)), where=cnt > 0)
    worst = per_frame.max() / scale * 100
    mean = per_frame[cnt > 0].mean() / scale * 100 if (cnt > 0).any() else 0.0
    contact_frac = 100 * float((cnt > 0).mean())
    return worst, mean, contact_frac


def metrics(world, scale, foot_mask=None, laplacian=None, cur_exp=None, mesh_exp=None):
    z = world[..., 2]
    sliding_worst, sliding_mean, _ = _sliding(world, scale, mask=None)
    out = dict(sliding_worst=sliding_worst, sliding_mean=sliding_mean, root_motion_pct=root_motion_pct(world, scale))
    if foot_mask is not None:
        fw, fm, fc = _sliding(world, scale, mask=foot_mask)
        out["foot_sliding_worst"] = fw
        out["foot_sliding_mean"] = fm
        out["foot_contact_frac"] = fc
    minz = z.min(1)
    pen = np.clip(-minz, 0, None)
    acc = np.linalg.norm(world[10:] - 2 * world[5:-5] + world[:-10], axis=-1)
    jerk = np.linalg.norm(world[3:] - 3 * world[2:-1] + 3 * world[1:-2] - world[:-3], axis=-1)
    out.update(pen_max=pen.max() / scale * 100, pen_frames=100 * (pen > 0.005 * scale).mean(),
                accel_mean=acc.mean() / scale * 100, accel_peak=acc.mean(1).max() / scale * 100,
                jerk_mean=jerk.mean() / scale * 100, jerk_peak=jerk.mean(1).max() / scale * 100)
    if laplacian is not None and cur_exp is not None:
        cur = curvature(world, laplacian, scale)
        out["cur_dev"] = 100 * np.mean(np.abs(cur - cur_exp) / cur_exp)
    if mesh_exp is not None:
        out["shape_dev"] = shape_dev(world, scale, mesh_exp)
    return out


def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {os.path.basename(__file__)} <character> <sequence_a> <sequence_b>")
        sys.exit(1)
    character, seq_a, seq_b = sys.argv[1], sys.argv[2], sys.argv[3]

    rest_pose = np.load(os.path.join(root_dir, character, "static_data", "rest_pose.npy")).astype(np.float32)
    scale = float(np.linalg.norm(rest_pose.max(0) - rest_pose.min(0)))
    foot_mask = foot_vertex_mask(character)
    if foot_mask is None:
        print("[warn] no foot/toe bones found by name -- foot_sliding_* will be unavailable")
    else:
        print(f"Foot vertices: {int(foot_mask.sum())} / {len(foot_mask)}")

    laplacian = build_laplacian(character, rest_pose)
    cur_exp = reference_curvature(character, seq_a, seq_b, laplacian, scale, num_frames=150)
    mesh_exp = reference_mesh_interp(character, seq_a, seq_b, num_frames=150)

    rows = []
    for name, prefix in METHODS:
        path = os.path.join(root_dir, character, f"{prefix}_{seq_a}_to_{seq_b}_FullMesh_{character}.npy")
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found (run its script first)")
            continue
        world = np.load(path).astype(np.float32)
        rows.append((name, metrics(world, scale, foot_mask, laplacian, cur_exp, mesh_exp)))

    if not rows:
        print("No .npy outputs found -- run the 4 blend scripts first.")
        return

    keys = ["shape_dev", "foot_sliding_worst", "foot_sliding_mean",
            "accel_mean", "accel_peak", "jerk_mean", "jerk_peak", "cur_dev"]
    header = f"{'method':<18}" + "".join(f"{k:>12}" for k in keys)
    print(f"\n{character}: {seq_a} -> {seq_b}\n")
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<18}" + "".join(f"{m[k]:>12.4f}" for k in keys))


if __name__ == "__main__":
    main()
