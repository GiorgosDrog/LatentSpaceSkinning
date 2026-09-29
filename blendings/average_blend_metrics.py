import os
import sys
import numpy as np

from measure_blend_metrics import (
    root_dir, METHODS, foot_vertex_mask, metrics, build_laplacian, reference_curvature, reference_mesh_interp,
)

def _pairs_for(character):
    import random
    if character == "monster":
        seqs = [f"Armature.{i:03d}" for i in range(1, 10)]
    else:
        excluded = {7, 15}
        seqs = [f"Armature.{i:03d}" for i in range(1, 18) if i not in excluded]
    rng = random.Random(1234)
    return [tuple(rng.sample(seqs, 2)) for _ in range(6)]

KEYS = ["shape_dev", "foot_sliding_worst", "foot_sliding_mean",
        "accel_mean", "accel_peak", "jerk_mean", "jerk_peak", "cur_dev"]

TABLES_DIR = r"E:\didaktoriko\diffusion_solution\paper_results\tables"

COL_LABELS = {
    "shape_dev": "Shape dev.\\ (\\%)",
    "foot_sliding_worst": "Foot skate worst", "foot_sliding_mean": "Foot skate mean",
    "root_motion_pct": "Root motion (\\%)",
    "accel_mean": "Accel mean", "accel_peak": "Accel peak",
    "jerk_mean": "Jerk mean", "jerk_peak": "Jerk peak", "cur_dev": "Cur.\\ dev.\\ (\\%)",
}


def _tex_escape(s):
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write_tex_table(character, per_method):
    lines = [r"\begin{tabular}{l" + "r" * len(KEYS) + "}", r"\toprule",
             "Method & " + " & ".join(COL_LABELS[k] for k in KEYS) + r" \\", r"\midrule"]
    for name, _ in METHODS:
        rows = per_method[name]
        if not rows:
            continue
        avg = {k: np.mean([r[k] for r in rows]) for k in KEYS}
        vals = " & ".join(f"{avg[k]:.4f}" for k in KEYS)
        lines.append(f"{_tex_escape(name)} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_path = os.path.join(TABLES_DIR, f"tab_synthesis_{character}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


def compute_per_method(character):
    PAIRS = _pairs_for(character)
    rest_pose = np.load(os.path.join(root_dir, character, "static_data", "rest_pose.npy")).astype(np.float32)
    scale = float(np.linalg.norm(rest_pose.max(0) - rest_pose.min(0)))
    foot_mask = foot_vertex_mask(character)
    laplacian = build_laplacian(character, rest_pose)

    per_method = {name: [] for name, _ in METHODS}
    cur_dev_per_pair = {name: {} for name, _ in METHODS}
    skipped = []
    for seq_a, seq_b in PAIRS:
        cur_exp = reference_curvature(character, seq_a, seq_b, laplacian, scale, num_frames=150)
        mesh_exp = reference_mesh_interp(character, seq_a, seq_b, num_frames=150)
        for name, prefix in METHODS:
            path = os.path.join(root_dir, character, f"{prefix}_{seq_a}_to_{seq_b}_FullMesh_{character}.npy")
            if not os.path.exists(path):
                skipped.append(f"[skip] {name} {seq_a}->{seq_b}: not found")
                continue
            world = np.load(path).astype(np.float32)
            m = metrics(world, scale, foot_mask, laplacian, cur_exp, mesh_exp)
            per_method[name].append(m)
            cur_dev_per_pair[name][(seq_a, seq_b)] = m["cur_dev"]
    return per_method, cur_dev_per_pair, skipped, PAIRS


def main():
    character = sys.argv[1] if len(sys.argv) > 1 else "michelle"
    per_method, cur_dev_per_pair, skipped, PAIRS = compute_per_method(character)

    header = f"{'method':<18}{'n':>4}" + "".join(f"{k:>12}" for k in KEYS)
    print(f"\n{character}: averaged over {len(PAIRS)} pairs (train_transition.py's own eval set)\n")
    print(header)
    print("-" * len(header))
    for name, _ in METHODS:
        rows = per_method[name]
        if not rows:
            continue
        avg = {k: np.mean([r[k] for r in rows]) for k in KEYS}
        print(f"{name:<18}{len(rows):>4}" + "".join(f"{avg[k]:>12.4f}" for k in KEYS))

    pair_labels = [f"{a}->{b}" for a, b in PAIRS]
    cur_header = f"{'method':<18}" + "".join(f"{p:>20}" for p in pair_labels)
    print(f"\n{character}: cur_dev per movement (pair)\n")
    print(cur_header)
    print("-" * len(cur_header))
    for name, _ in METHODS:
        vals = cur_dev_per_pair[name]
        if not vals:
            continue
        row = "".join(f"{vals.get((a, b), float('nan')):>20.4f}" for a, b in PAIRS)
        print(f"{name:<18}{row}")

    if skipped:
        print(f"\n{len(skipped)} missing output(s) (excluded from the averages above):")
        for line in skipped:
            print(f"  {line}")

    write_tex_table(character, per_method)


if __name__ == "__main__":
    main()
