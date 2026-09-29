import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation

from blend_utils import load_bones, resample_frames

ROOT = r"E:\didaktoriko\diffusion_solution"
DATASET = os.path.join(ROOT, "dataset")
N = 150


def relative_bones(bones):
    T = np.zeros(bones.shape[:2] + (4, 4), dtype=np.float64)
    T[..., :3, :] = bones
    T[..., 3, 3] = 1.0
    return np.einsum("fij,fbjk->fbik", np.linalg.inv(T[:, 0]), T)[..., :3, :].astype(np.float32)


def pose_distance_deg(rel_a, rel_b):
    A, B = rel_a[:, 1:, :, :3], rel_b[:, 1:, :, :3]
    rA = Rotation.from_matrix(A.reshape(-1, 3, 3))
    rB = Rotation.from_matrix(B.reshape(-1, 3, 3))
    ang = np.degrees(np.linalg.norm((rA.inv() * rB).as_rotvec(), axis=-1))
    return ang.reshape(A.shape[0], A.shape[1]).mean(axis=1)


def find_best_center(bones_a, bones_b, lo=0.2, hi=0.8):
    rel_a, rel_b = relative_bones(bones_a), relative_bones(bones_b)
    dist = pose_distance_deg(rel_a, rel_b)
    n = len(dist)
    lo_i, hi_i = int(lo * n), int(hi * n)
    best_i = lo_i + int(np.argmin(dist[lo_i:hi_i]))
    return best_i / n, dist


if __name__ == "__main__":
    character = sys.argv[1] if len(sys.argv) > 1 else "alpha"
    ia = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ib = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    anim_dir = os.path.join(DATASET, character, "animation_data")
    ba = resample_frames(load_bones(anim_dir, f"Armature.{ia:03d}"), N)
    bb = resample_frames(load_bones(anim_dir, f"Armature.{ib:03d}"), N)

    best_frac, dist = find_best_center(ba, bb)
    fixed_frac = 0.5

    print(f"{character}: clip {ia} vs clip {ib}, pose distance (deg) scanned over the clip")
    print(f"  at the FIXED transition center (t=0.50, current convention):  {dist[int(0.5 * N)]:.1f} deg")
    print(f"  BEST matching point found at t={best_frac:.3f}:                  {dist[int(best_frac * N)]:.1f} deg")
    print(f"  worst point in the scanned range:                              {dist[int(0.2 * N):int(0.8 * N)].max():.1f} deg")
    print(f"\n  -> new transition window centered on t={best_frac:.3f} instead of the fixed t=0.5")
