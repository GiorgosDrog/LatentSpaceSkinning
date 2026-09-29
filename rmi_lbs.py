import os
import sys
import numpy as np

from skeleton_blend_lbs import Skin
from blend_utils import resample_frames, orthonormalize

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
NUM_OUTPUT_FRAMES = 150


def main():
    character = sys.argv[1] if len(sys.argv) > 1 else "michelle"
    sequence_a = sys.argv[2] if len(sys.argv) > 2 else "Armature.016"
    sequence_b = sys.argv[3] if len(sys.argv) > 3 else "Armature.010"

    bones_path = os.path.join(root_dir, character, f"RMIBlend_{sequence_a}_to_{sequence_b}_Bones_{character}.npy")
    if not os.path.exists(bones_path):
        print(f"[missing] {bones_path} -- run infer_rmi.py first.")
        sys.exit(1)
    bones_world = np.load(bones_path).astype(np.float32)

    if bones_world.shape[0] != NUM_OUTPUT_FRAMES:
        bones_rs = resample_frames(bones_world, NUM_OUTPUT_FRAMES)
        J = bones_rs.shape[1]
        for j in range(J):
            bones_rs[:, j, :, :3] = np.stack([orthonormalize(r) for r in bones_rs[:, j, :, :3]])
        bones_world = bones_rs
        print(f"Resampled RMI output {50} -> {NUM_OUTPUT_FRAMES} frames to match the shared convention.")

    skin = Skin(character)
    mesh_vertices = skin.world(bones_world).astype(np.float32)

    out_path = os.path.join(root_dir, character, f"RMIBlend_{sequence_a}_to_{sequence_b}_FullMesh_{character}")
    np.save(out_path, mesh_vertices)
    print(f"Saved RMI+LBS mesh: {mesh_vertices.shape} -> {out_path}")


if __name__ == "__main__":
    main()
