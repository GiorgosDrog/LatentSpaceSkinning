import os
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT_BONE_INDEX = 0


def root_relative_mesh(mesh, bones):
    R_root = bones[:, ROOT_BONE_INDEX, :, :3]
    t_root = bones[:, ROOT_BONE_INDEX, :, 3]
    mesh_centered = mesh - t_root[:, None, :]
    return np.einsum('fji,fvj->fvi', R_root, mesh_centered).astype(np.float32)


class PCAMeshSequenceDataset(Dataset):
    def __init__(self, anim_prefixes):
        self.samples = []

        for prefix in anim_prefixes:
            anim_data_dir = os.path.dirname(prefix)
            char_root = os.path.dirname(anim_data_dir)
            static_dir = os.path.join(char_root, "static_data")
            base_name = os.path.basename(prefix)

            files = {
                "rest_surface": os.path.join(static_dir, "rest_pose.npy"),
                "mesh": os.path.join(anim_data_dir, f"{base_name}_FullMesh.npy"),
                "bones": os.path.join(anim_data_dir, f"{base_name}_Bones.npy"),
            }
            if not all(os.path.exists(f) for f in files.values()):
                print(f"Skipping incomplete animation components for prefix: {base_name}")
                continue

            rest_surface = np.load(files["rest_surface"]).astype(np.float32)
            mesh = np.load(files["mesh"]).astype(np.float32)
            bones_raw = np.load(files["bones"]).astype(np.float32)

            F = bones_raw.shape[0]
            B = bones_raw.shape[1] // 12
            bones = bones_raw.reshape(F, B, 3, 4)

            mesh_local = root_relative_mesh(mesh, bones)

            bbox_diag = np.linalg.norm(rest_surface.max(axis=0) - rest_surface.min(axis=0))
            scale = np.float32(bbox_diag)

            self.samples.append({
                "bone_matrices": torch.from_numpy(bones),
                "rest_pose": torch.from_numpy(rest_surface),
                "mesh_vertices": torch.from_numpy(mesh_local),
                "scale": torch.tensor(scale),
            })

        print(f"Loaded {len(self.samples)} PCA-experiment animation sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
