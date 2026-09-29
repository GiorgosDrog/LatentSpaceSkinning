import os
import numpy as np
import torch
from torch.utils.data import Dataset

from mixamo_to_rmi import (
    active_bone_names, rest_offsets, local_quats, foot_contact, fk_positions, ROOT_BONE_INDEX,
)
from mixamo_hierarchy import build_parent_indices

EXCLUDED_SEQUENCES = {7, 15}


class MixamoRMIDataset(Dataset):
    def __init__(self, root_dir, character, seq_len=50, offset=10, sequences=None):
        self.root_dir = root_dir
        self.character = character
        self.seq_len = seq_len
        self.offset = offset

        static_dir = os.path.join(root_dir, character, "static_data")
        anim_dir = os.path.join(root_dir, character, "animation_data")

        self.names, act = active_bone_names(root_dir, character)
        self.parents = build_parent_indices(self.names)
        bone_rest = np.load(os.path.join(static_dir, "bone_rest.npy")).astype(np.float64)[act]
        self.offsets_np = rest_offsets(bone_rest, self.parents)
        rest_pose = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
        self.scale = float(np.linalg.norm(rest_pose.max(0) - rest_pose.min(0)))

        foot_names = ["mixamorig:LeftFoot", "mixamorig:LeftToeBase",
                      "mixamorig:RightFoot", "mixamorig:RightToeBase"]
        self.foot_idx = [self.names.index(n) for n in foot_names]

        if sequences is None:
            sequences = [f"Armature.{i:03d}" for i in range(1, 18) if i not in EXCLUDED_SEQUENCES]
        self.sequences = sequences

        windows = []
        used_sequences = []
        for seq in sequences:
            path = os.path.join(anim_dir, f"{seq}_Bones.npy")
            if not os.path.exists(path):
                continue
            used_sequences.append(seq)
            bones_raw = np.load(path).astype(np.float32)
            F = bones_raw.shape[0]
            max_bones = bones_raw.shape[1] // 12
            bones_world = bones_raw.reshape(F, max_bones, 3, 4).astype(np.float64)
            if bones_world.shape[1] != len(self.parents):
                print(f"[skip] {seq}: {bones_world.shape[1]} bones, expected {len(self.parents)}")
                continue
            q = local_quats(bones_world, self.parents)
            root_p = bones_world[:, ROOT_BONE_INDEX, :, 3]
            contact = foot_contact(bones_world, self.foot_idx, scale=self.scale)
            for start in range(0, F - seq_len + 1, offset):
                q_w = q[start:start + seq_len].astype(np.float32)
                root_p_w = root_p[start:start + seq_len].astype(np.float32)
                glbl_p, _ = fk_positions(q_w, self.offsets_np, self.parents, root_p_w)
                windows.append(dict(
                    local_q=q_w,
                    root_p=root_p_w,
                    contact=contact[start:start + seq_len].astype(np.float32),
                    X=glbl_p.astype(np.float32),
                ))
        if not windows:
            raise RuntimeError(f"No windows built for {character} -- check seq_len ({seq_len}) "
                                f"against clip lengths, and that sequences exist.")
        self.windows = windows
        self.cur_seq_length = seq_len
        print(f"{character}: {len(used_sequences)} clips -> {len(windows)} windows of {seq_len} frames "
              f"(stride {offset})")

        J = len(self.names)
        x_glbl = np.stack([w["X"] for w in self.windows], axis=0)
        flat = x_glbl.reshape(x_glbl.shape[0], x_glbl.shape[1], -1).transpose(0, 2, 1)
        self.x_mean = torch.from_numpy(flat.mean(axis=(0, 2), keepdims=True).astype(np.float32))
        self.x_std = torch.from_numpy((flat.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        local_q = w["local_q"]
        root_p = w["root_p"]
        contact = w["contact"]
        return dict(
            local_q=local_q,
            root_v=(root_p[1:] - root_p[:-1]).astype(np.float32),
            contact=contact,
            root_p_offset=root_p[-1].astype(np.float32),
            local_q_offset=local_q[-1].astype(np.float32),
            target=local_q[-1].astype(np.float32),
            root_p=root_p,
            X=w["X"],
        )


if __name__ == "__main__":
    root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
    ds = MixamoRMIDataset(root_dir, "michelle", seq_len=50, offset=10)
    print(f"num_joints: {len(ds.names)}")
    print(f"x_mean: {ds.x_mean.numpy().ravel()}")
    print(f"x_std: {ds.x_std.numpy().ravel()}")
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")
