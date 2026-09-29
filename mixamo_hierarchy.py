import os
import numpy as np

MIXAMO_PARENT = {
    "Hips": None,
    "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1",
    "Neck": "Spine2", "Head": "Neck", "HeadTop_End": "Head",
    "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand", "LeftHandThumb2": "LeftHandThumb1", "LeftHandThumb3": "LeftHandThumb2", "LeftHandThumb4": "LeftHandThumb3",
    "LeftHandIndex1": "LeftHand", "LeftHandIndex2": "LeftHandIndex1", "LeftHandIndex3": "LeftHandIndex2", "LeftHandIndex4": "LeftHandIndex3",
    "LeftHandMiddle1": "LeftHand", "LeftHandMiddle2": "LeftHandMiddle1", "LeftHandMiddle3": "LeftHandMiddle2", "LeftHandMiddle4": "LeftHandMiddle3",
    "LeftHandRing1": "LeftHand", "LeftHandRing2": "LeftHandRing1", "LeftHandRing3": "LeftHandRing2", "LeftHandRing4": "LeftHandRing3",
    "LeftHandPinky1": "LeftHand", "LeftHandPinky2": "LeftHandPinky1", "LeftHandPinky3": "LeftHandPinky2", "LeftHandPinky4": "LeftHandPinky3",
    "RightShoulder": "Spine2", "RightArm": "RightShoulder", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand", "RightHandThumb2": "RightHandThumb1", "RightHandThumb3": "RightHandThumb2", "RightHandThumb4": "RightHandThumb3",
    "RightHandIndex1": "RightHand", "RightHandIndex2": "RightHandIndex1", "RightHandIndex3": "RightHandIndex2", "RightHandIndex4": "RightHandIndex3",
    "RightHandMiddle1": "RightHand", "RightHandMiddle2": "RightHandMiddle1", "RightHandMiddle3": "RightHandMiddle2", "RightHandMiddle4": "RightHandMiddle3",
    "RightHandRing1": "RightHand", "RightHandRing2": "RightHandRing1", "RightHandRing3": "RightHandRing2", "RightHandRing4": "RightHandRing3",
    "RightHandPinky1": "RightHand", "RightHandPinky2": "RightHandPinky1", "RightHandPinky3": "RightHandPinky2", "RightHandPinky4": "RightHandPinky3",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot", "LeftToe_End": "LeftToeBase",
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg", "RightToeBase": "RightFoot", "RightToe_End": "RightToeBase",
}


def _strip(name):
    return str(name).split(":", 1)[-1]


def build_parent_indices(bone_names):
    short = [_strip(n) for n in bone_names]
    index_of = {n: i for i, n in enumerate(short)}
    parents = np.full(len(short), -1, dtype=np.int64)
    for i, n in enumerate(short):
        p = MIXAMO_PARENT.get(n)
        if p is not None and p in index_of:
            parents[i] = index_of[p]
    return parents


def verify_against_rest_pose(bone_names, parents, bone_rest):
    pos = bone_rest[:, :, 3]
    lengths = []
    for i, p in enumerate(parents):
        if p < 0:
            continue
        d = float(np.linalg.norm(pos[i] - pos[p]))
        lengths.append((bone_names[i], bone_names[p], d))
    return lengths


if __name__ == "__main__":
    root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
    character = "michelle"
    static_dir = os.path.join(root_dir, character, "static_data")
    names = [str(n) for n in np.load(os.path.join(static_dir, "bone_names.npy"), allow_pickle=True)]
    bone_rest = np.load(os.path.join(static_dir, "bone_rest.npy")).astype(np.float64)
    parents = build_parent_indices(names)
    print(f"{(parents < 0).sum()} root(s), {len(names)} bones total")
    lengths = verify_against_rest_pose(names, parents, bone_rest)
    print("\nchild -> parent : bone length")
    for child, parent, d in lengths:
        flag = "  <-- SUSPICIOUS (near-zero or huge)" if d < 1e-4 or d > 200 else ""
        print(f"  {child:<22} -> {parent:<18} : {d:8.3f}{flag}")
