import os
import numpy as np
import torch

from model import VolumetricModelPCA
from blend_utils import load_bones, ROOT_BONE_INDEX
from dataloader import root_relative_mesh
from loss_functions import evaluate_metrics


root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
source_character = "amy"
source_sequence = "Armature.002"
target_character = "alpha"

device = "cuda" if torch.cuda.is_available() else "cpu"

source_static_dir = os.path.join(root_dir, source_character, "static_data")
source_animation_dir = os.path.join(root_dir, source_character, "animation_data")
target_static_dir = os.path.join(root_dir, target_character, "static_data")
target_animation_dir = os.path.join(root_dir, target_character, "animation_data")


def get_active_bone_order(static_dir):
    names = list(np.load(os.path.join(static_dir, "bone_names.npy"), allow_pickle=True))
    weights = np.load(os.path.join(static_dir, "skinning_weights.npy"))
    active_idx = np.where(weights.sum(axis=0) > 0)[0]
    return [names[i] for i in active_idx]


def build_bone_permutation(source_static_dir, target_static_dir):

    source_names = get_active_bone_order(source_static_dir)
    target_names = get_active_bone_order(target_static_dir)
    if set(source_names) != set(target_names):
        raise ValueError(
            f"Bone name sets differ between {source_character} and {target_character} "
            f"-- not a compatible skeleton (different rig topology). "
            f"Missing in source: {set(target_names) - set(source_names)}. "
            f"Missing in target: {set(source_names) - set(target_names)}."
        )
    name_to_source_idx = {n: i for i, n in enumerate(source_names)}
    perm = np.array([name_to_source_idx[n] for n in target_names], dtype=np.int64)
    print(f"Bone alignment: {len(perm)} bones matched by name "
          f"({'identity permutation -- same order already' if np.array_equal(perm, np.arange(len(perm))) else 'reordered'}).")
    return perm


target_ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{target_character}.pth")
if not os.path.exists(target_ckpt_path):
    raise FileNotFoundError(f"{target_ckpt_path} not found -- train the target character first.")
target_state = torch.load(target_ckpt_path, map_location=device)
target_pca_mean = target_state["pca_mean"].cpu().numpy()
target_pca_components = target_state["pca_components"].cpu().numpy()
target_num_bones = target_state["bone_encoder.input_norm.weight"].shape[0] // 12
target_num_layers = sum(1 for k in target_state if k.startswith("bone_encoder.lstm.weight_ih_l"))
target_hidden_size = target_state["bone_encoder.lstm.weight_ih_l0"].shape[0] // 4

target_model = VolumetricModelPCA(
    max_bones=target_num_bones, hidden_size=target_hidden_size, num_layers=target_num_layers,
    pca_mean=target_pca_mean, pca_components=target_pca_components,
).to(device)
target_model.load_state_dict(target_state)
target_model.eval()
print(f"Loaded target model: {target_ckpt_path} "
      f"(hidden_size={target_hidden_size}, num_layers={target_num_layers}, bones={target_num_bones})")

target_rest_pose = np.load(os.path.join(target_static_dir, "rest_pose.npy")).astype(np.float32)
assert target_rest_pose.shape[0] * 3 == target_pca_mean.shape[0], \
    "target rest_pose vertex count doesn't match target PCA basis -- stale/mismatched static_data."


source_rest_pose = np.load(os.path.join(source_static_dir, "rest_pose.npy")).astype(np.float32)
source_scale = np.linalg.norm(source_rest_pose.max(axis=0) - source_rest_pose.min(axis=0))
target_scale = np.linalg.norm(target_rest_pose.max(axis=0) - target_rest_pose.min(axis=0))
root_scale_ratio = float(target_scale / source_scale)
print(f"Body scale: {source_character}={source_scale:.4f}, {target_character}={target_scale:.4f} "
      f"-> root translation scale ratio = {root_scale_ratio:.4f}")

source_bone_rest = np.load(os.path.join(source_static_dir, "bone_rest.npy")).astype(np.float32)
target_bone_rest = np.load(os.path.join(target_static_dir, "bone_rest.npy")).astype(np.float32)
source_hip_rest_z = float(source_bone_rest[ROOT_BONE_INDEX, 2, 3])
target_hip_rest_z = float(target_bone_rest[ROOT_BONE_INDEX, 2, 3])
print(f"Hip rest height: {source_character}={source_hip_rest_z:.4f}, {target_character}={target_hip_rest_z:.4f}")


perm = build_bone_permutation(source_static_dir, target_static_dir)
source_bones_native = load_bones(source_animation_dir, source_sequence)
assert source_bones_native.shape[1] == len(perm), \
    f"source has {source_bones_native.shape[1]} bones but permutation expects {len(perm)}"
transferred_bones = source_bones_native[:, perm, :, :]
assert transferred_bones.shape[1] == target_num_bones, \
    f"realigned source has {transferred_bones.shape[1]} bones but target model expects {target_num_bones}"

bones_t = torch.from_numpy(transferred_bones).unsqueeze(0).to(device)
with torch.no_grad():
    pred_disp = target_model(bones_t)

rest_pose_t = torch.from_numpy(target_rest_pose).unsqueeze(0).to(device)
pred_local = rest_pose_t.unsqueeze(1) + pred_disp


R_root = transferred_bones[:, ROOT_BONE_INDEX, :, :3]
t_root_src = transferred_bones[:, ROOT_BONE_INDEX, :, 3]
t_root_xy = t_root_src[:, :2] * root_scale_ratio
t_root_z = target_hip_rest_z + (t_root_src[:, 2] - source_hip_rest_z) * root_scale_ratio
t_root = np.concatenate([t_root_xy, t_root_z[:, None]], axis=1)
R_root_t = torch.from_numpy(R_root).unsqueeze(0).to(device)
t_root_t = torch.from_numpy(t_root).unsqueeze(0).to(device)
world_pos = torch.matmul(pred_local, R_root_t.transpose(-1, -2)) + t_root_t.unsqueeze(-2)

mesh_vertices = world_pos.squeeze(0).cpu().numpy().astype(np.float32)


FLOOR_Z = 0.0
min_z_per_frame = mesh_vertices[..., 2].min(axis=1)
violation_per_frame = np.clip(FLOOR_Z - min_z_per_frame, a_min=0.0, a_max=None)
frames_with_violations = int((violation_per_frame > 0).sum())
if frames_with_violations > 0:
    print(f"Floor guarantee: lifting {frames_with_violations}/{mesh_vertices.shape[0]} frames "
          f"(worst penetration: {violation_per_frame.max():.4f} units).")
    mesh_vertices[..., 2] += violation_per_frame[:, None]

out_path = os.path.join(root_dir, target_character,
                         f"AnimTransfer_{source_character}_{source_sequence}_to_{target_character}_FullMesh")
np.save(out_path, mesh_vertices)
print(f"Saved transferred motion: {mesh_vertices.shape} -> {out_path}.npy")


target_gt_mesh_path = os.path.join(target_animation_dir, f"{source_sequence}_FullMesh.npy")
target_gt_bones_path = os.path.join(target_animation_dir, f"{source_sequence}_Bones.npy")
if os.path.exists(target_gt_mesh_path) and os.path.exists(target_gt_bones_path):
    target_gt_mesh = np.load(target_gt_mesh_path).astype(np.float32)
    target_gt_bones = load_bones(target_animation_dir, source_sequence)

    assert target_gt_mesh.shape[0] == pred_disp.shape[1], (
        f"frame count mismatch: target's own {source_sequence} has {target_gt_mesh.shape[0]} frames, "
        f"transferred motion has {pred_disp.shape[1]} -- not the same underlying motion length, "
        f"skipping quantitative comparison."
    )

    target_gt_local = root_relative_mesh(target_gt_mesh, target_gt_bones)
    target_disp_gt = target_gt_local - target_rest_pose[None]
    target_disp_gt_t = torch.from_numpy(target_disp_gt).unsqueeze(0).to(device)

    scale = np.float32(np.linalg.norm(target_rest_pose.max(axis=0) - target_rest_pose.min(axis=0)))
    scale_t = torch.tensor([scale], device=device)

    gt_root_rotation = torch.from_numpy(target_gt_bones[:, ROOT_BONE_INDEX, :, :3]).unsqueeze(0).to(device)
    gt_root_translation = torch.from_numpy(target_gt_bones[:, ROOT_BONE_INDEX, :, 3]).unsqueeze(0).to(device)

    metrics = evaluate_metrics(
        pred_disp, target_disp_gt_t, scale_t,
        rest_pose=rest_pose_t, root_rotation=gt_root_rotation, root_translation=gt_root_translation,
    )
    print("\n=== Transfer quality vs. target's own real motion for the same sequence ===")
    print(f"  DisPer: {metrics['DisPer']:.4f}%  |  ERMS: {metrics['ERMS']:.4f}%  |  "
          f"MaxAvg: {metrics['MaxAvg']:.6f}  |  FloorPenMax: {metrics['FloorPenMax']:.4f}")
    print(f"  (this number ALREADY includes whatever bias the checkpoint-selection caveat above "
          f"describes if source_sequence is a val sequence -- it is not a clean generalization bound)")
else:
    print(f"\nNo ground truth found for {target_character}/{source_sequence} -- "
          f"skipping quantitative comparison (qualitative Blender check only).")
