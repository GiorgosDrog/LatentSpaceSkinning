import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from dataloader import PCAMeshSequenceDataset
from model import VolumetricModelPCA
from loss_functions import make_total_loss

def count_parameters(model):
    module_params = {}
    total = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = params
        total += params
    return module_params, total

def unpadded_collate_fn(batch):
    out = {}
    for key in batch[0]:
        out[key] = default_collate([b[key] for b in batch])
    return out

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = "dragon"
animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")


if(character == "monster"):
    train_dataset = [
        os.path.join(animation_dir, f"Armature.{str(i).zfill(3)}")
        for i in range(1, 10)
    ]

    val_dataset = [
        os.path.join(animation_dir, f"Armature.{i:03d}") for i in [2,5]
    ]
else:
    EXCLUDED_SEQUENCES = {7, 15}
    train_dataset = [
        os.path.join(animation_dir, f"Armature.{str(i).zfill(3)}")
        for i in range(1, 2) if i not in EXCLUDED_SEQUENCES
    ]

    val_dataset = [
        os.path.join(animation_dir, f"Armature.{i:03d}") for i in [1] if i not in EXCLUDED_SEQUENCES
    ]

batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pca_mean_path = os.path.join(static_dir, "pca_mean.npy")
pca_components_path = os.path.join(static_dir, "pca_components.npy")

if not (os.path.exists(pca_mean_path) and os.path.exists(pca_components_path)):
    raise FileNotFoundError(
        f"Missing pca_mean.npy / pca_components.npy in {static_dir}. "
        f"Run compute_pca_basis.py first to generate them (offline, one-time step)."
    )

pca_mean = np.load(pca_mean_path)
pca_components = np.load(pca_components_path)
print(f"Loaded PCA basis: mean {pca_mean.shape}, components {pca_components.shape} "
      f"(K={pca_components.shape[0]})")

train_dataset = PCAMeshSequenceDataset(train_dataset)
val_dataset = PCAMeshSequenceDataset(val_dataset)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False,
    num_workers=0, pin_memory=True, collate_fn=unpadded_collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=True, collate_fn=unpadded_collate_fn
)

print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))

loss_fn = make_total_loss(w_vertex=1.0, w_smooth=0.02, w_l1=1.0, w_floor=0.0)

sample = train_dataset[0]
max_bones = sample["bone_matrices"].shape[1]
print(f"Active sample contains {sample['rest_pose'].shape[0]} vertices and {max_bones} bones.")

model = VolumetricModelPCA(
    max_bones=max_bones,
    hidden_size=64,
    num_layers=2,
    pca_mean=pca_mean,
    pca_components=pca_components,
).to(device)

print("Model created.")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
)

ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}.pth")
if os.path.exists(ckpt_path):
    print("Found existing checkpoint — loading weights only.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print("No saved model found — starting fresh training.")

print("\n==============================")
print("  MODEL PARAMETER COUNT")
print("==============================")
module_params, total_params = count_parameters(model)
for name, count in module_params.items():
    print(f"{name:25s}: {count:,} parameters")
print("--------------------------------")
print(f"Total trainable parameters: {total_params:,}")
print("================================\n")

epochs = 1500

model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=epochs,
    device=device,
    loss_fn=loss_fn,
    save_path=ckpt_path,
)
