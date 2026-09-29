import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from dataloader import PCAMeshSequenceDataset
from model import VolumetricModelPCA
from loss_functions import make_total_loss, evaluate_metrics

ROOT_BONE_INDEX = 0


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
character = "x_bot"
animation_dir = os.path.join(root_dir, character, "animation_data")
static_dir = os.path.join(root_dir, character, "static_data")

EXCLUDED_SEQUENCES = {7, 15}
train_dataset = [
    os.path.join(animation_dir, f"Armature.{str(i).zfill(3)}")
    for i in range(1, 18) if i not in EXCLUDED_SEQUENCES
]
val_dataset = [
    os.path.join(animation_dir, f"Armature.{i:03d}") for i in [2, 10, 16] if i not in EXCLUDED_SEQUENCES
]

batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

NOISE_STD = 0.05

pca_mean_path = os.path.join(static_dir, "pca_mean.npy")
pca_components_path = os.path.join(static_dir, "pca_components.npy")
if not (os.path.exists(pca_mean_path) and os.path.exists(pca_components_path)):
    raise FileNotFoundError(
        f"Missing pca_mean.npy / pca_components.npy in {static_dir}. "
        f"Run compute_pca_basis.py first."
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
    max_bones=max_bones, hidden_size=64, num_layers=2,
    pca_mean=pca_mean, pca_components=pca_components,
).to(device)
print("Model created.")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))

ckpt_path = os.path.join(root_dir, f"best_volumetric_model_pca_{character}_denoise.pth")
if os.path.exists(ckpt_path):
    print("Found existing denoise checkpoint — loading weights only.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print("No saved denoise model found — starting fresh training.")

print("\n==============================")
print("  MODEL PARAMETER COUNT")
print("==============================")
module_params, total_params = count_parameters(model)
for name, count in module_params.items():
    print(f"{name:25s}: {count:,} parameters")
print("--------------------------------")
print(f"Total trainable parameters: {total_params:,}")
print(f"Noise std on z (training only): {NOISE_STD}")
print("================================\n")


def forward_with_noise(model, bone_matrices, noise_std):
    B, F, B_max, _, _ = bone_matrices.shape
    bones_flat = bone_matrices.contiguous().view(B, F, B_max * 12)
    latent = model.bone_encoder(bones_flat)
    latent = model.attention(latent)
    if noise_std > 0:
        latent = latent + torch.randn_like(latent) * noise_std
    coeffs = model.decoder(latent)
    return model.reconstruct(coeffs)


def train_one_epoch_denoising(model, train_loader, optimizer, device, loss_fn, noise_std):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in train_loader:
        bones = batch["bone_matrices"].to(device)
        mesh_vertices = batch["mesh_vertices"].to(device)
        rest_pose = batch["rest_pose"].to(device)
        scale = batch["scale"].to(device)
        F = bones.shape[1]
        target = mesh_vertices - rest_pose.unsqueeze(1)
        root_rotation = bones[:, :, ROOT_BONE_INDEX, :, :3]
        root_translation = bones[:, :, ROOT_BONE_INDEX, :, 3]

        optimizer.zero_grad()
        pred_disp = forward_with_noise(model, bones, noise_std)
        loss = loss_fn(pred_disp, target, scale, rest_pose, root_rotation, root_translation)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * F
        total_samples += F
    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch_clean(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_erms = 0.0
    total_maxavg = 0.0
    total_disper = 0.0
    total_floorpen = 0.0
    val_batches = len(loader)

    for batch in loader:
        bones = batch["bone_matrices"].to(device)
        mesh_vertices = batch["mesh_vertices"].to(device)
        rest_pose = batch["rest_pose"].to(device)
        scale = batch["scale"].to(device)

        target = mesh_vertices - rest_pose.unsqueeze(1)
        root_rotation = bones[:, :, ROOT_BONE_INDEX, :, :3]
        root_translation = bones[:, :, ROOT_BONE_INDEX, :, 3]
        pred_disp = model(bones)

        loss = loss_fn(pred_disp, target, scale, rest_pose, root_rotation, root_translation)
        total_loss += loss.item()

        metrics = evaluate_metrics(pred_disp, target, scale, rest_pose, root_rotation, root_translation)
        total_erms += metrics["ERMS"]
        total_maxavg += metrics["MaxAvg"]
        total_disper += metrics["DisPer"]
        total_floorpen += metrics["FloorPenMax"]

    return {
        "loss": total_loss / val_batches,
        "ERMS": total_erms / val_batches,
        "MaxAvg": total_maxavg / val_batches,
        "DisPer": total_disper / val_batches,
        "FloorPenMax": total_floorpen / val_batches,
    }


epochs = 1500
best_val_loss = float("inf")
print("Starting denoising-regularized training loop...")

for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch_denoising(model, train_loader, optimizer, device, loss_fn, NOISE_STD)
    val = validate_one_epoch_clean(model, val_loader, device, loss_fn)

    print(
        f"[Epoch {epoch:04d}/{epochs:04d}] "
        f"Train: {train_loss:.6f} | Val: {val['loss']:.6f} || "
        f"ERMS: {val['ERMS']:.4f}% | DisPer: {val['DisPer']:.4f}% | MaxAvg: {val['MaxAvg']:.6f} | "
        f"FloorPenMax: {val['FloorPenMax']:.4f}"
    )

    if val["loss"] < best_val_loss:
        best_val_loss = val["loss"]
        torch.save(model.state_dict(), ckpt_path)
        print(f"  --> Saved new best checkpoint to: {ckpt_path}")
