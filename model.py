import math
import torch
import torch.nn as nn
from loss_functions import evaluate_metrics

ROOT_BONE_INDEX = 0

class BoneEncoder(nn.Module):
    def __init__(self, bone_feat_dim: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.input_norm = nn.LayerNorm(bone_feat_dim)
        self.lstm = nn.LSTM(
            input_size=bone_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, bone_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(bone_seq)
        out, _ = self.lstm(x)
        return out


class FrameAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, v)

        return self.norm(attended + x)


class PCADecoder(nn.Module):
    def __init__(self, hidden_size: int, num_pca_components: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, num_pca_components),
        )

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(latent_seq)
        out = self.norm(self.proj(out))
        coeffs = self.head(out)
        return coeffs


class VolumetricModelPCA(nn.Module):
    def __init__(self, max_bones: int, hidden_size: int, num_layers: int,
                 pca_mean, pca_components):
        super().__init__()
        self.bone_feat_dim = max_bones * 12
        self.num_components = pca_components.shape[0]
        self.num_vertices = pca_components.shape[1] // 3

        self.bone_encoder = BoneEncoder(self.bone_feat_dim, hidden_size, num_layers)
        self.attention = FrameAttention(hidden_size)
        self.decoder = PCADecoder(hidden_size, self.num_components, num_layers)

        pca_mean_t = torch.as_tensor(pca_mean, dtype=torch.float32)
        pca_components_t = torch.as_tensor(pca_components, dtype=torch.float32)
        self.register_buffer("pca_mean", pca_mean_t)
        self.register_buffer("pca_components", pca_components_t)

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        B, F, K = coeffs.shape
        flat = self.pca_mean.unsqueeze(0).unsqueeze(0) + coeffs @ self.pca_components
        return flat.view(B, F, self.num_vertices, 3)

    def forward(self, bone_matrices: torch.Tensor) -> torch.Tensor:
        B, F, B_max, _, _ = bone_matrices.shape
        bones_flat = bone_matrices.contiguous().view(B, F, B_max * 12)

        latent = self.bone_encoder(bones_flat)
        latent = self.attention(latent)
        coeffs = self.decoder(latent)
        return self.reconstruct(coeffs)

    def train_one_epoch(self, train_loader, optimizer, device, loss_fn, epoch=None):
        self.train()
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
            pred_disp = self.forward(bones)
            loss = loss_fn(pred_disp, target, scale, rest_pose, root_rotation, root_translation, epoch=epoch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * F
            total_samples += F

        return total_loss / total_samples

    def validate_one_epoch(self, loader, device, loss_fn):
        self.eval()
        total_loss = 0.0
        total_erms = 0.0
        total_maxavg = 0.0
        total_disper = 0.0
        total_floorpen = 0.0
        val_batches = len(loader)

        with torch.no_grad():
            for batch in loader:
                bones = batch["bone_matrices"].to(device)
                mesh_vertices = batch["mesh_vertices"].to(device)
                rest_pose = batch["rest_pose"].to(device)
                scale = batch["scale"].to(device)

                target = mesh_vertices - rest_pose.unsqueeze(1)
                root_rotation = bones[:, :, ROOT_BONE_INDEX, :, :3]
                root_translation = bones[:, :, ROOT_BONE_INDEX, :, 3]
                pred_disp = self.forward(bones)

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

    def train_model(self, train_loader, val_loader, optimizer, num_epochs, device, loss_fn, save_path):
        best_val_loss = float("inf")
        print("Starting PCA-decoder training loop...")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(train_loader, optimizer, device, loss_fn, epoch=epoch)
            val = self.validate_one_epoch(val_loader, device, loss_fn)

            print(
                f"[Epoch {epoch:04d}/{num_epochs:04d}] "
                f"Train: {train_loss:.6f} | Val: {val['loss']:.6f} || "
                f"ERMS: {val['ERMS']:.4f}% | DisPer: {val['DisPer']:.4f}% | MaxAvg: {val['MaxAvg']:.6f} | "
                f"FloorPenMax: {val['FloorPenMax']:.4f}"
            )

            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                torch.save(self.state_dict(), save_path)
                print(f"  --> Saved new best checkpoint to: {save_path}")
