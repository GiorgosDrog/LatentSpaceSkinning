import torch


def _normalize(pred_disp, target_disp, scale):
    scale_r = scale.view(-1, 1, 1, 1).clamp_min(1e-6)
    return pred_disp / scale_r, target_disp / scale_r


def vertex_rms_loss(pred_disp, target_disp):
    mse = (pred_disp - target_disp).pow(2).mean()
    return torch.sqrt(mse + 1e-8)


def smoothness_loss(pred_disp):
    if pred_disp.shape[1] < 2:
        return torch.tensor(0.0, device=pred_disp.device)
    diff = pred_disp[:, 1:] - pred_disp[:, :-1]
    return diff.pow(2).mean()


def disper_loss(pred_disp, target_disp):
    return torch.mean(torch.norm(pred_disp - target_disp, dim=-1))


def metric_maxavg(pred_disp, target_disp):
    diff = torch.norm(pred_disp - target_disp, dim=-1)
    max_per_frame = diff.max(dim=-1).values
    return max_per_frame.mean().item()


def metric_erms(pred_disp, target_disp):
    diff = pred_disp - target_disp
    B, F, V, _ = diff.shape
    frob = torch.norm(diff.reshape(B, -1), dim=1)
    denom = torch.sqrt(torch.tensor(3 * F * V, device=diff.device, dtype=diff.dtype))
    erms = 100.0 * (frob / denom)
    return erms.mean().item()


def floor_penetration_depth(pred_disp, rest_pose, root_rotation, root_translation, floor_z=0.0):
    local_pos = rest_pose.unsqueeze(1) + pred_disp
    world_pos = torch.matmul(local_pos, root_rotation.transpose(-1, -2)) + root_translation.unsqueeze(-2)
    world_z = world_pos[..., 2]
    return torch.clamp(floor_z - world_z, min=0.0)


def floor_penalty_loss(pred_disp, rest_pose, root_rotation, root_translation, scale, floor_z=0.0):
    depth = floor_penetration_depth(pred_disp, rest_pose, root_rotation, root_translation, floor_z)
    scale_r = scale.view(-1, 1, 1).clamp_min(1e-6)
    return (depth / scale_r).pow(2).mean()


def make_total_loss(w_vertex=1.0, w_smooth=0.02, w_l1=1.0, w_floor=0.05, floor_z=0.0,
                     floor_warmup_epochs=0):
    def loss_fn(pred_disp, target_disp, scale, rest_pose, root_rotation, root_translation, epoch=None):
        pred_n, target_n = _normalize(pred_disp, target_disp, scale)
        L_vertex = vertex_rms_loss(pred_n, target_n)
        L_smooth = smoothness_loss(pred_n)
        L_l1 = torch.mean(torch.abs(pred_n - target_n))
        L_floor = floor_penalty_loss(pred_disp, rest_pose, root_rotation, root_translation, scale, floor_z)
        if floor_warmup_epochs > 0 and epoch is not None:
            w_floor_eff = w_floor * min(1.0, epoch / floor_warmup_epochs)
        else:
            w_floor_eff = w_floor
        return w_vertex * L_vertex + w_smooth * L_smooth + w_l1 * L_l1 + w_floor_eff * L_floor
    return loss_fn


@torch.no_grad()
def evaluate_metrics(pred_disp, target_disp, scale, rest_pose=None, root_rotation=None,
                      root_translation=None, floor_z=0.0):
    pred_n, target_n = _normalize(pred_disp, target_disp, scale)
    metrics = {
        "DisPer": 100.0 * disper_loss(pred_n, target_n).item(),
        "MaxAvg": metric_maxavg(pred_n, target_n),
        "ERMS": metric_erms(pred_n, target_n),
    }
    if rest_pose is not None:
        depth = floor_penetration_depth(pred_disp, rest_pose, root_rotation, root_translation, floor_z)
        metrics["FloorPenMax"] = depth.max().item()
    return metrics


def floor_penalty_from_world_pos(world_pos, scale, floor_z=0.0):
    world_z = world_pos[..., 2]
    depth = torch.clamp(floor_z - world_z, min=0.0)
    scale_r = scale.view(-1, 1, 1).clamp_min(1e-6)
    return (depth / scale_r).pow(2).mean()


def smoothness_loss_world(world_pos, scale):
    if world_pos.shape[1] < 2:
        return torch.tensor(0.0, device=world_pos.device)
    scale_r = scale.view(-1, 1, 1, 1).clamp_min(1e-6)
    diff = (world_pos[:, 1:] - world_pos[:, :-1]) / scale_r
    return diff.pow(2).mean()


def contact_noslide_loss(world_pos, scale, floor_z=0.0, z_eps_frac=0.008, foot_mask=None):
    if foot_mask is not None:
        world_pos = world_pos[..., foot_mask, :]
    z_eps = z_eps_frac * scale.view(-1, 1, 1)
    z = world_pos[..., 2]
    contact = (z[:, :-1] < floor_z + z_eps) & (z[:, 1:] < floor_z + z_eps)

    horiz = world_pos[..., :2]
    horiz_disp = horiz[:, 1:] - horiz[:, :-1]
    scale_r = scale.view(-1, 1, 1, 1).clamp_min(1e-6)
    horiz_speed2 = ((horiz_disp / scale_r) ** 2).sum(dim=-1)

    masked = horiz_speed2 * contact.float()
    denom = contact.float().sum().clamp_min(1.0)
    return masked.sum() / denom


def contact_reference_velocity_loss(world_pos, world_a, world_b, weights, scale,
                                     floor_z=0.0, z_eps_frac=0.008, foot_mask=None):
    if foot_mask is not None:
        world_pos = world_pos[..., foot_mask, :]
        world_a = world_a[..., foot_mask, :]
        world_b = world_b[..., foot_mask, :]
    z_eps = z_eps_frac * scale.view(-1, 1, 1)
    za, zb = world_a[..., 2], world_b[..., 2]
    contact_a = (za[:, :-1] < floor_z + z_eps) & (za[:, 1:] < floor_z + z_eps)
    contact_b = (zb[:, :-1] < floor_z + z_eps) & (zb[:, 1:] < floor_z + z_eps)
    contact = contact_a | contact_b

    scale_r = scale.view(-1, 1, 1, 1).clamp_min(1e-6)
    horiz_blend = (world_pos[:, 1:, :, :2] - world_pos[:, :-1, :, :2]) / scale_r
    horiz_a = (world_a[:, 1:, :, :2] - world_a[:, :-1, :, :2]) / scale_r
    horiz_b = (world_b[:, 1:, :, :2] - world_b[:, :-1, :, :2]) / scale_r

    w = weights[1:].view(1, -1, 1, 1)
    horiz_ref = (1 - w) * horiz_a + w * horiz_b

    diff2 = ((horiz_blend - horiz_ref) ** 2).sum(dim=-1)
    masked = diff2 * contact.float()
    denom = contact.float().sum().clamp_min(1.0)
    return masked.sum() / denom


def foot_velocity_supervision_loss(world_pred, world_gt, scale, floor_z=0.0, z_eps_frac=0.008, foot_mask=None):
    if foot_mask is not None:
        world_pred = world_pred[..., foot_mask, :]
        world_gt = world_gt[..., foot_mask, :]
    z_eps = z_eps_frac * scale.view(-1, 1, 1)
    z_gt = world_gt[..., 2]
    contact = (z_gt[:, :-1] < floor_z + z_eps) & (z_gt[:, 1:] < floor_z + z_eps)

    scale_r = scale.view(-1, 1, 1, 1).clamp_min(1e-6)
    horiz_pred = (world_pred[:, 1:, :, :2] - world_pred[:, :-1, :, :2]) / scale_r
    horiz_gt = (world_gt[:, 1:, :, :2] - world_gt[:, :-1, :, :2]) / scale_r

    diff2 = ((horiz_pred - horiz_gt) ** 2).sum(dim=-1)
    masked = diff2 * contact.float()
    denom = contact.float().sum().clamp_min(1.0)
    return masked.sum() / denom


def build_knn_laplacian(rest_pose, k=8):
    import numpy as np
    from scipy.spatial import cKDTree
    V = rest_pose.shape[0]
    tree = cKDTree(rest_pose)
    _, idx = tree.query(rest_pose, k=k + 1)
    neighbours = idx[:, 1:]

    rows = np.repeat(np.arange(V), k)
    cols = neighbours.reshape(-1)
    off_diag_vals = np.full(rows.shape, -1.0 / k, dtype=np.float32)
    diag_idx = np.arange(V)
    all_rows = np.concatenate([rows, diag_idx])
    all_cols = np.concatenate([cols, diag_idx])
    all_vals = np.concatenate([off_diag_vals, np.ones(V, dtype=np.float32)])

    indices = torch.from_numpy(np.stack([all_rows, all_cols])).long()
    values = torch.from_numpy(all_vals)
    return torch.sparse_coo_tensor(indices, values, size=(V, V)).coalesce()


def _apply_laplacian(L, x):
    _, Fr, V, _ = x.shape
    flat = x.squeeze(0).permute(1, 0, 2).reshape(V, Fr * 3)
    out = torch.sparse.mm(L, flat)
    return out.reshape(V, Fr, 3).permute(1, 0, 2).unsqueeze(0)


def curvature_consistency_loss(mesh_blend, mesh_a, mesh_b, weights, laplacian):
    w = weights.view(1, -1, 1, 1)
    target = (1 - w) * _apply_laplacian(laplacian, mesh_a) + w * _apply_laplacian(laplacian, mesh_b)
    pred = _apply_laplacian(laplacian, mesh_blend)
    num = torch.norm(pred - target, dim=-1).pow(2).sum()
    den = torch.norm(target, dim=-1).pow(2).sum() + 1e-8
    return num / den


def make_synthesis_loss(w_floor=0.05, w_contact=0.5, w_smooth=0.02,
                         w_curvature=0.05, w_anchor=0.1, floor_z=0.0):
    def loss_fn(world_pos, scale, c_blend, c_anchor, mesh_blend_local=None,
                mesh_a_local=None, mesh_b_local=None, weights=None, laplacian=None):
        total = (w_floor * floor_penalty_from_world_pos(world_pos, scale, floor_z)
                 + w_contact * contact_noslide_loss(world_pos, scale, floor_z)
                 + w_smooth * smoothness_loss_world(world_pos, scale))

        scale_r = scale.view(-1, 1, 1).clamp_min(1e-6)
        anchor = (((c_blend - c_anchor) / scale_r) ** 2).mean()
        total = total + w_anchor * anchor

        if laplacian is not None and mesh_blend_local is not None:
            total = total + w_curvature * curvature_consistency_loss(
                mesh_blend_local, mesh_a_local, mesh_b_local, weights, laplacian)
        return total
    return loss_fn
