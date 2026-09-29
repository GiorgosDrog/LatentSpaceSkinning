import os
import numpy as np

ROOT_BONE_INDEX = 0


def root_relative_mesh(mesh, bones):
    R_root = bones[:, ROOT_BONE_INDEX, :, :3]
    t_root = bones[:, ROOT_BONE_INDEX, :, 3]
    mesh_centered = mesh - t_root[:, None, :]
    return np.einsum('fji,fvj->fvi', R_root, mesh_centered).astype(np.float32)


def build_pca_basis(anim_prefixes, num_components=64, report_ceiling_for=(8, 16, 32, 64, 128)):
    all_disp = []
    rest_surface = None
    scale = None

    for prefix in anim_prefixes:
        anim_data_dir = os.path.dirname(prefix)
        char_root = os.path.dirname(anim_data_dir)
        static_dir = os.path.join(char_root, "static_data")
        base_name = os.path.basename(prefix)

        if rest_surface is None:
            rest_surface = np.load(os.path.join(static_dir, "rest_pose.npy")).astype(np.float32)
            scale = np.linalg.norm(rest_surface.max(0) - rest_surface.min(0))

        mesh = np.load(os.path.join(anim_data_dir, f"{base_name}_FullMesh.npy")).astype(np.float32)
        bones_raw = np.load(os.path.join(anim_data_dir, f"{base_name}_Bones.npy")).astype(np.float32)

        if mesh.shape[1] != rest_surface.shape[0]:
            raise ValueError(
                f"Vertex count mismatch for {base_name}: rest_pose.npy has "
                f"{rest_surface.shape[0]} vertices but {base_name}_FullMesh.npy has "
                f"{mesh.shape[1]} vertices. These must come from the SAME mesh export "
                f"-- rest_pose.npy is likely stale, from a different mesh, or a proxy "
                f"surface (e.g. a volumetric/tetrahedra representation) rather than the "
                f"actual skinned mesh. Re-run extract_static_data.py for this character "
                f"so rest_pose.npy matches FullMesh.npy's vertex count."
            )

        F = bones_raw.shape[0]
        B = bones_raw.shape[1] // 12
        bones = bones_raw.reshape(F, B, 3, 4)

        mesh_local = root_relative_mesh(mesh, bones)
        disp = (mesh_local - rest_surface[None, :, :]).reshape(F, -1)
        all_disp.append(disp)
        print(f"  {base_name}: {F} frames")

    X = np.concatenate(all_disp, axis=0)
    print(f"\nPCA input matrix: {X.shape} ({X.shape[0]} frames, {X.shape[1]} = V*3)")

    mean = X.mean(axis=0)
    X_centered = X - mean[None, :]

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    max_k = Vt.shape[0]
    total_var = (X_centered ** 2).sum()

    print("\nReconstruction ceiling by K (perfect coefficient prediction):")
    for K in report_ceiling_for:
        if K > max_k:
            break
        comp = Vt[:K]
        coeffs = X_centered @ comp.T
        recon = mean[None, :] + coeffs @ comp
        err = np.linalg.norm((recon - X).reshape(X.shape[0], -1, 3), axis=-1)
        disper_ceiling = 100.0 * err.mean() / scale
        explained = (S[:K] ** 2).sum() / total_var
        print(f"  K={K:4d}: explains {100*explained:6.2f}% variance | DisPer ceiling = {disper_ceiling:.4f}%")

    K = min(num_components, max_k)
    components = Vt[:K].astype(np.float32)
    print(f"\nUsing K={K} components for training.")

    return mean.astype(np.float32), components


if __name__ == "__main__":
    root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
    character = "dragon"
    animation_dir = os.path.join(root_dir, character, "animation_data")

    if(character == "monster"):
        train_prefixes = [
            os.path.join(animation_dir, f"Armature.{str(i).zfill(3)}")
            for i in range(1, 10)
        ]
    else:
        EXCLUDED_SEQUENCES = {7,15}    
        train_prefixes = [
            os.path.join(animation_dir, f"Armature.{str(i).zfill(3)}")
            for i in range(1, 2) if i not in EXCLUDED_SEQUENCES
        ]

    NUM_COMPONENTS = 64
    mean, components = build_pca_basis(train_prefixes, num_components=NUM_COMPONENTS)

    out_dir = os.path.join(root_dir, character, "static_data")
    np.save(os.path.join(out_dir, "pca_mean.npy"), mean)
    np.save(os.path.join(out_dir, "pca_components.npy"), components)
    print(f"\nSaved pca_mean.npy {mean.shape} and pca_components.npy {components.shape} to {out_dir}")
