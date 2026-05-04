from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import PatientTrajectory, Vocab
from .model import GrailModel


def _pca_2d(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 2:
        return x[:, :2]
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    comp = vt[:2].T
    return x_centered @ comp


def project_embeddings_2d(model: GrailModel) -> np.ndarray:
    with torch.no_grad():
        # Keep manifold ops on the model device, then move to CPU for numpy/PCA.
        emb = model.embeddings.detach()
        tangent = model.manifold.logmap0(emb).detach().cpu().numpy()
    return _pca_2d(tangent)


def plot_poincare_disk_embeddings(
    model: GrailModel,
    vocab: Vocab,
    out_path: str,
    max_points: int = 6000,
    seed: int = 13,
) -> None:
    coords = project_embeddings_2d(model)
    node_ids = np.arange(coords.shape[0])
    if len(node_ids) > max_points:
        rng = np.random.default_rng(seed)
        node_ids = np.sort(rng.choice(node_ids, size=max_points, replace=False))
    coords = coords[node_ids]

    colors: List[str] = []
    palette: Dict[str, str] = {"dx": "#1f77b4", "proc": "#ff7f0e", "med": "#2ca02c", "lab": "#d62728"}
    for nid in node_ids:
        colors.append(palette.get(vocab.node_type.get(int(nid), "other"), "#7f7f7f"))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    circ = plt.Circle((0.0, 0.0), 1.0, fill=False, linewidth=1.2, color="#444444", alpha=0.8)
    ax.add_patch(circ)
    ax.scatter(coords[:, 0], coords[:, 1], s=4, c=colors, alpha=0.6)
    ax.set_title("Node Embeddings (2D Tangent PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_patient_trajectories(
    model: GrailModel,
    trajectories: List[PatientTrajectory],
    out_path: str,
    max_patients: int = 32,
    seed: int = 13,
) -> None:
    elig = [traj for traj in trajectories if len(traj.visits) >= 2]
    if not elig:
        return
    rng = np.random.default_rng(seed)
    if len(elig) > max_patients:
        idxs = rng.choice(np.arange(len(elig)), size=max_patients, replace=False)
        selected = [elig[int(i)] for i in idxs]
    else:
        selected = elig

    all_emb = model.embeddings.detach()
    points = []
    with torch.no_grad():
        for traj in selected:
            visit_pts = []
            for visit in traj.visits:
                codes = [c for values in visit.values() for c in values]
                if not codes:
                    continue
                code_tensor = torch.tensor([codes], device=all_emb.device)
                z = model.barycenter(code_tensor).squeeze(0)
                visit_pts.append(model.manifold.logmap0(z).cpu().numpy())
            if len(visit_pts) >= 2:
                points.append(np.stack(visit_pts, axis=0))

    if not points:
        return

    mat = np.concatenate(points, axis=0)
    mat_2d = _pca_2d(mat)

    rebuilt = []
    offset = 0
    for seq in points:
        n = seq.shape[0]
        rebuilt.append(mat_2d[offset:offset + n])
        offset += n

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    for seq in rebuilt:
        ax.plot(seq[:, 0], seq[:, 1], alpha=0.45, linewidth=1.2)
        ax.scatter(seq[0, 0], seq[0, 1], s=10, alpha=0.65)
        ax.scatter(seq[-1, 0], seq[-1, 1], s=10, alpha=0.9)
    ax.set_title("Patient Trajectories (Visit Barycenters in 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
