import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx
import torch

from grail.data import build_patient_trajectories, load_lab_groups, load_table, node_display
from grail.model import GrailModel


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _unique_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        p = f"{path}_v{idx}"
        if not os.path.exists(p):
            return p
        idx += 1


def _node_modality_ids(vocab, modality_cols: Dict[str, str]) -> Tuple[torch.Tensor, Dict[str, int]]:
    modalities = list(modality_cols.keys())
    for m in sorted(set(vocab.node_type.values())):
        if m not in modalities:
            modalities.append(m)
    mod_to_idx = {m: i for i, m in enumerate(modalities)}
    arr = torch.zeros(len(vocab.id_to_node), dtype=torch.long)
    for nid, m in vocab.node_type.items():
        arr[int(nid)] = int(mod_to_idx.get(m, 0))
    return arr, mod_to_idx


def _node_label(vocab, node_id: int, max_len: int = 42) -> str:
    text = str(node_display(vocab, int(node_id)))
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _visit_codes(visit: dict) -> List[int]:
    return [int(c) for vals in visit.values() for c in vals]


def _visit_barycenter(model: GrailModel, visit: dict) -> Optional[torch.Tensor]:
    codes = _visit_codes(visit)
    if not codes:
        return None
    with torch.no_grad():
        code_tensor = torch.tensor([codes], device=model.embeddings.device)
        return model.barycenter(code_tensor).squeeze(0)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 2:
        return x[:, :2]
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    return x_centered @ vt[:2].T


def _project_to_disk(points: np.ndarray, radius_scale: float = 0.95) -> np.ndarray:
    pts2 = _pca_2d(points)
    r = np.linalg.norm(pts2, axis=1, keepdims=True)
    if np.all(r < 1e-12):
        return pts2
    q = pts2 / np.maximum(r, 1e-12)
    # Smoothly squash to open unit disk while preserving rank order of radii.
    s = np.tanh(r / np.percentile(r, 90))
    return q * s * radius_scale


def _mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x)
    y2 = np.sum(y * y)
    xy = np.sum(x * y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / max(den, 1e-12)


def _mobius_scalar_mul(t: float, x: np.ndarray) -> np.ndarray:
    nx = float(np.linalg.norm(x))
    if nx < 1e-12:
        return x.copy()
    return np.tanh(t * np.arctanh(min(nx, 1 - 1e-9))) * x / nx


def _poincare_geodesic(x: np.ndarray, y: np.ndarray, n: int = 80) -> np.ndarray:
    # geodesic gamma(t) = x ⊕ ( t ⊗ ( (-x) ⊕ y ) )
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx >= 1.0:
        x = x / (nx + 1e-12) * 0.999
    if ny >= 1.0:
        y = y / (ny + 1e-12) * 0.999
    minus_x = -x
    v = _mobius_add(minus_x, y)
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        p = _mobius_add(x, _mobius_scalar_mul(float(t), v))
        npn = np.linalg.norm(p)
        if npn >= 1.0:
            p = p / (npn + 1e-12) * 0.999
        pts.append(p)
    return np.asarray(pts)


def _modality_palette() -> Dict[str, str]:
    return {"dx": "#1f77b4", "proc": "#ff7f0e", "med": "#2ca02c", "lab": "#d62728"}


def _edge_kind(edge_type: str) -> str:
    t = str(edge_type)
    if t.startswith("hier:"):
        return "hier"
    if t.startswith("cross:"):
        return "pmi"
    return "other"


def _node_depths_by_modality(graph: nx.DiGraph, vocab) -> Dict[int, int]:
    depths: Dict[int, int] = {}
    for mod, root in vocab.modality_roots.items():
        root = int(root)
        if root not in graph:
            continue
        depths[root] = 0
        q = [root]
        while q:
            u = q.pop(0)
            du = depths.get(u, 0)
            for v in graph.successors(u):
                ed = graph.get_edge_data(u, v, default={})
                if _edge_kind(ed.get("type", "")) != "hier":
                    continue
                vv = int(v)
                if vocab.node_type.get(vv) != mod:
                    continue
                if vv not in depths or du + 1 < depths[vv]:
                    depths[vv] = du + 1
                    q.append(vv)
    return depths


def _build_tension_subgraph(graph: nx.DiGraph, vocab, trajectories, max_nodes: int, seed: int) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    freq: Dict[int, int] = {}
    for traj in trajectories:
        for visit in traj.visits:
            for vals in visit.values():
                for c in vals:
                    cc = int(c)
                    freq[cc] = freq.get(cc, 0) + 1

    keep = set(int(v) for v in vocab.modality_roots.values())
    mods = list(vocab.modality_roots.keys())
    budget_per_mod = max(10, (max_nodes - len(keep)) // max(1, len(mods)))
    for mod in mods:
        nodes = [int(n) for n, m in vocab.node_type.items() if m == mod and n in graph]
        nodes.sort(key=lambda n: freq.get(n, 0), reverse=True)
        chosen = nodes[:budget_per_mod]
        for n in chosen:
            keep.add(n)
            # Add ancestor chain to preserve local hierarchy context.
            curr = n
            for _ in range(12):
                preds = [
                    int(p)
                    for p in graph.predecessors(curr)
                    if _edge_kind(graph.get_edge_data(p, curr, default={}).get("type", "")) == "hier"
                ]
                if not preds:
                    break
                parent = preds[0]
                keep.add(parent)
                curr = parent

    # Keep strongest cross-modal PMI links touching selected nodes.
    cross_edges = []
    for u, v, data in graph.edges(data=True):
        if _edge_kind(data.get("type", "")) != "pmi":
            continue
        uu = int(u)
        vv = int(v)
        if uu in keep or vv in keep:
            w = float(data.get("weight", 0.0))
            cross_edges.append((w, uu, vv))
    cross_edges.sort(key=lambda x: x[0], reverse=True)
    for _, u, v in cross_edges[: max(100, max_nodes)]:
        keep.add(int(u))
        keep.add(int(v))

    if len(keep) > max_nodes:
        # Keep roots + frequency-heavy + PMI endpoints.
        roots = set(int(v) for v in vocab.modality_roots.values())
        rest = [n for n in keep if n not in roots]
        rest.sort(key=lambda n: freq.get(n, 0), reverse=True)
        top = rest[: max_nodes - len(roots)]
        keep = roots.union(top)
        # small random diversity for ties
        if len(keep) < max_nodes and len(rest) > len(top):
            cand = [n for n in rest[len(top):] if n not in keep]
            if cand:
                extra_n = min(max_nodes - len(keep), len(cand))
                extra = rng.choice(np.array(cand), size=extra_n, replace=False).tolist()
                keep.update(int(x) for x in extra)

    return graph.subgraph(sorted(keep)).copy()


def plot_hierarchy_tension_graph(
    graph: nx.DiGraph,
    vocab,
    trajectories,
    out_path: str,
    max_nodes: int = 260,
    seed: int = 13,
) -> None:
    sub = _build_tension_subgraph(graph, vocab, trajectories, max_nodes=max_nodes, seed=seed)
    depths = _node_depths_by_modality(sub, vocab)
    palette = _modality_palette()
    mods = sorted(set(vocab.node_type.get(int(n), "other") for n in sub.nodes()))
    mod_to_x = {m: i * 3.3 for i, m in enumerate(mods)}

    rng = np.random.default_rng(seed)
    pos = {}
    for n in sub.nodes():
        nn = int(n)
        m = vocab.node_type.get(nn, "other")
        x0 = mod_to_x.get(m, 0.0)
        d = depths.get(nn, 4)
        x = x0 + float(rng.normal(0.0, 0.30))
        y = -float(d) + float(rng.normal(0.0, 0.12))
        pos[nn] = (x, y)

    hier_edges = []
    pmi_edges = []
    for u, v, data in sub.edges(data=True):
        kind = _edge_kind(data.get("type", ""))
        if kind == "hier":
            hier_edges.append((int(u), int(v)))
        elif kind == "pmi":
            pmi_edges.append((int(u), int(v), float(data.get("weight", 0.0))))

    fig, ax = plt.subplots(figsize=(13.5, 9.0), dpi=260)
    nx.draw_networkx_edges(
        sub,
        pos,
        edgelist=hier_edges,
        edge_color="#8c8c8c",
        width=0.7,
        alpha=0.45,
        arrows=False,
        ax=ax,
    )
    if pmi_edges:
        pmi_simple = [(u, v) for u, v, _ in pmi_edges]
        w = np.array([max(0.1, ww) for _, _, ww in pmi_edges], dtype=float)
        w_norm = 0.6 + 1.6 * (w - w.min()) / max((w.max() - w.min()), 1e-8)
        nx.draw_networkx_edges(
            sub,
            pos,
            edgelist=pmi_simple,
            edge_color="#d62728",
            width=w_norm.tolist(),
            alpha=0.38,
            arrows=False,
            ax=ax,
            connectionstyle="arc3,rad=0.08",
        )

    node_colors = [palette.get(vocab.node_type.get(int(n), "other"), "#777777") for n in sub.nodes()]
    node_sizes = []
    for n in sub.nodes():
        n = int(n)
        if n in vocab.modality_roots.values():
            node_sizes.append(260)
        else:
            node_sizes.append(46)
    nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.4, edgecolors="#222222", ax=ax, alpha=0.95)

    # Labels: roots + top frequency leaves + strong PMI endpoints.
    freq: Dict[int, int] = {}
    for traj in trajectories:
        for visit in traj.visits:
            for vals in visit.values():
                for c in vals:
                    cc = int(c)
                    if cc in sub:
                        freq[cc] = freq.get(cc, 0) + 1
    label_nodes = set(int(v) for v in vocab.modality_roots.values() if int(v) in sub)
    top_freq = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:42]
    label_nodes.update(int(n) for n, _ in top_freq)
    for u, v, _ in sorted(pmi_edges, key=lambda x: x[2], reverse=True)[:24]:
        label_nodes.add(int(u))
        label_nodes.add(int(v))
    labels = {int(n): _node_label(vocab, int(n), 30) for n in label_nodes if int(n) in sub}
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=6, font_color="#111111", ax=ax)

    legend = [
        Line2D([0], [0], marker="o", color="w", label="Diagnosis node", markerfacecolor=palette.get("dx", "#1f77b4"), markeredgecolor="#222", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Procedure node", markerfacecolor=palette.get("proc", "#ff7f0e"), markeredgecolor="#222", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Medication node", markerfacecolor=palette.get("med", "#2ca02c"), markeredgecolor="#222", markersize=6),
        Line2D([0], [0], color="#8c8c8c", lw=1.5, label="Hierarchy edge"),
        Line2D([0], [0], color="#d62728", lw=1.8, label="Cross-modal PMI edge"),
    ]
    if "lab" in mods:
        legend.insert(
            3,
            Line2D([0], [0], marker="o", color="w", label="Lab node", markerfacecolor=palette.get("lab", "#d62728"), markeredgecolor="#222", markersize=6),
        )
    ax.legend(handles=legend, loc="upper right", fontsize=8, frameon=True)
    ax.set_title("Hierarchy-Preserved Graph with Cross-Modal PMI Tension (sampled subgraph)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_global_poincare_map(model: GrailModel, vocab, trajectories, out_path: str, max_points: int = 9000, seed: int = 13) -> None:
    with torch.no_grad():
        tang = model.manifold.logmap0(model.embeddings).detach().cpu().numpy()
    coords = _project_to_disk(tang)
    ids = np.arange(coords.shape[0])
    rng = np.random.default_rng(seed)
    if len(ids) > max_points:
        keep = np.sort(rng.choice(ids, size=max_points, replace=False))
    else:
        keep = ids

    freq = {}
    for traj in trajectories:
        for visit in traj.visits:
            for vals in visit.values():
                for c in vals:
                    c = int(c)
                    freq[c] = freq.get(c, 0) + 1
    top_nodes = [n for n, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:25]]
    roots = sorted(vocab.modality_roots.values())
    highlights = set(top_nodes + roots)

    palette = _modality_palette()
    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=240)
    circle = plt.Circle((0, 0), 1.0, edgecolor="black", facecolor="none", linewidth=1.2, alpha=0.8)
    ax.add_patch(circle)

    c = [palette.get(vocab.node_type.get(int(i), "other"), "#777777") for i in keep]
    ax.scatter(coords[keep, 0], coords[keep, 1], s=6, c=c, alpha=0.2, linewidths=0)

    for nid in highlights:
        if 0 <= int(nid) < coords.shape[0]:
            x, y = coords[int(nid)]
            col = palette.get(vocab.node_type.get(int(nid), "other"), "#222222")
            ax.scatter([x], [y], s=42, c=col, edgecolors="black", linewidths=0.4, zorder=5)
            ax.annotate(_node_label(vocab, int(nid), 38), (x, y), fontsize=6, xytext=(4, 2), textcoords="offset points")

    ax.set_title("Global Hyperbolic Map (Poincare disk projection)")
    ax.set_xlabel("disk-x")
    ax.set_ylabel("disk-y")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_patient_geodesic_trajectory(model: GrailModel, vocab, traj, disk_coords: np.ndarray, out_path: str) -> None:
    points = []
    visits_idx = []
    for i, visit in enumerate(traj.visits):
        codes = _visit_codes(visit)
        if not codes:
            continue
        with torch.no_grad():
            code_tensor = torch.tensor([codes], device=model.embeddings.device)
            center = model.barycenter(code_tensor).squeeze(0)
            d = model.manifold.dist(center.unsqueeze(0), model.embeddings).squeeze(0)
            nid = int(torch.argmin(d).item())
        points.append(disk_coords[nid])
        visits_idx.append(i)
    if len(points) < 2:
        return

    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=240)
    circle = plt.Circle((0, 0), 1.0, edgecolor="black", facecolor="none", linewidth=1.2, alpha=0.8)
    ax.add_patch(circle)

    points = np.asarray(points)
    for i in range(len(points) - 1):
        geo = _poincare_geodesic(points[i], points[i + 1], n=80)
        ax.plot(geo[:, 0], geo[:, 1], color="#1f77b4", alpha=0.7, linewidth=1.6)

    ax.scatter(points[:, 0], points[:, 1], s=30, c="#222222", alpha=0.9)
    ax.scatter(points[0:1, 0], points[0:1, 1], s=70, c="#2ca02c", marker="o", edgecolors="black", linewidths=0.4, label="start")
    ax.scatter(points[-1:, 0], points[-1:, 1], s=80, c="#d62728", marker="s", edgecolors="black", linewidths=0.4, label="end")
    for j, p in enumerate(points):
        ax.annotate(f"V{visits_idx[j] + 1}", (p[0], p[1]), fontsize=7, xytext=(3, 1), textcoords="offset points")

    ax.set_title(f"Patient Trajectory with Hyperbolic Geodesic Arcs (subject_id={traj.subject_id})")
    ax.set_xlabel("disk-x")
    ax.set_ylabel("disk-y")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_query_neighborhood(model: GrailModel, vocab, traj, visit_index: int, out_path: str, top_k: int = 60, annotate_k: int = 25) -> None:
    if visit_index < 0 or visit_index >= len(traj.visits):
        return
    visit = traj.visits[visit_index]
    codes = _visit_codes(visit)
    if not codes:
        return

    with torch.no_grad():
        center = model.barycenter(torch.tensor([codes], device=model.embeddings.device)).squeeze(0)
        d = model.manifold.dist(center.unsqueeze(0), model.embeddings).squeeze(0)
        k = min(int(top_k), int(d.numel()))
        nn = torch.topk(-d, k=k).indices.cpu().numpy().astype(int)

    palette = _modality_palette()
    in_query = set(int(c) for c in codes)
    rows = []
    for rank, nid in enumerate(nn, start=1):
        rows.append((rank, nid, float(d[nid].item()), nid in in_query, vocab.node_type.get(int(nid), "other")))

    fig, ax = plt.subplots(figsize=(12.0, 5.8), dpi=240)
    x = np.arange(1, len(rows) + 1)
    y = np.array([r[2] for r in rows], dtype=float)
    cols = ["#111111" if r[3] else palette.get(r[4], "#999999") for r in rows]
    ax.bar(x, y, color=cols, alpha=0.9, width=0.9)
    ax.plot(x, y, color="#1f77b4", alpha=0.65, linewidth=1.2)

    for r in rows[: min(annotate_k, len(rows))]:
        rank, nid, distv, hit, _ = r
        tag = "*" if hit else ""
        ax.annotate(f"{rank}:{_node_label(vocab, nid, 26)}{tag}", (rank, distv), fontsize=6, rotation=65, xytext=(0, 3), textcoords="offset points")

    overlap = sum(1 for r in rows if r[3])
    ax.set_title(
        f"Query Neighborhood (subject={traj.subject_id}, visit={visit_index + 1}) | "
        f"top-{len(rows)} overlap-with-visit={overlap}"
    )
    ax.set_xlabel("retrieval rank")
    ax.set_ylabel("hyperbolic distance to query barycenter")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trajectory_diagnostics(model: GrailModel, traj, out_path: str) -> None:
    centers = []
    visit_ids = []
    for i, visit in enumerate(traj.visits):
        z = _visit_barycenter(model, visit)
        if z is None:
            continue
        centers.append(z)
        visit_ids.append(i + 1)
    if len(centers) < 2:
        return

    with torch.no_grad():
        radial = [float(model.manifold.dist0(z.unsqueeze(0)).item()) for z in centers]
        step = [float(model.manifold.dist(centers[i].unsqueeze(0), centers[i + 1].unsqueeze(0)).item()) for i in range(len(centers) - 1)]
        drift = [float(model.manifold.dist(centers[0].unsqueeze(0), z.unsqueeze(0)).item()) for z in centers]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), dpi=220, sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(visit_ids, radial, marker="o", color="#1f77b4")
    ax1.set_ylabel("dist(origin)")
    ax1.set_title("Radial Norm by Visit")
    ax1.grid(alpha=0.2)

    ax2.plot(visit_ids[1:], step, marker="o", color="#ff7f0e")
    ax2.set_ylabel("dist(V_t, V_t+1)")
    ax2.set_title("Hyperbolic Step Length")
    ax2.grid(alpha=0.2)

    ax3.plot(visit_ids, drift, marker="o", color="#2ca02c")
    ax3.set_ylabel("dist(V1, V_t)")
    ax3.set_title("Drift from First Visit")
    ax3.set_xlabel("visit index")
    ax3.grid(alpha=0.2)

    fig.suptitle(f"Trajectory Diagnostics (subject_id={traj.subject_id})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_embeddings_3d_html(model: GrailModel, vocab, trajectories, out_path: str, max_points: int = 12000, seed: int = 13) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        _log("plotly is not installed; skipping 3D HTML visualization.")
        return False

    with torch.no_grad():
        tang = model.manifold.logmap0(model.embeddings).detach().cpu().numpy()
    n = tang.shape[0]
    ids = np.arange(n)
    rng = np.random.default_rng(seed)
    if n > max_points:
        ids = np.sort(rng.choice(ids, size=max_points, replace=False))
    x = tang[ids]
    if x.shape[1] < 3:
        x = np.pad(x, ((0, 0), (0, 3 - x.shape[1])), mode="constant")
    else:
        x_center = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_center, full_matrices=False)
        x = x_center @ vt[:3].T

    freq = {}
    for traj in trajectories:
        for visit in traj.visits:
            for vals in visit.values():
                for c in vals:
                    c = int(c)
                    freq[c] = freq.get(c, 0) + 1
    top_nodes = set(n for n, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:30])
    root_nodes = set(int(v) for v in vocab.modality_roots.values())
    highlight = top_nodes.union(root_nodes)

    palette = _modality_palette()
    colors = [palette.get(vocab.node_type.get(int(i), "other"), "#777777") for i in ids]
    sizes = [7 if int(i) in highlight else 3 for i in ids]
    opacities = [0.9 if int(i) in highlight else 0.2 for i in ids]
    text = []
    for i in ids:
        nid = int(i)
        tag = ""
        if nid in root_nodes:
            tag += " [root]"
        if nid in top_nodes:
            tag += " [freq]"
        text.append(f"{nid}: {_node_label(vocab, nid, 80)}{tag}")

    trace = go.Scatter3d(
        x=x[:, 0],
        y=x[:, 1],
        z=x[:, 2],
        mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.75),
        text=text,
        hovertemplate="%{text}<extra></extra>",
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title="Interactive 3D Tangent-Space Embedding View",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return True


def _select_trajectory(trajectories, subject_id: Optional[str]):
    if subject_id is not None:
        for t in trajectories:
            if str(t.subject_id) == str(subject_id):
                return t
        return None
    traj_sorted = sorted(trajectories, key=lambda t: len(t.visits), reverse=True)
    return traj_sorted[0] if traj_sorted else None


def _select_trajectory_batch(trajectories, n: int, seed: int):
    eligible = [t for t in trajectories if len(t.visits) >= 2]
    if not eligible:
        return []
    eligible = sorted(eligible, key=lambda t: len(t.visits), reverse=True)
    if len(eligible) <= n:
        return eligible
    # Deterministic sample from strongest trajectories.
    top_pool = eligible[: max(n * 3, n)]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(len(top_pool)), size=n, replace=False))
    return [top_pool[int(i)] for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualization suite for GRAIL hyperbolic artifacts.")
    parser.add_argument("--artifact-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--format", type=str, default="pickle")
    parser.add_argument("--label-map", type=str, default=None)
    parser.add_argument("--lab-groups", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--visit-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--neighbor-top-k", type=int, default=60)
    parser.add_argument("--annotate-k", type=int, default=25)
    parser.add_argument("--graph-max-nodes", type=int, default=260)
    parser.add_argument("--skip-graph-viz", action="store_true")
    parser.add_argument("--save-3d-html", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.artifact_path):
        raise FileNotFoundError(f"Artifact not found: {args.artifact_path}")

    _log(f"Loading artifact: {args.artifact_path}")
    artifact = torch.load(args.artifact_path, map_location="cpu", weights_only=False)
    vocab = artifact["vocab"]
    edge_type_ids = artifact["edge_type_ids"]
    cfg = artifact["model_cfg"]
    data_cfg = artifact.get("data_cfg", {})
    modality_cols = data_cfg.get("modality_cols", {"dx": "icd_code", "proc": "pro_code", "med": "ndc", "lab": "lab_test"})

    node_mod_ids, mod_to_idx = _node_modality_ids(vocab, modality_cols)
    state = artifact.get("model_state", {})
    num_modalities = int(state["modality_weight_logits"].numel()) if "modality_weight_logits" in state else max(1, len(mod_to_idx))

    model = GrailModel(
        num_nodes=len(vocab.node_to_id),
        num_edge_types=max(1, len(edge_type_ids)),
        embedding_dim=cfg["embedding_dim"],
        curvature=cfg["curvature"],
        manifold=cfg["manifold"],
        margin=cfg["margin"],
        softmax_beta=cfg["softmax_beta"],
        barycenter_heads=cfg.get("barycenter_heads", 1),
        barycenter_mixture=cfg.get("barycenter_mixture", False),
        use_central_attention=cfg.get("use_central_attention", False),
        use_idf_weighting=cfg.get("use_idf_weighting", False),
        use_modality_weighting=cfg.get("use_modality_weighting", False),
        central_pos_decay=cfg.get("central_pos_decay", 1.0),
        node_modality_ids=node_mod_ids,
        num_modalities=num_modalities,
        multi_cone_retrieval=cfg.get("multi_cone_retrieval", False),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        _log(f"Artifact state compatibility: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(args.device)
    model.eval()

    subject_id_col = data_cfg.get("subject_id_col", "subject_id")
    time_col = data_cfg.get("time_col", "admittime")
    text_mode = data_cfg.get("text_mode", "hybrid")
    label_map = args.label_map if args.label_map is not None else data_cfg.get("label_map_path", None)

    if args.lab_groups:
        _ = load_lab_groups(args.lab_groups)

    _log(f"Loading data table: {args.data_path}")
    df = load_table(args.data_path, args.format)
    trajectories = build_patient_trajectories(
        df,
        subject_id_col=subject_id_col,
        time_col=time_col,
        modality_cols=modality_cols,
        vocab=vocab,
        text_mode=text_mode,
        label_map_path=label_map,
    )
    if not trajectories:
        raise ValueError("No trajectories built from provided data/artifact settings.")

    selected = _select_trajectory(trajectories, args.subject_id) if args.subject_id is not None else None

    ts = datetime.now().strftime("viz_%Y%m%d_%H%M%S")
    default_out = os.path.join(os.path.dirname(os.path.abspath(args.artifact_path)), ts)
    out_dir = _unique_dir(args.out_dir or default_out)
    os.makedirs(out_dir, exist_ok=False)
    _log(f"Saving outputs to: {out_dir}")

    graph_obj = artifact.get("graph", None)
    if graph_obj is not None and not args.skip_graph_viz:
        p0 = os.path.join(out_dir, "00_graph_hierarchy_tension.png")
        plot_hierarchy_tension_graph(
            graph_obj,
            vocab,
            trajectories,
            p0,
            max_nodes=max(80, int(args.graph_max_nodes)),
            seed=args.seed,
        )
        _log(f"Saved: {p0}")
    elif graph_obj is None:
        _log("Artifact has no graph object; skipping graph-structure visualization.")

    with torch.no_grad():
        tang_all = model.manifold.logmap0(model.embeddings).detach().cpu().numpy()
    disk_coords = _project_to_disk(tang_all)

    p1 = os.path.join(out_dir, "01_global_poincare_map.png")
    plot_global_poincare_map(model, vocab, trajectories, p1, seed=args.seed)
    _log(f"Saved: {p1}")

    if selected is not None:
        batch = [selected]
    else:
        batch = _select_trajectory_batch(trajectories, max(1, int(args.num_samples)), seed=args.seed)
    if not batch:
        raise ValueError("No eligible trajectories with at least 2 visits found.")
    _log(f"Selected {len(batch)} sample trajectory(ies) for per-patient plots.")

    for i, traj in enumerate(batch, start=1):
        visit_idx = min(max(int(args.visit_index), 0), max(len(traj.visits) - 1, 0))
        p2 = os.path.join(out_dir, f"02_sample{i:02d}_patient_{traj.subject_id}_geodesic_trajectory.png")
        plot_patient_geodesic_trajectory(model, vocab, traj, disk_coords, p2)
        _log(f"Saved: {p2}")

        p3 = os.path.join(out_dir, f"03_sample{i:02d}_patient_{traj.subject_id}_visit_{visit_idx + 1}_query_neighborhood.png")
        plot_query_neighborhood(
            model,
            vocab,
            traj,
            visit_idx,
            p3,
            top_k=args.neighbor_top_k,
            annotate_k=args.annotate_k,
        )
        _log(f"Saved: {p3}")

        p4 = os.path.join(out_dir, f"04_sample{i:02d}_patient_{traj.subject_id}_trajectory_diagnostics.png")
        plot_trajectory_diagnostics(model, traj, p4)
        _log(f"Saved: {p4}")

    if args.save_3d_html:
        p5 = os.path.join(out_dir, "05_embeddings_3d_interactive.html")
        if plot_embeddings_3d_html(model, vocab, trajectories, p5, seed=args.seed):
            _log(f"Saved: {p5}")


if __name__ == "__main__":
    main()
