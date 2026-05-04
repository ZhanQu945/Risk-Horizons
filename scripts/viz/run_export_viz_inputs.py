import argparse
import json
import os
from collections import deque
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch


def _load_artifact(path: str) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_embeddings(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "embeddings" in state:
        t = state["embeddings"]
    elif "embeddings.weight" in state:
        t = state["embeddings.weight"]
    else:
        keys = [k for k in state.keys() if "embedding" in k]
        raise KeyError(f"Cannot find embeddings tensor in state_dict. keys(sample)={keys[:5]}")
    return t.detach().cpu().float()


def _hierarchy_subgraph(graph: nx.DiGraph) -> nx.DiGraph:
    hg = nx.DiGraph()
    hg.add_nodes_from(graph.nodes())
    for u, v, d in graph.edges(data=True):
        if str(d.get("type", "")).startswith("hier:"):
            hg.add_edge(int(u), int(v), **d)
    return hg


def _compute_depths(vocab, hier_graph: nx.DiGraph) -> Dict[int, int]:
    depths: Dict[int, int] = {int(n): -1 for n in hier_graph.nodes()}
    roots = {int(v) for v in vocab.modality_roots.values()}
    for r in roots:
        if r not in hier_graph:
            continue
        q = deque([(r, 0)])
        while q:
            node, d = q.popleft()
            if depths.get(node, -1) != -1 and depths[node] <= d:
                continue
            depths[node] = d
            for ch in hier_graph.successors(node):
                q.append((int(ch), d + 1))
    return depths


def _build_selected_paths(vocab, hier_graph: nx.DiGraph, depths: Dict[int, int], n_paths_per_modality: int) -> List[List[int]]:
    paths: List[List[int]] = []
    node_type = vocab.node_type
    for mod, root in vocab.modality_roots.items():
        root = int(root)
        if root not in hier_graph:
            continue
        mod_nodes = [int(n) for n, m in node_type.items() if m == mod and n in hier_graph]
        leaves = [n for n in mod_nodes if hier_graph.out_degree(n) == 0 and depths.get(n, -1) >= 0]
        leaves = sorted(leaves, key=lambda n: depths.get(n, -1), reverse=True)
        chosen = leaves[: max(1, int(n_paths_per_modality))]
        for leaf in chosen:
            try:
                p = nx.shortest_path(hier_graph, source=root, target=leaf)
            except nx.NetworkXNoPath:
                continue
            if len(p) >= 2:
                paths.append([int(x) for x in p])
    return paths


def _save_embeddings_csv(path: str, emb: torch.Tensor) -> None:
    cols = {"node_id": list(range(emb.size(0)))}
    for j in range(emb.size(1)):
        cols[f"dim{j}"] = emb[:, j].tolist()
    pd.DataFrame(cols).to_csv(path, index=False)


def export_inputs(
    artifact_full_path: str,
    artifact_hier_path: str,
    out_dir: str,
    n_paths_per_modality: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    art_full = _load_artifact(artifact_full_path)
    art_hier = _load_artifact(artifact_hier_path)

    vocab_full = art_full["vocab"]
    graph_full = art_full["graph"]
    graph_hier = art_hier["graph"]

    emb_full = _extract_embeddings(art_full["model_state"])
    emb_hier = _extract_embeddings(art_hier["model_state"])
    if emb_full.size(0) != len(vocab_full.id_to_node):
        raise ValueError("Full embedding size does not match full vocab size.")
    if emb_hier.size(0) != len(art_hier["vocab"].id_to_node):
        raise ValueError("Hierarchy embedding size does not match hierarchy vocab size.")

    # For comparison figures we need node ids aligned. We enforce same vocab size for direct comparison.
    if emb_full.size(0) != emb_hier.size(0):
        raise ValueError(
            "Full and hierarchy-only artifacts have different vocab sizes; cannot compare directly. "
            "Run both with identical preprocessing/split/min_code_freq/text_mode."
        )

    hier_sub = _hierarchy_subgraph(graph_full)
    depths = _compute_depths(vocab_full, hier_sub)

    node_rows = []
    for nid, key in vocab_full.id_to_node.items():
        nid = int(nid)
        node_key = str(key)
        code_label = node_key.split(":", 1)[1] if ":" in node_key else node_key
        raw_label = getattr(vocab_full, "id_to_label", {}).get(nid, key)
        text_label = str(raw_label) if raw_label is not None else ""
        if text_label.strip() == "" or text_label.lower() == "nan":
            text_label = code_label
        modality = str(vocab_full.node_type.get(nid, "other"))
        node_rows.append(
            {
                "node_id": nid,
                "node_key": node_key,
                "code_label": code_label,
                "text_label": text_label,
                # Backward-compatible generic label used by visualization script.
                "label": text_label,
                "modality": modality,
                "depth": int(depths.get(nid, -1)),
            }
        )
    node_df = pd.DataFrame(node_rows).sort_values("node_id")
    node_df.to_csv(os.path.join(out_dir, "node_metadata.csv"), index=False)
    # Explicit alias for downstream inspection workflows.
    node_df.to_csv(os.path.join(out_dir, "node_metadata_all_nodes.csv"), index=False)

    edge_rows = []
    for u, v, d in graph_full.edges(data=True):
        et = str(d.get("type", ""))
        rel = "hierarchy" if et.startswith("hier:") else ("cross_modal" if et.startswith("cross:") else "other")
        edge_rows.append(
            {
                "source": int(u),
                "target": int(v),
                "edge_type": et,
                "relation_type": rel,
            }
        )
    pd.DataFrame(edge_rows).to_csv(os.path.join(out_dir, "graph_edges.csv"), index=False)

    _save_embeddings_csv(os.path.join(out_dir, "embeddings_full.csv"), emb_full)
    _save_embeddings_csv(os.path.join(out_dir, "embeddings_hier_only.csv"), emb_hier)

    sel_paths = _build_selected_paths(vocab_full, hier_sub, depths, n_paths_per_modality=n_paths_per_modality)
    with open(os.path.join(out_dir, "selected_paths.json"), "w") as f:
        json.dump({"paths": sel_paths}, f, indent=2)

    meta = {
        "artifact_full_path": artifact_full_path,
        "artifact_hier_path": artifact_hier_path,
        "n_nodes": int(len(vocab_full.id_to_node)),
        "n_edges_full": int(graph_full.number_of_edges()),
        "n_hierarchy_paths_selected": int(len(sel_paths)),
    }
    with open(os.path.join(out_dir, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved visualization inputs to: {out_dir}")
    print(
        "Node metadata saved with columns: "
        "node_id,node_key,code_label,text_label,label,modality,depth "
        f"(n={len(node_df)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export visualization inputs from GRAIL artifacts.")
    ap.add_argument("--artifact-full", type=str, required=True)
    ap.add_argument("--artifact-hier", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-paths-per-modality", type=int, default=2)
    args = ap.parse_args()
    export_inputs(
        artifact_full_path=args.artifact_full,
        artifact_hier_path=args.artifact_hier,
        out_dir=args.out_dir,
        n_paths_per_modality=args.n_paths_per_modality,
    )


if __name__ == "__main__":
    main()
