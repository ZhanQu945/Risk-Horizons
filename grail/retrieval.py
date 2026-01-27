from typing import Dict, List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

from .data import Vocab
from .model import GrailModel


def _descendants(graph: nx.DiGraph, node: int, max_depth: int = 3) -> List[int]:
    descendants = []
    frontier = [(node, 0)]
    seen = {node}
    while frontier:
        curr, depth = frontier.pop()
        if depth >= max_depth:
            continue
        for child in graph.successors(curr):
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            frontier.append((child, depth + 1))
    return descendants


def risk_cone_filter(
    model: GrailModel,
    candidate_indices: torch.Tensor,
    root_idx: int,
    barycenter: torch.Tensor,
    angle_thresh: float,
) -> torch.Tensor:
    if model.manifold_type == "euclidean" or angle_thresh <= -1.0:
        return candidate_indices
    z_root = model.embeddings[root_idx]
    vec_root = model.manifold.logmap0(z_root)
    vec_curr = model.manifold.logmap0(barycenter)
    direction = vec_curr - vec_root
    z_cands = model.embeddings[candidate_indices]
    vec_cands = model.manifold.logmap0(z_cands)
    cos_sim = F.cosine_similarity(vec_cands, direction.unsqueeze(0), dim=1)
    mask = cos_sim > angle_thresh
    return candidate_indices[mask]


def build_risk_horizon(
    graph: nx.DiGraph,
    vocab: Vocab,
    model: GrailModel,
    barycenter: torch.Tensor,
    central_code: int,
    max_desc_depth: int,
    max_candidates: int,
    cone_aperture: float,
    cone_bonus: float,
) -> Dict[str, List[Tuple[int, float]]]:
    results: Dict[str, List[Tuple[int, float]]] = {m: [] for m in vocab.modality_roots.keys()}
    # gather candidates: descendants, associates, predictive neighbors
    candidates = set(_descendants(graph, central_code, max_depth=max_desc_depth))
    for _, v, data in graph.out_edges(central_code, data=True):
        if data.get("type", "").startswith("cross:"):
            candidates.add(v)
    for u, _, data in graph.in_edges(central_code, data=True):
        if data.get("type", "").startswith("cross:"):
            candidates.add(u)

    if not candidates:
        return results

    cand_tensor = torch.tensor(sorted(candidates), device=barycenter.device)
    for modality, root_idx in vocab.modality_roots.items():
        mask = [vocab.node_type[idx] == modality for idx in cand_tensor.tolist()]
        if not any(mask):
            continue
        mod_cands = cand_tensor[torch.tensor(mask, device=barycenter.device)]
        filtered = risk_cone_filter(model, mod_cands, root_idx, barycenter, cone_aperture)
        if filtered.numel() == 0:
            filtered = mod_cands
        dists = model.manifold.dist(barycenter.unsqueeze(0), model.embeddings[filtered]).squeeze(0)
        scores = (-dists).view(-1)
        if model.manifold_type == "poincare":
            scores = scores + cone_bonus
        k = min(max_candidates, scores.numel())
        topk = torch.topk(scores, k=k).indices
        topk = topk.view(-1)
        chosen = filtered[topk]
        results[modality] = [(int(idx), float(scores[score_idx].item())) for idx, score_idx in zip(chosen, topk)]
    return results
