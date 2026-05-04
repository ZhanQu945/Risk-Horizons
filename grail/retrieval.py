from typing import Dict, List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

from .data import Vocab
from .model import GrailModel

_LOCAL_CAND_CACHE: Dict[Tuple[int, int, int, int], Tuple[int, ...]] = {}


def _collect_local_candidates(
    graph: nx.DiGraph,
    central_code: int,
    max_desc_depth: int,
    max_local_candidates: int = 20000,
) -> set:
    if central_code not in graph:
        return set()
    candidates = set(
        _descendants(
            graph,
            central_code,
            max_depth=max_desc_depth,
            max_nodes=max_local_candidates,
        )
    )
    if len(candidates) >= max_local_candidates:
        return candidates
    for _, v, data in graph.out_edges(central_code, data=True):
        if data.get("type", "").startswith("cross:"):
            candidates.add(v)
            if len(candidates) >= max_local_candidates:
                return candidates
    for u, _, data in graph.in_edges(central_code, data=True):
        if data.get("type", "").startswith("cross:"):
            candidates.add(u)
            if len(candidates) >= max_local_candidates:
                return candidates
    return candidates


def _collect_local_candidates_cached(
    graph: nx.DiGraph,
    central_code: int,
    max_desc_depth: int,
    max_local_candidates: int,
) -> set:
    key = (id(graph), int(central_code), int(max_desc_depth), int(max_local_candidates))
    cached = _LOCAL_CAND_CACHE.get(key)
    if cached is not None:
        return set(cached)
    cands = _collect_local_candidates(
        graph,
        central_code=central_code,
        max_desc_depth=max_desc_depth,
        max_local_candidates=max_local_candidates,
    )
    _LOCAL_CAND_CACHE[key] = tuple(sorted(int(x) for x in cands))
    return cands


def _edge_lag(edge_type: str) -> int:
    s = str(edge_type)
    if "lag" not in s:
        return -1
    try:
        return int(s.rsplit("lag", 1)[1])
    except Exception:
        return -1


def _collect_desc_assoc_pred_candidates(
    graph: nx.DiGraph,
    central_code: int,
    max_desc_depth: int,
    max_local_candidates: int,
) -> set:
    if central_code not in graph:
        return set()
    cands = set(_descendants(graph, central_code, max_depth=max_desc_depth, max_nodes=max_local_candidates))
    if len(cands) >= max_local_candidates:
        return cands
    # Assoc: same-visit stitched neighbors (lag=0), Pred: lag>0
    for _, v, data in graph.out_edges(central_code, data=True):
        et = str(data.get("type", ""))
        if et.startswith("cross:"):
            lg = _edge_lag(et)
            if lg >= 0:
                cands.add(int(v))
                if len(cands) >= max_local_candidates:
                    return cands
    # include reverse stitched links as local typed neighborhood
    for u, _, data in graph.in_edges(central_code, data=True):
        et = str(data.get("type", ""))
        if et.startswith("cross:"):
            lg = _edge_lag(et)
            if lg >= 0:
                cands.add(int(u))
                if len(cands) >= max_local_candidates:
                    return cands
    return cands


def _descendants(graph: nx.DiGraph, node: int, max_depth: int = 3, max_nodes: int = 20000) -> List[int]:
    if node not in graph:
        return []
    descendants = []
    frontier = [(node, 0)]
    seen = {node}
    while frontier:
        curr, depth = frontier.pop()
        if depth >= max_depth:
            continue
        if curr not in graph:
            continue
        for child in graph.successors(curr):
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            if len(descendants) >= max_nodes:
                return descendants
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


def risk_cone_membership_mask(
    model: GrailModel,
    candidate_indices: torch.Tensor,
    root_idx: int,
    barycenter: torch.Tensor,
    angle_thresh: float,
) -> torch.Tensor:
    if model.manifold_type == "euclidean" or angle_thresh <= -1.0:
        return torch.ones(candidate_indices.numel(), dtype=torch.bool, device=candidate_indices.device)
    z_root = model.embeddings[root_idx]
    vec_root = model.manifold.logmap0(z_root)
    vec_curr = model.manifold.logmap0(barycenter)
    direction = vec_curr - vec_root
    z_cands = model.embeddings[candidate_indices]
    vec_cands = model.manifold.logmap0(z_cands)
    cos_sim = F.cosine_similarity(vec_cands, direction.unsqueeze(0), dim=1)
    return cos_sim > angle_thresh


def build_risk_horizon(
    graph: nx.DiGraph,
    vocab: Vocab,
    model: GrailModel,
    barycenter: torch.Tensor,
    central_code: int,
    max_desc_depth: int,
    max_candidates: int,
    max_local_candidates: int,
    cone_aperture: float,
    cone_bonus: float,
    retrieval_pool_mode: str = "localmix",
    cone_mode: str = "filter",
    cone_empty_fallback: bool = True,
    allowed_nodes_by_modality: Dict[str, set] = None,
    filter_stats: Dict[str, float] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    results: Dict[str, List[Tuple[int, float]]] = {m: [] for m in vocab.modality_roots.keys()}
    # gather candidates: descendants, associates, predictive neighbors
    if retrieval_pool_mode == "desc_assoc_pred":
        candidates = _collect_desc_assoc_pred_candidates(
            graph,
            central_code,
            max_desc_depth,
            max_local_candidates,
        )
    else:
        candidates = _collect_local_candidates_cached(
            graph,
            central_code,
            max_desc_depth,
            max_local_candidates,
        )

    if not candidates:
        return results

    cand_tensor = torch.tensor(sorted(candidates), device=barycenter.device)
    for modality, root_idx in vocab.modality_roots.items():
        mask = [vocab.node_type[idx] == modality for idx in cand_tensor.tolist()]
        if not any(mask):
            continue
        mod_cands = cand_tensor[torch.tensor(mask, device=barycenter.device)]
        if allowed_nodes_by_modality is not None:
            before_n = int(mod_cands.numel())
            allow = allowed_nodes_by_modality.get(modality, set())
            if allow:
                allow_mask = [int(idx) in allow for idx in mod_cands.tolist()]
                if not any(allow_mask):
                    if filter_stats is not None:
                        filter_stats["total_candidates"] = float(filter_stats.get("total_candidates", 0.0) + before_n)
                        filter_stats["dropped_candidates"] = float(filter_stats.get("dropped_candidates", 0.0) + before_n)
                    continue
                mod_cands = mod_cands[torch.tensor(allow_mask, device=barycenter.device)]
            if filter_stats is not None:
                after_n = int(mod_cands.numel())
                filter_stats["total_candidates"] = float(filter_stats.get("total_candidates", 0.0) + before_n)
                filter_stats["dropped_candidates"] = float(
                    filter_stats.get("dropped_candidates", 0.0) + max(0, before_n - after_n)
                )
        if cone_mode == "score_bonus":
            in_cone = risk_cone_membership_mask(model, mod_cands, root_idx, barycenter, cone_aperture)
            dists = model.dist(barycenter.unsqueeze(0), model.embeddings[mod_cands]).squeeze(0)
            scores = (-dists).view(-1)
            scores = scores + cone_bonus * in_cone.to(scores.dtype)
            filtered = mod_cands
        else:
            filtered = risk_cone_filter(model, mod_cands, root_idx, barycenter, cone_aperture)
            if filtered.numel() == 0 and cone_empty_fallback:
                filtered = mod_cands
            dists = model.dist(barycenter.unsqueeze(0), model.embeddings[filtered]).squeeze(0)
            scores = (-dists).view(-1)
            if model.manifold_type == "poincare":
                scores = scores + cone_bonus
        k = min(max_candidates, scores.numel())
        topk = torch.topk(scores, k=k).indices
        topk = topk.view(-1)
        chosen = filtered[topk]
        results[modality] = [(int(idx), float(scores[score_idx].item())) for idx, score_idx in zip(chosen, topk)]
    return results


def build_risk_horizon_multi_cone(
    graph: nx.DiGraph,
    vocab: Vocab,
    model: GrailModel,
    head_barycenters: torch.Tensor,
    central_codes: List[int],
    max_desc_depth: int,
    max_candidates: int,
    max_local_candidates: int,
    cone_aperture: float,
    cone_bonus: float,
    head_weights: torch.Tensor = None,
    retrieval_pool_mode: str = "localmix",
    cone_mode: str = "filter",
    cone_empty_fallback: bool = True,
    allowed_nodes_by_modality: Dict[str, set] = None,
    filter_stats: Dict[str, float] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    # head_barycenters: (H, D), central_codes: len H
    results: Dict[str, List[Tuple[int, float]]] = {m: [] for m in vocab.modality_roots.keys()}
    if head_barycenters.ndim != 2:
        return results
    num_heads = head_barycenters.size(0)
    if num_heads == 0:
        return results
    if head_weights is None:
        head_weights = torch.ones(num_heads, device=head_barycenters.device, dtype=head_barycenters.dtype) / num_heads
    else:
        head_weights = head_weights.to(head_barycenters.device, dtype=head_barycenters.dtype)
        head_weights = head_weights / head_weights.sum().clamp_min(1e-8)

    candidates = set()
    for c in central_codes:
        if int(c) not in graph:
            continue
        if retrieval_pool_mode == "desc_assoc_pred":
            candidates.update(
                _collect_desc_assoc_pred_candidates(graph, int(c), max_desc_depth, max_local_candidates)
            )
        else:
            candidates.update(_collect_local_candidates_cached(graph, int(c), max_desc_depth, max_local_candidates))
        if len(candidates) >= max_local_candidates:
            break
    if not candidates:
        return results

    cand_tensor = torch.tensor(sorted(candidates), device=head_barycenters.device)
    for modality, root_idx in vocab.modality_roots.items():
        mask = [vocab.node_type[idx] == modality for idx in cand_tensor.tolist()]
        if not any(mask):
            continue
        mod_cands = cand_tensor[torch.tensor(mask, device=head_barycenters.device)]
        if allowed_nodes_by_modality is not None:
            before_n = int(mod_cands.numel())
            allow = allowed_nodes_by_modality.get(modality, set())
            if allow:
                allow_mask = [int(idx) in allow for idx in mod_cands.tolist()]
                if not any(allow_mask):
                    if filter_stats is not None:
                        filter_stats["total_candidates"] = float(filter_stats.get("total_candidates", 0.0) + before_n)
                        filter_stats["dropped_candidates"] = float(filter_stats.get("dropped_candidates", 0.0) + before_n)
                    continue
                mod_cands = mod_cands[torch.tensor(allow_mask, device=head_barycenters.device)]
            if filter_stats is not None:
                after_n = int(mod_cands.numel())
                filter_stats["total_candidates"] = float(filter_stats.get("total_candidates", 0.0) + before_n)
                filter_stats["dropped_candidates"] = float(
                    filter_stats.get("dropped_candidates", 0.0) + max(0, before_n - after_n)
                )
        # accumulate weighted scores from head-specific cones
        combined = torch.zeros(mod_cands.numel(), device=head_barycenters.device, dtype=head_barycenters.dtype)
        for h in range(num_heads):
            bary_h = head_barycenters[h]
            if cone_mode == "score_bonus":
                keep_mask = risk_cone_membership_mask(model, mod_cands, root_idx, bary_h, cone_aperture)
                dists = model.dist(bary_h.unsqueeze(0), model.embeddings[mod_cands]).squeeze(0)
                scores_h = (-dists).view(-1)
                scores_h = scores_h + cone_bonus * keep_mask.to(scores_h.dtype)
            else:
                filtered = risk_cone_filter(model, mod_cands, root_idx, bary_h, cone_aperture)
                if filtered.numel() == 0 and cone_empty_fallback:
                    filtered = mod_cands
                dists = model.dist(bary_h.unsqueeze(0), model.embeddings[mod_cands]).squeeze(0)
                scores_h = (-dists).view(-1)
                if model.manifold_type == "poincare":
                    scores_h = scores_h + cone_bonus
                keep_mask = torch.zeros(mod_cands.numel(), device=head_barycenters.device, dtype=torch.bool)
                pos_map = {int(idx): i for i, idx in enumerate(mod_cands.tolist())}
                for idx in filtered.tolist():
                    keep_mask[pos_map[int(idx)]] = True
                scores_h = torch.where(keep_mask, scores_h, torch.full_like(scores_h, -1e9))
            combined = combined + head_weights[h] * scores_h

        k = min(max_candidates, combined.numel())
        topk = torch.topk(combined, k=k).indices.view(-1)
        chosen = mod_cands[topk]
        results[modality] = [(int(idx), float(combined[pos].item())) for idx, pos in zip(chosen, topk)]
    return results
