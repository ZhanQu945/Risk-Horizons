import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from .data import PatientTrajectory, Vocab


def _iter_lagged_pairs(
    trajectories: Iterable[PatientTrajectory],
    max_lag: int,
    modality_src: str,
    modality_tgt: str,
    include_intra_visit: bool,
) -> Iterable[Tuple[int, int]]:
    start_lag = 0 if include_intra_visit else 1
    for traj in trajectories:
        visits = traj.visits
        for t in range(len(visits)):
            for lag in range(start_lag, max_lag + 1):
                if t + lag >= len(visits):
                    continue
                src_codes = visits[t].get(modality_src, [])
                tgt_codes = visits[t + lag].get(modality_tgt, [])
                for src in src_codes:
                    for tgt in tgt_codes:
                        if lag == 0 and src == tgt:
                            continue
                        yield lag, src, tgt


def build_cross_modal_edges(
    trajectories: List[PatientTrajectory],
    vocab: Vocab,
    max_lag: int,
    include_intra_visit: bool,
    pmi_thresh: float,
    min_support: int,
    top_m_per_node: int,
    use_ppmi: bool,
    include_intra_modality_pmi: bool = False,
    require_joint_lag01: bool = False,
    joint_lag01_min_support: int = 3,
) -> List[Tuple[int, int, str, float]]:
    edges: List[Tuple[int, int, str, float]] = []
    modalities = sorted(vocab.modality_roots.keys())
    for m_src in modalities:
        for m_tgt in modalities:
            if m_src == m_tgt and not include_intra_modality_pmi:
                continue
            pair_counts: Dict[Tuple[int, int, int], int] = Counter()
            src_counts: Dict[Tuple[int, int], int] = Counter()
            tgt_counts: Dict[Tuple[int, int], int] = Counter()
            total_by_lag: Dict[int, int] = Counter()
            for lag, src, tgt in _iter_lagged_pairs(
                trajectories,
                max_lag,
                m_src,
                m_tgt,
                include_intra_visit,
            ):
                pair_counts[(lag, src, tgt)] += 1
                src_counts[(lag, src)] += 1
                tgt_counts[(lag, tgt)] += 1
                total_by_lag[lag] += 1
            # compute PMI and prune per source
            scored: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
            for (lag, src, tgt), count in pair_counts.items():
                if count < min_support:
                    continue
                if require_joint_lag01:
                    c0 = int(pair_counts.get((0, src, tgt), 0))
                    c1 = int(pair_counts.get((1, src, tgt), 0))
                    if c0 < joint_lag01_min_support or c1 < joint_lag01_min_support:
                        continue
                total = total_by_lag[lag]
                if total == 0:
                    continue
                p_ab = count / total
                p_a = src_counts[(lag, src)] / total
                p_b = tgt_counts[(lag, tgt)] / total
                pmi = math.log(p_ab / (p_a * p_b + 1e-12))
                if use_ppmi:
                    pmi = max(0.0, pmi)
                if pmi > pmi_thresh:
                    scored[(src, lag)].append((tgt, pmi))
            for (src, lag), items in scored.items():
                items.sort(key=lambda x: x[1], reverse=True)
                for tgt, pmi in items[:top_m_per_node]:
                    edge_type = f"cross:{m_src}->{m_tgt}:lag{lag}"
                    edges.append((src, tgt, edge_type, pmi))
    return edges


def build_graph(
    trajectories: List[PatientTrajectory],
    vocab: Vocab,
    max_lag: int,
    include_intra_visit: bool,
    pmi_thresh: float,
    min_support: int,
    top_m_per_node: int,
    use_ppmi: bool,
    bootstrap_samples: int,
    bootstrap_keep_q: float,
    include_intra_modality_pmi: bool = False,
    require_joint_lag01: bool = False,
    joint_lag01_min_support: int = 3,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(vocab.id_to_node.keys())

    # hierarchy edges
    from .data import build_hierarchy_edges
    for u, v, edge_type in build_hierarchy_edges(vocab):
        graph.add_edge(u, v, type=edge_type, weight=1.0)

    if bootstrap_samples <= 1:
        edges = build_cross_modal_edges(
            trajectories,
            vocab,
            max_lag,
            include_intra_visit,
            pmi_thresh,
            min_support,
            top_m_per_node,
            use_ppmi,
            include_intra_modality_pmi=include_intra_modality_pmi,
            require_joint_lag01=require_joint_lag01,
            joint_lag01_min_support=joint_lag01_min_support,
        )
    else:
        counts: Counter = Counter()
        for _ in range(bootstrap_samples):
            sampled = [random.choice(trajectories) for _ in range(len(trajectories))]
            edges_b = build_cross_modal_edges(
                sampled,
                vocab,
                max_lag,
                include_intra_visit,
                pmi_thresh,
                min_support,
                top_m_per_node,
                use_ppmi,
                include_intra_modality_pmi=include_intra_modality_pmi,
                require_joint_lag01=require_joint_lag01,
                joint_lag01_min_support=joint_lag01_min_support,
            )
            for u, v, edge_type, weight in edges_b:
                counts[(u, v, edge_type)] += 1
        keep_threshold = bootstrap_keep_q * bootstrap_samples
        edges = []
        for (u, v, edge_type), freq in counts.items():
            if freq >= keep_threshold:
                edges.append((u, v, edge_type, float(freq)))
    for u, v, edge_type, weight in edges:
        graph.add_edge(u, v, type=edge_type, weight=weight)
    return graph


def graph_to_edge_list(graph: nx.DiGraph) -> List[Tuple[int, int, str]]:
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append((u, v, data.get("type", "unknown")))
    return edges


def edge_type_map(edges: List[Tuple[int, int, str]]) -> Dict[str, int]:
    types = sorted({t for _, _, t in edges})
    return {t: i for i, t in enumerate(types)}
