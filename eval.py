from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import networkx as nx


def recall_at_k(pred: List[int], truth: List[int], k: int) -> float:
    pred_k = set(pred[:k])
    truth_set = set(truth)
    if not truth_set:
        return 0.0
    return len(pred_k & truth_set) / len(truth_set)


def precision_at_k(pred: List[int], truth: List[int], k: int) -> float:
    pred_k = pred[:k]
    if not pred_k:
        return 0.0
    truth_set = set(truth)
    return len(set(pred_k) & truth_set) / len(pred_k)


def ndcg_at_k(pred: List[int], truth: List[int], k: int) -> float:
    truth_set = set(truth)
    dcg = 0.0
    for i, p in enumerate(pred[:k]):
        if p in truth_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(truth_set))))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def mrr(pred: List[int], truth: List[int]) -> float:
    truth_set = set(truth)
    for i, p in enumerate(pred):
        if p in truth_set:
            return 1.0 / (i + 1)
    return 0.0


def micro_f1(scores: List[float], truth: List[int], threshold: float) -> float:
    pred = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def tune_threshold(scores_list: List[List[float]], truth_list: List[List[int]]) -> float:
    all_scores = sorted({s for scores in scores_list for s in scores})
    if not all_scores:
        return 0.5
    best_th = all_scores[0]
    best_f1 = -1.0
    for th in all_scores:
        f1s = [micro_f1(scores, truth, th) for scores, truth in zip(scores_list, truth_list)]
        avg_f1 = float(np.mean(f1s)) if f1s else 0.0
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_th = th
    return best_th


def hierarchy_distance(graph: nx.DiGraph, pred: int, truth: int) -> int:
    try:
        return nx.shortest_path_length(graph.to_undirected(), pred, truth)
    except nx.NetworkXNoPath:
        return 999


def ancestor_match(graph: nx.DiGraph, pred: int, truth: int, max_depth: int = 2) -> int:
    pred_anc = {pred}
    frontier = [pred]
    for _ in range(max_depth):
        next_frontier = []
        for node in frontier:
            parents = list(graph.predecessors(node))
            for p in parents:
                if p not in pred_anc:
                    pred_anc.add(p)
                    next_frontier.append(p)
        frontier = next_frontier
    return int(truth in pred_anc)


def aggregate_metrics(metric_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    acc = defaultdict(list)
    for metrics in metric_list:
        for key, val in metrics.items():
            acc[key].append(val)
    return {k: float(np.mean(v)) for k, v in acc.items()}
