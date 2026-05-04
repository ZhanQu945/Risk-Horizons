import argparse
import json
import os
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from grail.config import DataConfig
from grail.data import (
    build_hierarchy_edges,
    build_patient_trajectories,
    build_vocab,
    filter_min_visits,
    load_lab_groups,
    load_table,
    split_patients,
)


def _split_file_path(data_path: str, seed: int, train_frac: float, val_frac: float, min_visits: int) -> str:
    base = os.path.dirname(os.path.abspath(data_path))
    stem = os.path.splitext(os.path.basename(data_path))[0]
    split_dir = os.path.join(base, "splits")
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, f"{stem}_seed{seed}_tr{train_frac:.2f}_val{val_frac:.2f}_mv{int(min_visits)}.json")


def _split_by_subject_ids(trajectories, split_payload: Dict[str, List[str]]):
    train_ids = set(map(str, split_payload.get("train_subject_ids", [])))
    val_ids = set(map(str, split_payload.get("val_subject_ids", [])))
    test_ids = set(map(str, split_payload.get("test_subject_ids", [])))
    train_traj, val_traj, test_traj = [], [], []
    for traj in trajectories:
        sid = str(traj.subject_id)
        if sid in train_ids:
            train_traj.append(traj)
        elif sid in val_ids:
            val_traj.append(traj)
        elif sid in test_ids:
            test_traj.append(traj)
    return train_traj, val_traj, test_traj


def _get_or_create_split(trajectories, data_cfg: DataConfig, seed: int):
    path = _split_file_path(data_cfg.data_path, seed, data_cfg.train_frac, data_cfg.val_frac, data_cfg.min_visits)
    if os.path.exists(path):
        with open(path, "r") as f:
            payload = json.load(f)
        train_traj, val_traj, test_traj = _split_by_subject_ids(trajectories, payload)
        return path, train_traj, val_traj, test_traj, True

    train_traj, val_traj, test_traj = split_patients(trajectories, data_cfg.train_frac, data_cfg.val_frac, seed)
    payload = {
        "seed": seed,
        "train_frac": data_cfg.train_frac,
        "val_frac": data_cfg.val_frac,
        "n_total": len(trajectories),
        "train_subject_ids": [str(t.subject_id) for t in train_traj],
        "val_subject_ids": [str(t.subject_id) for t in val_traj],
        "test_subject_ids": [str(t.subject_id) for t in test_traj],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path, train_traj, val_traj, test_traj, False


def _hier_depths(vocab) -> Tuple[Dict[int, int], Dict[str, Dict[int, int]], Dict[str, Dict[int, int]], Dict[str, Dict[int, int]]]:
    # Build hierarchy adjacency only (parent->child)
    edges = build_hierarchy_edges(vocab)
    children = defaultdict(list)
    parents = defaultdict(list)
    for u, v, _ in edges:
        u = int(u)
        v = int(v)
        children[u].append(v)
        parents[v].append(u)

    depth_by_node: Dict[int, int] = {}
    node_counts_by_mod_depth: Dict[str, Dict[int, int]] = {}
    leaf_counts_by_mod_depth: Dict[str, Dict[int, int]] = {}
    edge_counts_by_mod_parent_depth: Dict[str, Dict[int, int]] = {}

    for modality, root_id in vocab.modality_roots.items():
        root_id = int(root_id)
        q = deque([(root_id, 0)])
        seen = {root_id}
        mod_depth = {}
        while q:
            node, d = q.popleft()
            mod_depth[node] = d
            for ch in children.get(node, []):
                if ch not in seen:
                    seen.add(ch)
                    q.append((ch, d + 1))

        # Count nodes at depth for this modality
        counts = defaultdict(int)
        leaf_counts = defaultdict(int)
        edge_counts = defaultdict(int)
        for nid, d in mod_depth.items():
            counts[d] += 1
            if len(children.get(nid, [])) == 0:
                leaf_counts[d] += 1
            depth_by_node[nid] = d
        for u, v, _ in edges:
            u = int(u)
            v = int(v)
            if u in mod_depth and v in mod_depth:
                edge_counts[mod_depth[u]] += 1

        node_counts_by_mod_depth[modality] = {int(k): int(v) for k, v in sorted(counts.items())}
        leaf_counts_by_mod_depth[modality] = {int(k): int(v) for k, v in sorted(leaf_counts.items())}
        edge_counts_by_mod_parent_depth[modality] = {int(k): int(v) for k, v in sorted(edge_counts.items())}

    return depth_by_node, node_counts_by_mod_depth, leaf_counts_by_mod_depth, edge_counts_by_mod_parent_depth


def _collect_nodes(trajs) -> set:
    s = set()
    for tr in trajs:
        for v in tr.visits:
            for codes in v.values():
                for c in codes:
                    s.add(int(c))
    return s


def _counts_by_depth(nodes: set, depth_by_node: Dict[int, int]) -> Dict[int, int]:
    out = defaultdict(int)
    for nid in nodes:
        d = depth_by_node.get(int(nid))
        if d is not None:
            out[d] += 1
    return {int(k): int(v) for k, v in sorted(out.items())}


def _collect_nodes_by_modality(trajs) -> Dict[str, set]:
    out: Dict[str, set] = defaultdict(set)
    for tr in trajs:
        for v in tr.visits:
            for m, codes in v.items():
                for c in codes:
                    out[str(m)].add(int(c))
    return out


def _aggregate_depth_counts(seed_reports: List[Dict[str, object]], split_name: str) -> Dict[str, Dict[str, float]]:
    depth_keys = set()
    for r in seed_reports:
        depth_keys.update(int(k) for k in r["split_depth_counts"][split_name].keys())
    agg = {}
    for d in sorted(depth_keys):
        vals = []
        for r in seed_reports:
            vals.append(float(r["split_depth_counts"][split_name].get(str(d), r["split_depth_counts"][split_name].get(d, 0))))
        arr = np.asarray(vals, dtype=float)
        agg[str(d)] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return agg


def _run_dataset(name: str, data_path: str, label_map: str, lab_groups: str, seeds: List[int], text_mode: str, min_visits: int, min_code_freq: int, disable_labs: bool):
    data_cfg = DataConfig(data_path=data_path, format="pickle", text_mode=text_mode, label_map_path=label_map)
    data_cfg.min_visits = int(min_visits)
    data_cfg.min_code_freq = int(min_code_freq)
    if disable_labs and "lab" in data_cfg.modality_cols:
        data_cfg.modality_cols = {k: v for k, v in data_cfg.modality_cols.items() if k != "lab"}

    df = load_table(data_cfg.data_path, data_cfg.format)
    lab_to_group = None if disable_labs else load_lab_groups(lab_groups)
    vocab = build_vocab(
        df,
        data_cfg.modality_cols,
        min_code_freq=data_cfg.min_code_freq,
        top_k_per_modality=data_cfg.top_k_per_modality,
        lab_to_group=lab_to_group,
        text_mode=data_cfg.text_mode,
        label_map_path=data_cfg.label_map_path,
    )
    trajectories = build_patient_trajectories(
        df,
        data_cfg.subject_id_col,
        data_cfg.time_col,
        data_cfg.modality_cols,
        vocab,
        text_mode=data_cfg.text_mode,
        label_map_path=data_cfg.label_map_path,
    )
    trajectories = filter_min_visits(trajectories, data_cfg.min_visits)

    depth_by_node, node_by_mod_depth, leaf_by_mod_depth, edge_by_mod_parent_depth = _hier_depths(vocab)

    seed_reports = []
    for seed in seeds:
        split_path, train_traj, val_traj, test_traj, reused = _get_or_create_split(trajectories, data_cfg, int(seed))
        split_depth_counts = {
            "train": _counts_by_depth(_collect_nodes(train_traj), depth_by_node),
            "val": _counts_by_depth(_collect_nodes(val_traj), depth_by_node),
            "test": _counts_by_depth(_collect_nodes(test_traj), depth_by_node),
        }
        seed_reports.append(
            {
                "seed": int(seed),
                "split_file": split_path,
                "split_reused": bool(reused),
                "split_depth_counts": split_depth_counts,
            }
        )

    all_nodes = _collect_nodes(trajectories)
    all_nodes_by_mod = _collect_nodes_by_modality(trajectories)
    overall_depth_counts_by_mod = {
        m: _counts_by_depth(nodes, depth_by_node) for m, nodes in sorted(all_nodes_by_mod.items())
    }

    # label coverage diagnostics for textual interpretability
    label_coverage = {}
    for m in data_cfg.modality_cols.keys():
        total = 0
        text_like = 0
        for nid, node in vocab.id_to_node.items():
            mod, code = str(node).split(":", 1)
            if mod != m:
                continue
            total += 1
            label = str(vocab.id_to_label.get(int(nid), code))
            if label != code:
                text_like += 1
        label_coverage[m] = {
            "total_nodes": int(total),
            "text_labeled_nodes": int(text_like),
            "text_label_rate": float(text_like / max(total, 1)),
        }

    return {
        "dataset": name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "data_path": data_path,
            "label_map": label_map,
            "lab_groups": lab_groups,
            "text_mode": text_mode,
            "min_visits": int(min_visits),
            "min_code_freq": int(min_code_freq),
            "disable_labs": bool(disable_labs),
            "seeds": [int(s) for s in seeds],
        },
        "hierarchy_level_stats": {
            "node_counts_by_modality_depth": node_by_mod_depth,
            "leaf_counts_by_modality_depth": leaf_by_mod_depth,
            "edge_counts_by_modality_parent_depth": edge_by_mod_parent_depth,
            "label_coverage": label_coverage,
        },
        "split_observed_depth_counts_per_seed": seed_reports,
        "split_observed_depth_counts_5seed_aggregate": {
            "train": _aggregate_depth_counts(seed_reports, "train"),
            "val": _aggregate_depth_counts(seed_reports, "val"),
            "test": _aggregate_depth_counts(seed_reports, "test"),
        },
        "overall_observed_depth_counts_after_filtering_before_split": {
            "pooled_all_modalities": _counts_by_depth(all_nodes, depth_by_node),
            "by_modality": overall_depth_counts_by_mod,
        },
    }


def _print_summary(payload: Dict[str, object]) -> None:
    print(f"=== {payload['dataset']} hierarchy depth summary ===")
    h = payload["hierarchy_level_stats"]
    for modality, counts in h["node_counts_by_modality_depth"].items():
        joined = ", ".join([f"d{d}:{c}" for d, c in counts.items()])
        print(f"{modality} node counts by depth | {joined}")
    for modality, counts in h["edge_counts_by_modality_parent_depth"].items():
        joined = ", ".join([f"d{d}->d{int(d)+1}:{c}" for d, c in counts.items()])
        print(f"{modality} hierarchy edges by parent depth | {joined}")
    overall = payload.get("overall_observed_depth_counts_after_filtering_before_split", {})
    pooled = overall.get("pooled_all_modalities", {})
    if pooled:
        joined = ", ".join([f"d{d}:{c}" for d, c in pooled.items()])
        print(f"overall observed nodes by depth (post-filter, pre-split) | {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report hierarchy/prefix depth statistics for vocabulary and split-observed nodes.")
    parser.add_argument("--mimic-data-path", type=str, default="./Processed/mimiciv/ehr_tables_full_withlab.pkl")
    parser.add_argument("--mimic-label-map", type=str, default="./Processed/mimiciv/ehr_label_map.json")
    parser.add_argument("--mimic-lab-groups", type=str, default="./Processed/mimiciv/ehr_lab_groups.csv")
    parser.add_argument("--eicu-data-path", type=str, default="./Processed/eicu/ehr_tables_full_withlab.pkl")
    parser.add_argument("--eicu-label-map", type=str, default="./Processed/eicu/ehr_label_map.json")
    parser.add_argument("--eicu-lab-groups", type=str, default="./Processed/eicu/ehr_lab_groups.csv")
    parser.add_argument("--seeds", type=str, default="13,21,42,87,123")
    parser.add_argument("--text-mode", type=str, default="text", choices=["hybrid", "code", "text"])
    parser.add_argument("--min-visits", type=int, default=2)
    parser.add_argument("--min-code-freq", type=int, default=0)
    parser.add_argument("--disable-lab-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--run-name", type=str, default="hierarchy_level_stats_5seeds")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    out_dir = os.path.join(args.output_root, f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    mimic = _run_dataset(
        "mimiciv",
        args.mimic_data_path,
        args.mimic_label_map,
        args.mimic_lab_groups,
        seeds,
        args.text_mode,
        args.min_visits,
        args.min_code_freq,
        bool(args.disable_lab_tests),
    )
    eicu = _run_dataset(
        "eicu",
        args.eicu_data_path,
        args.eicu_label_map,
        args.eicu_lab_groups,
        seeds,
        args.text_mode,
        args.min_visits,
        args.min_code_freq,
        bool(args.disable_lab_tests),
    )

    with open(os.path.join(out_dir, "hierarchy_level_stats_mimiciv.json"), "w") as f:
        json.dump(mimic, f, indent=2)
    with open(os.path.join(out_dir, "hierarchy_level_stats_eicu.json"), "w") as f:
        json.dump(eicu, f, indent=2)

    _print_summary(mimic)
    _print_summary(eicu)
    print(f"Saved reports to: {out_dir}")


if __name__ == "__main__":
    main()
