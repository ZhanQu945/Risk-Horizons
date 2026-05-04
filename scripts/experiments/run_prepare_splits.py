import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np

from grail.config import DataConfig
from grail.data import build_patient_trajectories, build_vocab, filter_min_visits, load_lab_groups, load_table, split_patients


def _unique_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        cand = f"{path}_v{idx}"
        if not os.path.exists(cand):
            return cand
        idx += 1


def _default_split_file(data_path: str, seed: int, train_frac: float, val_frac: float, min_visits: int) -> str:
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


def get_or_create_patient_split(
    trajectories,
    data_path: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    min_visits: int,
):
    path = _default_split_file(data_path, seed, train_frac, val_frac, min_visits)
    if os.path.exists(path):
        with open(path, "r") as f:
            payload = json.load(f)
        train_traj, val_traj, test_traj = _split_by_subject_ids(trajectories, payload)
        return path, train_traj, val_traj, test_traj, True

    train_traj, val_traj, test_traj = split_patients(trajectories, train_frac, val_frac, seed)
    payload = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "n_total": len(trajectories),
        "train_subject_ids": [str(t.subject_id) for t in train_traj],
        "val_subject_ids": [str(t.subject_id) for t in val_traj],
        "test_subject_ids": [str(t.subject_id) for t in test_traj],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path, train_traj, val_traj, test_traj, False


def _collect_nodes(trajs) -> set:
    s = set()
    for tr in trajs:
        for v in tr.visits:
            for codes in v.values():
                for c in codes:
                    s.add(int(c))
    return s


def _collect_transition_edges(trajs) -> set:
    e = set()
    for tr in trajs:
        vs = tr.visits
        for t in range(len(vs) - 1):
            src = [int(c) for vals in vs[t].values() for c in vals]
            dst = [int(c) for vals in vs[t + 1].values() for c in vals]
            for u in src:
                for v in dst:
                    e.add((u, v))
    return e


def _cov(a: set, b: set) -> Dict[str, float]:
    overlap = len(a & b)
    unseen = len(a - b)
    return {
        "size": int(len(a)),
        "overlap_with_train": int(overlap),
        "overlap_rate_with_train": float(overlap / max(len(a), 1)),
        "unseen_vs_train": int(unseen),
        "unseen_rate_vs_train": float(unseen / max(len(a), 1)),
    }


def _split_coverage_stats(train_traj, val_traj, test_traj) -> Dict[str, object]:
    n_tr = _collect_nodes(train_traj)
    n_va = _collect_nodes(val_traj)
    n_te = _collect_nodes(test_traj)
    e_tr = _collect_transition_edges(train_traj)
    e_va = _collect_transition_edges(val_traj)
    e_te = _collect_transition_edges(test_traj)
    return {
        "definition": {
            "nodes": "Unique event node ids appearing in visits.",
            "transition_edges": "Directed lag-1 event transitions (u in visit t -> v in visit t+1), pooled over modalities.",
        },
        "nodes": {
            "train": {"size": int(len(n_tr))},
            "val": _cov(n_va, n_tr),
            "test": _cov(n_te, n_tr),
            "val_test_overlap": int(len(n_va & n_te)),
            "val_test_overlap_rate_over_union": float(len(n_va & n_te) / max(len(n_va | n_te), 1)),
        },
        "transition_edges": {
            "train": {"size": int(len(e_tr))},
            "val": _cov(e_va, e_tr),
            "test": _cov(e_te, e_tr),
            "val_test_overlap": int(len(e_va & e_te)),
            "val_test_overlap_rate_over_union": float(len(e_va & e_te) / max(len(e_va | e_te), 1)),
        },
    }


def _prepare_dataset(name: str, data_path: str, label_map: str, lab_groups: str, seeds: List[int], text_mode: str, min_visits: int, min_code_freq: int, disable_labs: bool) -> Dict[str, object]:
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

    per_seed = []
    for seed in seeds:
        split_path, train_traj, val_traj, test_traj, reused = get_or_create_patient_split(
            trajectories,
            data_path=data_cfg.data_path,
            seed=int(seed),
            train_frac=data_cfg.train_frac,
            val_frac=data_cfg.val_frac,
            min_visits=data_cfg.min_visits,
        )
        per_seed.append(
            {
                "seed": int(seed),
                "split_file": split_path,
                "split_reused": bool(reused),
                "counts": {
                    "n_patients_train": int(len(train_traj)),
                    "n_patients_val": int(len(val_traj)),
                    "n_patients_test": int(len(test_traj)),
                    "n_visits_train": int(sum(len(t.visits) for t in train_traj)),
                    "n_visits_val": int(sum(len(t.visits) for t in val_traj)),
                    "n_visits_test": int(sum(len(t.visits) for t in test_traj)),
                },
                "split_coverage": _split_coverage_stats(train_traj, val_traj, test_traj),
            }
        )
    def _agg(path: str) -> Dict[str, float]:
        vals = []
        for sr in per_seed:
            cur = sr
            for p in path.split("."):
                cur = cur[p]
            vals.append(float(cur))
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std(ddof=0)) if arr.size else 0.0,
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
        }
    aggregate_5seed = {
        "nodes_val_overlap_rate_with_train": _agg("split_coverage.nodes.val.overlap_rate_with_train"),
        "nodes_test_overlap_rate_with_train": _agg("split_coverage.nodes.test.overlap_rate_with_train"),
        "nodes_val_unseen_rate_vs_train": _agg("split_coverage.nodes.val.unseen_rate_vs_train"),
        "nodes_test_unseen_rate_vs_train": _agg("split_coverage.nodes.test.unseen_rate_vs_train"),
        "edges_val_overlap_rate_with_train": _agg("split_coverage.transition_edges.val.overlap_rate_with_train"),
        "edges_test_overlap_rate_with_train": _agg("split_coverage.transition_edges.test.overlap_rate_with_train"),
        "edges_val_unseen_rate_vs_train": _agg("split_coverage.transition_edges.val.unseen_rate_vs_train"),
        "edges_test_unseen_rate_vs_train": _agg("split_coverage.transition_edges.test.unseen_rate_vs_train"),
        "n_patients_train": _agg("counts.n_patients_train"),
        "n_patients_val": _agg("counts.n_patients_val"),
        "n_patients_test": _agg("counts.n_patients_test"),
        "n_visits_train": _agg("counts.n_visits_train"),
        "n_visits_val": _agg("counts.n_visits_val"),
        "n_visits_test": _agg("counts.n_visits_test"),
    }
    return {
        "dataset": name,
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
        "seed_reports": per_seed,
        "aggregate_5seed": aggregate_5seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/reuse 5-seed patient splits and save split coverage metrics for MIMIC/eICU.")
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
    parser.add_argument(
        "--disable-lab-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable lab modality for split prep/metrics (default: true).",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if 13 not in seeds:
        raise ValueError("Seed 13 must be included in --seeds.")
    if len(seeds) != 5:
        raise ValueError(f"Expected exactly 5 seeds, got {len(seeds)} from: {seeds}")

    ts = datetime.now().strftime("prepare_splits_%Y%m%d_%H%M%S")
    out_dir = _unique_dir(args.out_dir or os.path.join("./outputs", ts))
    os.makedirs(out_dir, exist_ok=False)

    mimic = _prepare_dataset(
        name="mimiciv",
        data_path=args.mimic_data_path,
        label_map=args.mimic_label_map,
        lab_groups=args.mimic_lab_groups,
        seeds=seeds,
        text_mode=args.text_mode,
        min_visits=args.min_visits,
        min_code_freq=args.min_code_freq,
        disable_labs=bool(args.disable_lab_tests),
    )
    eicu = _prepare_dataset(
        name="eicu",
        data_path=args.eicu_data_path,
        label_map=args.eicu_label_map,
        lab_groups=args.eicu_lab_groups,
        seeds=seeds,
        text_mode=args.text_mode,
        min_visits=args.min_visits,
        min_code_freq=args.min_code_freq,
        disable_labs=bool(args.disable_lab_tests),
    )

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "datasets": [mimic, eicu],
    }
    json_path = os.path.join(out_dir, "prepared_splits_and_coverage_5seed.json")
    md_path = os.path.join(out_dir, "prepared_splits_and_coverage_5seed.md")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Prepared Splits and Coverage (5 Seeds)",
        "",
        f"Generated: {payload['generated_at']}",
        f"Seeds: {seeds}",
        "",
        "Seed 13 split is reused if already present; missing seeds are created under Processed/*/splits.",
        "",
    ]
    for ds in payload["datasets"]:
        lines.append(f"## {ds['dataset']}")
        for sr in ds["seed_reports"]:
            lines.append(
                f"- seed={sr['seed']} reused={sr['split_reused']} split={sr['split_file']}"
            )
        ag = ds["aggregate_5seed"]
        lines.append("- 5-seed aggregates:")
        lines.append(
            f"  - val node overlap with train: {100*ag['nodes_val_overlap_rate_with_train']['mean']:.2f}% ± {100*ag['nodes_val_overlap_rate_with_train']['std']:.2f}%"
        )
        lines.append(
            f"  - test node overlap with train: {100*ag['nodes_test_overlap_rate_with_train']['mean']:.2f}% ± {100*ag['nodes_test_overlap_rate_with_train']['std']:.2f}%"
        )
        lines.append(
            f"  - val edge overlap with train: {100*ag['edges_val_overlap_rate_with_train']['mean']:.2f}% ± {100*ag['edges_val_overlap_rate_with_train']['std']:.2f}%"
        )
        lines.append(
            f"  - test edge overlap with train: {100*ag['edges_test_overlap_rate_with_train']['mean']:.2f}% ± {100*ag['edges_test_overlap_rate_with_train']['std']:.2f}%"
        )
        lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("=== 5-seed overlap aggregates ===")
    for ds in payload["datasets"]:
        ag = ds["aggregate_5seed"]
        print(f"[{ds['dataset']}]")
        print(
            "  val node overlap: "
            f"{100*ag['nodes_val_overlap_rate_with_train']['mean']:.2f}% "
            f"+/- {100*ag['nodes_val_overlap_rate_with_train']['std']:.2f}%"
        )
        print(
            "  test node overlap: "
            f"{100*ag['nodes_test_overlap_rate_with_train']['mean']:.2f}% "
            f"+/- {100*ag['nodes_test_overlap_rate_with_train']['std']:.2f}%"
        )
        print(
            "  val edge overlap: "
            f"{100*ag['edges_val_overlap_rate_with_train']['mean']:.2f}% "
            f"+/- {100*ag['edges_val_overlap_rate_with_train']['std']:.2f}%"
        )
        print(
            "  test edge overlap: "
            f"{100*ag['edges_test_overlap_rate_with_train']['mean']:.2f}% "
            f"+/- {100*ag['edges_test_overlap_rate_with_train']['std']:.2f}%"
        )
        print(
            "  val unseen node rate: "
            f"{100*ag['nodes_val_unseen_rate_vs_train']['mean']:.2f}% "
            f"+/- {100*ag['nodes_val_unseen_rate_vs_train']['std']:.2f}%"
        )
        print(
            "  test unseen node rate: "
            f"{100*ag['nodes_test_unseen_rate_vs_train']['mean']:.2f}% "
            f"+/- {100*ag['nodes_test_unseen_rate_vs_train']['std']:.2f}%"
        )
        print(
            "  val unseen edge rate: "
            f"{100*ag['edges_val_unseen_rate_vs_train']['mean']:.2f}% "
            f"+/- {100*ag['edges_val_unseen_rate_vs_train']['std']:.2f}%"
        )
        print(
            "  test unseen edge rate: "
            f"{100*ag['edges_test_unseen_rate_vs_train']['mean']:.2f}% "
            f"+/- {100*ag['edges_test_unseen_rate_vs_train']['std']:.2f}%"
        )

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Out dir: {out_dir}")


if __name__ == "__main__":
    main()
