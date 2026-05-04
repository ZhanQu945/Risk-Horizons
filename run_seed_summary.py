import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def _find_latest_run_dir(outputs_root: str, run_name_prefix: str) -> str:
    pattern = os.path.join(outputs_root, f"*{run_name_prefix}*")
    cands = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not cands:
        return ""
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def _load_metrics(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        obj = json.load(f)
    out: Dict[str, float] = {}
    for k, v in obj.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _dig(obj: Dict[str, object], path: str):
    cur = obj
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        return json.load(f)


def _pick_sample_outputs(run_dir: str) -> str:
    cands = [
        os.path.join(run_dir, "sample_outputs_geometry.json"),
        os.path.join(run_dir, "sample_outputs_all.json"),
        os.path.join(run_dir, "sample_outputs.json"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate metrics across multiple seeded runs.")
    ap.add_argument("--outputs-root", type=str, default="./outputs")
    ap.add_argument("--run-prefix", type=str, required=True, help="Prefix used in run-name per seed.")
    ap.add_argument("--seeds", type=str, default="13,21,42,87,123")
    ap.add_argument("--metrics-file", type=str, default="grail_geometry_metrics.json")
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--out-summary-csv", type=str, default="", help="Optional aggregate CSV with mean,std,mean±std.")
    ap.add_argument("--out-split-summary-csv", type=str, default="", help="Optional aggregate CSV for split stats.")
    ap.add_argument(
        "--check-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check whether sample prediction files exist and record file paths.",
    )
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    by_seed: Dict[int, Dict[str, float]] = {}
    run_dirs: Dict[int, str] = {}
    split_by_seed: Dict[int, Dict[str, float]] = {}
    aux_by_seed: Dict[int, Dict[str, object]] = {}

    split_extract: List[Tuple[str, str]] = [
        ("train_patients", "train.patients"),
        ("val_patients", "val.patients"),
        ("test_patients", "test.patients"),
        ("train_visits", "train.visits"),
        ("val_visits", "val.visits"),
        ("test_visits", "test.visits"),
        ("nodes_val_overlap_rate", "split_coverage.nodes.val.overlap_rate_with_train"),
        ("nodes_test_overlap_rate", "split_coverage.nodes.test.overlap_rate_with_train"),
        ("nodes_val_unseen_rate", "split_coverage.nodes.val.unseen_rate_vs_train"),
        ("nodes_test_unseen_rate", "split_coverage.nodes.test.unseen_rate_vs_train"),
        ("edges_val_overlap_rate", "split_coverage.transition_edges.val.overlap_rate_with_train"),
        ("edges_test_overlap_rate", "split_coverage.transition_edges.test.overlap_rate_with_train"),
        ("edges_val_unseen_rate", "split_coverage.transition_edges.val.unseen_rate_vs_train"),
        ("edges_test_unseen_rate", "split_coverage.transition_edges.test.unseen_rate_vs_train"),
    ]

    for seed in seeds:
        run_name = f"{args.run_prefix}_seed{seed}"
        run_dir = _find_latest_run_dir(args.outputs_root, run_name)
        if not run_dir:
            continue
        metrics_path = os.path.join(run_dir, args.metrics_file)
        if not os.path.exists(metrics_path):
            continue
        by_seed[seed] = _load_metrics(metrics_path)
        run_dirs[seed] = run_dir
        split_path = os.path.join(run_dir, "dataset_split_stats.json")
        runtime_path = os.path.join(run_dir, "runtime_summary.json")
        sample_path = _pick_sample_outputs(run_dir) if args.check_samples else ""
        split_vals: Dict[str, float] = {}
        if os.path.exists(split_path):
            sp = _load_json(split_path)
            for outk, p in split_extract:
                v = _dig(sp, p)
                if isinstance(v, (int, float)):
                    split_vals[f"split/{outk}"] = float(v)
        split_by_seed[seed] = split_vals
        aux_by_seed[seed] = {
            "dataset_split_stats_path": split_path if os.path.exists(split_path) else "",
            "runtime_summary_path": runtime_path if os.path.exists(runtime_path) else "",
            "sample_outputs_path": sample_path,
            "sample_outputs_exists": bool(sample_path and os.path.exists(sample_path)),
        }

    if not by_seed:
        raise SystemExit("No seed runs found for aggregation.")

    keys = sorted({k for m in by_seed.values() for k in m.keys()})
    rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [by_seed[s][k] for s in sorted(by_seed.keys()) if k in by_seed[s]]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": float(arr.size),
        }
    for seed in sorted(by_seed.keys()):
        row: Dict[str, object] = {"seed": seed, "run_dir": run_dirs.get(seed, "")}
        row.update(by_seed[seed])
        row.update(split_by_seed.get(seed, {}))
        row.update(aux_by_seed.get(seed, {}))
        rows.append(row)

    split_summary: Dict[str, Dict[str, float]] = {}
    split_keys = sorted({k for m in split_by_seed.values() for k in m.keys()})
    for k in split_keys:
        vals = [split_by_seed[s][k] for s in sorted(split_by_seed.keys()) if k in split_by_seed[s]]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        split_summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": float(arr.size),
        }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "run_prefix": args.run_prefix,
                "seeds_requested": seeds,
                "seeds_found": sorted(by_seed.keys()),
                "run_dirs": {str(k): v for k, v in run_dirs.items()},
                "summary": summary,
                "split_summary": split_summary,
                "per_seed": rows,
            },
            f,
            indent=2,
        )

    # lightweight CSV writer (no pandas dependency)
    csv_keys = [
        "seed",
        "run_dir",
        "dataset_split_stats_path",
        "runtime_summary_path",
        "sample_outputs_path",
        "sample_outputs_exists",
    ] + keys + split_keys
    with open(args.out_csv, "w") as f:
        f.write(",".join(csv_keys) + "\n")
        for row in rows:
            vals = []
            for k in csv_keys:
                v = row.get(k, "")
                vals.append(str(v))
            f.write(",".join(vals) + "\n")

    out_summary_csv = args.out_summary_csv
    if not out_summary_csv:
        root, ext = os.path.splitext(args.out_csv)
        out_summary_csv = f"{root}_summary{ext or '.csv'}"
    with open(out_summary_csv, "w") as f:
        f.write("metric,mean,std,mean_pm_std,min,max,n\n")
        for k in sorted(summary.keys()):
            s = summary[k]
            mean = float(s["mean"])
            std = float(s["std"])
            f.write(
                f"{k},{mean},{std},\"{mean:.6f} ± {std:.6f}\",{s['min']},{s['max']},{int(s['n'])}\n"
            )

    out_split_summary_csv = args.out_split_summary_csv
    if not out_split_summary_csv:
        root, ext = os.path.splitext(args.out_csv)
        out_split_summary_csv = f"{root}_split_summary{ext or '.csv'}"
    with open(out_split_summary_csv, "w") as f:
        f.write("split_metric,mean,std,mean_pm_std,min,max,n\n")
        for k in sorted(split_summary.keys()):
            s = split_summary[k]
            mean = float(s["mean"])
            std = float(s["std"])
            f.write(
                f"{k},{mean},{std},\"{mean:.6f} ± {std:.6f}\",{s['min']},{s['max']},{int(s['n'])}\n"
            )

    print(f"Saved seed summary JSON: {args.out_json}")
    print(f"Saved seed summary CSV: {args.out_csv}")
    print(f"Saved aggregate summary CSV: {out_summary_csv}")
    print(f"Saved split aggregate summary CSV: {out_split_summary_csv}")


if __name__ == "__main__":
    main()
