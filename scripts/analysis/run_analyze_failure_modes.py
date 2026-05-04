import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


def _safe_mean(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _bin_label(x: float, edges: List[float]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return f"[{edges[i]}, {edges[i+1]})"
    return f"[{edges[-1]}, inf)"


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc failure mode analysis from sample_outputs*.json")
    p.add_argument("--sample-path", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    with open(args.sample_path, "r") as f:
        samples = json.load(f)

    if not isinstance(samples, list) or not samples:
        raise ValueError("sample file is empty or invalid")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.sample_path))
    os.makedirs(out_dir, exist_ok=True)

    overall = defaultdict(list)
    by_mod_visitlen = defaultdict(lambda: defaultdict(list))
    by_mod_truthsize = defaultdict(lambda: defaultdict(list))
    hard_cases = []

    for s in samples:
        pvm = s.get("per_visit_metrics", {})
        history = s.get("history", {})
        truth_next = s.get("truth_next", {})
        preds = s.get("predictions", {})

        visit_len = sum(len(v) for v in history.values())

        for m in ["dx", "proc", "med", "lab"]:
            r5 = float(pvm.get(f"{m}/recall@5", np.nan))
            r10 = float(pvm.get(f"{m}/recall@10", np.nan))
            nd10 = float(pvm.get(f"{m}/ndcg@10", np.nan))
            mrr = float(pvm.get(f"{m}/mrr", np.nan))
            if np.isnan(r10):
                continue

            truth_size = len(truth_next.get(m, []))
            pred_topk = preds.get(m, [])[: args.topk]

            overall[f"{m}/recall@5"].append(r5)
            overall[f"{m}/recall@10"].append(r10)
            overall[f"{m}/ndcg@10"].append(nd10)
            overall[f"{m}/mrr"].append(mrr)

            vb = _bin_label(float(visit_len), [0, 10, 25, 50, 100])
            tb = _bin_label(float(truth_size), [0, 1, 3, 5, 10])
            by_mod_visitlen[m][vb].append(r10)
            by_mod_truthsize[m][tb].append(r10)

            if r10 == 0.0:
                hard_cases.append(
                    {
                        "subject_id": s.get("subject_id"),
                        "visit_index": s.get("visit_index"),
                        "modality": m,
                        "history_size": int(visit_len),
                        "truth_size": int(truth_size),
                        "truth_next": truth_next.get(m, []),
                        "pred_topk": pred_topk,
                        "metrics": {
                            "recall@5": r5,
                            "recall@10": r10,
                            "ndcg@10": nd10,
                            "mrr": mrr,
                        },
                    }
                )

    summary = {
        "n_samples": int(len(samples)),
        "overall": {k: _safe_mean(v) for k, v in overall.items()},
        "by_visit_size_bin_recall10": {
            m: {b: _safe_mean(vals) for b, vals in bins.items()} for m, bins in by_mod_visitlen.items()
        },
        "by_truth_size_bin_recall10": {
            m: {b: _safe_mean(vals) for b, vals in bins.items()} for m, bins in by_mod_truthsize.items()
        },
        "hard_case_count": int(len(hard_cases)),
    }

    summary_path = os.path.join(out_dir, "failure_mode_summary.json")
    hard_path = os.path.join(out_dir, "failure_mode_hard_cases.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(hard_path, "w") as f:
        json.dump(hard_cases, f, indent=2)

    print(f"Saved: {summary_path}")
    print(f"Saved: {hard_path}")


if __name__ == "__main__":
    main()
