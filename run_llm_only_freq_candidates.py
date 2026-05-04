import argparse
import difflib
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from grail.config import DataConfig, EvalConfig
from grail.data import (
    _med_label_keys,
    _proc_label_keys,
    build_patient_trajectories,
    build_hierarchy_edges,
    build_vocab,
    filter_min_visits,
    load_lab_groups,
    load_table,
    node_display,
    split_patients,
)
from grail.eval import aggregate_metrics, ancestor_match, hierarchy_distance, mrr, ndcg_at_k, precision_at_k, recall_at_k
from grail.llm import LLMReranker


def _log(msg: str) -> None:
    print(msg, flush=True)


def _log_label_coverage(vocab) -> None:
    by_mod = {}
    for nid, node in vocab.id_to_node.items():
        modality, code = str(node).split(":", 1)
        if code == "ROOT":
            continue
        label = str(getattr(vocab, "id_to_label", {}).get(int(nid), code))
        st = by_mod.setdefault(modality, {"total": 0, "text_like": 0})
        st["total"] += 1
        if label != code:
            st["text_like"] += 1
    _log("Label coverage by modality (non-root nodes):")
    for modality in sorted(by_mod.keys()):
        total = by_mod[modality]["total"]
        text_like = by_mod[modality]["text_like"]
        pct = 100.0 * text_like / max(total, 1)
        _log(f"  {modality}: text_like={text_like}/{total} ({pct:.1f}%)")


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.shape[0]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def summarize_trajectories(trajectories, modality_cols: Dict[str, str]) -> Dict[str, object]:
    visits_per_patient = [len(traj.visits) for traj in trajectories]
    modality_counts = {m: [] for m in modality_cols.keys()}
    for traj in trajectories:
        for visit in traj.visits:
            for m in modality_counts.keys():
                modality_counts[m].append(len(visit.get(m, [])))
    return {
        "patients": int(len(trajectories)),
        "visits": int(sum(visits_per_patient)),
        "visits_per_patient": _summary_stats(visits_per_patient),
        "events_per_visit": {m: _summary_stats(vals) for m, vals in modality_counts.items()},
    }


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


def _split_coverage_stats(train_traj, val_traj, test_traj) -> Dict[str, object]:
    n_tr = _collect_nodes(train_traj)
    n_va = _collect_nodes(val_traj)
    n_te = _collect_nodes(test_traj)
    e_tr = _collect_transition_edges(train_traj)
    e_va = _collect_transition_edges(val_traj)
    e_te = _collect_transition_edges(test_traj)

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


def save_split_stats(run_output_dir: str, modality_cols: Dict[str, str], train_traj, val_traj, test_traj) -> str:
    payload = {
        "train": summarize_trajectories(train_traj, modality_cols),
        "val": summarize_trajectories(val_traj, modality_cols),
        "test": summarize_trajectories(test_traj, modality_cols),
        "all": summarize_trajectories(train_traj + val_traj + test_traj, modality_cols),
        "split_coverage": _split_coverage_stats(train_traj, val_traj, test_traj),
    }
    path = os.path.join(run_output_dir, "dataset_split_stats.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


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
    split_file: Optional[str] = None,
):
    path = split_file or _default_split_file(data_path, seed, train_frac, val_frac, min_visits)
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


def _unique_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        candidate = f"{path}_v{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _canonical(vocab, idx: int) -> str:
    return str(vocab.id_to_node.get(int(idx), str(idx)))


def _build_token_to_id(vocab) -> Dict[str, int]:
    mp = {}
    for idx, tok in vocab.id_to_node.items():
        mp[str(tok).strip().lower()] = int(idx)
    return mp


def _norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s\.\-\_:/()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_code(s: str) -> str:
    return str(s).strip().upper().replace(" ", "")


def _extract_code_candidates(raw: str) -> List[str]:
    raw = str(raw).strip()
    cands = [raw]
    # Hybrid strings often contain explicit code in trailing parentheses.
    for m in re.finditer(r"\(([^()]{1,30})\)", raw):
        cands.append(m.group(1))
    # Prefix format: dx:XXX / proc:XXX / med:XXX
    if ":" in raw:
        cands.append(raw.split(":", 1)[1].strip())
    out = []
    seen = set()
    for c in cands:
        cc = _norm_code(c)
        if cc and cc not in seen:
            out.append(cc)
            seen.add(cc)
    return out


def _build_modality_indices(vocab):
    idx = {}
    for modality in ("dx", "proc", "med"):
        idx[modality] = {
            "canonical": {},
            "code_variants": {},
            "label_exact": {},
            "label_values": [],
            "label_to_ids": {},
        }
    for nid, node in vocab.id_to_node.items():
        nid = int(nid)
        m, code = str(node).split(":", 1)
        if m not in idx:
            continue
        label = str(getattr(vocab, "id_to_label", {}).get(nid, code))
        idx[m]["canonical"][str(node).lower()] = nid
        # Code variant lookup.
        code_vars = [_norm_code(code)]
        if m == "proc":
            code_vars.extend([_norm_code(x) for x in _proc_label_keys(code)])
        elif m == "med":
            code_vars.extend([_norm_code(x) for x in _med_label_keys(code)])
        for cv in code_vars:
            if cv:
                idx[m]["code_variants"].setdefault(cv, []).append(nid)
        # Text/hybrid lookup.
        nl = _norm_text(label)
        if nl:
            idx[m]["label_exact"].setdefault(nl, []).append(nid)
            idx[m]["label_to_ids"].setdefault(nl, []).append(nid)
            idx[m]["label_values"].append(nl)
    return idx


def _map_token_to_ids(token: str, modality: str, token_to_id: Dict[str, int], mod_idx) -> List[int]:
    out = []
    tok = str(token).strip()
    if not tok:
        return out
    # 1) exact canonical node token.
    ex = mod_idx[modality]["canonical"].get(tok.lower())
    if ex is not None:
        out.append(int(ex))
    # 2) code-based matching (accept code/text/hybrid).
    for code_cand in _extract_code_candidates(tok):
        ids = mod_idx[modality]["code_variants"].get(code_cand, [])
        out.extend([int(i) for i in ids])
    # 3) exact label matching.
    ntext = _norm_text(tok)
    out.extend([int(i) for i in mod_idx[modality]["label_exact"].get(ntext, [])])
    # 4) fuzzy label fallback for near-matches (text/hybrid generation).
    if not out and ntext and len(ntext) >= 6 and mod_idx[modality]["label_values"]:
        close = difflib.get_close_matches(ntext, mod_idx[modality]["label_values"], n=1, cutoff=0.90)
        if close:
            out.extend([int(i) for i in mod_idx[modality]["label_to_ids"].get(close[0], [])])
    # Dedupe preserving order.
    ded = []
    seen = set()
    for i in out:
        if i not in seen:
            ded.append(i)
            seen.add(i)
    return ded


def _build_prompt(
    history_codes: List[List[str]],
    target_modality: str,
    candidate_source: str = "structured retrieval",
) -> str:
    lines = [
        "You are a clinical event prediction assistant.",
        "",
        "Task:",
        "Given a patient's longitudinal visit history, estimate the probability that a candidate event will occur in the NEXT visit.",
        "",
        "Modalities:",
        "- dx: diagnosis",
        "- proc: procedure",
        "- med: medication",
        "",
        "Instructions:",
        "- Consider temporal progression (earlier -> later visits).",
        "- Consider cross-modality consistency (diagnoses, procedures, medications).",
        "- Favor clinically plausible developments over unrelated or overly generic events.",
        "- Do not rely solely on frequency or co-occurrence; reason about progression.",
        "",
        "Output:",
        "Return exactly one floating-point number between 0 and 1.",
        "Do not output any other text.",
        "",
        "Patient history (oldest -> newest):",
    ]
    for t, codes in enumerate(history_codes, 1):
        lines.append(f"Visit {t}: {', '.join(codes)}")
    lines.extend(
        [
            "",
            f"Candidate ({target_modality}):",
            "",
            "Candidate context:",
            f"- Proposed by upstream structured retrieval ({candidate_source})",
            "- Assumed to be clinically plausible; refine its likelihood",
            "",
            "Score:",
        ]
    )
    return "\n".join(lines)


def _build_frequency_candidate_pools(
    train_traj,
    k_per_modality: Dict[str, int],
    policy: str = "top",
    random_seed: int = 13,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[int, int]]]:
    modalities = [str(m) for m in k_per_modality.keys()]
    freq = {m: {} for m in modalities}
    for traj in train_traj:
        for visit in traj.visits:
            for m in modalities:
                for c in visit.get(m, []):
                    ci = int(c)
                    freq[m][ci] = int(freq[m].get(ci, 0) + 1)
    rng = random.Random(int(random_seed))
    pools: Dict[str, List[int]] = {}
    for m in modalities:
        items = list(freq[m].items())
        if policy == "bottom":
            items = sorted(items, key=lambda x: (x[1], x[0]))
        elif policy == "random":
            items = sorted(items, key=lambda x: x[0])
            rng.shuffle(items)
        else:
            items = sorted(items, key=lambda x: (-x[1], x[0]))
        pools[m] = [int(idx) for idx, _ in items[: int(k_per_modality.get(m, 80))]]
    return pools, freq


def _parse_json(text: str) -> Dict[str, List[str]]:
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        text = text[s : e + 1]
    try:
        obj = json.loads(text)
    except Exception:
        # Fallback parser for non-JSON outputs:
        #   dx: a, b
        #   proc: c, d
        #   med: e, f
        out = {"dx": [], "proc": [], "med": []}
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        for ln in lines:
            low = ln.lower()
            for m in ("dx", "proc", "med"):
                if low.startswith(f"{m}:"):
                    payload = ln.split(":", 1)[1]
                    toks = [t.strip().strip('"').strip("'") for t in re.split(r"[,\;\|]", payload) if t.strip()]
                    out[m].extend(toks)
        return out
    out = {}
    for m in ("dx", "proc", "med"):
        vals = obj.get(m, [])
        if isinstance(vals, list):
            out[m] = [str(x).strip() for x in vals if str(x).strip()]
        else:
            out[m] = []
    return out


def evaluate_zero_shot(
    trajectories,
    vocab,
    eval_cfg: EvalConfig,
    reranker: LLMReranker,
    token_to_id: Dict[str, int],
    sample_path: str,
    max_new_tokens: int,
    sample_limit: int = 50,
    candidate_pool_ids: Dict[str, List[int]] = None,
    train_code_freq: Dict[str, Dict[int, int]] = None,
):
    metrics = []
    samples = []
    total_steps = sum(max(0, len(traj.visits) - 1) for traj in trajectories)
    step = 0
    _log(f"Eval steps (test only): {total_steps}")
    h_graph = nx.DiGraph()
    h_graph.add_nodes_from(vocab.id_to_node.keys())
    for u, v, _ in build_hierarchy_edges(vocab):
        h_graph.add_edge(int(u), int(v))
    freq_bucket = {m: {"truth_freq_mean": [], "recall10": []} for m in eval_cfg.eval_modalities}

    for traj in trajectories:
        visits = traj.visits
        if len(visits) < 2:
            continue
        for t in range(len(visits) - 1):
            step += 1
            current = visits[t]
            next_visit = visits[t + 1]

            hist_struct = []
            for back in range(max(0, t - eval_cfg.max_history_visits + 1), t + 1):
                vv = visits[back]
                hist_struct.append(
                    {
                        "dx": [_canonical(vocab, c) for c in vv.get("dx", [])],
                        "proc": [_canonical(vocab, c) for c in vv.get("proc", [])],
                        "med": [_canonical(vocab, c) for c in vv.get("med", [])],
                    }
                )
            history_codes = []
            for v in hist_struct:
                merged = []
                for m in ("dx", "proc", "med"):
                    vals = v.get(m, [])
                    if vals:
                        merged.append(f"{m}: " + ", ".join(vals))
                history_codes.append(" | ".join(merged))

            sample_metrics = {}
            pred_map = {}
            sample_prompts = {}
            for modality in eval_cfg.eval_modalities:
                candidate_ids = [int(x) for x in (candidate_pool_ids.get(modality, []) if candidate_pool_ids else [])]
                candidate_texts = [node_display(vocab, c) for c in candidate_ids]
                prompt = _build_prompt(history_codes, target_modality=modality, candidate_source="frequency retrieval")
                sample_prompts[modality] = prompt
                scores = reranker.score(prompt, candidate_texts) if candidate_texts else []
                if scores:
                    order = list(np.argsort(np.asarray(scores))[::-1])
                    preds = [candidate_ids[i] for i in order]
                else:
                    preds = []
                pred_map[modality] = preds
                truth = next_visit.get(modality, [])
                for k in eval_cfg.top_ks:
                    sample_metrics[f"{modality}/recall@{k}"] = recall_at_k(preds, truth, k)
                    sample_metrics[f"{modality}/precision@{k}"] = precision_at_k(preds, truth, k)
                    sample_metrics[f"{modality}/ndcg@{k}"] = ndcg_at_k(preds, truth, k)
                sample_metrics[f"{modality}/mrr"] = mrr(preds, truth)
                if train_code_freq is not None and truth:
                    tf = [float(train_code_freq.get(modality, {}).get(int(tid), 0)) for tid in truth]
                    mean_tf = float(np.mean(tf)) if tf else 0.0
                    freq_bucket[modality]["truth_freq_mean"].append(mean_tf)
                    freq_bucket[modality]["recall10"].append(float(sample_metrics.get(f"{modality}/recall@10", 0.0)))
                if preds and truth:
                    best_pred = preds[0]
                    best_truth = truth[0]
                    sample_metrics[f"{modality}/hier_dist"] = float(hierarchy_distance(h_graph, best_pred, best_truth))
                    sample_metrics[f"{modality}/ancestor_match@2"] = float(ancestor_match(h_graph, best_pred, best_truth, 2))
                if modality == "dx" and preds and truth:
                    primary = truth[0]
                    for k in eval_cfg.top_ks:
                        sample_metrics[f"dx_primary/top@{k}"] = float(primary in preds[:k])
                    sample_metrics["dx_primary/mrr"] = mrr(preds, [primary])

            metrics.append(sample_metrics)
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "subject_id": traj.subject_id,
                        "visit_index": t,
                        "prompt_by_modality": sample_prompts,
                        "candidate_pool": {
                            m: [node_display(vocab, c) for c in (candidate_pool_ids.get(m, []) if candidate_pool_ids else [])]
                            for m in eval_cfg.eval_modalities
                        },
                        "history": {m: [node_display(vocab, c) for c in current.get(m, [])] for m in current.keys()},
                        "truth_next": {m: [node_display(vocab, c) for c in next_visit.get(m, [])] for m in next_visit.keys()},
                        "predictions": {
                            m: [node_display(vocab, idx) for idx in pred_map.get(m, [])[: max(eval_cfg.top_ks)]]
                            for m in eval_cfg.eval_modalities
                        },
                    }
                )
            if step % 20 == 0:
                _log(f"Eval progress: {step}/{total_steps}")

    with open(sample_path, "w") as f:
        json.dump(samples, f, indent=2)
    out = aggregate_metrics(metrics)
    for m in eval_cfg.eval_modalities:
        xf = freq_bucket[m]["truth_freq_mean"]
        yr = freq_bucket[m]["recall10"]
        if len(xf) >= 8 and len(yr) == len(xf):
            x = np.asarray(xf, dtype=float)
            y = np.asarray(yr, dtype=float)
            q = np.quantile(x, [0.25, 0.5, 0.75]).tolist()
            bins = np.digitize(x, q, right=True)
            for b in range(4):
                mask = bins == b
                if np.any(mask):
                    out[f"{m}/freq_bin{b+1}_recall@10"] = float(np.mean(y[mask]))
                    out[f"{m}/freq_bin{b+1}_n"] = float(np.sum(mask))
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)

    parser = argparse.ArgumentParser(description="LLM-only zero-shot next-visit prediction with frequency-ranked candidate constraints.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--format", type=str, default="pickle")
    parser.add_argument("--lab-groups", type=str, default=None)
    parser.add_argument("--text-mode", type=str, default="hybrid", choices=["hybrid", "code", "text"])
    parser.add_argument("--label-map", type=str, default=None)
    parser.add_argument("--llm-backend", type=str, default="hf", choices=["hf", "openai", "heuristic"])
    parser.add_argument("--llm-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--llm-scoring-mode",
        type=str,
        default="pointwise",
        choices=["pointwise", "batched_pairwise", "listwise"],
        help="LLM candidate scoring mode.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--max-history-visits", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--candidate-k-dx", type=int, default=None)
    parser.add_argument("--candidate-k-proc", type=int, default=None)
    parser.add_argument("--candidate-k-med", type=int, default=None)
    parser.add_argument("--candidate-policy", type=str, default="top", choices=["top", "bottom", "random"])
    parser.add_argument("--candidate-random-seed", type=int, default=13)
    parser.add_argument("--min-code-freq", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument(
        "--save-all-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-visit outputs for all evaluation steps (default: true). Use --no-save-all-samples to disable.",
    )
    parser.add_argument("--enable-lab-tests", action="store_true")
    parser.add_argument("--disable-lab-tests", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-sample-size", type=int, default=2000)
    args = parser.parse_args()

    _log("=== LLM-Only Frequency-Candidate Run Start ===")
    run_start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log(f"Start time: {run_start_ts}")
    _log(f"Data path: {args.data_path}")
    _log(f"Text mode: {args.text_mode}")
    _log(f"LLM backend: {args.llm_backend}")
    _log(f"LLM model: {args.llm_model}")
    _log(f"LLM scoring mode: {args.llm_scoring_mode}")

    cache_base = os.environ.get("GRAIL_CACHE_BASE", "./cache")
    os.makedirs(cache_base, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_base, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_base, "datasets"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_base, "torch"))

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_cfg = DataConfig(data_path=args.data_path, format=args.format, text_mode=args.text_mode, label_map_path=args.label_map)
    data_cfg.seed = args.seed
    data_cfg.min_code_freq = args.min_code_freq
    eval_cfg = EvalConfig(max_history_visits=args.max_history_visits)

    labs_enabled = bool(args.enable_lab_tests) and not bool(args.disable_lab_tests)
    if not labs_enabled:
        data_cfg.modality_cols = {k: v for k, v in data_cfg.modality_cols.items() if k != "lab"}
        eval_cfg.eval_modalities = tuple(m for m in eval_cfg.eval_modalities if m != "lab")

    if args.debug:
        data_cfg.min_visits = 2

    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_tag = f"{args.run_name}_{timestamp}" if args.run_name else timestamp
    run_name = f"llm_only_freqcand_{args.text_mode}_{run_tag}"
    output_root = os.environ.get("GRAIL_OUTPUT_ROOT", "./outputs")
    run_output_dir = _unique_dir(os.path.join(output_root, run_name))
    os.makedirs(run_output_dir, exist_ok=False)

    wall_start = time.perf_counter()
    stage_times: Dict[str, float] = {}

    t_stage = time.perf_counter()
    df = load_table(data_cfg.data_path, data_cfg.format)
    stage_times["load_table_sec"] = float(time.perf_counter() - t_stage)
    _log(f"Loaded table: {len(df)} rows")

    if args.debug:
        counts = df.groupby(data_cfg.subject_id_col)[data_cfg.visit_id_col].nunique()
        eligible = counts[counts >= 2].index.tolist()
        if eligible:
            sample_size = min(args.debug_sample_size, len(eligible))
            sampled_ids = np.random.RandomState(data_cfg.seed).choice(eligible, size=sample_size, replace=False)
            df = df[df[data_cfg.subject_id_col].isin(sampled_ids)].reset_index(drop=True)
            _log(f"Debug mode: sampled {sample_size} patients with >=2 visits ({len(df)} rows)")

    lab_to_group = load_lab_groups(args.lab_groups) if labs_enabled else None

    t_stage = time.perf_counter()
    vocab = build_vocab(
        df,
        data_cfg.modality_cols,
        min_code_freq=data_cfg.min_code_freq,
        top_k_per_modality=data_cfg.top_k_per_modality,
        lab_to_group=lab_to_group,
        text_mode=data_cfg.text_mode,
        label_map_path=data_cfg.label_map_path,
    )
    stage_times["build_vocab_sec"] = float(time.perf_counter() - t_stage)
    _log_label_coverage(vocab)

    t_stage = time.perf_counter()
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
    stage_times["build_trajectories_sec"] = float(time.perf_counter() - t_stage)

    split_file_path, train_traj, val_traj, test_traj, reused_split = get_or_create_patient_split(
        trajectories,
        data_path=data_cfg.data_path,
        seed=data_cfg.seed,
        train_frac=data_cfg.train_frac,
        val_frac=data_cfg.val_frac,
        min_visits=data_cfg.min_visits,
        split_file=args.split_file,
    )
    _log(f"Patient split file: {split_file_path} ({'reused' if reused_split else 'created'})")
    _log(f"Split sizes | train: {len(train_traj)} | val: {len(val_traj)} | test: {len(test_traj)}")
    k_map = {
        "dx": int(args.candidate_k_dx if args.candidate_k_dx is not None else args.candidate_k),
        "proc": int(args.candidate_k_proc if args.candidate_k_proc is not None else args.candidate_k),
        "med": int(args.candidate_k_med if args.candidate_k_med is not None else args.candidate_k),
    }
    candidate_pool_ids, train_code_freq = _build_frequency_candidate_pools(
        train_traj,
        k_map,
        policy=str(args.candidate_policy),
        random_seed=int(args.candidate_random_seed),
    )
    _log(
        f"Candidate pool sizes (train-frequency, policy={args.candidate_policy}): "
        f"dx={len(candidate_pool_ids['dx'])}, "
        f"proc={len(candidate_pool_ids['proc'])}, "
        f"med={len(candidate_pool_ids['med'])}"
    )

    split_stats_path = save_split_stats(run_output_dir, data_cfg.modality_cols, train_traj, val_traj, test_traj)
    _log(f"Saved split stats: {split_stats_path}")

    reranker = LLMReranker(
        backend=args.llm_backend,
        model_name=args.llm_model,
        scoring_mode=str(args.llm_scoring_mode),
    )
    token_to_id = _build_token_to_id(vocab)

    sample_path = os.path.join(run_output_dir, "sample_outputs_llm_only.json")
    sample_limit_eval = int(args.sample_limit)
    if args.save_all_samples:
        sample_limit_eval = int(sum(max(0, len(t.visits) - 1) for t in test_traj))
    elif sample_limit_eval <= 0:
        sample_limit_eval = 0
    t_stage = time.perf_counter()
    metrics = evaluate_zero_shot(
        test_traj,
        vocab,
        eval_cfg,
        reranker,
        token_to_id,
        sample_path=sample_path,
        max_new_tokens=max(64, int(args.max_new_tokens)),
        sample_limit=sample_limit_eval,
        candidate_pool_ids=candidate_pool_ids,
        train_code_freq=train_code_freq,
    )
    stage_times["evaluate_sec"] = float(time.perf_counter() - t_stage)

    metrics_path = os.path.join(run_output_dir, "llm_only_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    wall_total = float(time.perf_counter() - wall_start)
    runtime_payload = {
        "start_time": run_start_ts,
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "text_mode": args.text_mode,
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
        "llm_scoring_mode": str(args.llm_scoring_mode),
        "mode": "zero_shot_frequency_candidates_test_only",
        "candidate_k": k_map,
        "candidate_policy": str(args.candidate_policy),
        "candidate_random_seed": int(args.candidate_random_seed),
        "labs_enabled": bool(labs_enabled),
        "data_path": data_cfg.data_path,
        "run_output_dir": run_output_dir,
        "split_file": split_file_path,
        "counts": {
            "n_patients_total": len(trajectories),
            "n_patients_train": len(train_traj),
            "n_patients_val": len(val_traj),
            "n_patients_test": len(test_traj),
            "n_visits_train": int(sum(len(t.visits) for t in train_traj)),
            "n_visits_val": int(sum(len(t.visits) for t in val_traj)),
            "n_visits_test": int(sum(len(t.visits) for t in test_traj)),
            "n_visit_pairs_test": int(sum(max(0, len(t.visits) - 1) for t in test_traj)),
        },
        "stage_times_sec": stage_times,
        "total_wall_time_sec": wall_total,
    }
    runtime_path = os.path.join(run_output_dir, "llm_only_runtime.json")
    with open(runtime_path, "w") as f:
        json.dump(runtime_payload, f, indent=2)

    _log(f"Saved metrics: {metrics_path}")
    _log(f"Saved runtime report: {runtime_path}")
    _log("Timing summary:")
    for k, v in sorted(stage_times.items()):
        _log(f"  {k}: {v:.2f}s")
    _log(f"  total_wall_time_sec: {wall_total:.2f}s")
    for key, val in sorted(metrics.items()):
        _log(f"{key}: {val:.4f}")
    _log("=== LLM-Only Frequency-Candidate Run Complete ===")


if __name__ == "__main__":
    main()
