import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import torch

from grail.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    RerankConfig,
    TrainConfig,
)
from grail.data import (
    build_patient_trajectories,
    build_vocab,
    filter_min_visits,
    load_lab_groups,
    load_table,
    node_display,
    split_patients,
)
from grail.eval import (
    aggregate_metrics,
    ancestor_match,
    hierarchy_distance,
    micro_f1,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    tune_threshold,
)
from grail.graph import build_graph, edge_type_map, graph_to_edge_list
from grail.llm import LLMReranker
from grail.visualize import plot_patient_trajectories, plot_poincare_disk_embeddings
from grail.baselines import (
    BehrtModel,
    MedBertModel,
    NextVisitDataset,
    RetainModel,
    VisitTransformer,
    GramModel,
    HaloNextVisitModel,
    HybridSeqModel,
    init_sequence_embeddings_from_hyperbolic,
    build_ancestor_map,
    train_sequence_model,
    evaluate_sequence_model,
    pretrain_code_mlm,
    pretrain_masked_visits,
    tune_thresholds_sequence,
)
from grail.model import GrailModel
from grail.retrieval import build_risk_horizon, build_risk_horizon_multi_cone
from grail.train import train_model


def _load_baseline_pretrained_checkpoint(model: torch.nn.Module, ckpt_path: str) -> Dict[str, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")
    cleaned = {}
    for k, v in state.items():
        nk = str(k)
        for prefix in ("module.", "model.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    _log(
        f"Loaded pretrained baseline checkpoint: {ckpt_path} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return {"missing": len(missing), "unexpected": len(unexpected)}


def _freeze_baseline_backbone(model: torch.nn.Module) -> int:
    trainable = 0
    for name, p in model.named_parameters():
        # keep prediction head trainable
        if name.startswith("out."):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
    return int(trainable)


def evaluate_copy_last_baseline(
    trajectories,
    vocab,
    top_ks: Tuple[int, ...],
    sample_path: Optional[str] = None,
    sample_limit: int = 0,
    train_seen_by_modality: Optional[Dict[str, set]] = None,
) -> Dict[str, float]:
    metrics = []
    samples = []
    eval_modalities = sorted(set(vocab.node_type.values()))
    split_stats = {
        m: {
            "persistent_recall@5": [],
            "persistent_recall@10": [],
            "novel_recall@5": [],
            "novel_recall@10": [],
            "persistent_hit_rate": [],
            "novel_hit_rate": [],
            "seen_train_recall@5": [],
            "seen_train_recall@10": [],
            "seen_train_hit_rate": [],
            "unseen_train_recall@5": [],
            "unseen_train_recall@10": [],
            "unseen_train_hit_rate": [],
            "novel_seen_train_recall@5": [],
            "novel_seen_train_recall@10": [],
            "novel_seen_train_hit_rate": [],
            "novel_unseen_train_recall@5": [],
            "novel_unseen_train_recall@10": [],
            "novel_unseen_train_hit_rate": [],
            "persistent_seen_train_recall@5": [],
            "persistent_seen_train_recall@10": [],
            "persistent_seen_train_hit_rate": [],
            "persistent_unseen_train_recall@5": [],
            "persistent_unseen_train_recall@10": [],
            "persistent_unseen_train_hit_rate": [],
        }
        for m in eval_modalities
    }
    if train_seen_by_modality is None:
        train_seen_by_modality = {m: set() for m in eval_modalities}
    modality_node_ids = {
        m: [idx for idx, mm in vocab.node_type.items() if mm == m]
        for m in eval_modalities
    }
    for traj in trajectories:
        visits = traj.visits
        if len(visits) < 2:
            continue
        for t in range(len(visits) - 1):
            current = visits[t]
            nxt = visits[t + 1]
            sample_metrics = {}
            pred_map = {}
            for m in eval_modalities:
                preds = list(dict.fromkeys([int(x) for x in current.get(m, [])]))
                truth = [int(x) for x in nxt.get(m, [])]
                pred_map[m] = preds
                idxs = modality_node_ids.get(m, [])
                curr_set = set(int(x) for x in current.get(m, []))
                truth_set = set(int(x) for x in truth)
                pers_truth = sorted(list(truth_set & curr_set))
                novel_truth = sorted(list(truth_set - curr_set))
                seen_set = train_seen_by_modality.get(m, set())
                seen_truth = sorted(list(truth_set & seen_set))
                unseen_truth = sorted(list(truth_set - seen_set))
                novel_seen_truth = sorted(list(set(novel_truth) & seen_set))
                novel_unseen_truth = sorted(list(set(novel_truth) - seen_set))
                pers_seen_truth = sorted(list(set(pers_truth) & seen_set))
                pers_unseen_truth = sorted(list(set(pers_truth) - seen_set))
                for k in top_ks:
                    sample_metrics[f"{m}/recall@{k}"] = recall_at_k(preds, truth, k)
                    sample_metrics[f"{m}/precision@{k}"] = precision_at_k(preds, truth, k)
                    sample_metrics[f"{m}/ndcg@{k}"] = ndcg_at_k(preds, truth, k)
                sample_metrics[f"{m}/mrr"] = mrr(preds, truth)
                if idxs:
                    score_vals = [1.0 if idx in curr_set else 0.0 for idx in idxs]
                    truth_flags = [1 if idx in truth_set else 0 for idx in idxs]
                    sample_metrics[f"{m}/micro_f1"] = micro_f1(score_vals, truth_flags, 0.5)
                if m == "dx" and preds and truth:
                    primary = truth[0]
                    for k in top_ks:
                        sample_metrics[f"dx_primary/top@{k}"] = float(primary in preds[:k])
                    sample_metrics["dx_primary/mrr"] = mrr(preds, [primary])
                if pers_truth:
                    split_stats[m]["persistent_recall@5"].append(float(recall_at_k(preds, pers_truth, 5)))
                    split_stats[m]["persistent_recall@10"].append(float(recall_at_k(preds, pers_truth, 10)))
                    split_stats[m]["persistent_hit_rate"].append(float(len(set(preds[:10]) & set(pers_truth)) > 0))
                if novel_truth:
                    split_stats[m]["novel_recall@5"].append(float(recall_at_k(preds, novel_truth, 5)))
                    split_stats[m]["novel_recall@10"].append(float(recall_at_k(preds, novel_truth, 10)))
                    split_stats[m]["novel_hit_rate"].append(float(len(set(preds[:10]) & set(novel_truth)) > 0))
                if seen_truth:
                    split_stats[m]["seen_train_recall@5"].append(float(recall_at_k(preds, seen_truth, 5)))
                    split_stats[m]["seen_train_recall@10"].append(float(recall_at_k(preds, seen_truth, 10)))
                    split_stats[m]["seen_train_hit_rate"].append(float(len(set(preds[:10]) & set(seen_truth)) > 0))
                if unseen_truth:
                    split_stats[m]["unseen_train_recall@5"].append(float(recall_at_k(preds, unseen_truth, 5)))
                    split_stats[m]["unseen_train_recall@10"].append(float(recall_at_k(preds, unseen_truth, 10)))
                    split_stats[m]["unseen_train_hit_rate"].append(float(len(set(preds[:10]) & set(unseen_truth)) > 0))
                if novel_seen_truth:
                    split_stats[m]["novel_seen_train_recall@5"].append(float(recall_at_k(preds, novel_seen_truth, 5)))
                    split_stats[m]["novel_seen_train_recall@10"].append(float(recall_at_k(preds, novel_seen_truth, 10)))
                    split_stats[m]["novel_seen_train_hit_rate"].append(float(len(set(preds[:10]) & set(novel_seen_truth)) > 0))
                if novel_unseen_truth:
                    split_stats[m]["novel_unseen_train_recall@5"].append(float(recall_at_k(preds, novel_unseen_truth, 5)))
                    split_stats[m]["novel_unseen_train_recall@10"].append(float(recall_at_k(preds, novel_unseen_truth, 10)))
                    split_stats[m]["novel_unseen_train_hit_rate"].append(float(len(set(preds[:10]) & set(novel_unseen_truth)) > 0))
                if pers_seen_truth:
                    split_stats[m]["persistent_seen_train_recall@5"].append(float(recall_at_k(preds, pers_seen_truth, 5)))
                    split_stats[m]["persistent_seen_train_recall@10"].append(float(recall_at_k(preds, pers_seen_truth, 10)))
                    split_stats[m]["persistent_seen_train_hit_rate"].append(float(len(set(preds[:10]) & set(pers_seen_truth)) > 0))
                if pers_unseen_truth:
                    split_stats[m]["persistent_unseen_train_recall@5"].append(float(recall_at_k(preds, pers_unseen_truth, 5)))
                    split_stats[m]["persistent_unseen_train_recall@10"].append(float(recall_at_k(preds, pers_unseen_truth, 10)))
                    split_stats[m]["persistent_unseen_train_hit_rate"].append(float(len(set(preds[:10]) & set(pers_unseen_truth)) > 0))
            metrics.append(sample_metrics)
            if sample_limit > 0 and len(samples) < sample_limit:
                samples.append(
                    {
                        "subject_id": traj.subject_id,
                        "visit_index": int(t),
                        "per_visit_metrics": sample_metrics,
                        "predictions": {m: [node_display(vocab, c) for c in pred_map.get(m, [])] for m in pred_map.keys()},
                        "truth_next": {m: [node_display(vocab, c) for c in nxt.get(m, [])] for m in nxt.keys()},
                    }
                )
    if sample_path and sample_limit > 0:
        with open(sample_path, "w") as f:
            json.dump(samples, f, indent=2)
    out = aggregate_metrics(metrics)
    for m in eval_modalities:
        for key, vals in split_stats[m].items():
            if vals:
                out[f"{m}/{key}"] = float(np.mean(vals))
                out[f"{m}/{key}_n"] = float(len(vals))
    # Macro summaries over modalities.
    for metric_name in (
        "persistent_recall@5",
        "persistent_recall@10",
        "novel_recall@5",
        "novel_recall@10",
        "persistent_hit_rate",
        "novel_hit_rate",
    ):
        vals = []
        for m in eval_modalities:
            k = f"{m}/{metric_name}"
            if k in out and np.isfinite(out[k]):
                vals.append(float(out[k]))
        if vals:
            out[f"overall/{metric_name}_macro"] = float(np.mean(vals))
    _macro_additional_split_metrics(out, eval_modalities)
    return out


class FineTunedLLMScorer:
    def __init__(self, checkpoint_path: str, device: str = "cpu", max_length: int = 2048):
        from transformers import AutoTokenizer
        from run_llm_finetune import LLMClassifier

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})
        label_spaces = ckpt["label_spaces"]
        llm_model = args.get("llm_model")
        if not llm_model:
            raise ValueError("Finetuned LLM checkpoint missing args.llm_model")
        finetune_backbone = bool(args.get("finetune_backbone", False))
        head_hidden_dim = int(args.get("head_hidden_dim", 512))
        self.model = LLMClassifier(
            model_name=llm_model,
            label_spaces=label_spaces,
            head_hidden_dim=head_hidden_dim,
            finetune_backbone=finetune_backbone,
            torch_dtype=torch.float16 if str(device).startswith("cuda") else None,
        )
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_spaces = {m: [int(x) for x in label_spaces.get(m, [])] for m in ("dx", "proc", "med")}
        self.label_index = {m: {nid: i for i, nid in enumerate(self.label_spaces[m])} for m in ("dx", "proc", "med")}
        self.max_length = int(max_length)

    def _history_text(self, visits, t: int, vocab, max_history_visits: int) -> str:
        start = max(0, int(t) - int(max_history_visits) + 1)
        lines = ["Patient history visits:"]
        for i in range(start, t + 1):
            vv = visits[i]
            dx = [str(vocab.id_to_node.get(int(c), c)) for c in vv.get("dx", [])]
            proc = [str(vocab.id_to_node.get(int(c), c)) for c in vv.get("proc", [])]
            med = [str(vocab.id_to_node.get(int(c), c)) for c in vv.get("med", [])]
            lines.append(f"Visit {i - start + 1}")
            lines.append(f"dx: [{', '.join(dx)}]")
            lines.append(f"proc: [{', '.join(proc)}]")
            lines.append(f"med: [{', '.join(med)}]")
        lines.append("Predict next-visit events.")
        return "\n".join(lines)

    def score_candidates(
        self,
        visits,
        t: int,
        vocab,
        modality: str,
        candidate_ids: List[int],
        max_history_visits: int,
    ) -> List[float]:
        if modality not in self.label_spaces:
            return [0.0 for _ in candidate_ids]
        txt = self._history_text(visits, t, vocab, max_history_visits)
        tok = self.tokenizer(
            [txt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)[modality][0]
        scores = []
        li = self.label_index[modality]
        for nid in candidate_ids:
            j = li.get(int(nid))
            if j is None:
                scores.append(-1e9)
            else:
                scores.append(float(logits[j].item()))
        return scores


def build_prompt(
    history_codes: List[List[str]],
    candidates: Dict[str, List[str]],
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


def build_prompt_template() -> str:
    return "\n".join(
        [
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
            "Visit {t}: {code_1}, {code_2}, ...",
            "...",
            "",
            "Candidate ({target_modality}):",
            "{candidate_text}",
            "",
            "Candidate context:",
            "- Proposed by upstream structured retrieval ({candidate_source})",
            "- Assumed to be clinically plausible; refine its likelihood",
            "",
            "Score:",
        ]
    )


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


def _sample_split_rows(df, subject_col: str, time_col: str, ids: List[str], n_patients: int = 8, max_visits: int = 4):
    if df.empty or not ids:
        return df.head(0).copy()
    pick = ids[: min(n_patients, len(ids))]
    out = df[df[subject_col].astype(str).isin(set(map(str, pick)))].copy()
    out[time_col] = np.where(out[time_col].isna(), "", out[time_col].astype(str))
    out = out.sort_values([subject_col, time_col])
    out = out.groupby(subject_col, as_index=False).head(max_visits)
    return out


def _node_modality_ids(vocab, modality_cols: Dict[str, str]) -> Tuple[torch.Tensor, Dict[str, int]]:
    modalities = list(modality_cols.keys())
    for m in sorted(set(vocab.node_type.values())):
        if m not in modalities:
            modalities.append(m)
    mod_to_idx = {m: i for i, m in enumerate(modalities)}
    arr = torch.zeros(len(vocab.id_to_node), dtype=torch.long)
    for nid, m in vocab.node_type.items():
        arr[int(nid)] = int(mod_to_idx.get(m, 0))
    return arr, mod_to_idx


def _compute_visit_idf(train_traj, num_nodes: int) -> torch.Tensor:
    n_visits = 0
    df = np.zeros(num_nodes, dtype=np.int64)
    for traj in train_traj:
        for visit in traj.visits:
            n_visits += 1
            uniq = set()
            for vals in visit.values():
                for c in vals:
                    if 0 <= int(c) < num_nodes:
                        uniq.add(int(c))
            for c in uniq:
                df[c] += 1
    idf = np.log((n_visits + 1.0) / (df + 1.0)) + 1.0
    return torch.tensor(idf, dtype=torch.float32)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


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


def summarize_trajectories(trajectories, modality_cols: Dict[str, str]) -> Dict[str, object]:
    visits_per_patient = [len(traj.visits) for traj in trajectories]
    modality_counts = {m: [] for m in modality_cols.keys()}
    all_event_counts = []
    for traj in trajectories:
        for visit in traj.visits:
            total = 0
            for m in modality_counts.keys():
                c = len(visit.get(m, []))
                modality_counts[m].append(c)
                total += c
            all_event_counts.append(total)
    return {
        "patients": int(len(trajectories)),
        "visits": int(sum(visits_per_patient)),
        "visits_per_patient": _summary_stats(visits_per_patient),
        "events_per_visit": {m: _summary_stats(vals) for m, vals in modality_counts.items()},
        "all_events_per_visit": _summary_stats(all_event_counts),
    }


def summarize_vocab(vocab, modality_cols: Dict[str, str]) -> Dict[str, object]:
    modality_keys = list(modality_cols.keys())
    for m in sorted(set(vocab.node_type.values())):
        if m not in modality_keys:
            modality_keys.append(m)
    by_modality = {}
    for m in modality_keys:
        by_modality[m] = int(sum(1 for _, t in vocab.node_type.items() if t == m))
    return {
        "unique_nodes": int(len(vocab.node_to_id)),
        "nodes_by_modality": by_modality,
    }


def summarize_graph(graph, edge_type_ids: Dict[str, int]) -> Dict[str, object]:
    if graph is None:
        return {
            "unique_nodes": 0,
            "unique_edges": 0,
            "unique_edge_types": 0,
            "edges_by_type": {},
        }
    edges_by_type: Dict[str, int] = {}
    for _, _, d in graph.edges(data=True):
        et = str(d.get("edge_type", "unknown"))
        edges_by_type[et] = int(edges_by_type.get(et, 0) + 1)
    return {
        "unique_nodes": int(graph.number_of_nodes()),
        "unique_edges": int(graph.number_of_edges()),
        "unique_edge_types": int(len(edge_type_ids)),
        "edges_by_type": edges_by_type,
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
    # Directed lag-1 event transitions: all events in visit t -> all events in visit t+1.
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


def _augment_metrics_with_summaries(
    metrics: Dict[str, float],
    eval_modalities: Tuple[str, ...],
    stage_times: Dict[str, float],
    train_stats: Dict[str, float],
    n_eval_steps: int,
) -> Dict[str, float]:
    out = dict(metrics)

    def _macro(metric_suffix: str, out_key: str) -> None:
        vals = []
        for m in eval_modalities:
            k = f"{m}/{metric_suffix}"
            if k in metrics and np.isfinite(metrics[k]):
                vals.append(float(metrics[k]))
        if vals:
            out[out_key] = float(np.mean(vals))

    # Required ranking metrics (macro over modalities).
    _macro("recall@5", "overall/recall@5_macro")
    _macro("recall@10", "overall/recall@10_macro")
    _macro("ndcg@10", "overall/ndcg@10_macro")
    _macro("mrr", "overall/mrr_macro")

    # Extra useful ranking summaries.
    _macro("precision@5", "overall/precision@5_macro")
    _macro("precision@10", "overall/precision@10_macro")
    _macro("recall@20", "overall/recall@20_macro")
    _macro("ndcg@5", "overall/ndcg@5_macro")
    _macro("ndcg@20", "overall/ndcg@20_macro")
    _macro("hier_dist", "overall/hier_dist_macro")
    _macro("ancestor_match@2", "overall/ancestor_match@2_macro")
    _macro("grounded_rate", "overall/grounded_rate_macro")
    _macro("unsupported_rate", "overall/unsupported_rate_macro")
    _macro("micro_f1", "overall/micro_f1_macro")
    _macro("dx_primary/mrr", "overall/dx_primary_mrr")
    _macro("dx_primary/top@5", "overall/dx_primary_top@5")
    _macro("dx_primary/top@10", "overall/dx_primary_top@10")
    _macro("dx_primary/top@20", "overall/dx_primary_top@20")

    # Efficiency metrics.
    eval_sec = float(stage_times.get("evaluate_sec", 0.0))
    if eval_sec > 0 and n_eval_steps > 0:
        out["eff/eval_steps_per_sec"] = float(n_eval_steps / eval_sec)
        out["eff/eval_ms_per_step"] = float(1000.0 * eval_sec / n_eval_steps)
    if "tune_thresholds_sec" in stage_times and n_eval_steps > 0:
        th_sec = float(stage_times["tune_thresholds_sec"])
        if th_sec > 0:
            out["eff/threshold_steps_per_sec"] = float(n_eval_steps / th_sec)
            out["eff/threshold_ms_per_step"] = float(1000.0 * th_sec / n_eval_steps)
    if "train_model_sec" in stage_times:
        out["eff/train_wall_sec"] = float(stage_times["train_model_sec"])
    if "build_graph_sec" in stage_times:
        out["eff/build_graph_sec"] = float(stage_times["build_graph_sec"])
    if "wall_time_sec" in train_stats:
        out["eff/train_wall_sec_reported"] = float(train_stats["wall_time_sec"])
    out["eff/n_eval_steps"] = float(n_eval_steps)
    out["eff/total_stage_time_sec"] = float(sum(float(v) for v in stage_times.values()))

    return out


def _compute_train_code_freq(train_traj) -> Dict[int, int]:
    freq: Dict[int, int] = {}
    for traj in train_traj:
        for visit in traj.visits:
            for codes in visit.values():
                for c in codes:
                    i = int(c)
                    freq[i] = int(freq.get(i, 0) + 1)
    return freq


def _safe_corr(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if float(np.std(xa)) < 1e-12 or float(np.std(ya)) < 1e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _observed_nodes_by_modality(trajectories, eval_modalities: Tuple[str, ...]) -> Dict[str, set]:
    out: Dict[str, set] = {m: set() for m in eval_modalities}
    for traj in trajectories:
        for visit in traj.visits:
            for m in eval_modalities:
                for idx in visit.get(m, []):
                    out[m].add(int(idx))
    return out


def _macro_additional_split_metrics(out: Dict[str, float], eval_modalities: List[str]) -> None:
    metric_names = (
        "seen_train_recall@5",
        "seen_train_recall@10",
        "seen_train_hit_rate",
        "unseen_train_recall@5",
        "unseen_train_recall@10",
        "unseen_train_hit_rate",
        "novel_seen_train_recall@5",
        "novel_seen_train_recall@10",
        "novel_seen_train_hit_rate",
        "novel_unseen_train_recall@5",
        "novel_unseen_train_recall@10",
        "novel_unseen_train_hit_rate",
        "persistent_seen_train_recall@5",
        "persistent_seen_train_recall@10",
        "persistent_seen_train_hit_rate",
        "persistent_unseen_train_recall@5",
        "persistent_unseen_train_recall@10",
        "persistent_unseen_train_hit_rate",
    )
    for metric_name in metric_names:
        vals = []
        for m in eval_modalities:
            k = f"{m}/{metric_name}"
            if k in out and np.isfinite(out[k]):
                vals.append(float(out[k]))
        if vals:
            out[f"overall/{metric_name}_macro"] = float(np.mean(vals))


def _has_cross_edges(graph) -> bool:
    if graph is None:
        return False
    for _, _, d in graph.edges(data=True):
        if str(d.get("type", "")).startswith("cross:"):
            return True
    return False


def _augment_horizon_no_stitch(
    horizon: Dict[str, List[Tuple[int, float]]],
    graph,
    vocab,
    model: GrailModel,
    barycenter: torch.Tensor,
    all_codes: List[int],
    max_desc_depth: int,
    max_candidates: int,
    max_local_candidates: int,
    cone_aperture: float,
    cone_bonus: float,
    retrieval_pool_mode: str,
    cone_mode: str,
    cone_empty_fallback: bool,
    allowed_nodes_by_modality: Dict[str, set] = None,
    filter_stats: Dict[str, float] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    # When no stitched cross-modal edges exist, single-central retrieval can collapse to one modality.
    # Fallback: union horizons from multiple current-visit codes.
    if _has_cross_edges(graph):
        return horizon
    out = {m: list(v) for m, v in horizon.items()}
    needs = [m for m, lst in out.items() if len(lst) == 0]
    if not needs:
        return out
    merged = {m: {} for m in out.keys()}
    # Seed with existing candidates.
    for m, lst in out.items():
        for idx, sc in lst:
            merged[m][int(idx)] = float(sc)
    seed_codes = list(dict.fromkeys(int(c) for c in all_codes))[:32]
    for c in seed_codes:
        hz = build_risk_horizon(
            graph,
            vocab,
            model,
            barycenter,
            int(c),
            max_desc_depth=max_desc_depth,
            max_candidates=max_candidates,
            max_local_candidates=max_local_candidates,
            cone_aperture=cone_aperture,
            cone_bonus=cone_bonus,
            retrieval_pool_mode=retrieval_pool_mode,
            cone_mode=cone_mode,
            cone_empty_fallback=cone_empty_fallback,
            allowed_nodes_by_modality=allowed_nodes_by_modality,
            filter_stats=filter_stats,
        )
        for m, lst in hz.items():
            mm = merged[m]
            for idx, sc in lst:
                idx = int(idx)
                sc = float(sc)
                if idx not in mm or sc > mm[idx]:
                    mm[idx] = sc
    for m in needs:
        cand = sorted(merged[m].items(), key=lambda x: x[1], reverse=True)[:max_candidates]
        out[m] = [(int(i), float(s)) for i, s in cand]
    return out


def _select_central_code(
    model: GrailModel,
    barycenter: torch.Tensor,
    visits_t: Dict[str, List[int]],
    vocab,
    ce_mode: str = "nearest",
    ce_candidate_pool: str = "global",
    ce_softmax_beta: float = 5.0,
) -> int:
    dists = model.dist(barycenter, model.embeddings).squeeze(0)
    if ce_mode != "softmap":
        return int(torch.argmin(dists).item())

    if ce_candidate_pool == "restricted":
        mods = [m for m, vals in visits_t.items() if vals]
        cand = []
        for idx, m in vocab.node_type.items():
            if m in mods:
                cand.append(int(idx))
        for rid in vocab.modality_roots.values():
            cand.append(int(rid))
        cand = sorted(set(cand))
        if cand:
            cand_t = torch.tensor(cand, device=dists.device, dtype=torch.long)
            cd = dists[cand_t]
            logits = -float(ce_softmax_beta) * cd
            probs = torch.softmax(logits, dim=0)
            return int(cand_t[int(torch.argmax(probs).item())].item())
    logits = -float(ce_softmax_beta) * dists
    probs = torch.softmax(logits, dim=0)
    return int(torch.argmax(probs).item())


def save_split_stats(
    run_output_dir: str,
    modality_cols: Dict[str, str],
    vocab,
    train_traj,
    val_traj,
    test_traj,
) -> str:
    payload = {
        "train": summarize_trajectories(train_traj, modality_cols),
        "val": summarize_trajectories(val_traj, modality_cols),
        "test": summarize_trajectories(test_traj, modality_cols),
        "all": summarize_trajectories(train_traj + val_traj + test_traj, modality_cols),
        "vocab": summarize_vocab(vocab, modality_cols),
        "split_coverage": _split_coverage_stats(train_traj, val_traj, test_traj),
    }
    path = os.path.join(run_output_dir, "dataset_split_stats.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _default_split_file(
    data_path: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    min_visits: int,
) -> str:
    base = os.path.dirname(os.path.abspath(data_path))
    stem = os.path.splitext(os.path.basename(data_path))[0]
    split_dir = os.path.join(base, "splits")
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(
        split_dir,
        f"{stem}_seed{seed}_tr{train_frac:.2f}_val{val_frac:.2f}_mv{int(min_visits)}.json",
    )


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
    split_dir = os.path.dirname(os.path.abspath(path))
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
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


def save_geometry_artifacts(
    path: str,
    model: GrailModel,
    graph,
    vocab,
    edge_type_ids: Dict[str, int],
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "graph": graph,
        "vocab": vocab,
        "edge_type_ids": edge_type_ids,
        "model_cfg": {
            "embedding_dim": model_cfg.embedding_dim,
            "curvature": model_cfg.curvature,
            "manifold": model_cfg.manifold,
            "margin": model_cfg.margin,
            "softmax_beta": model_cfg.softmax_beta,
            "barycenter_heads": model_cfg.barycenter_heads,
            "barycenter_mixture": model_cfg.barycenter_mixture,
            "use_central_attention": model_cfg.use_central_attention,
            "use_idf_weighting": model_cfg.use_idf_weighting,
            "use_modality_weighting": model_cfg.use_modality_weighting,
            "central_pos_decay": model_cfg.central_pos_decay,
            "multi_cone_retrieval": model_cfg.multi_cone_retrieval,
        },
        "data_cfg": {
            "text_mode": data_cfg.text_mode,
            "label_map_path": data_cfg.label_map_path,
            "subject_id_col": data_cfg.subject_id_col,
            "visit_id_col": data_cfg.visit_id_col,
            "time_col": data_cfg.time_col,
            "modality_cols": data_cfg.modality_cols,
        },
    }
    torch.save(payload, path)


def load_geometry_artifacts(path: str) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _unique_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        candidate = f"{path}_v{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _non_overwrite_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{root}_{ts}{ext}"
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        candidate = f"{root}_{ts}_v{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def evaluate(
    model: GrailModel,
    graph,
    trajectories,
    vocab,
    eval_cfg: EvalConfig,
    rerank_cfg: RerankConfig,
    model_cfg: ModelConfig,
    max_local_candidates: int = 20000,
    ce_mode: str = "nearest",
    ce_candidate_pool: str = "global",
    ce_softmax_beta: float = 5.0,
    retrieval_pool_mode: str = "localmix",
    cone_mode: str = "filter",
    cone_empty_fallback: bool = True,
    thresholds: Dict[str, float] = None,
    sample_limit: int = 50,
    sample_path: str = None,
    train_code_freq: Dict[int, int] = None,
    train_seen_by_modality: Optional[Dict[str, set]] = None,
    prompt_example_path: str = None,
) -> Dict[str, float]:
    model.eval()
    reranker = LLMReranker(
        backend=rerank_cfg.backend,
        model_name=rerank_cfg.model_name,
        scoring_mode=getattr(rerank_cfg, "scoring_mode", "pointwise"),
    )
    ft_scorer = None
    if rerank_cfg.backend == "ftcls":
        ft_ckpt = getattr(rerank_cfg, "ft_llm_checkpoint", None)
        if not ft_ckpt:
            raise ValueError("llm-backend=ftcls requires --ft-llm-checkpoint")
        ft_scorer = FineTunedLLMScorer(
            checkpoint_path=ft_ckpt,
            device=model.embeddings.device.type if hasattr(model.embeddings.device, "type") else str(model.embeddings.device),
            max_length=int(getattr(rerank_cfg, "ft_max_length", 2048)),
        )
    use_llm = rerank_cfg.backend != "none"
    node_disp_cache: Dict[int, str] = {}
    undir_graph = graph.to_undirected() if graph is not None else None

    def _disp(idx: int) -> str:
        i = int(idx)
        if i not in node_disp_cache:
            node_disp_cache[i] = node_display(vocab, i)
        return node_disp_cache[i]

    metrics = []
    samples = []
    llm_calls = 0
    llm_candidates_scored = 0
    llm_score_sec = 0.0
    prompt_token_lens: List[int] = []
    prompt_char_lens: List[int] = []
    prompt_word_lens: List[int] = []
    prompt_example_written = False
    freq_perf = {
        m: {
            "truth_freq_mean": [],
            "recall10": [],
            "mrr": [],
        }
        for m in eval_cfg.eval_modalities
    }
    split_stats = {
        m: {
            "persistent_recall@5": [],
            "persistent_recall@10": [],
            "novel_recall@5": [],
            "novel_recall@10": [],
            "persistent_hit_rate": [],
            "novel_hit_rate": [],
            "seen_train_recall@5": [],
            "seen_train_recall@10": [],
            "seen_train_hit_rate": [],
            "unseen_train_recall@5": [],
            "unseen_train_recall@10": [],
            "unseen_train_hit_rate": [],
            "novel_seen_train_recall@5": [],
            "novel_seen_train_recall@10": [],
            "novel_seen_train_hit_rate": [],
            "novel_unseen_train_recall@5": [],
            "novel_unseen_train_recall@10": [],
            "novel_unseen_train_hit_rate": [],
            "persistent_seen_train_recall@5": [],
            "persistent_seen_train_recall@10": [],
            "persistent_seen_train_hit_rate": [],
            "persistent_unseen_train_recall@5": [],
            "persistent_unseen_train_recall@10": [],
            "persistent_unseen_train_hit_rate": [],
        }
        for m in eval_cfg.eval_modalities
    }
    total_steps = sum(max(0, len(traj.visits) - 1) for traj in trajectories)
    allowed_nodes = _observed_nodes_by_modality(trajectories, tuple(eval_cfg.eval_modalities))
    if train_seen_by_modality is None:
        train_seen_by_modality = {m: set() for m in eval_cfg.eval_modalities}
    filter_stats: Dict[str, float] = {"total_candidates": 0.0, "dropped_candidates": 0.0}
    invalid_topk_postfilter = 0.0
    for m in eval_cfg.eval_modalities:
        _log(f"Prediction node filter [{m}]: {len(allowed_nodes.get(m, set()))} observed nodes")
    step = 0
    last_log = time.time()
    _log(f"Evaluation steps: {total_steps}")
    for traj in trajectories:
        visits = traj.visits
        if len(visits) < 2:
            continue
        for t in range(len(visits) - 1):
            step_t0 = time.time()
            step += 1
            if step % 50 == 0:
                now = time.time()
                rate = 50 / max(now - last_log, 1e-6)
                _log(f"Eval progress: {step}/{total_steps} ({rate:.2f} steps/s)")
                last_log = now
            current = visits[t]
            next_visit = visits[t + 1]
            all_codes = [code for codes in current.values() for code in codes]
            if not all_codes:
                continue
            code_tensor = torch.tensor([all_codes], device=model.embeddings.device)
            barycenter = model.barycenter(code_tensor)
            if model_cfg.multi_cone_retrieval and model.barycenter_heads > 1:
                heads = model.head_barycenters(code_tensor).squeeze(0)  # (H, D)
                central_codes = []
                for h in range(heads.size(0)):
                    d_h = model.dist(heads[h].unsqueeze(0), model.embeddings)
                    central_codes.append(int(torch.argmin(d_h).item()))
                if model.barycenter_mixture:
                    head_w = torch.softmax(model.bary_head_logits, dim=0)
                else:
                    head_w = torch.ones(model.barycenter_heads, device=heads.device) / model.barycenter_heads
                horizon = build_risk_horizon_multi_cone(
                    graph,
                    vocab,
                    model,
                    heads,
                    central_codes,
                    max_desc_depth=3,
                    max_candidates=rerank_cfg.max_candidates,
                    max_local_candidates=max_local_candidates,
                    cone_aperture=model_cfg.cone_aperture,
                    cone_bonus=model_cfg.cone_bonus,
                    head_weights=head_w,
                    retrieval_pool_mode=retrieval_pool_mode,
                    cone_mode=cone_mode,
                    cone_empty_fallback=cone_empty_fallback,
                    allowed_nodes_by_modality=allowed_nodes,
                    filter_stats=filter_stats,
                )
            else:
                central_code = _select_central_code(
                    model,
                    barycenter,
                    current,
                    vocab,
                    ce_mode=ce_mode,
                    ce_candidate_pool=ce_candidate_pool,
                    ce_softmax_beta=ce_softmax_beta,
                )
                horizon = build_risk_horizon(
                    graph,
                    vocab,
                    model,
                    barycenter.squeeze(0),
                    central_code,
                    max_desc_depth=3,
                    max_candidates=rerank_cfg.max_candidates,
                    max_local_candidates=max_local_candidates,
                    cone_aperture=model_cfg.cone_aperture,
                    cone_bonus=model_cfg.cone_bonus,
                    retrieval_pool_mode=retrieval_pool_mode,
                    cone_mode=cone_mode,
                    cone_empty_fallback=cone_empty_fallback,
                    allowed_nodes_by_modality=allowed_nodes,
                    filter_stats=filter_stats,
                )
            horizon = _augment_horizon_no_stitch(
                horizon,
                graph,
                vocab,
                model,
                barycenter.squeeze(0),
                all_codes,
                max_desc_depth=3,
                max_candidates=rerank_cfg.max_candidates,
                max_local_candidates=max_local_candidates,
                cone_aperture=model_cfg.cone_aperture,
                cone_bonus=model_cfg.cone_bonus,
                retrieval_pool_mode=retrieval_pool_mode,
                cone_mode=cone_mode,
                cone_empty_fallback=cone_empty_fallback,
                allowed_nodes_by_modality=allowed_nodes,
                filter_stats=filter_stats,
            )
            history_codes = []
            scored_candidates = {}
            step_prompt = ""
            if use_llm:
                for back in range(max(0, t - eval_cfg.max_history_visits + 1), t + 1):
                    hist_codes = [_disp(c) for m in visits[back].values() for c in m]
                    history_codes.append(hist_codes[:rerank_cfg.top_m_history])
            score_map = {}
            for modality, cand_list in horizon.items():
                if not cand_list:
                    continue
                combined = []
                if use_llm:
                    codes = [_disp(idx) for idx, _ in cand_list]
                    prompt = build_prompt(history_codes, {modality: codes}, target_modality=modality)
                    if not step_prompt:
                        step_prompt = prompt
                    prompt_char_lens.append(len(prompt))
                    prompt_word_lens.append(len(prompt.split()))
                    if rerank_cfg.backend == "hf":
                        try:
                            reranker._load_hf()
                            enc = reranker._tokenizer(prompt, return_tensors=None, add_special_tokens=True)
                            prompt_token_lens.append(int(len(enc.get("input_ids", []))))
                        except Exception:
                            pass
                    if prompt_example_path and not prompt_example_written:
                        with open(prompt_example_path, "w") as f:
                            f.write(prompt)
                        prompt_example_written = True
                    t_llm0 = time.perf_counter()
                    if rerank_cfg.backend == "ftcls":
                        cand_ids = [int(idx) for idx, _ in cand_list]
                        llm_scores = ft_scorer.score_candidates(
                            visits=visits,
                            t=t,
                            vocab=vocab,
                            modality=modality,
                            candidate_ids=cand_ids,
                            max_history_visits=eval_cfg.max_history_visits,
                        )
                    else:
                        llm_scores = reranker.score(prompt, codes)
                    llm_score_sec += float(time.perf_counter() - t_llm0)
                    llm_calls += 1
                    llm_candidates_scored += int(len(cand_list))
                    for (idx, geo_score), llm_score in zip(cand_list, llm_scores):
                        score = rerank_cfg.lambda_llm * llm_score + (1 - rerank_cfg.lambda_llm) * geo_score
                        if float(model_cfg.persistence_bonus) != 0.0 and int(idx) in set(current.get(modality, [])):
                            score += float(model_cfg.persistence_bonus)
                        combined.append((idx, score))
                else:
                    for idx, geo_score in cand_list:
                        score = float(geo_score)
                        if float(model_cfg.persistence_bonus) != 0.0 and int(idx) in set(current.get(modality, [])):
                            score += float(model_cfg.persistence_bonus)
                        combined.append((idx, score))
                combined.sort(key=lambda x: x[1], reverse=True)
                scored_candidates[modality] = [idx for idx, _ in combined]
                allow_set = allowed_nodes.get(modality, set())
                if allow_set:
                    for idx in scored_candidates[modality][: max(eval_cfg.top_ks)]:
                        if int(idx) not in allow_set:
                            invalid_topk_postfilter += 1.0
                score_map[modality] = combined

            sample_metrics = {}
            for modality in eval_cfg.eval_modalities:
                preds = scored_candidates.get(modality, [])
                truth = next_visit.get(modality, [])
                curr_set = set(int(x) for x in current.get(modality, []))
                truth_set = set(int(x) for x in truth)
                pers_truth = sorted(list(truth_set & curr_set))
                novel_truth = sorted(list(truth_set - curr_set))
                seen_set = train_seen_by_modality.get(modality, set())
                seen_truth = sorted(list(truth_set & seen_set))
                unseen_truth = sorted(list(truth_set - seen_set))
                novel_seen_truth = sorted(list(set(novel_truth) & seen_set))
                novel_unseen_truth = sorted(list(set(novel_truth) - seen_set))
                pers_seen_truth = sorted(list(set(pers_truth) & seen_set))
                pers_unseen_truth = sorted(list(set(pers_truth) - seen_set))
                for k in eval_cfg.top_ks:
                    sample_metrics[f"{modality}/recall@{k}"] = recall_at_k(preds, truth, k)
                    sample_metrics[f"{modality}/precision@{k}"] = precision_at_k(preds, truth, k)
                    sample_metrics[f"{modality}/ndcg@{k}"] = ndcg_at_k(preds, truth, k)
                sample_metrics[f"{modality}/mrr"] = mrr(preds, truth)
                if train_code_freq is not None:
                    tf = [float(train_code_freq.get(int(tid), 0)) for tid in truth]
                    mean_tf = float(np.mean(tf)) if tf else 0.0
                    freq_perf[modality]["truth_freq_mean"].append(mean_tf)
                    freq_perf[modality]["recall10"].append(float(recall_at_k(preds, truth, 10)))
                    freq_perf[modality]["mrr"].append(float(sample_metrics[f"{modality}/mrr"]))
                if thresholds is not None:
                    combined = score_map.get(modality, [])
                    scores = [s for _, s in combined]
                    truth_flags = [1 if idx in truth else 0 for idx, _ in combined]
                    if scores:
                        th = thresholds.get(modality, 0.5)
                        sample_metrics[f"{modality}/micro_f1"] = micro_f1(scores, truth_flags, th)
                if preds:
                    history_code_set = {code for codes in current.values() for code in codes}
                    horizon_codes = {idx for idx, _ in score_map.get(modality, [])}
                    grounded = sum(1 for p in preds if p in history_code_set or p in horizon_codes)
                    unsupported = sum(1 for p in preds if p not in history_code_set and p not in horizon_codes)
                    sample_metrics[f"{modality}/grounded_rate"] = grounded / max(len(preds), 1)
                    sample_metrics[f"{modality}/unsupported_rate"] = unsupported / max(len(preds), 1)
                if preds and truth:
                    pred_top = preds[0]
                    if undir_graph is not None:
                        distances = []
                        for truth_code in truth:
                            try:
                                distances.append(nx.shortest_path_length(undir_graph, pred_top, truth_code))
                            except nx.NetworkXNoPath:
                                distances.append(999)
                    else:
                        distances = [hierarchy_distance(graph, pred_top, tt) for tt in truth]
                    sample_metrics[f"{modality}/hier_dist"] = float(np.mean(distances))
                    sample_metrics[f"{modality}/ancestor_match@2"] = float(
                        np.mean([ancestor_match(graph, pred_top, t, max_depth=2) for t in truth])
                    )
                if modality == "dx" and preds and truth:
                    primary = truth[0]
                    for k in eval_cfg.top_ks:
                        sample_metrics[f"dx_primary/top@{k}"] = float(primary in preds[:k])
                    sample_metrics["dx_primary/mrr"] = mrr(preds, [primary])
                if pers_truth:
                    split_stats[modality]["persistent_recall@5"].append(float(recall_at_k(preds, pers_truth, 5)))
                    split_stats[modality]["persistent_recall@10"].append(float(recall_at_k(preds, pers_truth, 10)))
                    split_stats[modality]["persistent_hit_rate"].append(float(len(set(preds[:10]) & set(pers_truth)) > 0))
                if novel_truth:
                    split_stats[modality]["novel_recall@5"].append(float(recall_at_k(preds, novel_truth, 5)))
                    split_stats[modality]["novel_recall@10"].append(float(recall_at_k(preds, novel_truth, 10)))
                    split_stats[modality]["novel_hit_rate"].append(float(len(set(preds[:10]) & set(novel_truth)) > 0))
                if seen_truth:
                    split_stats[modality]["seen_train_recall@5"].append(float(recall_at_k(preds, seen_truth, 5)))
                    split_stats[modality]["seen_train_recall@10"].append(float(recall_at_k(preds, seen_truth, 10)))
                    split_stats[modality]["seen_train_hit_rate"].append(float(len(set(preds[:10]) & set(seen_truth)) > 0))
                if unseen_truth:
                    split_stats[modality]["unseen_train_recall@5"].append(float(recall_at_k(preds, unseen_truth, 5)))
                    split_stats[modality]["unseen_train_recall@10"].append(float(recall_at_k(preds, unseen_truth, 10)))
                    split_stats[modality]["unseen_train_hit_rate"].append(float(len(set(preds[:10]) & set(unseen_truth)) > 0))
                if novel_seen_truth:
                    split_stats[modality]["novel_seen_train_recall@5"].append(float(recall_at_k(preds, novel_seen_truth, 5)))
                    split_stats[modality]["novel_seen_train_recall@10"].append(float(recall_at_k(preds, novel_seen_truth, 10)))
                    split_stats[modality]["novel_seen_train_hit_rate"].append(float(len(set(preds[:10]) & set(novel_seen_truth)) > 0))
                if novel_unseen_truth:
                    split_stats[modality]["novel_unseen_train_recall@5"].append(float(recall_at_k(preds, novel_unseen_truth, 5)))
                    split_stats[modality]["novel_unseen_train_recall@10"].append(float(recall_at_k(preds, novel_unseen_truth, 10)))
                    split_stats[modality]["novel_unseen_train_hit_rate"].append(float(len(set(preds[:10]) & set(novel_unseen_truth)) > 0))
                if pers_seen_truth:
                    split_stats[modality]["persistent_seen_train_recall@5"].append(float(recall_at_k(preds, pers_seen_truth, 5)))
                    split_stats[modality]["persistent_seen_train_recall@10"].append(float(recall_at_k(preds, pers_seen_truth, 10)))
                    split_stats[modality]["persistent_seen_train_hit_rate"].append(float(len(set(preds[:10]) & set(pers_seen_truth)) > 0))
                if pers_unseen_truth:
                    split_stats[modality]["persistent_unseen_train_recall@5"].append(float(recall_at_k(preds, pers_unseen_truth, 5)))
                    split_stats[modality]["persistent_unseen_train_recall@10"].append(float(recall_at_k(preds, pers_unseen_truth, 10)))
                    split_stats[modality]["persistent_unseen_train_hit_rate"].append(float(len(set(preds[:10]) & set(pers_unseen_truth)) > 0))
            metrics.append(sample_metrics)
            if len(samples) < sample_limit:
                sample_entry = {
                    "subject_id": traj.subject_id,
                    "visit_index": t,
                    "prompt": step_prompt,
                    "per_visit_metrics": sample_metrics,
                    "history": {m: [node_display(vocab, c) for c in current.get(m, [])] for m in current.keys()},
                    "truth_next": {m: [_disp(c) for c in next_visit.get(m, [])] for m in next_visit.keys()},
                    "predictions": {
                        m: [_disp(c) for c in scored_candidates.get(m, [])[: max(eval_cfg.top_ks)]]
                        for m in eval_cfg.eval_modalities
                    },
                    "candidates": {
                        m: [
                            {"code": _disp(idx), "score": float(score)}
                            for idx, score in score_map.get(m, [])[: max(eval_cfg.top_ks)]
                        ]
                        for m in eval_cfg.eval_modalities
                    },
                }
                samples.append(sample_entry)
            step_dt = time.time() - step_t0
            if step_dt > 10.0:
                _log(
                    f"Slow eval step detected: step={step}/{total_steps} "
                    f"subject_id={traj.subject_id} visit_idx={t} took {step_dt:.2f}s"
                )
    if sample_path:
        with open(sample_path, "w") as f:
            json.dump(samples, f, indent=2)
    out = aggregate_metrics(metrics)
    total_cands = float(filter_stats.get("total_candidates", 0.0))
    dropped_cands = float(filter_stats.get("dropped_candidates", 0.0))
    out["filter/total_candidates"] = total_cands
    out["filter/dropped_candidates"] = dropped_cands
    out["filter/drop_rate"] = float(dropped_cands / total_cands) if total_cands > 0 else 0.0
    out["filter/invalid_topk_postfilter"] = float(invalid_topk_postfilter)
    if use_llm:
        out["llm/score_calls"] = float(llm_calls)
        out["llm/candidates_scored"] = float(llm_candidates_scored)
        out["llm/score_time_sec"] = float(llm_score_sec)
        out["llm/candidates_per_call"] = float(llm_candidates_scored / max(llm_calls, 1))
        out["llm/score_ms_per_call"] = float(1000.0 * llm_score_sec / max(llm_calls, 1))
        out["llm/score_ms_per_candidate"] = float(1000.0 * llm_score_sec / max(llm_candidates_scored, 1))
        if prompt_char_lens:
            arr = np.asarray(prompt_char_lens, dtype=float)
            out["llm/prompt_chars_mean"] = float(arr.mean())
            out["llm/prompt_chars_median"] = float(np.median(arr))
            out["llm/prompt_chars_p90"] = float(np.quantile(arr, 0.90))
            out["llm/prompt_chars_max"] = float(arr.max())
        if prompt_word_lens:
            arr = np.asarray(prompt_word_lens, dtype=float)
            out["llm/prompt_words_mean"] = float(arr.mean())
            out["llm/prompt_words_median"] = float(np.median(arr))
            out["llm/prompt_words_p90"] = float(np.quantile(arr, 0.90))
            out["llm/prompt_words_max"] = float(arr.max())
        if prompt_token_lens:
            arr = np.asarray(prompt_token_lens, dtype=float)
            out["llm/prompt_tokens_mean"] = float(arr.mean())
            out["llm/prompt_tokens_median"] = float(np.median(arr))
            out["llm/prompt_tokens_p90"] = float(np.quantile(arr, 0.90))
            out["llm/prompt_tokens_max"] = float(arr.max())
    for m in eval_cfg.eval_modalities:
        x = freq_perf[m]["truth_freq_mean"]
        y_r = freq_perf[m]["recall10"]
        y_m = freq_perf[m]["mrr"]
        out[f"{m}/freq_corr_truthfreq_recall@10"] = _safe_corr(x, y_r)
        out[f"{m}/freq_corr_truthfreq_mrr"] = _safe_corr(x, y_m)
        for key, vals in split_stats[m].items():
            if vals:
                out[f"{m}/{key}"] = float(np.mean(vals))
                out[f"{m}/{key}_n"] = float(len(vals))
    for metric_name in (
        "persistent_recall@5",
        "persistent_recall@10",
        "novel_recall@5",
        "novel_recall@10",
        "persistent_hit_rate",
        "novel_hit_rate",
    ):
        vals = []
        for m in eval_cfg.eval_modalities:
            k = f"{m}/{metric_name}"
            if k in out and np.isfinite(out[k]):
                vals.append(float(out[k]))
        if vals:
            out[f"overall/{metric_name}_macro"] = float(np.mean(vals))
    _macro_additional_split_metrics(out, list(eval_cfg.eval_modalities))
    return out


def tune_thresholds_for_eval(
    model: GrailModel,
    graph,
    trajectories,
    vocab,
    eval_cfg: EvalConfig,
    rerank_cfg: RerankConfig,
    model_cfg: ModelConfig,
    max_local_candidates: int = 20000,
    ce_mode: str = "nearest",
    ce_candidate_pool: str = "global",
    ce_softmax_beta: float = 5.0,
    retrieval_pool_mode: str = "localmix",
    cone_mode: str = "filter",
    cone_empty_fallback: bool = True,
    max_steps: int = 0,
) -> Dict[str, float]:
    model.eval()
    reranker = LLMReranker(
        backend=rerank_cfg.backend,
        model_name=rerank_cfg.model_name,
        scoring_mode=getattr(rerank_cfg, "scoring_mode", "pointwise"),
    )
    ft_scorer = None
    if rerank_cfg.backend == "ftcls":
        ft_ckpt = getattr(rerank_cfg, "ft_llm_checkpoint", None)
        if not ft_ckpt:
            raise ValueError("llm-backend=ftcls requires --ft-llm-checkpoint")
        ft_scorer = FineTunedLLMScorer(
            checkpoint_path=ft_ckpt,
            device=model.embeddings.device.type if hasattr(model.embeddings.device, "type") else str(model.embeddings.device),
            max_length=int(getattr(rerank_cfg, "ft_max_length", 2048)),
        )
    use_llm = rerank_cfg.backend != "none"
    node_disp_cache: Dict[int, str] = {}

    def _disp(idx: int) -> str:
        i = int(idx)
        if i not in node_disp_cache:
            node_disp_cache[i] = node_display(vocab, i)
        return node_disp_cache[i]

    scores_by_modality = {m: [] for m in eval_cfg.eval_modalities}
    truth_by_modality = {m: [] for m in eval_cfg.eval_modalities}
    total_steps = sum(max(0, len(traj.visits) - 1) for traj in trajectories)
    allowed_nodes = _observed_nodes_by_modality(trajectories, tuple(eval_cfg.eval_modalities))
    filter_stats: Dict[str, float] = {"total_candidates": 0.0, "dropped_candidates": 0.0}
    for m in eval_cfg.eval_modalities:
        _log(f"Threshold node filter [{m}]: {len(allowed_nodes.get(m, set()))} observed nodes")
    step = 0
    kept_steps = 0
    last_log = time.time()
    _log(f"Threshold tuning steps: {total_steps}")
    stop_early = False
    for traj in trajectories:
        visits = traj.visits
        if len(visits) < 2:
            continue
        for t in range(len(visits) - 1):
            if max_steps > 0 and kept_steps >= max_steps:
                stop_early = True
                break
            step_t0 = time.time()
            step += 1
            if step % 50 == 0:
                now = time.time()
                rate = 50 / max(now - last_log, 1e-6)
                _log(f"Threshold tuning progress: {step}/{total_steps} ({rate:.2f} steps/s)")
                last_log = now
            current = visits[t]
            next_visit = visits[t + 1]
            all_codes = [code for codes in current.values() for code in codes]
            if not all_codes:
                continue
            code_tensor = torch.tensor([all_codes], device=model.embeddings.device)
            barycenter = model.barycenter(code_tensor)
            if model_cfg.multi_cone_retrieval and model.barycenter_heads > 1:
                heads = model.head_barycenters(code_tensor).squeeze(0)
                central_codes = []
                for h in range(heads.size(0)):
                    d_h = model.dist(heads[h].unsqueeze(0), model.embeddings)
                    central_codes.append(int(torch.argmin(d_h).item()))
                if model.barycenter_mixture:
                    head_w = torch.softmax(model.bary_head_logits, dim=0)
                else:
                    head_w = torch.ones(model.barycenter_heads, device=heads.device) / model.barycenter_heads
                horizon = build_risk_horizon_multi_cone(
                    graph,
                    vocab,
                    model,
                    heads,
                    central_codes,
                    max_desc_depth=3,
                    max_candidates=rerank_cfg.max_candidates,
                    max_local_candidates=max_local_candidates,
                    cone_aperture=model_cfg.cone_aperture,
                    cone_bonus=model_cfg.cone_bonus,
                    head_weights=head_w,
                    retrieval_pool_mode=retrieval_pool_mode,
                    cone_mode=cone_mode,
                    cone_empty_fallback=cone_empty_fallback,
                    allowed_nodes_by_modality=allowed_nodes,
                    filter_stats=filter_stats,
                )
            else:
                central_code = _select_central_code(
                    model,
                    barycenter,
                    current,
                    vocab,
                    ce_mode=ce_mode,
                    ce_candidate_pool=ce_candidate_pool,
                    ce_softmax_beta=ce_softmax_beta,
                )
                horizon = build_risk_horizon(
                    graph,
                    vocab,
                    model,
                    barycenter.squeeze(0),
                    central_code,
                    max_desc_depth=3,
                    max_candidates=rerank_cfg.max_candidates,
                    max_local_candidates=max_local_candidates,
                    cone_aperture=model_cfg.cone_aperture,
                    cone_bonus=model_cfg.cone_bonus,
                    retrieval_pool_mode=retrieval_pool_mode,
                    cone_mode=cone_mode,
                    cone_empty_fallback=cone_empty_fallback,
                    allowed_nodes_by_modality=allowed_nodes,
                    filter_stats=filter_stats,
                )
            horizon = _augment_horizon_no_stitch(
                horizon,
                graph,
                vocab,
                model,
                barycenter.squeeze(0),
                all_codes,
                max_desc_depth=3,
                max_candidates=rerank_cfg.max_candidates,
                max_local_candidates=max_local_candidates,
                cone_aperture=model_cfg.cone_aperture,
                cone_bonus=model_cfg.cone_bonus,
                retrieval_pool_mode=retrieval_pool_mode,
                cone_mode=cone_mode,
                cone_empty_fallback=cone_empty_fallback,
                allowed_nodes_by_modality=allowed_nodes,
                filter_stats=filter_stats,
            )
            history_codes = []
            if use_llm:
                for back in range(max(0, t - eval_cfg.max_history_visits + 1), t + 1):
                    hist_codes = [_disp(c) for m in visits[back].values() for c in m]
                    history_codes.append(hist_codes[:rerank_cfg.top_m_history])
            for modality, cand_list in horizon.items():
                if not cand_list:
                    continue
                combined = []
                if use_llm:
                    codes = [_disp(idx) for idx, _ in cand_list]
                    prompt = build_prompt(history_codes, {modality: codes}, target_modality=modality)
                    if rerank_cfg.backend == "ftcls":
                        cand_ids = [int(idx) for idx, _ in cand_list]
                        llm_scores = ft_scorer.score_candidates(
                            visits=visits,
                            t=t,
                            vocab=vocab,
                            modality=modality,
                            candidate_ids=cand_ids,
                            max_history_visits=eval_cfg.max_history_visits,
                        )
                    else:
                        llm_scores = reranker.score(prompt, codes)
                    for (idx, geo_score), llm_score in zip(cand_list, llm_scores):
                        score = rerank_cfg.lambda_llm * llm_score + (1 - rerank_cfg.lambda_llm) * geo_score
                        if float(model_cfg.persistence_bonus) != 0.0 and int(idx) in set(current.get(modality, [])):
                            score += float(model_cfg.persistence_bonus)
                        combined.append((idx, score))
                else:
                    for idx, geo_score in cand_list:
                        score = float(geo_score)
                        if float(model_cfg.persistence_bonus) != 0.0 and int(idx) in set(current.get(modality, [])):
                            score += float(model_cfg.persistence_bonus)
                        combined.append((idx, score))
                scores = [s for _, s in combined]
                truth = next_visit.get(modality, [])
                truth_flags = [1 if idx in truth else 0 for idx, _ in combined]
                scores_by_modality[modality].append(scores)
                truth_by_modality[modality].append(truth_flags)
            kept_steps += 1
            step_dt = time.time() - step_t0
            if step_dt > 10.0:
                _log(
                    f"Slow threshold step detected: step={step}/{total_steps} "
                    f"subject_id={traj.subject_id} visit_idx={t} took {step_dt:.2f}s"
                )
        if stop_early:
            break
    if max_steps > 0:
        _log(f"Threshold tuning used {kept_steps} steps (max_steps={max_steps}).")
    total_cands = float(filter_stats.get("total_candidates", 0.0))
    dropped_cands = float(filter_stats.get("dropped_candidates", 0.0))
    drop_rate = float(dropped_cands / total_cands) if total_cands > 0 else 0.0
    _log(
        f"Threshold filter audit: total_candidates={int(total_cands)} "
        f"dropped={int(dropped_cands)} drop_rate={drop_rate:.6f}"
    )
    thresholds = {}
    for modality in eval_cfg.eval_modalities:
        thresholds[modality] = tune_threshold(scores_by_modality[modality], truth_by_modality[modality])
    return thresholds


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--format", type=str, default="pickle")
    parser.add_argument("--lab-groups", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--manifold", type=str, default="poincare")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--halo-safe-mode",
        action="store_true",
        help="Enable HALO-specific vocabulary shrinking safeguards to avoid OOM on large vocabularies.",
    )
    parser.add_argument(
        "--halo-min-code-freq",
        type=int,
        default=100,
        help="When --halo-safe-mode is on and baseline=halo, enforce at least this min code frequency.",
    )
    parser.add_argument(
        "--halo-max-codes-per-modality",
        type=int,
        default=2500,
        help="When --halo-safe-mode is on and baseline=halo, cap direct codes per modality before ancestors.",
    )
    parser.add_argument("--text-mode", type=str, default="hybrid", choices=["hybrid", "code", "text"])
    parser.add_argument("--label-map", type=str, default=None)
    parser.add_argument(
        "--baseline",
        type=str,
        default="grail",
        choices=[
            "grail",
            "transformer",
            "retain",
            "gram",
            "behrt",
            "medbert",
            "halo",
            "copy_last",
            "grail_seq_transformer",
            "grail_seq_retain",
            "grail_seq_gram",
            "grail_seq_hybrid",
            "llm_only_hyper",
            "llm_only_euclid",
            "hyper_no_stitch",
            "euclid_grail",
            "euclid_rag",
        ],
    )
    parser.add_argument("--llm-backend", type=str, default="hf", choices=["hf", "openai", "heuristic", "ftcls", "none"])
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument(
        "--llm-scoring-mode",
        type=str,
        default="pointwise",
        choices=["pointwise", "batched_pairwise", "listwise"],
        help="LLM candidate scoring mode. pointwise=1 candidate/forward, batched_pairwise=batched candidate NLL, listwise=single generation over candidate list.",
    )
    parser.add_argument("--ft-llm-checkpoint", type=str, default=None)
    parser.add_argument("--ft-llm-max-length", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-visits", type=int, default=None, help="Minimum visits per patient retained after trajectory build.")
    parser.add_argument("--min-code-freq", type=int, default=None, help="Minimum code frequency used for vocab construction.")
    parser.add_argument(
        "--baseline-pretrained-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path for baseline backbones (behrt/medbert).",
    )
    parser.add_argument(
        "--baseline-freeze-backbone",
        action="store_true",
        help="Freeze pretrained baseline backbone and train only prediction head.",
    )
    parser.add_argument(
        "--baseline-skip-mlm-pretrain",
        action="store_true",
        help="Skip in-project MLM pretraining stage for behrt/medbert baselines.",
    )
    parser.add_argument(
        "--grail-seq-freeze-embedding",
        action="store_true",
        help="For grail_seq_* baselines, freeze sequence embedding layer after hyperbolic initialization.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-stitch", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-cone", action="store_true")
    parser.add_argument("--pipeline-stage", type=str, default="all", choices=["all", "geometry", "llm"])
    parser.add_argument("--geometry-artifact-path", type=str, default=None)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--enable-lab-tests", action="store_true")
    parser.add_argument(
        "--disable-lab-tests",
        action="store_true",
        help="Disable lab modality for this run. Labs are disabled by default unless --enable-lab-tests is set.",
    )
    parser.add_argument("--barycenter-heads", type=int, default=1)
    parser.add_argument("--barycenter-mixture", action="store_true")
    parser.add_argument("--central-attention", action="store_true")
    parser.add_argument("--central-idf", action="store_true")
    parser.add_argument("--central-modality-weight", action="store_true")
    parser.add_argument("--central-pos-decay", type=float, default=1.0)
    parser.add_argument("--multi-cone-retrieval", action="store_true")
    parser.add_argument(
        "--exclude-intra-visit-pmi",
        action="store_true",
        help="If set, lagged PMI starts at lag=1 (cross-visit only).",
    )
    parser.add_argument("--pmi-thresh", type=float, default=None)
    parser.add_argument(
        "--include-intra-modality-pmi",
        action="store_true",
        help="Enable intra-modality PMI stitching edges (dx->dx, proc->proc, med->med).",
    )
    parser.add_argument(
        "--require-joint-lag01",
        action="store_true",
        help="Keep PMI edges only if pair support passes in both lag0 and lag1.",
    )
    parser.add_argument(
        "--joint-lag01-min-support",
        type=int,
        default=3,
        help="Minimum pair count required in both lag0 and lag1 when --require-joint-lag01 is enabled.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--bootstrap-keep-q", type=float, default=None)
    parser.add_argument("--cone-aperture", type=float, default=None)
    parser.add_argument("--cone-bonus", type=float, default=None)
    parser.add_argument(
        "--persistence-bonus",
        type=float,
        default=0.0,
        help="Additive bonus for candidates that already appear in current visit (simple carry-over prior).",
    )
    parser.add_argument("--lambda-llm", type=float, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--max-history-visits",
        type=int,
        default=None,
        help="Maximum number of history visits used during evaluation/reranking context.",
    )
    parser.add_argument(
        "--max-local-candidates",
        type=int,
        default=20000,
        help="Hard cap on local retrieval candidate expansion per step to avoid pathological graph blow-ups.",
    )
    parser.add_argument("--ce-mode", type=str, default="nearest", choices=["nearest", "softmap"])
    parser.add_argument("--ce-candidate-pool", type=str, default="global", choices=["global", "restricted"])
    parser.add_argument("--ce-softmax-beta", type=float, default=5.0)
    parser.add_argument("--retrieval-pool-mode", type=str, default="localmix", choices=["localmix", "desc_assoc_pred"])
    parser.add_argument("--cone-mode", type=str, default="filter", choices=["filter", "score_bonus"])
    parser.add_argument("--disable-cone-empty-fallback", action="store_true")
    parser.add_argument("--paper-faithful-profile", action="store_true")
    parser.add_argument(
        "--skip-threshold-tuning",
        action="store_true",
        help="Skip validation threshold tuning. Useful for quick runs when focusing on ranking metrics (recall@k/ndcg/mrr).",
    )
    parser.add_argument(
        "--threshold-tune-max-steps",
        type=int,
        default=0,
        help="If >0, cap threshold tuning to this many validation steps for faster runs.",
    )
    parser.add_argument(
        "--threshold-tune-no-llm",
        action="store_true",
        help="Disable LLM scoring during threshold tuning (uses geometry-only scores for thresholds). Final evaluation still uses configured LLM backend.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=50,
        help="Number of per-visit samples to store. <=0 disables sample output unless --save-all-samples is set.",
    )
    parser.add_argument(
        "--save-all-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-visit outputs for all evaluation steps (default: true). Use --no-save-all-samples to disable.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stopping patience (epochs) based on validation ranking metric. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation score improvement to reset early stopping patience.",
    )
    parser.add_argument(
        "--early-stop-max-traj",
        type=int,
        default=400,
        help="Max number of validation trajectories used for early-stop/model-selection monitoring.",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="recall@10",
        choices=["recall@10", "ndcg@10", "mrr"],
        help="Validation metric used for model selection in early stopping.",
    )
    args = parser.parse_args()

    _log("=== GRAIL Run Start ===")
    run_start_ts = _now()
    _log(f"Start time: {run_start_ts}")
    _log(f"Data path: {args.data_path}")
    _log(f"Text mode: {args.text_mode}")
    _log(f"Baseline: {args.baseline}")
    _log(f"Pipeline stage: {args.pipeline_stage}")
    _log(f"Seed: {args.seed}")
    _log(f"LLM backend: {args.llm_backend}")
    labs_enabled = bool(args.enable_lab_tests) and not bool(args.disable_lab_tests)
    _log(f"Labs enabled: {labs_enabled}")
    _log(
        "Central weighting: "
        f"attention={args.central_attention}, "
        f"idf={args.central_idf}, "
        f"modality={args.central_modality_weight}, "
        f"pos_decay={args.central_pos_decay}"
    )
    _log(f"Multi-cone retrieval: {args.multi_cone_retrieval}")
    if args.llm_model:
        print(f"LLM model: {args.llm_model}")
    if args.lambda_llm is not None:
        print(f"LLM lambda override: {args.lambda_llm}")

    # Redirect heavy caches to output path to avoid home quota.
    cache_base = os.environ.get("GRAIL_CACHE_BASE", "./cache")
    os.makedirs(cache_base, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_base, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_base, "datasets"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_base, "torch"))
    _log(f"Cache base: {cache_base}")
    set_global_seed(args.seed)
    wall_start = time.perf_counter()
    stage_times: Dict[str, float] = {}

    data_cfg = DataConfig(data_path=args.data_path, format=args.format, text_mode=args.text_mode, label_map_path=args.label_map)
    data_cfg.seed = args.seed
    if args.min_visits is not None:
        data_cfg.min_visits = int(args.min_visits)
    if args.min_code_freq is not None:
        data_cfg.min_code_freq = int(args.min_code_freq)
    if args.baseline == "halo" and args.halo_safe_mode:
        if args.min_code_freq is None:
            data_cfg.min_code_freq = max(int(data_cfg.min_code_freq), int(args.halo_min_code_freq))
        safe_topk = int(args.halo_max_codes_per_modality)
        data_cfg.top_k_per_modality = {m: safe_topk for m in data_cfg.modality_cols.keys()}
        _log(
            "HALO safe mode enabled: "
            f"min_code_freq={data_cfg.min_code_freq}, "
            f"top_k_per_modality={data_cfg.top_k_per_modality}"
        )
    _log(f"Data filters: min_visits={data_cfg.min_visits}, min_code_freq={data_cfg.min_code_freq}")
    graph_cfg = GraphConfig()
    model_cfg = ModelConfig(embedding_dim=args.embedding_dim, manifold=args.manifold)
    model_cfg.barycenter_heads = max(1, args.barycenter_heads)
    model_cfg.barycenter_mixture = bool(args.barycenter_mixture)
    model_cfg.use_central_attention = bool(args.central_attention)
    model_cfg.use_idf_weighting = bool(args.central_idf)
    model_cfg.use_modality_weighting = bool(args.central_modality_weight)
    model_cfg.central_pos_decay = float(args.central_pos_decay)
    model_cfg.multi_cone_retrieval = bool(args.multi_cone_retrieval)
    model_cfg.persistence_bonus = float(args.persistence_bonus)
    graph_cfg.include_intra_visit = not args.exclude_intra_visit_pmi
    graph_cfg.include_intra_modality_pmi = bool(args.include_intra_modality_pmi)
    graph_cfg.require_joint_lag01 = bool(args.require_joint_lag01)
    graph_cfg.joint_lag01_min_support = int(args.joint_lag01_min_support)
    if args.no_stitch:
        graph_cfg.max_lag = 0
        graph_cfg.pmi_thresh = 1e9
    _log(
        "Lagged PMI mode: "
        f"{'within+cross-visit' if graph_cfg.include_intra_visit else 'cross-visit only'} "
        f"(max_lag={graph_cfg.max_lag})"
    )
    _log(
        "PMI stitching options: "
        f"intra_modality={graph_cfg.include_intra_modality_pmi}, "
        f"joint_lag01={graph_cfg.require_joint_lag01}, "
        f"joint_lag01_min_support={graph_cfg.joint_lag01_min_support}"
    )
    _log(f"Persistence bonus: {model_cfg.persistence_bonus}")
    if args.no_denoise:
        model_cfg.alpha_mask = 0.0
    if args.no_cone:
        model_cfg.cone_aperture = -1.0
    train_cfg = TrainConfig(epochs=args.epochs, device=args.device)
    if args.batch_size is not None:
        train_cfg.batch_size = int(args.batch_size)
    eval_cfg = EvalConfig()
    rerank_cfg = RerankConfig(backend=args.llm_backend, model_name=args.llm_model)
    rerank_cfg.scoring_mode = str(args.llm_scoring_mode)
    rerank_cfg.ft_llm_checkpoint = args.ft_llm_checkpoint
    rerank_cfg.ft_max_length = int(args.ft_llm_max_length)
    if args.pmi_thresh is not None:
        graph_cfg.pmi_thresh = float(args.pmi_thresh)
    if args.bootstrap_samples is not None:
        graph_cfg.bootstrap_samples = int(args.bootstrap_samples)
    if args.bootstrap_keep_q is not None:
        graph_cfg.bootstrap_keep_q = float(args.bootstrap_keep_q)
    if args.cone_aperture is not None:
        model_cfg.cone_aperture = float(args.cone_aperture)
    if args.cone_bonus is not None:
        model_cfg.cone_bonus = float(args.cone_bonus)
    if args.lambda_llm is not None:
        rerank_cfg.lambda_llm = float(args.lambda_llm)
    if args.max_candidates is not None:
        rerank_cfg.max_candidates = int(args.max_candidates)
    if args.max_history_visits is not None:
        eval_cfg.max_history_visits = int(args.max_history_visits)
    if args.paper_faithful_profile:
        args.ce_mode = "softmap"
        args.ce_candidate_pool = "restricted"
        args.retrieval_pool_mode = "desc_assoc_pred"
        args.cone_mode = "score_bonus"
        args.disable_cone_empty_fallback = False
    if args.debug:
        train_cfg.epochs = 2
        train_cfg.batch_size = 64
        eval_cfg.max_history_visits = 3
        graph_cfg.top_m_per_node = 10
        data_cfg.min_visits = 2
    config = ExperimentConfig(
        data=data_cfg,
        graph=graph_cfg,
        model=model_cfg,
        train=train_cfg,
        eval=eval_cfg,
        rerank=rerank_cfg,
    )

    def _apply_disable_labs() -> None:
        if labs_enabled:
            return
        if "lab" in config.data.modality_cols:
            config.data.modality_cols = {k: v for k, v in config.data.modality_cols.items() if k != "lab"}
        if "lab" in config.eval.eval_modalities:
            config.eval.eval_modalities = tuple(m for m in config.eval.eval_modalities if m != "lab")
        print("Lab modality disabled for this run.")

    _apply_disable_labs()
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.run_name:
        run_tag = f"{args.run_name}_{timestamp}"
    else:
        run_tag = timestamp
    config.run_name = f"grail_{config.data.text_mode}_{config.model.manifold}_{run_tag}"

    run_output_dir = _unique_dir(os.path.join(config.output_dir, config.run_name))
    os.makedirs(run_output_dir, exist_ok=False)
    artifact_path = args.geometry_artifact_path or os.path.join(run_output_dir, "geometry_artifacts.pt")
    if args.pipeline_stage == "llm" and not args.geometry_artifact_path:
        raise ValueError(
            "For --pipeline-stage llm, please pass --geometry-artifact-path explicitly "
            "to select which frozen geometry artifact to use."
        )
    if args.pipeline_stage in ("all", "geometry"):
        artifact_path = _non_overwrite_path(artifact_path)
    artifact_prefetched: Optional[Dict[str, object]] = None

    t_stage = time.perf_counter()
    df = load_table(config.data.data_path, config.data.format)
    stage_times["load_table_sec"] = float(time.perf_counter() - t_stage)
    _log(f"Loaded table: {len(df)} rows")
    _log("Table sample:")
    print(df.head(2).to_string())
    if args.debug:
        counts = df.groupby(config.data.subject_id_col)[config.data.visit_id_col].nunique()
        eligible = counts[counts >= 2].index.tolist()
        if eligible:
            sample_size = min(args.debug_sample_size, len(eligible))
            sampled_ids = np.random.RandomState(config.data.seed).choice(eligible, size=sample_size, replace=False)
            df = df[df[config.data.subject_id_col].isin(sampled_ids)].reset_index(drop=True)
            print(f"Debug mode: sampled {sample_size} patients with >=2 visits ({len(df)} rows)")
        else:
            print("Debug mode: no patients with >=2 visits found in sample")
    lab_to_group = load_lab_groups(args.lab_groups)
    use_saved_vocab = args.pipeline_stage == "llm" and args.baseline in (
        "grail",
        "hyper_no_stitch",
        "euclid_grail",
        "euclid_rag",
        "llm_only_hyper",
        "llm_only_euclid",
    )
    t_stage = time.perf_counter()
    if use_saved_vocab:
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Geometry artifact not found: {artifact_path}")
        artifact_prefetched = load_geometry_artifacts(artifact_path)
        saved_data_cfg = artifact_prefetched.get("data_cfg", {})
        if saved_data_cfg:
            config.data.text_mode = saved_data_cfg.get("text_mode", config.data.text_mode)
            config.data.label_map_path = saved_data_cfg.get("label_map_path", config.data.label_map_path)
            config.data.subject_id_col = saved_data_cfg.get("subject_id_col", config.data.subject_id_col)
            config.data.visit_id_col = saved_data_cfg.get("visit_id_col", config.data.visit_id_col)
            config.data.time_col = saved_data_cfg.get("time_col", config.data.time_col)
            config.data.modality_cols = saved_data_cfg.get("modality_cols", config.data.modality_cols)
            _apply_disable_labs()
        vocab = artifact_prefetched["vocab"]
        print(f"Loaded vocab from artifact for llm stage: {artifact_path}")
    else:
        vocab = build_vocab(
            df,
            config.data.modality_cols,
            min_code_freq=config.data.min_code_freq,
            top_k_per_modality=config.data.top_k_per_modality,
            lab_to_group=lab_to_group,
            text_mode=config.data.text_mode,
            label_map_path=config.data.label_map_path,
        )
    stage_times["build_or_load_vocab_sec"] = float(time.perf_counter() - t_stage)
    _log(f"Vocab size: {len(vocab.node_to_id)}")
    sample_nodes = [(k, node_display(vocab, k, with_modality=True)) for k, _ in list(vocab.id_to_node.items())[:5]]
    print(f"Vocab sample: {sample_nodes}")
    _log_label_coverage(vocab)
    t_stage = time.perf_counter()
    trajectories = build_patient_trajectories(
        df,
        config.data.subject_id_col,
        config.data.time_col,
        config.data.modality_cols,
        vocab,
        text_mode=config.data.text_mode,
        label_map_path=config.data.label_map_path,
    )
    trajectories = filter_min_visits(trajectories, config.data.min_visits)
    stage_times["build_trajectories_sec"] = float(time.perf_counter() - t_stage)
    _log(f"Trajectories: {len(trajectories)}")
    if not trajectories:
        print("No trajectories after filtering; adjust debug sampling or min_visits.")
        return
    visit_counts = [len(traj.visits) for traj in trajectories]
    print(
        "Trajectory length (visits per patient): "
        f"min={min(visit_counts)} max={max(visit_counts)} "
        f"median={np.median(visit_counts):.2f} mean={np.mean(visit_counts):.2f}"
    )
    modality_counts = {m: [] for m in config.data.modality_cols.keys()}
    for traj in trajectories:
        for visit in traj.visits:
            for m in modality_counts.keys():
                modality_counts[m].append(len(visit.get(m, [])))
    print("Code density per visit (per modality):")
    for m, counts in modality_counts.items():
        if not counts:
            continue
        print(
            f"  {m}: min={min(counts)} max={max(counts)} "
            f"median={np.median(counts):.2f} mean={np.mean(counts):.2f}"
        )
    if trajectories:
        print("Trajectory sample (first patient, first 2 visits):")
        print(trajectories[0].visits[:2])
    split_file_path, train_traj, val_traj, test_traj, reused_split = get_or_create_patient_split(
        trajectories,
        data_path=config.data.data_path,
        seed=config.data.seed,
        train_frac=config.data.train_frac,
        val_frac=config.data.val_frac,
        min_visits=config.data.min_visits,
        split_file=args.split_file,
    )
    print(f"Patient split file: {split_file_path} ({'reused' if reused_split else 'created'})")
    print(f"Split sizes | train: {len(train_traj)} | val: {len(val_traj)} | test: {len(test_traj)}")

    split_samples_dir = os.path.join(run_output_dir, "split_samples")
    os.makedirs(split_samples_dir, exist_ok=True)
    train_ids = [str(t.subject_id) for t in train_traj]
    val_ids = [str(t.subject_id) for t in val_traj]
    test_ids = [str(t.subject_id) for t in test_traj]
    _sample_split_rows(df, config.data.subject_id_col, config.data.time_col, train_ids).to_csv(
        os.path.join(split_samples_dir, "train_sample.csv"), index=False
    )
    _sample_split_rows(df, config.data.subject_id_col, config.data.time_col, val_ids).to_csv(
        os.path.join(split_samples_dir, "val_sample.csv"), index=False
    )
    _sample_split_rows(df, config.data.subject_id_col, config.data.time_col, test_ids).to_csv(
        os.path.join(split_samples_dir, "test_sample.csv"), index=False
    )

    if args.baseline in ("hyper_no_stitch",):
        graph_cfg.max_lag = 0
        graph_cfg.pmi_thresh = 1e9
    if args.baseline in ("euclid_grail", "euclid_rag", "llm_only_euclid"):
        model_cfg.manifold = "euclidean"
    if args.baseline == "euclid_rag":
        model_cfg.cone_aperture = -1.0
    if args.baseline in ("llm_only_hyper", "llm_only_euclid"):
        rerank_cfg.lambda_llm = 1.0
        if args.llm_backend == "none":
            raise ValueError("llm_only_* baselines require --llm-backend hf/openai/heuristic/ftcls, not none.")

    split_stats_path = save_split_stats(run_output_dir, config.data.modality_cols, vocab, train_traj, val_traj, test_traj)
    print(f"Saved split stats: {split_stats_path}")
    train_code_freq = _compute_train_code_freq(train_traj)
    train_seen_by_modality = _observed_nodes_by_modality(train_traj, tuple(config.eval.eval_modalities))

    stage = args.pipeline_stage
    model: Optional[GrailModel] = None
    graph = None
    edge_type_ids: Dict[str, int] = {}
    train_stats: Dict[str, float] = {}

    if args.baseline in ("grail", "hyper_no_stitch", "euclid_grail", "euclid_rag", "llm_only_hyper", "llm_only_euclid"):
        if stage in ("all", "geometry"):
            t_stage = time.perf_counter()
            graph = build_graph(
                train_traj,
                vocab,
                max_lag=config.graph.max_lag,
                include_intra_visit=config.graph.include_intra_visit,
                pmi_thresh=config.graph.pmi_thresh,
                min_support=config.graph.min_support,
                top_m_per_node=config.graph.top_m_per_node,
                use_ppmi=config.graph.use_ppmi,
                bootstrap_samples=config.graph.bootstrap_samples,
                bootstrap_keep_q=config.graph.bootstrap_keep_q,
                include_intra_modality_pmi=config.graph.include_intra_modality_pmi,
                require_joint_lag01=config.graph.require_joint_lag01,
                joint_lag01_min_support=config.graph.joint_lag01_min_support,
            )
            stage_times["build_graph_sec"] = float(time.perf_counter() - t_stage)
            print(f"Graph nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")
            graph_edges = list(graph.edges(data=True))[:5]
            print(f"Graph sample edges: {graph_edges}")

            edges = graph_to_edge_list(graph)
            edge_type_ids = edge_type_map(edges)
            edge_triplets = [(u, v, edge_type_ids[t]) for u, v, t in edges]
            print(f"Edge types: {len(edge_type_ids)}")
            print(f"Edge type sample: {list(edge_type_ids.items())[:5]}")

            _log("Training GRAIL model...")
            node_mod_ids, mod_to_idx = _node_modality_ids(vocab, config.data.modality_cols)
            model = GrailModel(
                num_nodes=len(vocab.node_to_id),
                num_edge_types=len(edge_type_ids),
                embedding_dim=config.model.embedding_dim,
                curvature=config.model.curvature,
                manifold=config.model.manifold,
                margin=config.model.margin,
                softmax_beta=config.model.softmax_beta,
                barycenter_heads=config.model.barycenter_heads,
                barycenter_mixture=config.model.barycenter_mixture,
                use_central_attention=config.model.use_central_attention,
                use_idf_weighting=config.model.use_idf_weighting,
                use_modality_weighting=config.model.use_modality_weighting,
                central_pos_decay=config.model.central_pos_decay,
                node_modality_ids=node_mod_ids,
                num_modalities=len(mod_to_idx),
                multi_cone_retrieval=config.model.multi_cone_retrieval,
            )
            if config.model.use_idf_weighting:
                model.set_idf_weights(_compute_visit_idf(train_traj, len(vocab.node_to_id)))
            t_stage = time.perf_counter()
            _log(
                f"Train start: epochs={config.train.epochs}, batch_size={config.train.batch_size}, "
                f"num_nodes={len(vocab.node_to_id)}, num_edges={len(edge_triplets)}"
            )
            val_score_fn = None
            if args.early_stop_patience > 0:
                val_subset = val_traj[: max(1, int(args.early_stop_max_traj))]
                eval_cfg_monitor = copy.deepcopy(config.eval)
                eval_cfg_monitor.top_ks = (10,)
                rerank_cfg_monitor = copy.deepcopy(config.rerank)
                rerank_cfg_monitor.backend = "none"
                rerank_cfg_monitor.lambda_llm = 0.0
                _log(
                    f"Early stopping enabled: patience={args.early_stop_patience}, "
                    f"min_delta={args.early_stop_min_delta}, metric={args.early_stop_metric}, "
                    f"val_subset={len(val_subset)} trajectories"
                )

                def _val_score_fn(m: GrailModel, epoch: int) -> float:
                    metrics_val = evaluate(
                        m,
                        graph,
                        val_subset,
                        vocab,
                        eval_cfg_monitor,
                        rerank_cfg_monitor,
                        config.model,
                        thresholds=None,
                        sample_limit=0,
                        sample_path=None,
                        train_code_freq=train_code_freq,
                        train_seen_by_modality=train_seen_by_modality,
                    )
                    modalities = list(eval_cfg_monitor.eval_modalities)
                    if args.early_stop_metric == "recall@10":
                        vals = [float(metrics_val.get(f"{mm}/recall@10", 0.0)) for mm in modalities]
                    elif args.early_stop_metric == "ndcg@10":
                        vals = [float(metrics_val.get(f"{mm}/ndcg@10", 0.0)) for mm in modalities]
                    else:
                        vals = [float(metrics_val.get(f"{mm}/mrr", 0.0)) for mm in modalities]
                    vals = [v for v in vals if np.isfinite(v)]
                    return float(np.mean(vals)) if vals else 0.0

                val_score_fn = _val_score_fn
            train_stats = train_model(
                model,
                edge_triplets,
                train_traj,
                num_nodes=len(vocab.node_to_id),
                neg_samples=config.model.neg_samples,
                alpha_mask=config.model.alpha_mask,
                mask_ratio=config.model.mask_ratio,
                batch_size=config.train.batch_size,
                epochs=config.train.epochs,
                lr=config.train.lr,
                weight_decay=config.train.weight_decay,
                device=config.train.device,
                log_every=config.train.log_every,
                val_score_fn=val_score_fn,
                early_stop_patience=max(0, int(args.early_stop_patience)),
                early_stop_min_delta=float(args.early_stop_min_delta),
            )
            stage_times["train_model_sec"] = float(time.perf_counter() - t_stage)
            _log(f"Train done in {stage_times['train_model_sec']:.2f}s")
            save_geometry_artifacts(artifact_path, model, graph, vocab, edge_type_ids, config.model, config.data)
            _log(f"Saved geometry artifacts: {artifact_path}")

        if stage == "llm":
            artifact = artifact_prefetched if artifact_prefetched is not None else load_geometry_artifacts(artifact_path)
            graph = artifact["graph"]
            vocab = artifact["vocab"]
            edge_type_ids = artifact["edge_type_ids"]
            saved_cfg = artifact["model_cfg"]
            model = GrailModel(
                num_nodes=len(vocab.node_to_id),
                num_edge_types=len(edge_type_ids),
                embedding_dim=saved_cfg["embedding_dim"],
                curvature=saved_cfg["curvature"],
                manifold=saved_cfg["manifold"],
                margin=saved_cfg["margin"],
                softmax_beta=saved_cfg["softmax_beta"],
                barycenter_heads=saved_cfg.get("barycenter_heads", 1),
                barycenter_mixture=saved_cfg.get("barycenter_mixture", False),
                use_central_attention=saved_cfg.get("use_central_attention", False),
                use_idf_weighting=saved_cfg.get("use_idf_weighting", False),
                use_modality_weighting=saved_cfg.get("use_modality_weighting", False),
                central_pos_decay=saved_cfg.get("central_pos_decay", 1.0),
                node_modality_ids=_node_modality_ids(vocab, config.data.modality_cols)[0],
                num_modalities=len(config.data.modality_cols),
                multi_cone_retrieval=saved_cfg.get("multi_cone_retrieval", False),
            )
            if saved_cfg.get("use_idf_weighting", False):
                model.set_idf_weights(_compute_visit_idf(train_traj, len(vocab.node_to_id)))
            missing, unexpected = model.load_state_dict(artifact["model_state"], strict=False)
            if missing or unexpected:
                print(f"Artifact state compatibility: missing={len(missing)} unexpected={len(unexpected)}")
            model.to(config.train.device)
            _log(f"Loaded geometry artifacts: {artifact_path}")

        if stage == "geometry":
            config.rerank.backend = "none"

        if args.skip_threshold_tuning:
            thresholds = None
            _log("Skipping threshold tuning on validation set (--skip-threshold-tuning).")
        else:
            _log("Tuning thresholds on validation set...")
            t_stage = time.perf_counter()
            tune_rerank_cfg = config.rerank
            if args.threshold_tune_no_llm:
                tune_rerank_cfg = copy.deepcopy(config.rerank)
                tune_rerank_cfg.backend = "none"
                tune_rerank_cfg.lambda_llm = 0.0
                _log("Threshold tuning mode: geometry-only (--threshold-tune-no-llm).")
            thresholds = tune_thresholds_for_eval(
                model,
                graph,
                val_traj,
                vocab,
                config.eval,
                tune_rerank_cfg,
                config.model,
                max_local_candidates=max(1000, int(args.max_local_candidates)),
                ce_mode=args.ce_mode,
                ce_candidate_pool=args.ce_candidate_pool,
                ce_softmax_beta=float(args.ce_softmax_beta),
                retrieval_pool_mode=args.retrieval_pool_mode,
                cone_mode=args.cone_mode,
                cone_empty_fallback=(not bool(args.disable_cone_empty_fallback)),
                max_steps=max(0, int(args.threshold_tune_max_steps)),
            )
            stage_times["tune_thresholds_sec"] = float(time.perf_counter() - t_stage)
        _log(f"Evaluating GRAIL stage={stage}...")
        sample_path = os.path.join(run_output_dir, f"sample_outputs_{stage}.json")
        sample_limit_eval = int(args.sample_limit)
        if args.save_all_samples:
            sample_limit_eval = int(sum(max(0, len(t.visits) - 1) for t in test_traj))
        elif sample_limit_eval <= 0:
            sample_limit_eval = 0
        t_stage = time.perf_counter()
        metrics = evaluate(
            model,
            graph,
            test_traj,
            vocab,
            config.eval,
            config.rerank,
            config.model,
            max_local_candidates=max(1000, int(args.max_local_candidates)),
            ce_mode=args.ce_mode,
            ce_candidate_pool=args.ce_candidate_pool,
            ce_softmax_beta=float(args.ce_softmax_beta),
            retrieval_pool_mode=args.retrieval_pool_mode,
            cone_mode=args.cone_mode,
            cone_empty_fallback=(not bool(args.disable_cone_empty_fallback)),
            thresholds=thresholds,
            sample_limit=sample_limit_eval,
            sample_path=sample_path,
            train_code_freq=train_code_freq,
            train_seen_by_modality=train_seen_by_modality,
            prompt_example_path=(os.path.join(run_output_dir, "llm_prompt_example.txt") if config.rerank.backend != "none" else None),
        )
        stage_times["evaluate_sec"] = float(time.perf_counter() - t_stage)

        if args.save_viz and model is not None:
            viz_dir = os.path.join(run_output_dir, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            node_plot = os.path.join(viz_dir, "nodes_2d.png")
            traj_plot = os.path.join(viz_dir, "trajectories_2d.png")
            plot_poincare_disk_embeddings(model, vocab, node_plot)
            plot_patient_trajectories(model, test_traj, traj_plot)
            print(f"Saved visualizations: {node_plot} | {traj_plot}")
    else:
        if stage != "all":
            raise ValueError("pipeline-stage geometry/llm is only supported for GRAIL baselines.")
        if args.baseline == "copy_last":
            _log("Evaluating baseline: copy_last (no training).")
            t_stage = time.perf_counter()
            sample_path = os.path.join(run_output_dir, "sample_outputs.json")
            sample_limit_eval = int(args.sample_limit)
            if args.save_all_samples:
                sample_limit_eval = int(sum(max(0, len(t.visits) - 1) for t in test_traj))
            elif sample_limit_eval <= 0:
                sample_limit_eval = 0
            metrics = evaluate_copy_last_baseline(
                test_traj,
                vocab,
                config.eval.top_ks,
                sample_path=sample_path,
                sample_limit=sample_limit_eval,
                train_seen_by_modality=train_seen_by_modality,
            )
            stage_times["evaluate_sec"] = float(time.perf_counter() - t_stage)
        else:
            if args.baseline == "halo" and args.batch_size is None and train_cfg.batch_size > 16:
                train_cfg.batch_size = 16
                _log("HALO memory safeguard: using batch_size=16 (override with --batch-size if desired).")
            print(f"Training baseline: {args.baseline} ...")
            t_stage = time.perf_counter()
            graph = build_graph(
                train_traj,
                vocab,
                max_lag=config.graph.max_lag,
                include_intra_visit=config.graph.include_intra_visit,
                pmi_thresh=config.graph.pmi_thresh,
                min_support=config.graph.min_support,
                top_m_per_node=config.graph.top_m_per_node,
                use_ppmi=config.graph.use_ppmi,
                bootstrap_samples=config.graph.bootstrap_samples,
                bootstrap_keep_q=config.graph.bootstrap_keep_q,
                include_intra_modality_pmi=config.graph.include_intra_modality_pmi,
                require_joint_lag01=config.graph.require_joint_lag01,
                joint_lag01_min_support=config.graph.joint_lag01_min_support,
            )
            stage_times["build_graph_sec"] = float(time.perf_counter() - t_stage)
            edges = graph_to_edge_list(graph)
            dataset_train = NextVisitDataset(train_traj, num_nodes=len(vocab.node_to_id))
            dataset_test = NextVisitDataset(test_traj, num_nodes=len(vocab.node_to_id))
            grail_for_init = None
            if args.baseline == "transformer":
                model = VisitTransformer(len(vocab.node_to_id), config.model.embedding_dim)
            elif args.baseline == "retain":
                model = RetainModel(len(vocab.node_to_id), config.model.embedding_dim)
            elif args.baseline == "gram":
                ancestor_map = build_ancestor_map(len(vocab.node_to_id), edges)
                model = GramModel(len(vocab.node_to_id), config.model.embedding_dim, ancestor_map)
            elif args.baseline == "halo":
                model = HaloNextVisitModel(
                    len(vocab.node_to_id),
                    max_visits=dataset_train.max_visits,
                    n_embd=config.model.embedding_dim,
                    n_layer=4,
                    n_head=4,
                )
            elif args.baseline == "behrt":
                model = BehrtModel(
                    len(vocab.node_to_id),
                    config.model.embedding_dim,
                    max_visits=dataset_train.max_visits,
                )
            elif args.baseline == "medbert":
                model = MedBertModel(
                    len(vocab.node_to_id),
                    config.model.embedding_dim,
                    max_visits=dataset_train.max_visits,
                )
            elif args.baseline in ("grail_seq_transformer", "grail_seq_retain", "grail_seq_gram", "grail_seq_hybrid"):
                _log("Stage 1/2: training hyperbolic GRAIL backbone for sequence initialization.")
                edges_typed = graph_to_edge_list(graph)
                edge_type_ids_local = edge_type_map(edges_typed)
                edge_triplets = [(u, v, edge_type_ids_local[t]) for u, v, t in edges_typed]
                node_mod_ids, mod_to_idx = _node_modality_ids(vocab, config.data.modality_cols)
                grail_for_init = GrailModel(
                    num_nodes=len(vocab.node_to_id),
                    num_edge_types=len(edge_type_ids_local),
                    embedding_dim=config.model.embedding_dim,
                    curvature=config.model.curvature,
                    manifold=config.model.manifold,
                    margin=config.model.margin,
                    softmax_beta=config.model.softmax_beta,
                    barycenter_heads=config.model.barycenter_heads,
                    barycenter_mixture=config.model.barycenter_mixture,
                    use_central_attention=config.model.use_central_attention,
                    use_idf_weighting=config.model.use_idf_weighting,
                    use_modality_weighting=config.model.use_modality_weighting,
                    central_pos_decay=config.model.central_pos_decay,
                    node_modality_ids=node_mod_ids,
                    num_modalities=len(mod_to_idx),
                    multi_cone_retrieval=config.model.multi_cone_retrieval,
                )
                if config.model.use_idf_weighting:
                    grail_for_init.set_idf_weights(_compute_visit_idf(train_traj, len(vocab.node_to_id)))
                t_grail = time.perf_counter()
                geo_stats = train_model(
                    grail_for_init,
                    edge_triplets,
                    train_traj,
                    num_nodes=len(vocab.node_to_id),
                    neg_samples=config.model.neg_samples,
                    alpha_mask=config.model.alpha_mask,
                    mask_ratio=config.model.mask_ratio,
                    batch_size=config.train.batch_size,
                    epochs=config.train.epochs,
                    lr=config.train.lr,
                    weight_decay=config.train.weight_decay,
                    device=config.train.device,
                    log_every=config.train.log_every,
                )
                stage_times["grail_init_train_sec"] = float(time.perf_counter() - t_grail)
                train_stats.update({f"grail_init_{k}": v for k, v in geo_stats.items()})
                if args.baseline == "grail_seq_transformer":
                    model = VisitTransformer(len(vocab.node_to_id), config.model.embedding_dim)
                elif args.baseline == "grail_seq_retain":
                    model = RetainModel(len(vocab.node_to_id), config.model.embedding_dim)
                elif args.baseline == "grail_seq_gram":
                    ancestor_map = build_ancestor_map(len(vocab.node_to_id), edges_typed)
                    model = GramModel(len(vocab.node_to_id), config.model.embedding_dim, ancestor_map)
                else:
                    model = HybridSeqModel(len(vocab.node_to_id), config.model.embedding_dim, max_visits=dataset_train.max_visits)
                init_sequence_embeddings_from_hyperbolic(
                    model,
                    grail_for_init.embeddings,
                    grail_for_init.manifold,
                    freeze_embedding=bool(args.grail_seq_freeze_embedding),
                )
                _log(
                    f"Initialized {args.baseline} embeddings from trained GRAIL backbone "
                    f"(freeze={bool(args.grail_seq_freeze_embedding)})."
                )
            else:
                raise ValueError(f"Unknown baseline: {args.baseline}")

            if args.baseline in ("behrt", "medbert") and args.baseline_pretrained_checkpoint:
                ckpt_stats = _load_baseline_pretrained_checkpoint(model, args.baseline_pretrained_checkpoint)
                train_stats["pretrained_ckpt_missing"] = float(ckpt_stats["missing"])
                train_stats["pretrained_ckpt_unexpected"] = float(ckpt_stats["unexpected"])
                if args.baseline_freeze_backbone:
                    n_trainable = _freeze_baseline_backbone(model)
                    _log(f"Frozen baseline backbone; trainable head params={n_trainable}")
                    train_stats["pretrained_head_trainable_params"] = float(n_trainable)

            if args.baseline in ("behrt", "medbert"):
                if args.baseline_skip_mlm_pretrain:
                    _log("Skipping in-project MLM pretraining for baseline (--baseline-skip-mlm-pretrain).")
                else:
                    print(f"Pretraining {args.baseline} with code-level MLM...")
                    t_stage = time.perf_counter()
                    pre_stats = pretrain_code_mlm(
                        model,
                        dataset_train,
                        epochs=2,
                        batch_size=config.train.batch_size,
                        lr=config.train.lr,
                        device=config.train.device,
                    )
                    stage_times["pretrain_mlm_sec"] = float(time.perf_counter() - t_stage)
                    train_stats.update({f"pretrain_{k}": v for k, v in pre_stats.items()})

            t_stage = time.perf_counter()
            seq_stats = train_sequence_model(
                model,
                dataset_train,
                epochs=config.train.epochs,
                batch_size=config.train.batch_size,
                lr=config.train.lr,
                device=config.train.device,
            )
            stage_times["train_model_sec"] = float(time.perf_counter() - t_stage)
            train_stats.update(seq_stats)
            print("Evaluating baseline...")
            sample_path = os.path.join(run_output_dir, "sample_outputs.json")
            dataset_val = NextVisitDataset(val_traj, num_nodes=len(vocab.node_to_id))
            if args.skip_threshold_tuning:
                thresholds = None
                _log("Skipping threshold tuning on validation set (--skip-threshold-tuning).")
            else:
                t_stage = time.perf_counter()
                thresholds = tune_thresholds_sequence(model, dataset_val, vocab, config.train.device)
                stage_times["tune_thresholds_sec"] = float(time.perf_counter() - t_stage)
            t_stage = time.perf_counter()
            sample_limit_eval = int(args.sample_limit)
            if args.save_all_samples:
                sample_limit_eval = int(sum(max(0, len(t.visits) - 1) for t in test_traj))
            elif sample_limit_eval <= 0:
                sample_limit_eval = 0
            metrics = evaluate_sequence_model(
                model,
                dataset_test,
                vocab,
                config.eval.top_ks,
                config.train.device,
                thresholds=thresholds,
                sample_path=sample_path,
                sample_limit=sample_limit_eval,
                graph=graph,
                train_seen_by_modality=train_seen_by_modality,
            )
            stage_times["evaluate_sec"] = float(time.perf_counter() - t_stage)
    n_eval_steps = int(sum(max(0, len(t.visits) - 1) for t in test_traj))
    metrics = _augment_metrics_with_summaries(
        metrics=metrics,
        eval_modalities=tuple(config.eval.eval_modalities),
        stage_times=stage_times,
        train_stats=train_stats,
        n_eval_steps=n_eval_steps,
    )

    metrics_path = os.path.join(run_output_dir, f"{args.baseline}_{stage}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    _log(f"Saved metrics: {metrics_path}")
    wall_total = float(time.perf_counter() - wall_start)
    runtime_payload = {
        "start_time": run_start_ts,
        "end_time": _now(),
        "seed": args.seed,
        "baseline": args.baseline,
        "stage": stage,
        "data_path": config.data.data_path,
        "llm_prompt_template": build_prompt_template(),
        "central_weighting": {
            "attention": bool(config.model.use_central_attention),
            "idf": bool(config.model.use_idf_weighting),
            "modality": bool(config.model.use_modality_weighting),
            "pos_decay": float(config.model.central_pos_decay),
            "multi_cone_retrieval": bool(config.model.multi_cone_retrieval),
        },
        "paper_alignment_flags": {
            "paper_faithful_profile": bool(args.paper_faithful_profile),
            "llm_scoring_mode": str(args.llm_scoring_mode),
            "ce_mode": str(args.ce_mode),
            "ce_candidate_pool": str(args.ce_candidate_pool),
            "ce_softmax_beta": float(args.ce_softmax_beta),
            "retrieval_pool_mode": str(args.retrieval_pool_mode),
            "cone_mode": str(args.cone_mode),
            "cone_empty_fallback": bool(not bool(args.disable_cone_empty_fallback)),
        },
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
            "n_visit_pairs_total": int(sum(max(0, len(t.visits) - 1) for t in trajectories)),
            "n_visit_pairs_train": int(sum(max(0, len(t.visits) - 1) for t in train_traj)),
            "n_visit_pairs_val": int(sum(max(0, len(t.visits) - 1) for t in val_traj)),
            "n_visit_pairs_test": int(sum(max(0, len(t.visits) - 1) for t in test_traj)),
        },
        "vocab": summarize_vocab(vocab, config.data.modality_cols),
        "graph": summarize_graph(graph, edge_type_ids),
        "stage_times_sec": stage_times,
        "train_stats": train_stats,
        "total_wall_time_sec": wall_total,
    }
    runtime_path = os.path.join(run_output_dir, f"{args.baseline}_{stage}_runtime.json")
    with open(runtime_path, "w") as f:
        json.dump(runtime_payload, f, indent=2)
    _log(f"Saved runtime report: {runtime_path}")
    _log("Timing summary:")
    for k, v in sorted(stage_times.items()):
        print(f"  {k}: {v:.2f}s")
    print(f"  total_wall_time_sec: {wall_total:.2f}s")
    for key, val in sorted(metrics.items()):
        print(f"{key}: {val:.4f}")
    _log(f"End time: {_now()}")
    _log("=== GRAIL Run Complete ===")


if __name__ == "__main__":
    main()
