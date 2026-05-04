import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from grail.config import DataConfig, EvalConfig
from grail.data import (
    build_patient_trajectories,
    build_vocab,
    filter_min_visits,
    load_lab_groups,
    load_table,
    node_display,
    split_patients,
)
from grail.eval import aggregate_metrics, mrr, ndcg_at_k, precision_at_k, recall_at_k


MODALITIES = ("dx", "proc", "med")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def save_split_stats(run_output_dir: str, modality_cols: Dict[str, str], train_traj, val_traj, test_traj) -> str:
    payload = {
        "train": summarize_trajectories(train_traj, modality_cols),
        "val": summarize_trajectories(val_traj, modality_cols),
        "test": summarize_trajectories(test_traj, modality_cols),
        "all": summarize_trajectories(train_traj + val_traj + test_traj, modality_cols),
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


def _visit_to_tokens(vocab, visit: Dict[str, List[int]]) -> Dict[str, List[str]]:
    out = {}
    for m in MODALITIES:
        out[m] = [str(vocab.id_to_node[int(c)]) for c in visit.get(m, [])]
    return out


def _build_input_text(vocab, visits: List[Dict[str, List[int]]], t: int, max_history_visits: int) -> str:
    start = max(0, t - max_history_visits + 1)
    lines = [
        "Patient history visits:",
    ]
    for i in range(start, t + 1):
        vv = _visit_to_tokens(vocab, visits[i])
        lines.append(f"Visit {i - start + 1}")
        lines.append(f"dx: [{', '.join(vv['dx'])}]")
        lines.append(f"proc: [{', '.join(vv['proc'])}]")
        lines.append(f"med: [{', '.join(vv['med'])}]")
    lines.append("Predict next-visit events.")
    return "\n".join(lines)


def _build_label_spaces(train_traj) -> Dict[str, List[int]]:
    per_mod = {m: set() for m in MODALITIES}
    for traj in train_traj:
        visits = traj.visits
        for t in range(len(visits) - 1):
            nxt = visits[t + 1]
            for m in MODALITIES:
                per_mod[m].update(int(x) for x in nxt.get(m, []))
    return {m: sorted(list(ids)) for m, ids in per_mod.items()}


class NextVisitTextDataset(Dataset):
    def __init__(self, trajectories, vocab, max_history_visits: int):
        self.samples = []
        for traj in trajectories:
            visits = traj.visits
            for t in range(len(visits) - 1):
                self.samples.append((traj.subject_id, t, visits))
        self.vocab = vocab
        self.max_history_visits = max_history_visits

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, t, visits = self.samples[idx]
        text = _build_input_text(self.vocab, visits, t, self.max_history_visits)
        next_visit = visits[t + 1]
        return {
            "subject_id": sid,
            "visit_index": int(t),
            "text": text,
            "next_visit": next_visit,
            "current_visit": visits[t],
        }


class LLMClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        label_spaces: Dict[str, List[int]],
        head_hidden_dim: int = 512,
        finetune_backbone: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            device_map=None,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
        hidden = int(getattr(self.backbone.config, "hidden_size", 4096))
        self.label_spaces = {m: list(v) for m, v in label_spaces.items()}
        self.heads = nn.ModuleDict()
        for m in MODALITIES:
            out_dim = len(self.label_spaces[m])
            self.heads[m] = nn.Sequential(
                nn.Linear(hidden, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(head_hidden_dim, out_dim),
            )
        # Keep head dtype aligned with backbone dtype (e.g., fp16 on CUDA).
        bb_dtype = next(self.backbone.parameters()).dtype
        self.heads.to(dtype=bb_dtype)
        if not finetune_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        last_idx = attention_mask.sum(dim=1).long().clamp(min=1) - 1
        pooled = h[torch.arange(h.size(0), device=h.device), last_idx]
        return {m: self.heads[m](pooled.to(next(self.heads[m].parameters()).dtype)) for m in MODALITIES}


def _collate(batch, tokenizer, label_index: Dict[str, Dict[int, int]], device: str, max_length: int):
    texts = [x["text"] for x in batch]
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    targets = {}
    for m in MODALITIES:
        y = torch.zeros((len(batch), len(label_index[m])), dtype=torch.float32)
        for i, item in enumerate(batch):
            for nid in item["next_visit"].get(m, []):
                j = label_index[m].get(int(nid))
                if j is not None:
                    y[i, j] = 1.0
        targets[m] = y.to(device)
    return {
        "subject_id": [x["subject_id"] for x in batch],
        "visit_index": [x["visit_index"] for x in batch],
        "prompt": [x["text"] for x in batch],
        "current_visit": [x["current_visit"] for x in batch],
        "next_visit": [x["next_visit"] for x in batch],
        "input_ids": tok["input_ids"].to(device),
        "attention_mask": tok["attention_mask"].to(device),
        "targets": targets,
    }


def _predict_ranked_ids(logits: torch.Tensor, label_space_ids: List[int]) -> List[int]:
    if logits.numel() == 0:
        return []
    order = torch.argsort(logits, descending=True).tolist()
    return [int(label_space_ids[i]) for i in order]


def evaluate_model(
    model,
    loader,
    label_spaces,
    eval_modalities,
    top_ks,
    device,
    sample_path: Optional[str] = None,
    sample_limit: int = 0,
    vocab=None,
):
    model.eval()
    metrics = []
    samples = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            bsz = batch["input_ids"].size(0)
            for i in range(bsz):
                sample_metrics = {}
                pred_map = {}
                truth_map = {}
                for m in eval_modalities:
                    ranked = _predict_ranked_ids(logits[m][i], label_spaces[m])
                    truth = batch["next_visit"][i].get(m, [])
                    pred_map[m] = ranked[: max(top_ks)]
                    truth_map[m] = truth
                    for k in top_ks:
                        sample_metrics[f"{m}/recall@{k}"] = recall_at_k(ranked, truth, k)
                        sample_metrics[f"{m}/precision@{k}"] = precision_at_k(ranked, truth, k)
                        sample_metrics[f"{m}/ndcg@{k}"] = ndcg_at_k(ranked, truth, k)
                    sample_metrics[f"{m}/mrr"] = mrr(ranked, truth)
                metrics.append(sample_metrics)
                if sample_limit > 0 and len(samples) < sample_limit:
                    if vocab is not None:
                        disp_pred = {m: [node_display(vocab, c) for c in pred_map.get(m, [])] for m in eval_modalities}
                        disp_truth = {m: [node_display(vocab, c) for c in truth_map.get(m, [])] for m in eval_modalities}
                    else:
                        disp_pred = pred_map
                        disp_truth = truth_map
                    samples.append(
                        {
                            "subject_id": batch["subject_id"][i],
                            "visit_index": int(batch["visit_index"][i]),
                            "prompt": batch["prompt"][i],
                            "per_visit_metrics": sample_metrics,
                            "predictions": disp_pred,
                            "truth_next": disp_truth,
                        }
                    )
    if sample_path and sample_limit > 0:
        with open(sample_path, "w") as f:
            json.dump(samples, f, indent=2)
    return aggregate_metrics(metrics)


def _augment_metrics_with_summaries(
    metrics: Dict[str, float],
    eval_modalities: Tuple[str, ...],
    stage_times: Dict[str, float],
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

    _macro("recall@5", "overall/recall@5_macro")
    _macro("recall@10", "overall/recall@10_macro")
    _macro("ndcg@10", "overall/ndcg@10_macro")
    _macro("mrr", "overall/mrr_macro")

    _macro("precision@5", "overall/precision@5_macro")
    _macro("precision@10", "overall/precision@10_macro")
    _macro("recall@20", "overall/recall@20_macro")
    _macro("ndcg@5", "overall/ndcg@5_macro")
    _macro("ndcg@20", "overall/ndcg@20_macro")

    eval_sec = float(stage_times.get("evaluate_sec", 0.0))
    if eval_sec > 0 and n_eval_steps > 0:
        out["eff/eval_steps_per_sec"] = float(n_eval_steps / eval_sec)
        out["eff/eval_ms_per_step"] = float(1000.0 * eval_sec / n_eval_steps)
    if "train_sec" in stage_times:
        out["eff/train_wall_sec"] = float(stage_times["train_sec"])
    out["eff/n_eval_steps"] = float(n_eval_steps)
    out["eff/total_stage_time_sec"] = float(sum(float(v) for v in stage_times.values()))
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)

    parser = argparse.ArgumentParser(description="LLM finetuning with multi-label classification heads for next-visit prediction.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--format", type=str, default="pickle")
    parser.add_argument("--lab-groups", type=str, default=None)
    parser.add_argument("--text-mode", type=str, default="hybrid", choices=["hybrid", "code", "text"])
    parser.add_argument("--label-map", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--run-name", type=str, default="mimic_llm_finetune_cls")
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--max-history-visits", type=int, default=1)
    parser.add_argument("--min-code-freq", type=int, default=0)
    parser.add_argument("--disable-lab-tests", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--loss-w-dx", type=float, default=2.0)
    parser.add_argument("--loss-w-proc", type=float, default=2.0)
    parser.add_argument("--loss-w-med", type=float, default=1.0)
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument(
        "--save-all-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-visit outputs for full test set by default. Use --no-save-all-samples to disable.",
    )
    args = parser.parse_args()

    _log("=== LLM Finetune (Classification Head) Run Start ===")
    _log(f"Data path: {args.data_path}")
    _log(f"Text mode: {args.text_mode}")
    _log(f"LLM model: {args.llm_model}")
    _log(f"Seed: {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_cfg = DataConfig(data_path=args.data_path, format=args.format, text_mode=args.text_mode, label_map_path=args.label_map)
    data_cfg.seed = args.seed
    data_cfg.min_code_freq = args.min_code_freq
    eval_cfg = EvalConfig(max_history_visits=args.max_history_visits)
    eval_cfg.eval_modalities = tuple([m for m in eval_cfg.eval_modalities if m in MODALITIES])

    if args.disable_lab_tests:
        data_cfg.modality_cols = {k: v for k, v in data_cfg.modality_cols.items() if k != "lab"}

    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_name = f"llm_finetune_cls_{args.text_mode}_{args.run_name}_{timestamp}"
    output_root = os.environ.get("GRAIL_OUTPUT_ROOT", "./outputs")
    run_output_dir = _unique_dir(os.path.join(output_root, run_name))
    os.makedirs(run_output_dir, exist_ok=False)

    stage_times: Dict[str, float] = {}
    t0 = time.perf_counter()
    df = load_table(data_cfg.data_path, data_cfg.format)
    stage_times["load_table_sec"] = float(time.perf_counter() - t0)
    _log(f"Loaded table: {len(df)} rows")

    lab_to_group = load_lab_groups(args.lab_groups)
    t0 = time.perf_counter()
    vocab = build_vocab(
        df,
        data_cfg.modality_cols,
        min_code_freq=data_cfg.min_code_freq,
        top_k_per_modality=data_cfg.top_k_per_modality,
        lab_to_group=lab_to_group,
        text_mode=data_cfg.text_mode,
        label_map_path=data_cfg.label_map_path,
    )
    stage_times["build_vocab_sec"] = float(time.perf_counter() - t0)
    _log(f"Vocab size: {len(vocab.node_to_id)}")

    t0 = time.perf_counter()
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
    stage_times["build_trajectories_sec"] = float(time.perf_counter() - t0)
    _log(f"Trajectories: {len(trajectories)}")

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

    split_stats_path = save_split_stats(run_output_dir, data_cfg.modality_cols, train_traj, val_traj, test_traj)
    _log(f"Saved split stats: {split_stats_path}")

    label_spaces = _build_label_spaces(train_traj)
    label_sizes = {m: len(v) for m, v in label_spaces.items()}
    _log(f"Label spaces: {label_sizes}")
    label_index = {m: {nid: i for i, nid in enumerate(label_spaces[m])} for m in MODALITIES}

    train_ds = NextVisitTextDataset(train_traj, vocab, args.max_history_visits)
    val_ds = NextVisitTextDataset(val_traj, vocab, args.max_history_visits)
    test_ds = NextVisitTextDataset(test_traj, vocab, args.max_history_visits)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    _log(f"Using device: {device}")

    dtype = torch.float16 if device.startswith("cuda") else None
    model = LLMClassifier(
        model_name=args.llm_model,
        label_spaces=label_spaces,
        head_hidden_dim=args.head_hidden_dim,
        finetune_backbone=args.finetune_backbone,
        torch_dtype=dtype,
    )
    model.to(device)

    collate = lambda b: _collate(b, tokenizer, label_index, device, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loss_w = {
        "dx": float(args.loss_w_dx),
        "proc": float(args.loss_w_proc),
        "med": float(args.loss_w_med),
    }

    best_state = None
    best_val = -1.0
    bad_epochs = 0
    t_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = 0.0
            for m in MODALITIES:
                if logits[m].numel() == 0:
                    continue
                loss = loss + loss_w.get(m, 1.0) * criterion(logits[m].float(), batch["targets"][m].float())
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            if step % 20 == 0:
                _log(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss={running/step:.4f}")

        val_metrics = evaluate_model(model, val_loader, label_spaces, eval_cfg.eval_modalities, eval_cfg.top_ks, device)
        val_score = float(np.mean([val_metrics.get(f"{m}/recall@10", 0.0) for m in eval_cfg.eval_modalities]))
        _log(f"Epoch {epoch}/{args.epochs} done | train_loss={running/max(len(train_loader),1):.4f} | val_recall@10_macro={val_score:.4f}")
        if val_score > best_val + 1e-6:
            best_val = val_score
            bad_epochs = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if args.early_stop_patience >= 0 and bad_epochs > args.early_stop_patience:
                _log(f"Early stopping at epoch {epoch} (patience={args.early_stop_patience})")
                break

    stage_times["train_sec"] = float(time.perf_counter() - t_train)

    if best_state is not None:
        model.load_state_dict(best_state)

    t_eval = time.perf_counter()
    sample_limit_eval = int(args.sample_limit)
    if args.save_all_samples:
        sample_limit_eval = int(len(test_ds))
    elif sample_limit_eval <= 0:
        sample_limit_eval = 0
    sample_path = os.path.join(run_output_dir, "sample_outputs_llm_finetune.json")
    test_metrics = evaluate_model(
        model,
        test_loader,
        label_spaces,
        eval_cfg.eval_modalities,
        eval_cfg.top_ks,
        device,
        sample_path=sample_path,
        sample_limit=sample_limit_eval,
        vocab=vocab,
    )
    stage_times["evaluate_sec"] = float(time.perf_counter() - t_eval)
    n_eval_steps = int(len(test_ds))
    test_metrics = _augment_metrics_with_summaries(
        metrics=test_metrics,
        eval_modalities=tuple(eval_cfg.eval_modalities),
        stage_times=stage_times,
        n_eval_steps=n_eval_steps,
    )

    metrics_path = os.path.join(run_output_dir, "llm_finetune_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    ckpt_path = os.path.join(run_output_dir, "llm_finetune_cls.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_spaces": label_spaces,
            "args": vars(args),
            "vocab": vocab,
            "split_file": split_file_path,
        },
        ckpt_path,
    )

    runtime = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "llm_model": args.llm_model,
        "finetune_backbone": bool(args.finetune_backbone),
        "split_file": split_file_path,
        "run_output_dir": run_output_dir,
        "label_sizes": label_sizes,
        "loss_weights": loss_w,
        "stage_times_sec": stage_times,
    }
    with open(os.path.join(run_output_dir, "llm_finetune_runtime.json"), "w") as f:
        json.dump(runtime, f, indent=2)

    _log(f"Saved metrics: {metrics_path}")
    _log(f"Saved checkpoint: {ckpt_path}")
    for k, v in sorted(test_metrics.items()):
        _log(f"{k}: {v:.4f}")
    _log("=== LLM Finetune (Classification Head) Run Complete ===")


if __name__ == "__main__":
    main()
