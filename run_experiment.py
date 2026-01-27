import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
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
from grail.baselines import (
    NextVisitDataset,
    RetainModel,
    VisitTransformer,
    GramModel,
    build_ancestor_map,
    train_sequence_model,
    evaluate_sequence_model,
    pretrain_masked_visits,
    tune_thresholds_sequence,
)
from grail.model import GrailModel
from grail.retrieval import build_risk_horizon
from grail.train import train_model


def build_prompt(history_codes: List[List[str]], candidates: Dict[str, List[str]]) -> str:
    lines = ["Patient history (central events):"]
    for t, codes in enumerate(history_codes, 1):
        lines.append(f"Visit {t}: {', '.join(codes)}")
    lines.append("Candidate next events by modality:")
    for modality, codes in candidates.items():
        lines.append(f"{modality}: {', '.join(codes)}")
    lines.append("Predict the next visit codes by modality.")
    return "\n".join(lines)


def evaluate(
    model: GrailModel,
    graph,
    trajectories,
    vocab,
    eval_cfg: EvalConfig,
    rerank_cfg: RerankConfig,
    model_cfg: ModelConfig,
    thresholds: Dict[str, float] = None,
    sample_limit: int = 50,
    sample_path: str = None,
) -> Dict[str, float]:
    model.eval()
    reranker = LLMReranker(backend=rerank_cfg.backend, model_name=rerank_cfg.model_name)
    metrics = []
    samples = []
    total_steps = sum(max(0, len(traj.visits) - 1) for traj in trajectories)
    step = 0
    last_log = time.time()
    print(f"Evaluation steps: {total_steps}")
    for traj in trajectories:
        visits = traj.visits
        if len(visits) < 2:
            continue
        for t in range(len(visits) - 1):
            step += 1
            if step % 200 == 0:
                now = time.time()
                rate = 200 / max(now - last_log, 1e-6)
                print(f"Eval progress: {step}/{total_steps} ({rate:.2f} steps/s)")
                last_log = now
            current = visits[t]
            next_visit = visits[t + 1]
            all_codes = [code for codes in current.values() for code in codes]
            if not all_codes:
                continue
            code_tensor = torch.tensor([all_codes])
            barycenter = model.barycenter(code_tensor)
            dists = model.manifold.dist(barycenter, model.embeddings)
            central_code = int(torch.argmin(dists).item())
            horizon = build_risk_horizon(
                graph,
                vocab,
                model,
                barycenter.squeeze(0),
                central_code,
                max_desc_depth=3,
                max_candidates=rerank_cfg.max_candidates,
                cone_aperture=model_cfg.cone_aperture,
                cone_bonus=model_cfg.cone_bonus,
            )
            history_codes = []
            for back in range(max(0, t - eval_cfg.max_history_visits + 1), t + 1):
                hist_codes = [vocab.id_to_node[c] for m in visits[back].values() for c in m]
                history_codes.append(hist_codes[:rerank_cfg.top_m_history])
            prompt_candidates = {}
            scored_candidates = {}
            for modality, cand_list in horizon.items():
                codes = [vocab.id_to_node[idx] for idx, _ in cand_list]
                prompt_candidates[modality] = codes
            prompt = build_prompt(history_codes, prompt_candidates)
            score_map = {}
            for modality, cand_list in horizon.items():
                if not cand_list:
                    continue
                codes = [vocab.id_to_node[idx] for idx, _ in cand_list]
                llm_scores = reranker.score(prompt, codes)
                combined = []
                for (idx, geo_score), llm_score in zip(cand_list, llm_scores):
                    score = rerank_cfg.lambda_llm * llm_score + (1 - rerank_cfg.lambda_llm) * geo_score
                    combined.append((idx, score))
                combined.sort(key=lambda x: x[1], reverse=True)
                scored_candidates[modality] = [idx for idx, _ in combined]
                score_map[modality] = combined

            sample_metrics = {}
            for modality in eval_cfg.eval_modalities:
                preds = scored_candidates.get(modality, [])
                truth = next_visit.get(modality, [])
                for k in eval_cfg.top_ks:
                    sample_metrics[f"{modality}/recall@{k}"] = recall_at_k(preds, truth, k)
                    sample_metrics[f"{modality}/precision@{k}"] = precision_at_k(preds, truth, k)
                    sample_metrics[f"{modality}/ndcg@{k}"] = ndcg_at_k(preds, truth, k)
                sample_metrics[f"{modality}/mrr"] = mrr(preds, truth)
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
                if modality in ("dx", "med") and preds and truth:
                    pred_top = preds[0]
                    distances = [hierarchy_distance(graph, pred_top, t) for t in truth]
                    sample_metrics[f"{modality}/hier_dist"] = float(np.mean(distances))
                    sample_metrics[f"{modality}/ancestor_match@2"] = float(
                        np.mean([ancestor_match(graph, pred_top, t, max_depth=2) for t in truth])
                    )
                if modality == "dx" and preds and truth:
                    primary = truth[0]
                    for k in eval_cfg.top_ks:
                        sample_metrics[f"dx_primary/top@{k}"] = float(primary in preds[:k])
                    sample_metrics["dx_primary/mrr"] = mrr(preds, [primary])
            metrics.append(sample_metrics)
            if len(samples) < sample_limit:
                sample_entry = {
                    "subject_id": traj.subject_id,
                    "visit_index": t,
                    "history": {m: [vocab.id_to_node[c] for c in current.get(m, [])] for m in current.keys()},
                    "truth_next": {m: [vocab.id_to_node[c] for c in next_visit.get(m, [])] for m in next_visit.keys()},
                    "predictions": {
                        m: [vocab.id_to_node[c] for c in scored_candidates.get(m, [])[: max(eval_cfg.top_ks)]]
                        for m in eval_cfg.eval_modalities
                    },
                    "candidates": {
                        m: [
                            {"code": vocab.id_to_node[idx], "score": float(score)}
                            for idx, score in score_map.get(m, [])[: max(eval_cfg.top_ks)]
                        ]
                        for m in eval_cfg.eval_modalities
                    },
                }
                samples.append(sample_entry)
    if sample_path:
        with open(sample_path, "w") as f:
            json.dump(samples, f, indent=2)
    return aggregate_metrics(metrics)


def tune_thresholds_for_eval(
    model: GrailModel,
    graph,
    trajectories,
    vocab,
    eval_cfg: EvalConfig,
    rerank_cfg: RerankConfig,
    model_cfg: ModelConfig,
) -> Dict[str, float]:
    model.eval()
    reranker = LLMReranker(backend=rerank_cfg.backend, model_name=rerank_cfg.model_name)
    scores_by_modality = {m: [] for m in eval_cfg.eval_modalities}
    truth_by_modality = {m: [] for m in eval_cfg.eval_modalities}
    for traj in trajectories:
        visits = traj.visits
        if len(visits) < 2:
            continue
        for t in range(len(visits) - 1):
            current = visits[t]
            next_visit = visits[t + 1]
            all_codes = [code for codes in current.values() for code in codes]
            if not all_codes:
                continue
            code_tensor = torch.tensor([all_codes])
            barycenter = model.barycenter(code_tensor)
            dists = model.manifold.dist(barycenter, model.embeddings)
            central_code = int(torch.argmin(dists).item())
            horizon = build_risk_horizon(
                graph,
                vocab,
                model,
                barycenter.squeeze(0),
                central_code,
                max_desc_depth=3,
                max_candidates=rerank_cfg.max_candidates,
                cone_aperture=model_cfg.cone_aperture,
                cone_bonus=model_cfg.cone_bonus,
            )
            prompt_candidates = {}
            for modality, cand_list in horizon.items():
                codes = [vocab.id_to_node[idx] for idx, _ in cand_list]
                prompt_candidates[modality] = codes
            prompt = build_prompt([], prompt_candidates)
            for modality, cand_list in horizon.items():
                if not cand_list:
                    continue
                codes = [vocab.id_to_node[idx] for idx, _ in cand_list]
                llm_scores = reranker.score(prompt, codes)
                combined = []
                for (idx, geo_score), llm_score in zip(cand_list, llm_scores):
                    score = rerank_cfg.lambda_llm * llm_score + (1 - rerank_cfg.lambda_llm) * geo_score
                    combined.append((idx, score))
                scores = [s for _, s in combined]
                truth = next_visit.get(modality, [])
                truth_flags = [1 if idx in truth else 0 for idx, _ in combined]
                scores_by_modality[modality].append(scores)
                truth_by_modality[modality].append(truth_flags)
    thresholds = {}
    for modality in eval_cfg.eval_modalities:
        thresholds[modality] = tune_threshold(scores_by_modality[modality], truth_by_modality[modality])
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--format", type=str, default="pickle")
    parser.add_argument("--lab-groups", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--manifold", type=str, default="poincare")
    parser.add_argument("--epochs", type=int, default=50)
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
            "hyper_no_stitch",
            "euclid_grail",
            "euclid_rag",
        ],
    )
    parser.add_argument("--llm-backend", type=str, default="hf", choices=["hf", "openai", "heuristic", "none"])
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-sample-size", type=int, default=2000)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-stitch", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-cone", action="store_true")
    args = parser.parse_args()

    print("=== GRAIL Run Start ===")
    print(f"Data path: {args.data_path}")
    print(f"Text mode: {args.text_mode}")
    print(f"Baseline: {args.baseline}")
    print(f"LLM backend: {args.llm_backend}")
    if args.llm_model:
        print(f"LLM model: {args.llm_model}")

    # Redirect heavy caches to output path to avoid home quota.
    cache_base = "/data/horse/ws/zhqu771g-GRAIL/cache"
    os.makedirs(cache_base, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_base, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_base, "datasets"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_base, "torch"))
    print(f"Cache base: {cache_base}")

    data_cfg = DataConfig(data_path=args.data_path, format=args.format, text_mode=args.text_mode, label_map_path=args.label_map)
    graph_cfg = GraphConfig()
    model_cfg = ModelConfig(embedding_dim=args.embedding_dim, manifold=args.manifold)
    if args.no_stitch:
        graph_cfg.max_lag = 0
        graph_cfg.pmi_thresh = 1e9
    if args.no_denoise:
        model_cfg.alpha_mask = 0.0
    if args.no_cone:
        model_cfg.cone_aperture = -1.0
    train_cfg = TrainConfig(epochs=args.epochs, device=args.device)
    eval_cfg = EvalConfig()
    rerank_cfg = RerankConfig(backend=args.llm_backend, model_name=args.llm_model)
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
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.run_name:
        run_tag = f"{args.run_name}_{timestamp}"
    else:
        run_tag = timestamp
    config.run_name = f"grail_{config.data.text_mode}_{config.model.manifold}_{run_tag}"

    run_output_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    df = load_table(config.data.data_path, config.data.format)
    print(f"Loaded table: {len(df)} rows")
    print("Table sample:")
    print(df.head(2).to_string())
    if args.debug:
        counts = df.groupby(config.data.subject_id_col)[config.data.visit_id_col].nunique()
        eligible = counts[counts >= 2].index.tolist()
        if eligible:
            sample_size = min(args.debug_sample_size, len(eligible))
            sampled_ids = np.random.RandomState(13).choice(eligible, size=sample_size, replace=False)
            df = df[df[config.data.subject_id_col].isin(sampled_ids)].reset_index(drop=True)
            print(f"Debug mode: sampled {sample_size} patients with >=2 visits ({len(df)} rows)")
        else:
            print("Debug mode: no patients with >=2 visits found in sample")
    lab_to_group = load_lab_groups(args.lab_groups)
    vocab = build_vocab(
        df,
        config.data.modality_cols,
        min_code_freq=config.data.min_code_freq,
        top_k_per_modality=config.data.top_k_per_modality,
        lab_to_group=lab_to_group,
        text_mode=config.data.text_mode,
        label_map_path=config.data.label_map_path,
    )
    print(f"Vocab size: {len(vocab.node_to_id)}")
    sample_nodes = list(vocab.id_to_node.items())[:5]
    print(f"Vocab sample: {sample_nodes}")
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
    print(f"Trajectories: {len(trajectories)}")
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
    train_traj, val_traj, test_traj = split_patients(
        trajectories,
        config.data.train_frac,
        config.data.val_frac,
        config.data.seed,
    )
    print(f"Split sizes | train: {len(train_traj)} | val: {len(val_traj)} | test: {len(test_traj)}")

    if args.baseline in ("hyper_no_stitch",):
        graph_cfg.max_lag = 0
        graph_cfg.pmi_thresh = 1e9
    if args.baseline in ("euclid_grail", "euclid_rag"):
        model_cfg.manifold = "euclidean"
    if args.baseline == "euclid_rag":
        model_cfg.cone_aperture = -1.0

    graph = build_graph(
        train_traj,
        vocab,
        max_lag=config.graph.max_lag,
        pmi_thresh=config.graph.pmi_thresh,
        min_support=config.graph.min_support,
        top_m_per_node=config.graph.top_m_per_node,
        use_ppmi=config.graph.use_ppmi,
        bootstrap_samples=config.graph.bootstrap_samples,
        bootstrap_keep_q=config.graph.bootstrap_keep_q,
    )
    print(f"Graph nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")
    graph_edges = list(graph.edges(data=True))[:5]
    print(f"Graph sample edges: {graph_edges}")

    edges = graph_to_edge_list(graph)
    edge_type_ids = edge_type_map(edges)
    edge_triplets = [(u, v, edge_type_ids[t]) for u, v, t in edges]
    print(f"Edge types: {len(edge_type_ids)}")
    print(f"Edge type sample: {list(edge_type_ids.items())[:5]}")

    if args.baseline in ("grail", "hyper_no_stitch", "euclid_grail", "euclid_rag"):
        print("Training GRAIL model...")
        model = GrailModel(
            num_nodes=len(vocab.node_to_id),
            num_edge_types=len(edge_type_ids),
            embedding_dim=config.model.embedding_dim,
            curvature=config.model.curvature,
            manifold=config.model.manifold,
            margin=config.model.margin,
            softmax_beta=config.model.softmax_beta,
        )

        train_model(
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
        )

        print("Tuning thresholds on validation set...")
        thresholds = tune_thresholds_for_eval(model, graph, val_traj, vocab, config.eval, config.rerank, config.model)
        print("Evaluating GRAIL...")
        sample_path = os.path.join(run_output_dir, "sample_outputs.json")
        metrics = evaluate(
            model,
            graph,
            test_traj,
            vocab,
            config.eval,
            config.rerank,
            config.model,
            thresholds=thresholds,
            sample_path=sample_path,
        )
    else:
        print(f"Training baseline: {args.baseline} ...")
        dataset_train = NextVisitDataset(train_traj, num_nodes=len(vocab.node_to_id))
        dataset_test = NextVisitDataset(test_traj, num_nodes=len(vocab.node_to_id))
        if args.baseline == "transformer":
            model = VisitTransformer(len(vocab.node_to_id), config.model.embedding_dim)
        elif args.baseline == "retain":
            model = RetainModel(len(vocab.node_to_id), config.model.embedding_dim)
        elif args.baseline == "gram":
            ancestor_map = build_ancestor_map(len(vocab.node_to_id), edges)
            model = GramModel(len(vocab.node_to_id), config.model.embedding_dim, ancestor_map)
        elif args.baseline in ("behrt", "medbert"):
            model = VisitTransformer(len(vocab.node_to_id), config.model.embedding_dim)
        else:
            raise ValueError(f"Unknown baseline: {args.baseline}")

        if args.baseline in ("behrt", "medbert"):
            print(f"Pretraining {args.baseline} with masked visit modeling...")
            pretrain_masked_visits(
                model,
                dataset_train,
                epochs=2,
                batch_size=config.train.batch_size,
                lr=config.train.lr,
                device=config.train.device,
            )

        train_sequence_model(
            model,
            dataset_train,
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            lr=config.train.lr,
            device=config.train.device,
        )
        print("Evaluating baseline...")
        sample_path = os.path.join(run_output_dir, "sample_outputs.json")
        dataset_val = NextVisitDataset(val_traj, num_nodes=len(vocab.node_to_id))
        thresholds = tune_thresholds_sequence(model, dataset_val, vocab, config.train.device)
        metrics = evaluate_sequence_model(
            model,
            dataset_test,
            vocab,
            config.eval.top_ks,
            config.train.device,
            thresholds=thresholds,
            sample_path=sample_path,
        )
    metrics_path = os.path.join(run_output_dir, f"{args.baseline}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    for key, val in sorted(metrics.items()):
        print(f"{key}: {val:.4f}")
    print("=== GRAIL Run Complete ===")


if __name__ == "__main__":
    main()
