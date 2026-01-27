This repository contains the official PyTorch implementation of **GRAIL**, a neuro-symbolic framework for longitudinal EHR forecasting. GRAIL leverages hyperbolic geometry to model hierarchical medical concepts and Risk Cones to retrieve structured context for downstream reasoning.

## 📂 Repository Structure

The code is modularized into the following components:

| File | Description |
| :--- | :--- |
| **`grail/`** | **Core library.** Data preprocessing, graph construction, hyperbolic modeling, retrieval, LLM reranking, and evaluation utilities. |
| **`run_experiment.py`** | **Main runner.** End-to-end training/evaluation with configurable hyperparameters and ablations. |
| **`data_gen.py`** | **Legacy simulation.** Synthetic example kept for reference. |
| **`graph_builder.py`** | **Legacy topology utility.** Simple PMI stitching reference implementation. |

## ✅ End-to-end pipeline

GRAIL is implemented end-to-end with the following stages:

1) **Data preprocessing**: load visit-level tables, filter codes, build vocab + deterministic hierarchies.  
2) **Graph construction**: add hierarchy edges and lagged cross-modal PMI edges (bootstrap optional).  
3) **Hyperbolic embeddings**: train typed edge reconstruction with negative sampling.  
4) **Central Events**: log–exp barycenters and soft assignment.  
5) **Risk Horizon**: cone-filtered retrieval across modalities.  
6) **Geometric–LLM reranking**: optional inference-time reranking.  
7) **Evaluation + ablations**: Recall/Precision/nDCG/MRR across modalities.

## 🚀 Quickstart

```bash
python run_experiment.py \\
  --data-path /data/horse/ws/zhqu771g-GRAIL/Processed/ehr_tables_full_withlab.pkl \\
  --format pickle \\
  --label-map /data/horse/ws/zhqu771g-GRAIL/Processed/ehr_label_map.json \\
  --text-mode hybrid \\
  --embedding-dim 64 \\
  --manifold poincare \\
  --epochs 50
```

The input table should contain one row per visit with at least:

```
subject_id, admittime, icd_code, pro_code, ndc, lab_test
```

Each modality column must be a list of codes for that visit.

Text/hybrid options:
- `--text-mode hybrid` (default): description + code when available
- `--text-mode code`: codes only (lab codes prefixed with `lab_`)
- `--text-mode text`: descriptions only (falls back to code if missing)

## 🧭 Recommended run order

1) **Preprocess MIMIC-IV raw tables**  
   Run `ehr_tables.py` to build visit-level tables and label maps.

2) **Train + evaluate GRAIL**  
   Run `run_experiment.py` to train embeddings and evaluate predictions.

## 📊 Outputs and metrics

Results are saved to `/data/horse/ws/zhqu771g-GRAIL/outputs` as JSON, e.g. `/data/horse/ws/zhqu771g-GRAIL/outputs/grail_hybrid_poincare_metrics.json`.
Metrics include per-modality:
- `recall@k`, `precision@k`, `ndcg@k`
- `mrr`
- `micro_f1` (threshold tuned on validation when scores available)
- hierarchy-aware diagnosis metrics: `dx/hier_dist`, `dx/ancestor_match@2`
- primary diagnosis: `dx_primary/top@k`, `dx_primary/mrr`
- grounding: `grounded_rate`, `unsupported_rate`

## 🧪 Baselines

Use `--baseline` to select baselines:
- `grail` (default)
- `transformer` (visit-token Transformer)
- `retain` (RETAIN)
- `gram` (ontology-aware GRAM-style)
- `behrt` (masked-visit pretraining + Transformer)
- `medbert` (masked-visit pretraining + Transformer)
- `hyper_no_stitch` (hyperbolic without cross-modal stitching)
- `euclid_grail` (Euclidean embeddings with GRAIL topology)
- `euclid_rag` (Euclidean retrieval + reranking)

Example:
```bash
python run_experiment.py \
  --data-path /data/horse/ws/zhqu771g-TRACE/Processed/ehr_tables_full_withlab.pkl \
  --format pickle \
  --label-map /data/horse/ws/zhqu771g-TRACE/Processed/ehr_label_map.json \
  --text-mode hybrid \
  --baseline retain
```

## 🧠 LLM reranking

Default LLM backend is `hf` with `meta-llama/Llama-3.1-8B-Instruct`.
Override with:
```bash
--llm-backend openai --llm-model gpt-4.1-mini
```

## 🔬 Ablations

Key switches (see `grail/config.py`) for ablation studies:

- **Euclidean baseline**: `ModelConfig(manifold="euclidean")`
- **No stitching**: `GraphConfig(max_lag=0, pmi_thresh=inf)` or skip `build_graph` cross edges
- **No denoising**: set `ModelConfig(alpha_mask=0.0)` to disable masked reconstruction
- **No cones**: set `ModelConfig(cone_aperture=1.0)` to accept all candidates
- **LLM reranking off**: `RerankConfig(backend="none")`

## 📌 Notes

- Lagged association statistics are computed on training patients only to prevent leakage.
- The LLM reranker is optional and defaults to a no-op for offline reproducibility.
