from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    data_path: str
    format: str = "pickle"
    subject_id_col: str = "subject_id"
    visit_id_col: str = "hadm_id"
    time_col: str = "admittime"
    modality_cols: Dict[str, str] = field(default_factory=lambda: {
        "dx": "icd_code",
        "proc": "pro_code",
        "med": "ndc",
        "lab": "lab_test",
    })
    min_visits: int = 2
    min_code_freq: int = 0
    top_k_per_modality: Optional[Dict[str, int]] = None
    text_mode: str = "hybrid"  # "hybrid" | "code" | "text"
    label_map_path: Optional[str] = None
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 13


@dataclass
class GraphConfig:
    max_lag: int = 1
    include_intra_visit: bool = True
    pmi_thresh: float = 0.0
    min_support: int = 3
    top_m_per_node: int = 50
    use_ppmi: bool = True
    bootstrap_samples: int = 0
    bootstrap_keep_q: float = 0.6
    include_intra_modality_pmi: bool = False
    require_joint_lag01: bool = False
    joint_lag01_min_support: int = 3


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    curvature: float = 1.0
    manifold: str = "poincare"  # or "euclidean"
    neg_samples: int = 10
    margin: float = 1.0
    alpha_mask: float = 1.0
    mask_ratio: float = 0.3
    softmax_beta: float = 5.0
    cone_aperture: float = 0.5
    cone_bonus: float = 1.0
    barycenter_heads: int = 1
    barycenter_mixture: bool = False
    use_central_attention: bool = False
    use_idf_weighting: bool = False
    use_modality_weighting: bool = False
    central_pos_decay: float = 1.0
    multi_cone_retrieval: bool = False
    persistence_bonus: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-2
    weight_decay: float = 0.0
    log_every: int = 5
    device: str = "cpu"


@dataclass
class EvalConfig:
    top_ks: Tuple[int, ...] = (5, 10, 20)
    primary_k: int = 10
    eval_modalities: Tuple[str, ...] = ("dx", "proc", "med", "lab")
    max_history_visits: int = 10


@dataclass
class RerankConfig:
    backend: str = "hf"  # "hf" | "openai" | "heuristic" | "none"
    lambda_llm: float = 0.6
    max_candidates: int = 40
    top_m_history: int = 3
    model_name: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class ExperimentConfig:
    data: DataConfig
    graph: GraphConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    rerank: RerankConfig
    output_dir: str = "./outputs"
    run_name: str = "grail"
