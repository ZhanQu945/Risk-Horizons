from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import PatientTrajectory, Vocab, node_display
from .eval import (
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


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


@dataclass
class SequenceBatch:
    visit_codes: torch.Tensor  # (B, T, C)
    visit_mask: torch.Tensor   # (B, T, C)
    target: torch.Tensor       # (B, N)


class NextVisitDataset(Dataset):
    def __init__(self, trajectories: List[PatientTrajectory], num_nodes: int, max_visits: int = 10, max_codes: int = 64):
        self.samples = []
        for traj in trajectories:
            visits = traj.visits
            for t in range(len(visits) - 1):
                hist = visits[: t + 1][-max_visits:]
                target = visits[t + 1]
                self.samples.append((hist, target))
        self.num_nodes = num_nodes
        self.max_visits = max_visits
        self.max_codes = max_codes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def collate(self, batch):
        batch_size = len(batch)
        max_vis = max(len(hist) for hist, _ in batch)
        max_codes = max(
            max((len(codes) for codes in visit.values()), default=1) for hist, _ in batch for visit in hist
        )
        max_codes = min(max_codes, self.max_codes)

        visit_codes = torch.zeros((batch_size, max_vis, max_codes), dtype=torch.long)
        visit_mask = torch.zeros((batch_size, max_vis, max_codes), dtype=torch.float)
        targets = torch.zeros((batch_size, self.num_nodes), dtype=torch.float)
        for i, (hist, target) in enumerate(batch):
            for t, visit in enumerate(hist):
                codes = []
                for vals in visit.values():
                    codes.extend(vals)
                codes = codes[:max_codes] if codes else [0]
                visit_codes[i, t, : len(codes)] = torch.tensor(codes)
                visit_mask[i, t, : len(codes)] = 1.0
            tgt_codes = []
            for vals in target.values():
                tgt_codes.extend(vals)
            if tgt_codes:
                targets[i, tgt_codes] = 1.0
        return SequenceBatch(visit_codes=visit_codes, visit_mask=visit_mask, target=targets)


class VisitEncoder(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, emb_dim)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # codes: (B, T, C), mask: (B, T, C)
        emb = self.embedding(codes)
        masked = emb * mask.unsqueeze(-1)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        visit_emb = masked.sum(dim=-2) / denom
        return visit_emb


class VisitTransformer(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, num_layers: int = 2, num_heads: int = 4, max_visits: int = 10):
        super().__init__()
        self.encoder = VisitEncoder(num_nodes, emb_dim)
        self.pos_emb = nn.Parameter(torch.randn(max_visits, emb_dim) * 0.01)
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(emb_dim, num_nodes)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        visit_emb = self.encoder(codes, mask)
        pos = self.pos_emb[: visit_emb.size(1)].unsqueeze(0)
        h = visit_emb + pos
        h = self.transformer(h)
        last = h[:, -1, :]
        return self.out(last)


class RetainModel(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        # RETAIN-style dual-RNN attention with explicit visit and variable attention.
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.gru_alpha = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.gru_beta = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.alpha_fc = nn.Linear(emb_dim, 1)
        self.beta_fc = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, num_nodes)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(codes)  # (B,T,C,D)
        masked = emb * mask.unsqueeze(-1)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        visit_emb = masked.sum(dim=-2) / denom  # (B,T,D)
        rev = torch.flip(visit_emb, dims=[1])  # reverse time
        g_alpha, _ = self.gru_alpha(rev)
        g_beta, _ = self.gru_beta(rev)
        alpha_logits = self.alpha_fc(g_alpha).squeeze(-1)  # (B,T)
        alpha = torch.softmax(alpha_logits, dim=1).unsqueeze(-1)  # (B,T,1)
        beta = torch.tanh(self.beta_fc(g_beta))  # (B,T,D)
        context = (alpha * beta * rev).sum(dim=1)
        return self.out(context)


class HybridSeqModel(nn.Module):
    """Fusion of visit-Transformer and RETAIN-style temporal heads."""

    def __init__(self, num_nodes: int, emb_dim: int, max_visits: int = 10):
        super().__init__()
        self.tf = VisitTransformer(num_nodes, emb_dim, max_visits=max_visits)
        self.ret = RetainModel(num_nodes, emb_dim)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z_tf = self.tf(codes, mask)
        z_ret = self.ret(codes, mask)
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_tf + (1.0 - alpha) * z_ret


def init_sequence_embeddings_from_hyperbolic(
    seq_model: nn.Module,
    node_embeddings: torch.Tensor,
    manifold,
    freeze_embedding: bool = False,
) -> None:
    """Initialize sequence embedding layer from hyperbolic node embeddings via tangent projection."""
    with torch.no_grad():
        src = node_embeddings.detach()
        if hasattr(manifold, "logmap0"):
            src = manifold.logmap0(src)
        src = src.detach().cpu()

    target = None
    if hasattr(seq_model, "tf") and hasattr(seq_model.tf, "encoder") and hasattr(seq_model.tf.encoder, "embedding"):
        target = seq_model.tf.encoder.embedding
    elif hasattr(seq_model, "ret") and hasattr(seq_model.ret, "embedding"):
        target = seq_model.ret.embedding
    elif hasattr(seq_model, "encoder") and hasattr(seq_model.encoder, "embedding"):
        target = seq_model.encoder.embedding
    elif hasattr(seq_model, "embedding"):
        target = seq_model.embedding
    elif hasattr(seq_model, "code_embedding"):
        target = seq_model.code_embedding
    if target is None:
        raise ValueError("Sequence model has no detectable code embedding layer.")
    if tuple(target.weight.shape) != tuple(src.shape):
        raise ValueError(
            f"Embedding shape mismatch: seq={tuple(target.weight.shape)} vs hyper={tuple(src.shape)}"
        )
    target.weight.data.copy_(src.to(dtype=target.weight.dtype))
    if freeze_embedding:
        target.weight.requires_grad = False
    # If this is HybridSeqModel, also initialize RETAIN embedding to the same values.
    if hasattr(seq_model, "ret") and hasattr(seq_model.ret, "embedding"):
        seq_model.ret.embedding.weight.data.copy_(src.to(dtype=seq_model.ret.embedding.weight.dtype))
        if freeze_embedding:
            seq_model.ret.embedding.weight.requires_grad = False


class BehrtModel(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, num_layers: int = 2, num_heads: int = 4, max_visits: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.max_visits = max_visits
        self.code_embedding = nn.Embedding(num_nodes, emb_dim)
        self.visit_pos_embedding = nn.Embedding(max_visits, emb_dim)
        self.modality_embedding = nn.Embedding(4, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(emb_dim, num_nodes)
        self.mlm_head = nn.Linear(emb_dim, num_nodes)

    def _modality_ids(self, c: int, device: torch.device) -> torch.Tensor:
        # Deterministic code-slot modality tags for flattened visit order (dx,proc,med,lab mixed).
        # We do not have token-level modality in collate, so encode a stable slot pattern.
        idx = torch.arange(c, device=device)
        return (idx % 4).long()

    def _encode_tokens(self, codes: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = codes.shape
        device = codes.device
        code_emb = self.code_embedding(codes)  # (B,T,C,D)
        pos_ids = torch.arange(t, device=device).clamp(max=self.max_visits - 1).view(1, t, 1).expand(b, t, c)
        pos_emb = self.visit_pos_embedding(pos_ids)
        mod_ids = self._modality_ids(c, device).view(1, 1, c).expand(b, t, c)
        mod_emb = self.modality_embedding(mod_ids)
        x = code_emb + pos_emb + mod_emb
        x = x.view(b, t * c, self.emb_dim)
        key_padding_mask = (mask.view(b, t * c) <= 0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B,1+T*C,D)
        cls_pad = torch.zeros((b, 1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return h, key_padding_mask

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self._encode_tokens(codes, mask)
        cls_h = h[:, 0, :]
        return self.out(cls_h)

    def mlm_logits(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self._encode_tokens(codes, mask)
        return self.mlm_head(h[:, 1:, :])  # (B,T*C,V)


class MedBertModel(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, num_layers: int = 2, num_heads: int = 4, max_visits: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.max_visits = max_visits
        self.code_embedding = nn.Embedding(num_nodes, emb_dim)
        self.visit_pos_embedding = nn.Embedding(max_visits, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(emb_dim, num_nodes)
        self.mlm_head = nn.Linear(emb_dim, num_nodes)

    def _encode_tokens(self, codes: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = codes.shape
        device = codes.device
        code_emb = self.code_embedding(codes)
        pos_ids = torch.arange(t, device=device).clamp(max=self.max_visits - 1).view(1, t, 1).expand(b, t, c)
        pos_emb = self.visit_pos_embedding(pos_ids)
        x = (code_emb + pos_emb).view(b, t * c, self.emb_dim)
        key_padding_mask = (mask.view(b, t * c) <= 0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_pad = torch.zeros((b, 1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return h, key_padding_mask

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self._encode_tokens(codes, mask)
        cls_h = h[:, 0, :]
        return self.out(cls_h)

    def mlm_logits(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self._encode_tokens(codes, mask)
        return self.mlm_head(h[:, 1:, :])


class GramModel(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, ancestor_map: torch.Tensor):
        super().__init__()
        self.ancestor_map = ancestor_map
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.attn = nn.Linear(emb_dim, 1)
        self.out = nn.Linear(emb_dim, num_nodes)

    def _code_embed(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (...)
        anc = self.ancestor_map[codes]
        anc_emb = self.embedding(anc)
        scores = self.attn(anc_emb).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return (anc_emb * weights.unsqueeze(-1)).sum(dim=-2)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # codes: (B, T, C)
        flat = codes.view(-1)
        emb = self._code_embed(flat).view(codes.size(0), codes.size(1), codes.size(2), -1)
        masked = emb * mask.unsqueeze(-1)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        visit_emb = masked.sum(dim=-2) / denom
        last = visit_emb[:, -1, :]
        return self.out(last)


def _load_halo_classes():
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "HALO_Inpatient" / "config.py"
    model_path = root / "HALO_Inpatient" / "model.py"
    if not cfg_path.exists() or not model_path.exists():
        return None, None
    spec_cfg = importlib.util.spec_from_file_location("halo_inpatient_config", str(cfg_path))
    mod_cfg = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(mod_cfg)
    spec_model = importlib.util.spec_from_file_location("halo_inpatient_model", str(model_path))
    mod_model = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(mod_model)
    return mod_model.HALOModel, mod_cfg.HALOConfig


class HaloNextVisitModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        max_visits: int = 10,
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
    ):
        super().__init__()
        HALOModel, HALOConfig = _load_halo_classes()
        if HALOModel is None:
            raise RuntimeError("HALO_Inpatient not found. Expected ./HALO_Inpatient with model.py and config.py")
        self.num_nodes = num_nodes
        self.start_idx = num_nodes
        self.end_idx = num_nodes + 1
        self.pad_idx = num_nodes + 2
        self.n_ctx = max_visits + 3
        cfg = HALOConfig(
            total_vocab_size=num_nodes + 3,
            code_vocab_size=num_nodes,
            label_vocab_size=0,
            special_vocab_size=3,
            n_positions=self.n_ctx,
            n_ctx=self.n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.halo = HALOModel(cfg)

    def _history_to_multihot(self, codes: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # codes/mask: (B,T,C)
        b, t, c = codes.shape
        device = codes.device
        visit_hot = torch.zeros((b, t, self.num_nodes), device=device)
        valid = mask > 0
        safe_codes = codes.clamp(min=0, max=self.num_nodes - 1)
        visit_hot.scatter_add_(2, safe_codes, valid.float())
        visit_hot = (visit_hot > 0).float()
        lengths = (mask.sum(dim=-1) > 0).sum(dim=-1).long().clamp(min=1, max=t)
        return visit_hot, lengths

    def _build_input(self, visit_hot: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, t, v = visit_hot.shape
        device = visit_hot.device
        x = torch.zeros((b, self.n_ctx, self.num_nodes + 3), device=device)
        x[:, :, self.pad_idx] = 1.0
        x[:, 0, self.pad_idx] = 0.0
        x[:, 0, self.start_idx] = 1.0
        for i in range(b):
            l = int(lengths[i].item())
            l = min(l, t, self.n_ctx - 3)
            x[i, 1:1 + l, self.pad_idx] = 0.0
            x[i, 1:1 + l, : self.num_nodes] = visit_hot[i, :l]
            end_pos = min(1 + l, self.n_ctx - 1)
            x[i, end_pos, self.pad_idx] = 0.0
            x[i, end_pos, self.end_idx] = 1.0
        return x

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        visit_hot, lengths = self._history_to_multihot(codes, mask)
        x = self._build_input(visit_hot, lengths)
        hidden = self.halo.transformer(x)
        logits = self.halo.ehr_head.sample(hidden, x).squeeze(1)  # (B, total_vocab)
        return logits[:, : self.num_nodes]


def build_ancestor_map(num_nodes: int, edges: List[Tuple[int, int, str]], max_anc: int = 6) -> torch.Tensor:
    parents = {i: [] for i in range(num_nodes)}
    for u, v, edge_type in edges:
        if edge_type.startswith("hier:"):
            parents[v].append(u)
    ancestor_map = torch.zeros((num_nodes, max_anc), dtype=torch.long)
    for node in range(num_nodes):
        anc = [node]
        frontier = [node]
        while frontier and len(anc) < max_anc:
            curr = frontier.pop()
            for p in parents.get(curr, []):
                if p not in anc:
                    anc.append(p)
                    frontier.append(p)
                if len(anc) >= max_anc:
                    break
        anc += [node] * (max_anc - len(anc))
        ancestor_map[node] = torch.tensor(anc)
    return ancestor_map


def train_sequence_model(
    model: nn.Module,
    dataset: NextVisitDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    _log(
        f"train_sequence_model start | epochs={epochs} batch_size={batch_size} steps_per_epoch={len(loader)}"
    )
    wall_start = time.perf_counter()
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        total = 0.0
        steps = len(loader)
        batch_log_every = max(1, min(100, steps // 10 if steps >= 10 else 1))
        _log(f"train_sequence_model epoch {epoch}/{epochs} start | steps={steps} | batch_log_every={batch_log_every}")
        for step_idx, batch in enumerate(loader, 1):
            optimizer.zero_grad()
            logits = model(batch.visit_codes.to(device), batch.visit_mask.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, batch.target.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            if step_idx % batch_log_every == 0 or step_idx == steps:
                _log(
                    f"train_sequence_model epoch {epoch}/{epochs} progress "
                    f"{step_idx}/{steps} | running_loss={total / max(step_idx, 1):.4f}"
                )
        mean_loss = total / max(len(loader), 1)
        epoch_losses.append(mean_loss)
        ep_sec = time.perf_counter() - ep_start
        elapsed = time.perf_counter() - wall_start
        eta = (elapsed / max(epoch, 1)) * (epochs - epoch)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            _log(
                f"Epoch {epoch}/{epochs} | Loss {mean_loss:.4f} | "
                f"epoch {ep_sec:.1f}s | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
            )
    wall_sec = time.perf_counter() - wall_start
    _log(f"train_sequence_model done | wall_time_sec={wall_sec:.2f}")
    return {
        "epochs": float(epochs),
        "wall_time_sec": float(wall_sec),
        "steps_per_epoch": float(len(loader)),
        "final_loss": float(epoch_losses[-1] if epoch_losses else 0.0),
    }


def pretrain_masked_visits(
    model: nn.Module,
    dataset: NextVisitDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    mask_ratio: float = 0.3,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    _log(
        f"pretrain_masked_visits start | epochs={epochs} batch_size={batch_size} steps_per_epoch={len(loader)}"
    )
    wall_start = time.perf_counter()
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        total = 0.0
        steps = len(loader)
        batch_log_every = max(1, min(100, steps // 10 if steps >= 10 else 1))
        _log(f"pretrain_masked_visits epoch {epoch}/{epochs} start | steps={steps} | batch_log_every={batch_log_every}")
        for step_idx, batch in enumerate(loader, 1):
            codes = batch.visit_codes.to(device)
            mask = batch.visit_mask.to(device)
            rand = torch.rand_like(mask)
            masked = (rand < mask_ratio) & (mask > 0)
            # zero out masked positions
            codes = codes.masked_fill(masked.bool(), 0)
            optimizer.zero_grad()
            logits = model(codes, mask)
            # predict masked codes across the vocabulary
            targets = torch.zeros_like(logits)
            for i in range(batch.visit_codes.size(0)):
                masked_codes = batch.visit_codes[i][masked[i].bool()].view(-1).unique()
                if masked_codes.numel() > 0:
                    targets[i, masked_codes] = 1.0
            loss = F.binary_cross_entropy_with_logits(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            if step_idx % batch_log_every == 0 or step_idx == steps:
                _log(
                    f"pretrain_masked_visits epoch {epoch}/{epochs} progress "
                    f"{step_idx}/{steps} | running_loss={total / max(step_idx, 1):.4f}"
                )
        mean_loss = total / max(len(loader), 1)
        epoch_losses.append(mean_loss)
        ep_sec = time.perf_counter() - ep_start
        elapsed = time.perf_counter() - wall_start
        eta = (elapsed / max(epoch, 1)) * (epochs - epoch)
        _log(
            f"Pretrain epoch {epoch}/{epochs} | Loss {mean_loss:.4f} | "
            f"epoch {ep_sec:.1f}s | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
        )
    wall_sec = time.perf_counter() - wall_start
    _log(f"pretrain_masked_visits done | wall_time_sec={wall_sec:.2f}")
    return {
        "epochs": float(epochs),
        "wall_time_sec": float(wall_sec),
        "steps_per_epoch": float(len(loader)),
        "final_loss": float(epoch_losses[-1] if epoch_losses else 0.0),
    }


def pretrain_code_mlm(
    model: nn.Module,
    dataset: NextVisitDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    mask_ratio: float = 0.15,
) -> Dict[str, float]:
    if not hasattr(model, "mlm_logits"):
        raise ValueError("Model does not implement mlm_logits for code-level MLM pretraining.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    _log(f"pretrain_code_mlm start | epochs={epochs} batch_size={batch_size} steps_per_epoch={len(loader)}")
    wall_start = time.perf_counter()
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            codes = batch.visit_codes.to(device)
            valid = batch.visit_mask.to(device) > 0
            rand = torch.rand_like(batch.visit_mask.to(device))
            masked = (rand < mask_ratio) & valid

            input_codes = codes.clone()
            input_codes[masked] = 0
            logits = model.mlm_logits(input_codes, batch.visit_mask.to(device))  # (B,T*C,V)
            target = codes.view(codes.size(0), -1)
            masked_flat = masked.view(masked.size(0), -1)

            loss_all = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction="none",
            ).view(target.size(0), target.size(1))
            loss = (loss_all * masked_flat.float()).sum() / masked_flat.float().sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        epoch_losses.append(total / max(len(loader), 1))
        _log(f"pretrain_code_mlm epoch {epoch}/{epochs} | loss={epoch_losses[-1]:.4f}")
    wall_sec = time.perf_counter() - wall_start
    _log(f"pretrain_code_mlm done | wall_time_sec={wall_sec:.2f}")
    return {
        "epochs": float(epochs),
        "wall_time_sec": float(wall_sec),
        "steps_per_epoch": float(len(loader)),
        "final_loss": float(epoch_losses[-1] if epoch_losses else 0.0),
    }


def evaluate_sequence_model(
    model: nn.Module,
    dataset: NextVisitDataset,
    vocab: Vocab,
    top_ks: Tuple[int, ...],
    device: str,
    thresholds: Dict[str, float] = None,
    sample_path: str = None,
    sample_limit: int = 50,
    graph: Optional[object] = None,
    train_seen_by_modality: Optional[Dict[str, set]] = None,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate)
    model.eval()
    metrics = []
    samples = []
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
        for m in vocab.modality_roots.keys()
    }
    if train_seen_by_modality is None:
        train_seen_by_modality = {m: set() for m in vocab.modality_roots.keys()}
    # Rare-event diagnostics: correlate per-sample performance with truth frequency.
    # Frequency is measured on the evaluation split to keep the baseline path self-contained.
    truth_freq_by_modality = {m: {} for m in vocab.modality_roots.keys()}
    for _, target in dataset.samples:
        for m in vocab.modality_roots.keys():
            for c in target.get(m, []):
                cid = int(c)
                truth_freq_by_modality[m][cid] = int(truth_freq_by_modality[m].get(cid, 0)) + 1
    freq_corr_rows = {m: {"x": [], "y_recall": [], "y_mrr": []} for m in vocab.modality_roots.keys()}
    _log(f"evaluate_sequence_model start | batches={len(loader)}")
    with torch.no_grad():
        for bidx, batch in enumerate(loader, 1):
            logits = model(batch.visit_codes.to(device), batch.visit_mask.to(device))
            scores = torch.sigmoid(logits).cpu()
            for i in range(scores.size(0)):
                sample_metrics = {}
                score_vec = scores[i]
                for modality in vocab.modality_roots.keys():
                    idxs = [idx for idx, m in vocab.node_type.items() if m == modality]
                    preds = idxs
                    preds.sort(key=lambda x: float(score_vec[x]), reverse=True)
                    truth = batch.target[i].nonzero().view(-1).tolist()
                    truth = [t for t in truth if vocab.node_type[t] == modality]
                    curr_codes = [
                        int(c)
                        for c, msk in zip(batch.visit_codes[i, -1], batch.visit_mask[i, -1])
                        if msk > 0 and vocab.node_type.get(int(c)) == modality
                    ]
                    curr_set = set(curr_codes)
                    truth_set = set(int(t) for t in truth)
                    pers_truth = sorted(list(truth_set & curr_set))
                    novel_truth = sorted(list(truth_set - curr_set))
                    seen_set = train_seen_by_modality.get(modality, set())
                    seen_truth = sorted(list(truth_set & seen_set))
                    unseen_truth = sorted(list(truth_set - seen_set))
                    novel_seen_truth = sorted(list(set(novel_truth) & seen_set))
                    novel_unseen_truth = sorted(list(set(novel_truth) - seen_set))
                    pers_seen_truth = sorted(list(set(pers_truth) & seen_set))
                    pers_unseen_truth = sorted(list(set(pers_truth) - seen_set))
                    for k in top_ks:
                        sample_metrics[f"{modality}/recall@{k}"] = recall_at_k(preds, truth, k)
                        sample_metrics[f"{modality}/precision@{k}"] = precision_at_k(preds, truth, k)
                        sample_metrics[f"{modality}/ndcg@{k}"] = ndcg_at_k(preds, truth, k)
                    sample_metrics[f"{modality}/mrr"] = mrr(preds, truth)
                    if graph is not None and preds and truth:
                        pred_top = preds[0]
                        dists = [hierarchy_distance(graph, pred_top, int(t)) for t in truth]
                        if dists:
                            sample_metrics[f"{modality}/tree_distance"] = float(np.mean(dists))
                        sample_metrics[f"{modality}/ancestor_match@2"] = float(
                            np.mean([ancestor_match(graph, pred_top, int(t), max_depth=2) for t in truth])
                        )
                    if thresholds is not None:
                        score_vals = [float(score_vec[idx]) for idx in idxs]
                        truth_flags = [1 if idx in truth else 0 for idx in idxs]
                        th = thresholds.get(modality, 0.5)
                        sample_metrics[f"{modality}/micro_f1"] = micro_f1(score_vals, truth_flags, th)
                    if modality == "dx" and preds and truth:
                        primary = truth[0]
                        for k in top_ks:
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
                    if truth:
                        tfreq = [truth_freq_by_modality[modality].get(int(t), 0) for t in truth]
                        x = float(np.log1p(np.mean(tfreq))) if tfreq else 0.0
                        freq_corr_rows[modality]["x"].append(x)
                        freq_corr_rows[modality]["y_recall"].append(float(recall_at_k(preds, truth, 10)))
                        freq_corr_rows[modality]["y_mrr"].append(float(mrr(preds, truth)))
                if len(samples) < sample_limit:
                    pred_map = {}
                    truth_map = {}
                    for mm in vocab.modality_roots.keys():
                        idxs = [idx for idx, m in vocab.node_type.items() if m == mm]
                        idxs.sort(key=lambda x: float(score_vec[x]), reverse=True)
                        pred_map[mm] = [node_display(vocab, c) for c in idxs[: max(top_ks)]]
                        truth_mod = batch.target[i].nonzero().view(-1).tolist()
                        truth_mod = [t for t in truth_mod if vocab.node_type[t] == mm]
                        truth_map[mm] = [node_display(vocab, c) for c in truth_mod]
                    samples.append(
                        {
                            "history_last_visit": [
                                node_display(vocab, int(c))
                                for c, m in zip(batch.visit_codes[i, -1], batch.visit_mask[i, -1])
                                if m > 0
                            ],
                            "predictions": pred_map,
                            "truth_next": truth_map,
                        }
                    )
                metrics.append(sample_metrics)
            if bidx % 10 == 0 or bidx == len(loader):
                _log(f"evaluate_sequence_model progress {bidx}/{len(loader)}")
    if sample_path:
        with open(sample_path, "w") as f:
            json.dump(samples, f, indent=2)
    _log("evaluate_sequence_model done")
    out = aggregate_metrics(metrics)
    def _safe_corr(x: List[float], y: List[float]) -> float:
        if len(x) < 3 or len(y) < 3:
            return float("nan")
        xv = np.asarray(x, dtype=float)
        yv = np.asarray(y, dtype=float)
        if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
            return float("nan")
        return float(np.corrcoef(xv, yv)[0, 1])

    for m in vocab.modality_roots.keys():
        for key, vals in split_stats[m].items():
            out[f"{m}/{key}"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{m}/{key}_n"] = float(len(vals))
        x = freq_corr_rows[m]["x"]
        y_r = freq_corr_rows[m]["y_recall"]
        y_m = freq_corr_rows[m]["y_mrr"]
        out[f"{m}/freq_corr_truthfreq_recall@10"] = _safe_corr(x, y_r)
        out[f"{m}/freq_corr_truthfreq_mrr"] = _safe_corr(x, y_m)
    for metric_name in (
        "persistent_recall@5",
        "persistent_recall@10",
        "novel_recall@5",
        "novel_recall@10",
        "persistent_hit_rate",
        "novel_hit_rate",
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
    ):
        vals = []
        for m in vocab.modality_roots.keys():
            k = f"{m}/{metric_name}"
            if k in out and np.isfinite(out[k]):
                vals.append(float(out[k]))
        out[f"overall/{metric_name}_macro"] = float(np.mean(vals)) if vals else float("nan")
    for metric_name in ("freq_corr_truthfreq_recall@10", "freq_corr_truthfreq_mrr"):
        vals = []
        for m in vocab.modality_roots.keys():
            k = f"{m}/{metric_name}"
            if k in out and np.isfinite(out[k]):
                vals.append(float(out[k]))
        out[f"overall/{metric_name}_macro"] = float(np.mean(vals)) if vals else float("nan")
    return out


def tune_thresholds_sequence(
    model: nn.Module,
    dataset: NextVisitDataset,
    vocab: Vocab,
    device: str,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate)
    model.eval()
    scores_by_modality = {m: [] for m in vocab.modality_roots.keys()}
    truth_by_modality = {m: [] for m in vocab.modality_roots.keys()}
    _log(f"tune_thresholds_sequence start | batches={len(loader)}")
    with torch.no_grad():
        for bidx, batch in enumerate(loader, 1):
            logits = model(batch.visit_codes.to(device), batch.visit_mask.to(device))
            scores = torch.sigmoid(logits).cpu()
            for i in range(scores.size(0)):
                score_vec = scores[i]
                truth = batch.target[i].nonzero().view(-1).tolist()
                for modality in vocab.modality_roots.keys():
                    idxs = [idx for idx, m in vocab.node_type.items() if m == modality]
                    truth_mod = [t for t in truth if vocab.node_type[t] == modality]
                    score_list = [float(score_vec[idx]) for idx in idxs]
                    truth_flags = [1 if idx in truth_mod else 0 for idx in idxs]
                    scores_by_modality[modality].append(score_list)
                    truth_by_modality[modality].append(truth_flags)
            if bidx % 10 == 0 or bidx == len(loader):
                _log(f"tune_thresholds_sequence progress {bidx}/{len(loader)}")
    thresholds = {}
    for modality in vocab.modality_roots.keys():
        thresholds[modality] = tune_threshold(scores_by_modality[modality], truth_by_modality[modality])
    _log("tune_thresholds_sequence done")
    return thresholds
