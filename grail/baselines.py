from dataclasses import dataclass
import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import PatientTrajectory, Vocab
from .eval import aggregate_metrics, micro_f1, mrr, ndcg_at_k, precision_at_k, recall_at_k, tune_threshold


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
        self.encoder = VisitEncoder(num_nodes, emb_dim)
        self.gru_alpha = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.gru_beta = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.out = nn.Linear(emb_dim, num_nodes)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        visit_emb = self.encoder(codes, mask)
        rev = torch.flip(visit_emb, dims=[1])
        alpha, _ = self.gru_alpha(rev)
        beta, _ = self.gru_beta(rev)
        alpha = torch.softmax(alpha, dim=1)
        context = (alpha * torch.tanh(beta) * rev).sum(dim=1)
        return self.out(context)


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
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch.visit_codes.to(device), batch.visit_mask.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, batch.target.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss {total / len(loader):.4f}")


def pretrain_masked_visits(
    model: nn.Module,
    dataset: NextVisitDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    mask_ratio: float = 0.3,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
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
        if epoch % 1 == 0:
            print(f"Pretrain epoch {epoch} | Loss {total / len(loader):.4f}")


def evaluate_sequence_model(
    model: nn.Module,
    dataset: NextVisitDataset,
    vocab: Vocab,
    top_ks: Tuple[int, ...],
    device: str,
    thresholds: Dict[str, float] = None,
    sample_path: str = None,
    sample_limit: int = 50,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate)
    model.eval()
    metrics = []
    samples = []
    with torch.no_grad():
        for batch in loader:
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
                    for k in top_ks:
                        sample_metrics[f"{modality}/recall@{k}"] = recall_at_k(preds, truth, k)
                        sample_metrics[f"{modality}/precision@{k}"] = precision_at_k(preds, truth, k)
                        sample_metrics[f"{modality}/ndcg@{k}"] = ndcg_at_k(preds, truth, k)
                    sample_metrics[f"{modality}/mrr"] = mrr(preds, truth)
                    if thresholds is not None:
                        scores = [float(score_vec[idx]) for idx in idxs]
                        truth_flags = [1 if idx in truth else 0 for idx in idxs]
                        th = thresholds.get(modality, 0.5)
                        sample_metrics[f"{modality}/micro_f1"] = micro_f1(scores, truth_flags, th)
                    if modality == "dx" and preds and truth:
                        primary = truth[0]
                        for k in top_ks:
                            sample_metrics[f"dx_primary/top@{k}"] = float(primary in preds[:k])
                        sample_metrics["dx_primary/mrr"] = mrr(preds, [primary])
                if len(samples) < sample_limit:
                    pred_map = {}
                    truth_map = {}
                    for mm in vocab.modality_roots.keys():
                        idxs = [idx for idx, m in vocab.node_type.items() if m == mm]
                        idxs.sort(key=lambda x: float(score_vec[x]), reverse=True)
                        pred_map[mm] = [vocab.id_to_node[c] for c in idxs[: max(top_ks)]]
                        truth_mod = batch.target[i].nonzero().view(-1).tolist()
                        truth_mod = [t for t in truth_mod if vocab.node_type[t] == mm]
                        truth_map[mm] = [vocab.id_to_node[c] for c in truth_mod]
                    samples.append(
                        {
                            "history_last_visit": [
                                vocab.id_to_node[int(c)]
                                for c, m in zip(batch.visit_codes[i, -1], batch.visit_mask[i, -1])
                                if m > 0
                            ],
                            "predictions": pred_map,
                            "truth_next": truth_map,
                        }
                    )
                metrics.append(sample_metrics)
    if sample_path:
        with open(sample_path, "w") as f:
            json.dump(samples, f, indent=2)
    return aggregate_metrics(metrics)


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
    with torch.no_grad():
        for batch in loader:
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
    thresholds = {}
    for modality in vocab.modality_roots.keys():
        thresholds[modality] = tune_threshold(scores_by_modality[modality], truth_by_modality[modality])
    return thresholds
