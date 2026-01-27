from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


class GrailModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_edge_types: int,
        embedding_dim: int,
        curvature: float = 1.0,
        manifold: str = "poincare",
        margin: float = 1.0,
        softmax_beta: float = 5.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.softmax_beta = softmax_beta
        self.manifold_type = manifold
        if manifold == "poincare":
            self.manifold = geoopt.PoincareBall(c=curvature)
        else:
            self.manifold = geoopt.Euclidean()
        self.embeddings = geoopt.ManifoldParameter(
            torch.randn(num_nodes, embedding_dim) * 1e-3,
            manifold=self.manifold,
        )
        self.type_bias = nn.Parameter(torch.zeros(num_edge_types))

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.manifold_type == "poincare":
            return self.manifold.dist(u, v)
        return torch.norm(u - v, dim=-1)

    def score(self, u: torch.Tensor, v: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        d = self.dist(u, v)
        return -d + self.type_bias[edge_type]

    def edge_loss(
        self,
        u_idx: torch.Tensor,
        v_idx: torch.Tensor,
        edge_type: torch.Tensor,
        neg_idx: torch.Tensor,
    ) -> torch.Tensor:
        u = self.embeddings[u_idx]
        v = self.embeddings[v_idx]
        neg = self.embeddings[neg_idx]
        pos_score = self.score(u, v, edge_type)
        neg_score = self.score(u.unsqueeze(1), neg, edge_type.unsqueeze(1).expand_as(neg_idx))
        loss_pos = -F.logsigmoid(pos_score)
        loss_neg = -F.logsigmoid(-neg_score + self.margin).mean(dim=1)
        return (loss_pos + loss_neg).mean()

    def barycenter(self, code_indices: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        emb = self.embeddings[code_indices]
        tangent = self.manifold.logmap0(emb)
        if weights is None:
            weights = torch.ones_like(code_indices, dtype=tangent.dtype)
        weights = weights.unsqueeze(-1)
        mean_tangent = (tangent * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)
        return self.manifold.expmap0(mean_tangent)

    def soft_assign(
        self,
        barycenter: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        cand_emb = self.embeddings[candidate_indices]
        dists = self.manifold.dist(barycenter.unsqueeze(1), cand_emb)
        scores = -self.softmax_beta * dists
        probs = F.softmax(scores, dim=-1)
        return probs

    def masked_visit_loss(
        self,
        visit_codes: torch.Tensor,
        mask: torch.Tensor,
        candidate_indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # visit_codes: (B, N)
        # mask: (B, N) bool for masked tokens
        # valid_mask: (B, N) bool for non-padding entries
        keep = (~mask) & valid_mask
        barycenter = self.barycenter(visit_codes, weights=keep.float())
        probs = self.soft_assign(barycenter, candidate_indices)
        # map visit codes to candidate indices positions
        cand_map = {int(idx): i for i, idx in enumerate(candidate_indices.tolist())}
        target_pos = []
        for codes, m, valid in zip(visit_codes, mask, valid_mask):
            masked_codes = codes[m & valid]
            positions = [cand_map[int(code)] for code in masked_codes if int(code) in cand_map]
            target_pos.append(positions)
        losses = []
        for b, positions in enumerate(target_pos):
            if not positions:
                continue
            logp = torch.log(probs[b] + 1e-8)
            losses.append(-logp[positions].mean())
        if not losses:
            return torch.tensor(0.0, device=visit_codes.device)
        return torch.stack(losses).mean()
