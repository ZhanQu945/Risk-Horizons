from typing import Optional

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
        barycenter_heads: int = 1,
        barycenter_mixture: bool = False,
        use_central_attention: bool = False,
        use_idf_weighting: bool = False,
        use_modality_weighting: bool = False,
        central_pos_decay: float = 1.0,
        multi_cone_retrieval: bool = False,
        node_modality_ids: Optional[torch.Tensor] = None,
        num_modalities: int = 0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.softmax_beta = softmax_beta
        self.barycenter_heads = max(1, int(barycenter_heads))
        self.barycenter_mixture = bool(barycenter_mixture and self.barycenter_heads > 1)
        self.use_central_attention = bool(use_central_attention)
        self.use_idf_weighting = bool(use_idf_weighting)
        self.use_modality_weighting = bool(use_modality_weighting)
        self.central_pos_decay = float(central_pos_decay)
        self.multi_cone_retrieval = bool(multi_cone_retrieval)
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
        self.bary_proj = nn.Parameter(torch.eye(embedding_dim).unsqueeze(0).repeat(self.barycenter_heads, 1, 1))
        self.bary_head_logits = nn.Parameter(torch.zeros(self.barycenter_heads))
        self.attn_vec = nn.Parameter(torch.zeros(embedding_dim))
        self.modality_weight_logits = nn.Parameter(torch.zeros(max(1, int(num_modalities))))
        if node_modality_ids is None:
            node_modality_ids = torch.zeros(num_nodes, dtype=torch.long)
        self.register_buffer("node_modality_ids", node_modality_ids.long(), persistent=False)
        self.register_buffer("idf_weights", torch.ones(num_nodes, dtype=torch.float32), persistent=False)

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.manifold_type == "poincare":
            return self.manifold.dist(u, v)
        return torch.norm(u - v, dim=-1)

    def _logmap0(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.manifold, "logmap0"):
            return self.manifold.logmap0(x)
        # Euclidean: tangent at origin is identity.
        return x

    def _expmap0(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.manifold, "expmap0"):
            return self.manifold.expmap0(x)
        # Euclidean: tangent at origin is identity.
        return x

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

    def set_idf_weights(self, idf_weights: Optional[torch.Tensor]) -> None:
        if idf_weights is None:
            self.idf_weights = torch.ones_like(self.idf_weights)
            return
        if idf_weights.shape[0] != self.num_nodes:
            raise ValueError(f"idf_weights size mismatch: {idf_weights.shape[0]} vs num_nodes={self.num_nodes}")
        self.idf_weights = idf_weights.to(self.idf_weights.device, dtype=self.idf_weights.dtype)

    def _effective_weights(self, code_indices: torch.Tensor, base_weights: Optional[torch.Tensor], tangent: torch.Tensor) -> torch.Tensor:
        if base_weights is None:
            w = torch.ones_like(code_indices, dtype=tangent.dtype)
        else:
            w = base_weights.to(tangent.dtype)

        if self.use_idf_weighting:
            idf = self.idf_weights[code_indices].to(tangent.dtype)
            w = w * idf

        if self.use_modality_weighting and self.modality_weight_logits.numel() > 0:
            mids = self.node_modality_ids[code_indices].long().clamp(min=0, max=self.modality_weight_logits.numel() - 1)
            mw = torch.exp(self.modality_weight_logits[mids]).to(tangent.dtype)
            w = w * mw

        if self.central_pos_decay < 1.0:
            n = code_indices.size(1)
            pos = torch.arange(n, device=code_indices.device, dtype=tangent.dtype).view(1, n)
            decay = torch.pow(torch.tensor(self.central_pos_decay, device=code_indices.device, dtype=tangent.dtype), pos)
            w = w * decay

        if self.use_central_attention:
            logits = (tangent * self.attn_vec.view(1, 1, -1)).sum(dim=-1)
            logits = logits.masked_fill(w <= 0, -1e9)
            attn = torch.softmax(logits, dim=-1)
            w = w * attn

        return w.clamp_min(0.0)

    def _head_barycenters(self, code_indices: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        emb = self.embeddings[code_indices]
        tangent = self._logmap0(emb)
        eff_w = self._effective_weights(code_indices, weights, tangent)
        w = eff_w.unsqueeze(-1)
        tangent_h = torch.einsum("bnd,hde->bhne", tangent, self.bary_proj)
        mean_tangent_h = (tangent_h * w.unsqueeze(1)).sum(dim=2) / (w.unsqueeze(1).sum(dim=2) + 1e-8)
        return self._expmap0(mean_tangent_h)

    def head_barycenters(self, code_indices: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        return self._head_barycenters(code_indices, weights=weights)

    def barycenter(self, code_indices: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        heads = self._head_barycenters(code_indices, weights=weights)
        if self.barycenter_heads == 1:
            return heads[:, 0, :]
        if self.barycenter_mixture:
            head_w = torch.softmax(self.bary_head_logits, dim=0).view(1, self.barycenter_heads, 1)
            head_tangent = self._logmap0(heads)
            mixed_tangent = (head_tangent * head_w).sum(dim=1)
            return self._expmap0(mixed_tangent)
        return heads.mean(dim=1)

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
