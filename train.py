from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import geoopt

from .data import PatientTrajectory
from .model import GrailModel


class EdgeDataset(Dataset):
    def __init__(self, edges: List[Tuple[int, int, int]]):
        self.edges = edges

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        return self.edges[idx]


class VisitDataset(Dataset):
    def __init__(self, trajectories: List[PatientTrajectory], max_codes: int = 128):
        self.visits = []
        for traj in trajectories:
            for visit in traj.visits:
                codes = []
                for vals in visit.values():
                    codes.extend(vals)
                if not codes:
                    continue
                codes = codes[:max_codes]
                self.visits.append(codes)

    def __len__(self) -> int:
        return len(self.visits)

    def __getitem__(self, idx: int) -> List[int]:
        return self.visits[idx]


def _collate_edges(batch):
    u, v, t = zip(*batch)
    return torch.tensor(u), torch.tensor(v), torch.tensor(t)


def _collate_visits(batch):
    max_len = max(len(v) for v in batch)
    padded = []
    mask = []
    for visit in batch:
        pad = visit + [visit[-1]] * (max_len - len(visit))
        padded.append(pad)
        mask.append([1] * len(visit) + [0] * (max_len - len(visit)))
    return torch.tensor(padded), torch.tensor(mask)


def train_model(
    model: GrailModel,
    edges: List[Tuple[int, int, int]],
    trajectories: List[PatientTrajectory],
    num_nodes: int,
    neg_samples: int,
    alpha_mask: float,
    mask_ratio: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    log_every: int,
) -> None:
    edge_loader = DataLoader(EdgeDataset(edges), batch_size=batch_size, shuffle=True, collate_fn=_collate_edges)
    visit_loader = DataLoader(VisitDataset(trajectories), batch_size=batch_size, shuffle=True, collate_fn=_collate_visits)
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        edge_iter = iter(edge_loader)
        visit_iter = iter(visit_loader)
        steps = max(len(edge_loader), len(visit_loader))
        total_loss = 0.0
        for _ in range(steps):
            optimizer.zero_grad()
            loss = 0.0
            try:
                u, v, t = next(edge_iter)
                u = u.to(device)
                v = v.to(device)
                t = t.to(device)
                neg = torch.randint(0, num_nodes, (u.size(0), neg_samples), device=device)
                loss = loss + model.edge_loss(u, v, t, neg)
            except StopIteration:
                pass

            try:
                visits, keep_mask = next(visit_iter)
                visits = visits.to(device)
                keep_mask = keep_mask.to(device)
                rand = torch.rand_like(keep_mask.float())
                mask = (rand < mask_ratio) & (keep_mask == 1)
                # candidate pool is the unique codes in batch
                candidates = torch.unique(visits)
                loss = loss + alpha_mask * model.masked_visit_loss(visits, mask, candidates, keep_mask == 1)
            except StopIteration:
                pass

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if epoch % log_every == 0:
            print(f"Epoch {epoch} | Loss {total_loss / steps:.4f}")
