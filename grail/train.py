from typing import Callable, Dict, List, Optional, Tuple
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import geoopt

from .data import PatientTrajectory
from .model import GrailModel


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


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
    val_score_fn: Optional[Callable[[GrailModel, int], float]] = None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
) -> Dict[str, float]:
    edge_loader = DataLoader(EdgeDataset(edges), batch_size=batch_size, shuffle=True, collate_fn=_collate_edges)
    visit_loader = DataLoader(VisitDataset(trajectories), batch_size=batch_size, shuffle=True, collate_fn=_collate_visits)
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    _log(
        f"train_model start | epochs={epochs} batch_size={batch_size} edge_batches={len(edge_loader)} "
        f"visit_batches={len(visit_loader)}"
    )

    wall_start = time.perf_counter()
    epoch_losses = []
    best_state = None
    best_epoch = 0
    best_val_score = float("-inf")
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        edge_iter = iter(edge_loader)
        visit_iter = iter(visit_loader)
        steps = max(len(edge_loader), len(visit_loader))
        total_loss = 0.0
        batch_log_every = max(1, min(100, steps // 10 if steps >= 10 else 1))
        _log(f"train_model epoch {epoch}/{epochs} start | steps={steps} | batch_log_every={batch_log_every}")
        for step_idx in range(steps):
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
            if (step_idx + 1) % batch_log_every == 0 or (step_idx + 1) == steps:
                _log(
                    f"train_model epoch {epoch}/{epochs} progress "
                    f"{step_idx + 1}/{steps} | running_loss={total_loss / max(step_idx + 1, 1):.4f}"
                )
        mean_loss = total_loss / max(steps, 1)
        epoch_losses.append(mean_loss)
        ep_sec = time.perf_counter() - ep_start
        elapsed = time.perf_counter() - wall_start
        eta = (elapsed / max(epoch, 1)) * (epochs - epoch)
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            _log(
                f"Epoch {epoch}/{epochs} | Loss {mean_loss:.4f} | "
                f"epoch {ep_sec:.1f}s | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
            )
        if val_score_fn is not None:
            score = float(val_score_fn(model, epoch))
            _log(f"Validation monitor | epoch={epoch} | score={score:.6f}")
            if score > (best_val_score + float(early_stop_min_delta)):
                best_val_score = score
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _log(f"Validation improved -> best epoch {best_epoch} score={best_val_score:.6f}")
            else:
                bad_epochs += 1
                _log(f"Validation did not improve | bad_epochs={bad_epochs}/{max(early_stop_patience, 0)}")
                if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
                    _log(f"Early stopping triggered at epoch {epoch}.")
                    break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        _log(f"Restored best model from epoch {best_epoch} (val_score={best_val_score:.6f}).")
    wall_sec = time.perf_counter() - wall_start
    _log(f"train_model done | wall_time_sec={wall_sec:.2f}")
    out = {
        "epochs": float(epochs),
        "wall_time_sec": float(wall_sec),
        "steps_per_epoch": float(max(len(edge_loader), len(visit_loader))),
        "final_loss": float(epoch_losses[-1] if epoch_losses else 0.0),
    }
    if best_state is not None:
        out["best_epoch"] = float(best_epoch)
        out["best_val_score"] = float(best_val_score)
    return out
