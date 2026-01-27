import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import json

import numpy as np
import pandas as pd


@dataclass
class Vocab:
    node_to_id: Dict[str, int]
    id_to_node: Dict[int, str]
    node_type: Dict[int, str]
    modality_roots: Dict[str, int]
    lab_to_group: Optional[Dict[str, str]] = None


@dataclass
class PatientTrajectory:
    subject_id: str
    visits: List[Dict[str, List[int]]]


def load_table(path: str, fmt: str) -> pd.DataFrame:
    if fmt == "pickle":
        return pd.read_pickle(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {fmt}")


def _normalize_icd(code: str) -> str:
    code = str(code).strip().upper()
    if code == "nan":
        return ""
    if "." in code:
        return code
    if len(code) > 3:
        return f"{code[:3]}.{code[3:]}"
    return code


def _icd_ancestors(code: str) -> List[str]:
    if not code:
        return []
    code = _normalize_icd(code)
    prefix = code.split(".")[0]
    ancestors = []
    if len(prefix) >= 1:
        ancestors.append(prefix[:1])
    if len(prefix) >= 3:
        ancestors.append(prefix[:3])
    if code not in ancestors:
        ancestors.append(code)
    # Unique in order
    seen = set()
    ordered = []
    for item in ancestors:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _proc_ancestors(code: str) -> List[str]:
    code = str(code).strip().upper()
    if not code:
        return []
    raw = code.replace(".", "")
    ancestors = []
    if len(raw) == 7:
        for level in range(1, 8):
            ancestors.append(raw[:level])
    elif len(raw) >= 2:
        ancestors.append(raw[:2])
        ancestors.append(raw)
    else:
        ancestors.append(raw)
    return list(dict.fromkeys(ancestors))


def _atc_ancestors(code: str) -> List[str]:
    code = str(code)
    if not code:
        return []
    levels = [1, 3, 4, 5]
    ancestors = []
    for level in levels:
        if len(code) >= level:
            ancestors.append(code[:level])
    if code not in ancestors:
        ancestors.append(code)
    return list(dict.fromkeys(ancestors))


def _lab_ancestors(code: str, lab_to_group: Optional[Dict[str, str]]) -> List[str]:
    code = str(code).strip()
    if not code:
        return []
    ancestors = []
    item_id = code.split("-", 1)[0]
    if lab_to_group and item_id in lab_to_group:
        ancestors.append(lab_to_group[item_id])
    ancestors.append(code)
    return list(dict.fromkeys(ancestors))


def _add_nodes(
    node_to_id: Dict[str, int],
    id_to_node: Dict[int, str],
    node_type: Dict[int, str],
    codes: Iterable[str],
    modality: str,
) -> List[int]:
    ids = []
    for code in codes:
        key = f"{modality}:{code}"
        if key not in node_to_id:
            idx = len(node_to_id)
            node_to_id[key] = idx
            id_to_node[idx] = key
            node_type[idx] = modality
        ids.append(node_to_id[key])
    return ids


def _format_label(code: str, desc: Optional[str], mode: str, modality: str) -> str:
    if mode == "code":
        if modality == "lab":
            if str(code).split("-", 1)[0].isdigit():
                return f"lab_{code}"
            return f"labcat_{str(code).replace(' ', '_')}"
        return code
    if mode == "text":
        return desc or code
    # hybrid
    if desc:
        return f"{desc} ({code})"
    return code


def _load_label_map(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def build_vocab(
    df: pd.DataFrame,
    modality_cols: Dict[str, str],
    min_code_freq: int = 25,
    top_k_per_modality: Optional[Dict[str, int]] = None,
    lab_to_group: Optional[Dict[str, str]] = None,
    text_mode: str = "hybrid",
    label_map_path: Optional[str] = None,
) -> Vocab:
    node_to_id: Dict[str, int] = {}
    id_to_node: Dict[int, str] = {}
    node_type: Dict[int, str] = {}
    modality_roots: Dict[str, int] = {}

    label_map = _load_label_map(label_map_path)
    for modality, col in modality_cols.items():
        counter = Counter()
        for codes in df[col]:
            counter.update(list(codes))
        # filter
        items = [c for c, cnt in counter.items() if cnt >= min_code_freq]
        if top_k_per_modality and modality in top_k_per_modality:
            items = [c for c, _ in counter.most_common(top_k_per_modality[modality])]
        # add roots
        root_id = _add_nodes(node_to_id, id_to_node, node_type, ["ROOT"], modality)[0]
        modality_roots[modality] = root_id
        # add codes and ancestors
        for code in items:
            if modality == "dx":
                ancestors = _icd_ancestors(code)
            elif modality == "proc":
                ancestors = _proc_ancestors(code)
            elif modality == "med":
                ancestors = _atc_ancestors(code)
            elif modality == "lab":
                ancestors = _lab_ancestors(code, lab_to_group)
            else:
                ancestors = [code]
            # attach labels if available
            mapped = []
            for anc in ancestors:
                desc = label_map.get(modality, {}).get(str(anc))
                mapped.append(_format_label(str(anc), desc, text_mode, modality))
            _add_nodes(node_to_id, id_to_node, node_type, mapped, modality)

    return Vocab(
        node_to_id=node_to_id,
        id_to_node=id_to_node,
        node_type=node_type,
        modality_roots=modality_roots,
        lab_to_group=lab_to_group,
    )


def build_hierarchy_edges(vocab: Vocab) -> List[Tuple[int, int, str]]:
    edges: List[Tuple[int, int, str]] = []
    for node_id, node in vocab.id_to_node.items():
        modality, code = node.split(":", 1)
        root_id = vocab.modality_roots[modality]
        if code == "ROOT":
            continue
        if modality == "dx":
            ancestors = _icd_ancestors(code)
        elif modality == "proc":
            ancestors = _proc_ancestors(code)
        elif modality == "med":
            ancestors = _atc_ancestors(code)
        elif modality == "lab":
            ancestors = _lab_ancestors(code, vocab.lab_to_group)
        else:
            ancestors = [code]
        # ensure root and chain
        parent = root_id
        for anc in ancestors:
            anc_id = vocab.node_to_id.get(f"{modality}:{anc}")
            if anc_id is None or anc_id == root_id:
                continue
            edges.append((parent, anc_id, f"hier:{modality}"))
            parent = anc_id
    return edges


def build_patient_trajectories(
    df: pd.DataFrame,
    subject_id_col: str,
    time_col: str,
    modality_cols: Dict[str, str],
    vocab: Vocab,
    text_mode: str = "hybrid",
    label_map_path: Optional[str] = None,
) -> List[PatientTrajectory]:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([subject_id_col, time_col])
    trajectories: List[PatientTrajectory] = []
    label_map = _load_label_map(label_map_path)
    for subject_id, group in df.groupby(subject_id_col):
        visits = []
        for _, row in group.iterrows():
            visit: Dict[str, List[int]] = {}
            for modality, col in modality_cols.items():
                codes = row[col]
                if not isinstance(codes, (list, tuple, np.ndarray)):
                    codes = []
                node_ids = []
                for code in codes:
                    desc = label_map.get(modality, {}).get(str(code))
                    label = _format_label(str(code), desc, text_mode, modality)
                    key = f"{modality}:{label}"
                    if key in vocab.node_to_id:
                        node_ids.append(vocab.node_to_id[key])
                visit[modality] = node_ids
            visits.append(visit)
        trajectories.append(PatientTrajectory(subject_id=str(subject_id), visits=visits))
    return trajectories


def split_patients(
    trajectories: List[PatientTrajectory],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[PatientTrajectory], List[PatientTrajectory], List[PatientTrajectory]]:
    random.seed(seed)
    ids = list(range(len(trajectories)))
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = [trajectories[i] for i in ids[:n_train]]
    val = [trajectories[i] for i in ids[n_train:n_train + n_val]]
    test = [trajectories[i] for i in ids[n_train + n_val:]]
    return train, val, test


def filter_min_visits(trajectories: List[PatientTrajectory], min_visits: int) -> List[PatientTrajectory]:
    return [traj for traj in trajectories if len(traj.visits) >= min_visits]


def load_lab_groups(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    df = pd.read_csv(path)
    if "itemid" in df.columns and "category" in df.columns:
        df["itemid"] = df["itemid"].astype(str)
        return dict(zip(df["itemid"], df["category"]))
    return None
