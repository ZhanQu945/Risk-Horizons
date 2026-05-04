import argparse
import json
import os
from typing import List

from grail.config import DataConfig
from grail.data import (
    build_patient_trajectories,
    build_vocab,
    filter_min_visits,
    load_lab_groups,
    load_table,
    split_patients,
)


def _default_split_file(data_path: str, seed: int, train_frac: float, val_frac: float, min_visits: int) -> str:
    base = os.path.dirname(os.path.abspath(data_path))
    stem = os.path.splitext(os.path.basename(data_path))[0]
    split_dir = os.path.join(base, "splits")
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, f"{stem}_seed{seed}_tr{train_frac:.2f}_val{val_frac:.2f}_mv{int(min_visits)}.json")


def _split_by_subject_ids(trajectories, payload):
    train_ids = set(map(str, payload.get("train_subject_ids", [])))
    val_ids = set(map(str, payload.get("val_subject_ids", [])))
    test_ids = set(map(str, payload.get("test_subject_ids", [])))
    train, val, test = [], [], []
    for tr in trajectories:
        sid = str(tr.subject_id)
        if sid in train_ids:
            train.append(tr)
        elif sid in val_ids:
            val.append(tr)
        elif sid in test_ids:
            test.append(tr)
    return train, val, test


def _get_or_create_split(trajectories, data_path: str, seed: int, train_frac: float, val_frac: float, min_visits: int):
    split_path = _default_split_file(data_path, seed, train_frac, val_frac, min_visits)
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            payload = json.load(f)
        train, val, test = _split_by_subject_ids(trajectories, payload)
        return split_path, train, val, test, True
    train, val, test = split_patients(trajectories, train_frac, val_frac, seed)
    payload = {
        "seed": int(seed),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "n_total": int(len(trajectories)),
        "train_subject_ids": [str(t.subject_id) for t in train],
        "val_subject_ids": [str(t.subject_id) for t in val],
        "test_subject_ids": [str(t.subject_id) for t in test],
    }
    with open(split_path, "w") as f:
        json.dump(payload, f, indent=2)
    return split_path, train, val, test, False


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare/reuse eICU split files for a set of seeds.")
    ap.add_argument("--data-path", type=str, default="./Processed/eicu/ehr_tables_full_withlab.pkl")
    ap.add_argument("--label-map", type=str, default="./Processed/eicu/ehr_label_map.json")
    ap.add_argument("--lab-groups", type=str, default="./Processed/eicu/ehr_lab_groups.csv")
    ap.add_argument("--seeds", type=str, default="13,21,42,87,123")
    ap.add_argument("--text-mode", type=str, default="text", choices=["text", "hybrid", "code"])
    ap.add_argument("--min-visits", type=int, default=2)
    ap.add_argument("--min-code-freq", type=int, default=0)
    ap.add_argument("--disable-lab-tests", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cfg = DataConfig(data_path=args.data_path, format="pickle", text_mode=args.text_mode, label_map_path=args.label_map)
    cfg.min_visits = int(args.min_visits)
    cfg.min_code_freq = int(args.min_code_freq)
    if args.disable_lab_tests and "lab" in cfg.modality_cols:
        cfg.modality_cols = {k: v for k, v in cfg.modality_cols.items() if k != "lab"}

    df = load_table(cfg.data_path, cfg.format)
    lab_to_group = None if args.disable_lab_tests else load_lab_groups(args.lab_groups)
    vocab = build_vocab(
        df,
        cfg.modality_cols,
        min_code_freq=cfg.min_code_freq,
        top_k_per_modality=cfg.top_k_per_modality,
        lab_to_group=lab_to_group,
        text_mode=cfg.text_mode,
        label_map_path=cfg.label_map_path,
    )
    trajs = build_patient_trajectories(
        df,
        cfg.subject_id_col,
        cfg.time_col,
        cfg.modality_cols,
        vocab,
        text_mode=cfg.text_mode,
        label_map_path=cfg.label_map_path,
    )
    trajs = filter_min_visits(trajs, cfg.min_visits)

    seeds: List[int] = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    for s in seeds:
        split_path, tr, va, te, reused = _get_or_create_split(
            trajs,
            data_path=cfg.data_path,
            seed=s,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            min_visits=cfg.min_visits,
        )
        flag = "reused" if reused else "created"
        print(
            f"[split] seed={s} {flag} path={split_path} "
            f"counts(train/val/test)={len(tr)}/{len(va)}/{len(te)}"
        )


if __name__ == "__main__":
    main()

