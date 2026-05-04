import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_EICU_ROOT = "./Data/eICU/eicu-collaborative-research-database-2.0"
DEFAULT_OUT_DIR = "./Processed/eicu"


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0, "std": 0.0, "zero_count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "zero_count": int((arr == 0).sum()),
    }


def _clean_text(raw: str) -> str:
    txt = str(raw).strip()
    if not txt or txt.lower() == "nan":
        return ""
    return " ".join(txt.split())


def _clean_code_list(values):
    if not isinstance(values, (list, tuple, np.ndarray)):
        return []
    out = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        out.append(s)
    return list(dict.fromkeys(out))


def _backup_if_exists(path: str) -> None:
    if not os.path.exists(path):
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.bak_{ts}"
    os.rename(path, bak)
    print(f"Backed up existing file: {bak}")


def _render_label(code: str, modality: str, mode: str, label_map: Dict[str, Dict[str, str]]) -> str:
    code = str(code)
    desc = label_map.get(modality, {}).get(code, "")
    if mode == "code":
        return code
    if mode == "text":
        return desc if desc else code
    return f"{desc} ({code})" if desc else code


def _render_codes(codes, modality: str, mode: str, label_map: Dict[str, Dict[str, str]]) -> List[str]:
    if not isinstance(codes, list):
        return []
    return [_render_label(c, modality, mode, label_map) for c in codes]


_ICD_RE = re.compile(r"^[A-TV-Z0-9][A-Z0-9.]{1,9}$")


def _extract_icd_first(raw: str) -> str:
    txt = _clean_text(raw)
    if not txt:
        return ""
    first = txt.split(",")[0].strip().upper()
    if _ICD_RE.fullmatch(first):
        return first
    return ""


def preprocess_eicu(
    eicu_root: str,
    output_dir: str,
    max_rows: int = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    patient = pd.read_csv(
        os.path.join(eicu_root, "patient.csv.gz"),
        usecols=[
            "patientunitstayid",
            "patienthealthsystemstayid",
            "uniquepid",
            "gender",
            "age",
            "hospitaladmitoffset",
            "unitdischargeoffset",
        ],
        nrows=max_rows,
    )

    apache = pd.read_csv(
        os.path.join(eicu_root, "apachePatientResult.csv.gz"),
        usecols=["patientunitstayid", "actualhospitalmortality", "unabridgedhosplos"],
        nrows=max_rows,
    )

    ad = patient.merge(apache, on="patientunitstayid", how="left")
    ad["subject_id"] = ad["uniquepid"].astype(str)
    ad["hadm_id"] = ad["patienthealthsystemstayid"].astype(str)
    ad["actualhospitalmortality"] = ad["actualhospitalmortality"].astype(str).str.strip().str.lower()
    ad["mortality_any"] = ad["actualhospitalmortality"].isin(["expired", "dead", "1", "true", "yes"]).astype(int)

    ad["stay_days"] = pd.to_numeric(ad["unabridgedhosplos"], errors="coerce")
    ad["stay_days"] = ad["stay_days"].fillna(pd.to_numeric(ad["unitdischargeoffset"], errors="coerce") / 1440.0)

    ad["hospitaladmitoffset"] = pd.to_numeric(ad["hospitaladmitoffset"], errors="coerce")
    hadm_time = (
        ad.groupby(["subject_id", "hadm_id"], as_index=False)["hospitaladmitoffset"]
        .min()
        .rename(columns={"hospitaladmitoffset": "hadm_offset"})
    )
    hadm_time = hadm_time.sort_values(["subject_id", "hadm_offset", "hadm_id"]).reset_index(drop=True)
    hadm_time["visit_rank"] = hadm_time.groupby("subject_id").cumcount()
    base = pd.to_datetime("2000-01-01")
    hadm_time["admittime"] = base + pd.to_timedelta(hadm_time["visit_rank"], unit="D")
    ad = ad.merge(hadm_time[["subject_id", "hadm_id", "admittime"]], on=["subject_id", "hadm_id"], how="left")

    # Medication (notebook-aligned: use drughiclseqno; keep first event per offset)
    med_raw = pd.read_csv(
        os.path.join(eicu_root, "medication.csv.gz"),
        usecols=["patientunitstayid", "drugstartoffset", "drughiclseqno", "drugname"],
        dtype={"drughiclseqno": "string", "drugname": "string"},
        nrows=max_rows,
    )
    med_raw["drughiclseqno"] = med_raw["drughiclseqno"].fillna("").str.strip()
    med_raw = med_raw[med_raw["drughiclseqno"].ne("") & med_raw["drughiclseqno"].ne("0")]
    med_raw["drugstartoffset"] = pd.to_numeric(med_raw["drugstartoffset"], errors="coerce").fillna(0).astype(int)
    med_raw = med_raw.sort_values(["patientunitstayid", "drugstartoffset"]) 
    med_raw = med_raw.groupby(["patientunitstayid", "drugstartoffset"], as_index=False).head(1)
    med_raw["ndc"] = "NDC_" + med_raw["drughiclseqno"].astype(str)
    med_text_map = (
        med_raw.assign(drugname_clean=med_raw["drugname"].map(_clean_text))
        .groupby(["ndc", "drugname_clean"]).size().reset_index(name="n")
        .sort_values(["ndc", "n"], ascending=[True, False])
        .drop_duplicates(subset=["ndc"]) 
    )
    med_text_map = dict(zip(med_text_map["ndc"], med_text_map["drugname_clean"]))
    med = med_raw.groupby("patientunitstayid", as_index=False)["ndc"].agg(list)

    # Diagnosis (ICD-only; notebook takes first ICD token)
    dx_raw = pd.read_csv(
        os.path.join(eicu_root, "diagnosis.csv.gz"),
        usecols=["patientunitstayid", "icd9code", "diagnosisstring"],
        dtype={"icd9code": "string", "diagnosisstring": "string"},
        nrows=max_rows,
    )
    dx_raw["dx_code"] = dx_raw["icd9code"].map(_extract_icd_first)
    dx_raw = dx_raw[dx_raw["dx_code"].ne("")]
    dx_raw["dx_desc"] = dx_raw["diagnosisstring"].map(_clean_text)
    dx_text_map = (
        dx_raw.groupby(["dx_code", "dx_desc"]).size().reset_index(name="n")
        .sort_values(["dx_code", "n"], ascending=[True, False])
        .drop_duplicates(subset=["dx_code"]) 
    )
    dx_text_map = dict(zip(dx_text_map["dx_code"], dx_text_map["dx_desc"]))
    dx = dx_raw.groupby("patientunitstayid", as_index=False)["dx_code"].agg(list).rename(columns={"dx_code": "icd_code"})

    # Procedure proxy from treatment with stable synthetic code
    tr_raw = pd.read_csv(
        os.path.join(eicu_root, "treatment.csv.gz"),
        usecols=["patientunitstayid", "treatmentstring"],
        dtype={"treatmentstring": "string"},
        nrows=max_rows,
    )
    tr_raw["proc_text"] = tr_raw["treatmentstring"].map(_clean_text)
    tr_raw = tr_raw[tr_raw["proc_text"].ne("")]
    tr_raw["proc_id"] = tr_raw.groupby("proc_text").ngroup()
    tr_raw["pro_code"] = "PROC_" + tr_raw["proc_id"].astype(str)
    proc_text_map_df = tr_raw[["pro_code", "proc_text"]].drop_duplicates()
    proc_text_map = dict(zip(proc_text_map_df["pro_code"], proc_text_map_df["proc_text"]))
    proc = tr_raw.groupby("patientunitstayid", as_index=False)["pro_code"].agg(list)

    # Labs (notebook-aligned binning by labname/itemid)
    lab_raw = pd.read_csv(
        os.path.join(eicu_root, "lab.csv.gz"),
        usecols=["patientunitstayid", "labname", "labresulttext"],
        dtype={"labname": "string", "labresulttext": "string"},
        nrows=max_rows,
    )
    lab_raw["labname_clean"] = lab_raw["labname"].map(_clean_text)
    lab_raw = lab_raw[lab_raw["labname_clean"].ne("")]
    lab_raw["itemid"] = lab_raw.groupby("labname_clean").ngroup()
    lab_raw = lab_raw.groupby(["patientunitstayid", "itemid"], as_index=False).head(1)

    def contains_text(vals: pd.Series) -> bool:
        for item in vals:
            try:
                float(item)
            except Exception:
                return True
        return False

    lab_raw["value_bin"] = "txt"
    for itemid in lab_raw["itemid"].unique():
        mask = lab_raw["itemid"] == itemid
        group = lab_raw.loc[mask, "labresulttext"].fillna("")
        if contains_text(group):
            lab_raw.loc[mask, "value_bin"] = group.astype(str).str.strip().replace("", "txt")
        else:
            values_numeric = pd.to_numeric(group, errors="coerce")
            if values_numeric.dropna().shape[0] < 5:
                lab_raw.loc[mask, "value_bin"] = group.astype(str).str.strip().replace("", "txt")
            else:
                binned = pd.qcut(values_numeric, q=5, labels=False, duplicates="drop")
                lab_raw.loc[mask, "value_bin"] = binned.astype(str)

    lab_raw["lab_test"] = "LAB_" + lab_raw["itemid"].astype(str) + "-" + lab_raw["value_bin"].astype(str)
    lab_text_map = (
        lab_raw.groupby(["lab_test", "labname_clean"]).size().reset_index(name="n")
        .sort_values(["lab_test", "n"], ascending=[True, False])
        .drop_duplicates(subset=["lab_test"]) 
    )
    lab_text_map = dict(zip(lab_text_map["lab_test"], lab_text_map["labname_clean"]))
    lab = lab_raw.groupby("patientunitstayid", as_index=False)["lab_test"].agg(list)

    # intersection on unit stay (same as notebook)
    med_key = med[["patientunitstayid"]].drop_duplicates()
    dx_key = dx[["patientunitstayid"]].drop_duplicates()
    proc_key = proc[["patientunitstayid"]].drop_duplicates()
    lab_key = lab[["patientunitstayid"]].drop_duplicates()
    ad_key = ad[["patientunitstayid", "subject_id", "hadm_id", "admittime", "stay_days"]].drop_duplicates()

    combined_key = med_key.merge(dx_key, on="patientunitstayid", how="inner")
    combined_key = combined_key.merge(proc_key, on="patientunitstayid", how="inner")
    combined_key = combined_key.merge(lab_key, on="patientunitstayid", how="inner")
    combined_key = combined_key.merge(ad_key[["patientunitstayid"]], on="patientunitstayid", how="inner")

    med = med.merge(combined_key, on="patientunitstayid", how="inner")
    dx = dx.merge(combined_key, on="patientunitstayid", how="inner")
    proc = proc.merge(combined_key, on="patientunitstayid", how="inner")
    lab = lab.merge(combined_key, on="patientunitstayid", how="inner")
    ad = ad.merge(combined_key, on="patientunitstayid", how="inner")

    # aggregate to encounter level (subject + health system stay)
    ad_enc = (
        ad.groupby(["subject_id", "hadm_id"], as_index=False)
        .agg(
            admittime=("admittime", "min"),
            stay_days=("stay_days", "first"),
            mortality_any=("mortality_any", "max"),
        )
        .sort_values(["subject_id", "hadm_id", "admittime"])
    )

    bridge = ad[["patientunitstayid", "subject_id", "hadm_id"]].drop_duplicates()

    dx_enc = bridge.merge(dx, on="patientunitstayid", how="inner").groupby(["subject_id", "hadm_id"], as_index=False)["icd_code"].sum()
    med_enc = bridge.merge(med, on="patientunitstayid", how="inner").groupby(["subject_id", "hadm_id"], as_index=False)["ndc"].sum()
    proc_enc = bridge.merge(proc, on="patientunitstayid", how="inner").groupby(["subject_id", "hadm_id"], as_index=False)["pro_code"].sum()
    lab_enc = bridge.merge(lab, on="patientunitstayid", how="inner").groupby(["subject_id", "hadm_id"], as_index=False)["lab_test"].sum()

    merged = ad_enc.merge(dx_enc, on=["subject_id", "hadm_id"], how="inner")
    merged = merged.merge(med_enc, on=["subject_id", "hadm_id"], how="inner")
    merged = merged.merge(proc_enc, on=["subject_id", "hadm_id"], how="inner")
    merged = merged.merge(lab_enc, on=["subject_id", "hadm_id"], how="inner")
    merged = merged.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    for col in ["icd_code", "pro_code", "ndc", "lab_test"]:
        merged[col] = merged[col].map(_clean_code_list)

    before_clean_rows = len(merged)
    empty_counts_before_drop = {col: int((merged[col].map(len) == 0).sum()) for col in ["icd_code", "pro_code", "ndc", "lab_test"]}
    mask_complete = (
        (merged["icd_code"].map(len) > 0)
        & (merged["pro_code"].map(len) > 0)
        & (merged["ndc"].map(len) > 0)
        & (merged["lab_test"].map(len) > 0)
    )
    merged = merged[mask_complete].copy()
    after_clean_rows = len(merged)

    table_path = os.path.join(output_dir, "ehr_tables_full_withlab.pkl")
    label_path = os.path.join(output_dir, "ehr_label_map.json")
    lab_group_path = os.path.join(output_dir, "ehr_lab_groups.csv")
    stats_path = os.path.join(output_dir, "preprocess_stats.json")
    sample_path = os.path.join(output_dir, "ehr_samples_50.csv")

    for p in [table_path, label_path, lab_group_path, stats_path, sample_path]:
        _backup_if_exists(p)

    merged.to_pickle(table_path)

    label_map = {"dx": {}, "proc": {}, "med": {}, "lab": {}}
    for code in pd.Series(merged["icd_code"].explode().dropna().unique()):
        label_map["dx"][str(code)] = _clean_text(dx_text_map.get(str(code), ""))
    for code in pd.Series(merged["pro_code"].explode().dropna().unique()):
        label_map["proc"][str(code)] = _clean_text(proc_text_map.get(str(code), ""))
    for code in pd.Series(merged["ndc"].explode().dropna().unique()):
        label_map["med"][str(code)] = _clean_text(med_text_map.get(str(code), ""))
    for code in pd.Series(merged["lab_test"].explode().dropna().unique()):
        label_map["lab"][str(code)] = _clean_text(lab_text_map.get(str(code), ""))

    with open(label_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # lab groups placeholder (itemid/category schema expected by loader)
    pd.DataFrame({"itemid": [], "category": []}).to_csv(lab_group_path, index=False)

    visit_counts = merged.groupby("subject_id")["hadm_id"].nunique().tolist()
    stats = {
        "dataset": "eICU",
        "rows": int(len(merged)),
        "patients": int(merged["subject_id"].nunique()),
        "visits": int(merged["hadm_id"].nunique()),
        "cleaning": {
            "rows_before_cleaning": int(before_clean_rows),
            "rows_after_cleaning": int(after_clean_rows),
            "rows_dropped_for_empty_modality": int(before_clean_rows - after_clean_rows),
            "empty_counts_before_drop": empty_counts_before_drop,
        },
        "stay_days": _summary(merged["stay_days"].tolist()),
        "visits_per_patient": _summary(visit_counts),
        "events_per_visit": {
            "dx": _summary(merged["icd_code"].map(len).tolist()),
            "proc": _summary(merged["pro_code"].map(len).tolist()),
            "med": _summary(merged["ndc"].map(len).tolist()),
            "lab": _summary(merged["lab_test"].map(len).tolist()),
        },
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # trajectory-oriented sample: multiple visits per patient
    max_rows_out = 50
    max_visits_per_patient = 5
    patient_order = (
        merged[["subject_id"]]
        .drop_duplicates()
        .sample(frac=1, random_state=13)["subject_id"]
        .tolist()
    )
    chunks = []
    rows = 0
    for pid in patient_order:
        part = merged[merged["subject_id"] == pid].sort_values("admittime").head(max_visits_per_patient)
        if part.empty:
            continue
        chunks.append(part)
        rows += len(part)
        if rows >= max_rows_out:
            break
    if chunks:
        sample = pd.concat(chunks, ignore_index=True).head(max_rows_out).copy()
    else:
        sample = merged.head(0).copy()
    sample = sample.sort_values(["subject_id", "admittime"]).copy()
    sample["patient_visit_rank"] = sample.groupby("subject_id").cumcount() + 1

    for modality, col in [("dx", "icd_code"), ("proc", "pro_code"), ("med", "ndc"), ("lab", "lab_test")]:
        sample[f"{modality}_code_view"] = sample[col].map(lambda x: _render_codes(x, modality, "code", label_map))
        sample[f"{modality}_text_view"] = sample[col].map(lambda x: _render_codes(x, modality, "text", label_map))
        sample[f"{modality}_hybrid_view"] = sample[col].map(lambda x: _render_codes(x, modality, "hybrid", label_map))

    sample.to_csv(sample_path, index=False)

    print(f"Saved: {table_path}")
    print(f"Saved: {label_path}")
    print(f"Saved: {lab_group_path}")
    print(f"Saved: {stats_path}")
    print(f"Saved: {sample_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess eICU into GRAIL visit table format.")
    parser.add_argument("--eicu-root", type=str, default=DEFAULT_EICU_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row cap per source table.")
    args = parser.parse_args()
    preprocess_eicu(args.eicu_root, args.output_dir, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
