import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

data_path = os.environ.get("GRAIL_MIMIC_DATA_PATH", "./Data")
output_dir = os.environ.get("GRAIL_PROCESSED_DIR", "./Processed/mimiciv")
os.makedirs(output_dir, exist_ok=True)


def _backup_if_exists(path):
    if not os.path.exists(path):
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.bak_{ts}"
    os.rename(path, bak)
    print(f"Backed up existing file: {bak}")

med_file = os.path.join(data_path, "prescriptions.csv.gz")
procedure_file = os.path.join(data_path, "procedures_icd.csv.gz")
diag_file = os.path.join(data_path, "diagnoses_icd.csv.gz")
lab_test_file = os.path.join(data_path, "labevents.csv.gz")

admission_file = os.path.join(data_path, "admissions.csv.gz")
patient_file = os.path.join(data_path, "patients.csv.gz")
# pharmacy = os.path.join(data_path, 'pharmacy.csv.gz')

drgcodes = os.path.join(data_path, 'drgcodes.csv.gz')
d_diag = os.path.join(data_path, 'd_icd_diagnoses.csv.gz')
d_proc = os.path.join(data_path, 'd_icd_procedures.csv.gz')
d_lab = os.path.join(data_path, 'd_labitems.csv.gz')

# drug code mapping files from GAMENet repo
ndc2atc_file = './Tools/Drug/ndc2atc_level4.csv' 
cid_atc = './Tools/Drug/drug-atc.csv'
ndc2rxnorm_file = './Tools/Drug/ndc2rxnorm_mapping.txt'

#use this function if the icd codes in the tables are not in the standard format, i.e., with missing decimal points
def normalize_icd_codes(df, table_name):
    def _normalize_code(row):
        code = str(row['icd_code']).strip()
        version = row['icd_version']
        
        if pd.isna(code) or code == 'nan':
            return None
        
        # ICD-9-CM Diagnosis (XXX.XX)
        if version == 9 and table_name == 'diagnoses':
            if len(code) >= 3:
                return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code[:3]
        
        # ICD-9-CM Procedure (XX.XX)
        elif version == 9 and table_name == 'procedures':
            if len(code) >= 2:
                return f"{code[:2]}.{code[2:]}" if len(code) > 2 else code[:2]
        
        # ICD-10-CM Diagnosis (XXX.XXX)
        elif version == 10 and table_name == 'diagnoses':
            if len(code) > 3:
                return f"{code[:3]}.{code[3:]}"
            else:
                return code
        
        # ICD-10-PCS Procedure (no decimals, 7 chars)
        elif version == 10 and table_name == 'procedures':
            return code
        
        return code  # Fallback (no change)
    
    df['icd_normalized'] = df.apply(_normalize_code, axis=1)
    return df


def normalize_icd_code_str(code, version, table_name):
    code = str(code).strip().upper()
    if code == "nan":
        return None
    if version == 9 and table_name == "diagnoses":
        if len(code) >= 3:
            return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code[:3]
    elif version == 9 and table_name == "procedures":
        if len(code) >= 2:
            return f"{code[:2]}.{code[2:]}" if len(code) > 2 else code[:2]
    elif version == 10 and table_name == "diagnoses":
        if len(code) > 3:
            return f"{code[:3]}.{code[3:]}"
        return code
    elif version == 10 and table_name == "procedures":
        return code
    return code


def _icd_ancestors(code):
    if not code:
        return []
    code = str(code).strip().upper()
    prefix = code.split(".")[0]
    ancestors = []
    if len(prefix) >= 1:
        ancestors.append(prefix[:1])
    if len(prefix) >= 3:
        ancestors.append(prefix[:3])
    if code not in ancestors:
        ancestors.append(code)
    return ancestors


def _proc_ancestors(code):
    if not code:
        return []
    raw = str(code).replace(".", "").strip().upper()
    ancestors = []
    if len(raw) == 7:
        for level in range(1, 8):
            ancestors.append(raw[:level])
    elif len(raw) >= 2:
        ancestors.append(raw[:2])
        ancestors.append(raw)
    else:
        ancestors.append(raw)
    return ancestors


def _load_icd_desc():
    try:
        from icdcodex import icd
    except Exception:
        return None
    return icd


def _icd_desc_lookup(icd_module, code):
    if icd_module is None:
        return None
    for attr in ("get_description", "desc", "description"):
        fn = getattr(icd_module, attr, None)
        if fn is None:
            continue
        try:
            return fn(code)
        except Exception:
            continue
    return None


def _atc_desc_map(path):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    code_col = None
    name_col = None
    for c in df.columns:
        if c.lower() in ("atc", "atc_code", "atc5", "atc4"):
            code_col = c
            break
    if code_col is None:
        for c in df.columns:
            if "atc" in c.lower() and "code" in c.lower():
                code_col = c
                break
    for c in df.columns:
        if "name" in c.lower() or "desc" in c.lower():
            name_col = c
            break
    if code_col is None or name_col is None:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        code = str(row[code_col]).strip()
        name = str(row[name_col]).strip()
        if code and code != "nan":
            mapping[code] = name
    return mapping


def process_diag():
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['seq_num'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['subject_id','hadm_id'], inplace=True)
    d_diag_pd = pd.read_csv(d_diag)
    diag_pd = pd.merge(diag_pd, d_diag_pd, on=['icd_code', 'icd_version'], how='left')
    diag_pd = normalize_icd_codes(diag_pd, "diagnoses")

    return diag_pd.reset_index(drop=True)

def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'icd_code':'category'})
    pro_pd['chartdate'] = pd.to_datetime(pro_pd['chartdate'], errors='coerce')
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['subject_id', 'hadm_id', 'chartdate', 'seq_num'], inplace=True)
    pro_pd.drop(columns=['seq_num'], inplace=True)
    pro_pd["icd_code"] = "PRO_" + pro_pd["icd_code"].astype(str)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)
    pro_pd['icd_code'] = pro_pd['icd_code'].str.replace('PRO_', '')
    d_proc_pd = pd.read_csv(d_proc)
    pro_pd = pd.merge(pro_pd, d_proc_pd, on=['icd_code', 'icd_version'], how='left')
    pro_pd = normalize_icd_codes(pro_pd, "procedures")  

    return pro_pd

def process_med():
    # med_pd = pd.read_csv(med_file, dtype={'ndc':'category'})
    med_pd = pd.read_csv(med_file, dtype={'gsn':'string', 'ndc':'category'})
    med_pd.drop(columns=['pharmacy_id', 'poe_id', 'poe_seq',
           'order_provider_id',
           'formulary_drug_cd', 'gsn', 'form_rx'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['ndc'] == '0'].index, axis=0, inplace=True)
    med_pd.ffill(inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['starttime'] = pd.to_datetime(med_pd['starttime'], format='%Y-%m-%d %H:%M:%S')    
    med_pd['stoptime'] = pd.to_datetime(med_pd['stoptime'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['subject_id', 'hadm_id', 'starttime'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)
    # def filter_first24hour_med(med_pd):
    #     med_pd_new = med_pd.drop(columns=['ndc'])
    #     med_pd_new = med_pd_new.groupby(by=['subject_id','hadm_id']).head(1).reset_index(drop=True)
    #     med_pd_new = pd.merge(med_pd_new, med_pd, on=['subject_id','hadm_id','starttime'])
    #     med_pd_new = med_pd_new.drop(columns=['starttime'])
    #     return med_pd_new
    # med_pd = filter_first24hour_med(med_pd)
    med_pd = med_pd.drop_duplicates()
    return med_pd.reset_index(drop=True)

def ndc2atc4(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['ndc'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['ndc', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4':'ndc'})
    med_pd['ndc'] = med_pd['ndc'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()    
    return med_pd.reset_index(drop=True)

def process_admission():
    ad_pd = pd.read_csv(admission_file)
    patient_pd = pd.read_csv(patient_file)
    
    ad_pd.drop(columns=['admission_type', 'admit_provider_id', 'admission_location',
       'discharge_location', 'insurance', 'language', 'marital_status', 'race',
       'hospital_expire_flag'], axis=1, inplace=True)
    patient_pd.drop(columns=['anchor_year', 'anchor_year_group'], axis=1, inplace=True)
    
    ad_pd["admittime"] = pd.to_datetime(ad_pd['admittime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    ad_pd["dischtime"] = pd.to_datetime(ad_pd['dischtime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')  # time for leaving hospital
    ad_pd["edregtime"] = pd.to_datetime(ad_pd['edregtime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    ad_pd["edouttime"] = pd.to_datetime(ad_pd['edouttime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    ad_pd = ad_pd.merge(patient_pd, on=['subject_id'], how='inner')
    ad_pd["dod"] = pd.to_datetime(ad_pd["dod"], errors='coerce')
    ad_pd["deathtime"] = pd.to_datetime(ad_pd["deathtime"], errors='coerce')
    
    # create features: age, death, number of days in this encounter, readmission (next visit)
    ad_pd["age"] = ad_pd['anchor_age']
    # ad_pd[ad_pd["age"] >= 300] = 90
    age = ad_pd["age"]
    bins = np.linspace(age.min(), age.max(), 20 + 1)
    ad_pd['age'] = pd.cut(age, bins=bins, labels=False, include_lowest=True)
    ad_pd['age'] = "AGE_" + ad_pd["age"].astype("str")
    
    ad_pd["death"] = ad_pd["dod"].notna()
    ad_pd["stay_days"] = (ad_pd["dischtime"] - ad_pd["admittime"]).dt.days
    ad_pd.sort_values(by=['subject_id', 'hadm_id', 'admittime'], inplace=True)
    ad_pd['next_visit'] = ad_pd.groupby('subject_id')['hadm_id'].shift(-1)
    ad_pd['readmission'] = ad_pd['next_visit'].notnull().astype(int)
    ad_pd.drop('next_visit', axis=1, inplace=True)
    ad_pd.drop(columns=['anchor_age'], axis=1, inplace=True)
    ad_pd.drop_duplicates(inplace=True)
    return ad_pd.reset_index(drop=True)

def process_lab_test(n_bins=5):
    lab_pd = pd.read_csv(lab_test_file)
    lab_pd = lab_pd.groupby(by=['subject_id','itemid']).head(1).reset_index(drop=True)  # only consider the first value
    lab_pd = lab_pd[lab_pd["valuenum"].notna()]
    lab_pd = lab_pd[lab_pd["hadm_id"].notna()]
    
    # lab_pd.drop(columns=['ROW_ID'], axis=1, inplace=True)
    
    def contains_text(group):
        for item in group:
            try:
                float(item)
            except ValueError:
                return True
        return False
    for itemid in lab_pd['itemid'].unique():
        group = lab_pd[lab_pd['itemid'] == itemid]['value']
        # if the lab test contains text value then directly copy the value
        if contains_text(group):
            lab_pd.loc[lab_pd['itemid'] == itemid, 'value_bin'] = group
        else:
            # value->numeric
            values_numeric = pd.to_numeric(group, errors='coerce')
            if len(values_numeric.dropna()) < n_bins:
                lab_pd.loc[lab_pd['itemid'] == itemid, 'value_bin'] = group
            else:
                # cut
    #             bins = np.linspace(values_numeric.min(), values_numeric.max(), n_bins + 1)
    #             lab_pd.loc[df['ITEMID'] == itemid, 'value_bin'] = pd.cut(values_numeric, bins=bins, labels=False, include_lowest=True)
                lab_pd.loc[lab_pd['itemid'] == itemid, 'value_bin'] = pd.qcut(values_numeric, q=n_bins, labels=False, duplicates='drop')
        
    lab_pd["itemid"] = lab_pd["itemid"].astype(str)
    lab_pd["value_bin"] = lab_pd["value_bin"].astype(str)
    lab_pd["lab_test"] = lab_pd[["itemid", "value_bin"]].apply("-".join, axis=1)
    
    lab_pd.drop(columns=['labevent_id', 'specimen_id', 'itemid',
           'order_provider_id', 'charttime', 'storetime', 'value', 'valuenum',
           'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority',
           'comments', 'value_bin'], axis=1, inplace=True)
    lab_pd.drop_duplicates(inplace=True)
    return lab_pd.reset_index(drop=True)

ad_pd = process_admission()
diag_pd = process_diag()
pro_pd = process_procedure()
med_pd = process_med()
med_pd = ndc2atc4(med_pd)
lab_pd = process_lab_test()

print("=== Raw Table Samples ===")
print("Admissions sample:")
print(ad_pd.head(2).to_string())
print("Diagnoses sample:")
print(diag_pd.head(2).to_string())
print("Procedures sample:")
print(pro_pd.head(2).to_string())
print("Medications sample:")
print(med_pd.head(2).to_string())
print("Labs sample:")
print(lab_pd.head(2).to_string())

med_pd_key = med_pd[['subject_id', 'hadm_id']].drop_duplicates()
diag_pd_key = diag_pd[['subject_id', 'hadm_id']].drop_duplicates()
pro_pd_key = pro_pd[['subject_id', 'hadm_id']].drop_duplicates()
lab_pd_key = lab_pd[['subject_id', 'hadm_id']].drop_duplicates()
ad_pd_key = ad_pd[['subject_id', 'hadm_id']].drop_duplicates()

# filter key
combined_key = med_pd_key.merge(diag_pd_key, on=['subject_id', 'hadm_id'], how='inner')
combined_key = combined_key.merge(pro_pd_key, on=['subject_id', 'hadm_id'], how='inner')
combined_key = combined_key.merge(lab_pd_key, on=['subject_id', 'hadm_id'], how='inner')
combined_key = combined_key.merge(ad_pd_key, on=['subject_id', 'hadm_id'], how='inner')
diag_pd = diag_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
med_pd = med_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
pro_pd = pro_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
lab_pd = lab_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
ad_pd = ad_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')

# flatten and merge
diag_pd = diag_pd.groupby(by=['subject_id','hadm_id'])['icd_normalized'].unique().reset_index().rename(columns={'icd_normalized':'icd_code'}) 
med_pd = med_pd.groupby(by=['subject_id', 'hadm_id']).agg({
    'ndc': lambda x: list(pd.unique(x)),
    'starttime': list,
    'stoptime': list
}).reset_index().rename(columns={'starttime': 'med_starttime', 'stoptime': 'med_stoptime'})
pro_pd = pro_pd.groupby(by=['subject_id','hadm_id']).agg({
    'icd_normalized': lambda x: list(pd.unique(x)),
    'chartdate': list
}).reset_index().rename(columns={'icd_normalized':'pro_code', 'chartdate': 'pro_chartdate'})
lab_pd = lab_pd.groupby(by=['subject_id','hadm_id'])['lab_test'].unique().reset_index()

diag_pd['icd_code'] = diag_pd['icd_code'].map(lambda x: list(x))
med_pd['ndc'] = med_pd['ndc'].map(lambda x: list(x))
pro_pd['pro_code'] = pro_pd['pro_code'].map(lambda x: list(x))
lab_pd['lab_test'] = lab_pd['lab_test'].map(lambda x: list(x))

data = diag_pd.merge(med_pd, on=['subject_id', 'hadm_id'], how='inner')
data = data.merge(pro_pd, on=['subject_id', 'hadm_id'], how='inner')
data = data.merge(lab_pd, on=['subject_id', 'hadm_id'], how='inner')
data = data.merge(ad_pd, on=['subject_id', 'hadm_id'], how='inner')
data = data.sort_values(by=['subject_id', 'admittime'])
data.drop(columns=['gender', 'age', 'death', 'readmission'], axis=1, inplace=True)

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

for col in ["icd_code", "pro_code", "ndc", "lab_test"]:
    data[col] = data[col].map(_clean_code_list)

before_clean_rows = len(data)
empty_counts_before_drop = {col: int((data[col].map(len) == 0).sum()) for col in ["icd_code", "pro_code", "ndc", "lab_test"]}
mask_complete = (
    (data["icd_code"].map(len) > 0)
    & (data["pro_code"].map(len) > 0)
    & (data["ndc"].map(len) > 0)
    & (data["lab_test"].map(len) > 0)
)
data = data[mask_complete].copy()
after_clean_rows = len(data)

table_path = os.path.join(output_dir, "ehr_tables_full_withlab.pkl")
lab_pickle_path = os.path.join(output_dir, "ehr_lab.pkl")
lab_group_path = os.path.join(output_dir, "ehr_lab_groups.csv")
label_map_path = os.path.join(output_dir, "ehr_label_map.json")
stats_path = os.path.join(output_dir, "preprocess_stats.json")
sample_path = os.path.join(output_dir, "ehr_samples_50.csv")

for p in [table_path, lab_pickle_path, lab_group_path, label_map_path, stats_path, sample_path]:
    _backup_if_exists(p)

data.to_pickle(table_path)
# ad_pd.to_pickle('./Processed/ehr_ad.pkl')
# diag_pd.to_pickle('./Processed/ehr_diag.pkl')
# pro_pd.to_pickle('./Processed/ehr_pro.pkl')
# med_pd.to_pickle('./Processed/ehr_med.pkl')
lab_pd.to_pickle(lab_pickle_path)

d_lab_pd = pd.read_csv(d_lab)
d_lab_pd["itemid"] = d_lab_pd["itemid"].astype(str)
d_lab_pd[["itemid", "category"]].drop_duplicates().to_csv(
    lab_group_path, index=False
)

# Build label map for text/hybrid vocab
icd_module = _load_icd_desc()
label_map = {"dx": {}, "proc": {}, "med": {}, "lab": {}}

# ICD diagnosis text
d_diag_pd = pd.read_csv(d_diag)
for _, row in d_diag_pd.iterrows():
    code = normalize_icd_code_str(row["icd_code"], row["icd_version"], "diagnoses")
    if not code:
        continue
    label_map["dx"][code] = str(row.get("long_title", "")).strip()

# ICD procedure text
d_proc_pd = pd.read_csv(d_proc)
for _, row in d_proc_pd.iterrows():
    code = normalize_icd_code_str(row["icd_code"], row["icd_version"], "procedures")
    if not code:
        continue
    label_map["proc"][code] = str(row.get("long_title", "")).strip()

# Add parent labels via icdcodex if available
for code in list(label_map["dx"].keys()):
    for anc in _icd_ancestors(code):
        if anc not in label_map["dx"]:
            desc = _icd_desc_lookup(icd_module, anc)
            if desc:
                label_map["dx"][anc] = str(desc).strip()
for code in list(label_map["proc"].keys()):
    for anc in _proc_ancestors(code):
        if anc not in label_map["proc"]:
            desc = _icd_desc_lookup(icd_module, anc)
            if desc:
                label_map["proc"][anc] = str(desc).strip()

# Lab text: itemid + value bin + category
d_lab_pd = pd.read_csv(d_lab)
d_lab_pd["itemid"] = d_lab_pd["itemid"].astype(str)
lab_label = dict(zip(d_lab_pd["itemid"], d_lab_pd["label"].astype(str)))
lab_cat = dict(zip(d_lab_pd["itemid"], d_lab_pd["category"].astype(str)))
lab_codes = pd.Series(lab_pd["lab_test"].explode().dropna().unique())
for lab_code in lab_codes:
    item_id, value_bin = str(lab_code).split("-", 1)
    desc = lab_label.get(item_id, item_id)
    label_map["lab"][lab_code] = f"{desc} [{value_bin}]"
    if item_id in lab_cat:
        label_map["lab"][lab_cat[item_id]] = lab_cat[item_id]

# Medication text: ATC code -> name
atc_desc = _atc_desc_map(cid_atc)
for code, name in atc_desc.items():
    label_map["med"][code] = name
    if len(code) >= 1:
        label_map["med"].setdefault(code[:1], name)
    if len(code) >= 3:
        label_map["med"].setdefault(code[:3], name)
    if len(code) >= 4:
        label_map["med"].setdefault(code[:4], name)
    if len(code) >= 5:
        label_map["med"].setdefault(code[:5], name)

with open(label_map_path, "w") as f:
    json.dump(label_map, f)

def _print_summary():
    summary_payload = {}
    print("=== Preprocessing Summary ===")
    print(f"Visits: {len(data)}")
    print(f"Patients: {data['subject_id'].nunique()}")
    print(f"Admissions: {data['hadm_id'].nunique()}")
    summary_payload["visits"] = int(len(data))
    summary_payload["patients"] = int(data["subject_id"].nunique())
    summary_payload["admissions"] = int(data["hadm_id"].nunique())
    summary_payload["cleaning"] = {
        "rows_before_cleaning": int(before_clean_rows),
        "rows_after_cleaning": int(after_clean_rows),
        "rows_dropped_for_empty_modality": int(before_clean_rows - after_clean_rows),
        "empty_counts_before_drop": empty_counts_before_drop,
    }
    print("Code counts per modality (per-visit stats):")
    def _stats(series):
        counts = series.map(len)
        return {
            "count": int(counts.shape[0]),
            "zero_count": int((counts == 0).sum()),
            "min": float(counts.min()),
            "max": float(counts.max()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
            "std": float(counts.std(ddof=0)),
            "p25": float(counts.quantile(0.25)),
            "p75": float(counts.quantile(0.75)),
            "p90": float(counts.quantile(0.90)),
        }
    summary_payload["events_per_visit"] = {}
    all_counts = pd.Series(
        data["icd_code"].map(len)
        + data["pro_code"].map(len)
        + data["ndc"].map(len)
        + data["lab_test"].map(len)
    )
    for name, col in [("dx", "icd_code"), ("proc", "pro_code"), ("med", "ndc"), ("lab", "lab_test")]:
        stats = _stats(data[col])
        summary_payload["events_per_visit"][name] = stats
        print(
            f"  {name}: min={stats['min']:.0f} max={stats['max']:.0f} "
            f"median={stats['median']:.2f} mean={stats['mean']:.2f} "
            f"std={stats['std']:.2f} zero={stats['zero_count']} "
            f"p25={stats['p25']:.2f} p75={stats['p75']:.2f} p90={stats['p90']:.2f}"
        )
    all_stats = {
        "count": int(all_counts.shape[0]),
        "zero_count": int((all_counts == 0).sum()),
        "min": float(all_counts.min()),
        "max": float(all_counts.max()),
        "median": float(all_counts.median()),
        "mean": float(all_counts.mean()),
        "std": float(all_counts.std(ddof=0)),
        "p25": float(all_counts.quantile(0.25)),
        "p75": float(all_counts.quantile(0.75)),
        "p90": float(all_counts.quantile(0.90)),
    }
    summary_payload["all_events_per_visit"] = all_stats
    print(
        "  all: "
        f"min={all_stats['min']:.0f} max={all_stats['max']:.0f} "
        f"median={all_stats['median']:.2f} mean={all_stats['mean']:.2f} "
        f"std={all_stats['std']:.2f} zero={all_stats['zero_count']} "
        f"p25={all_stats['p25']:.2f} p75={all_stats['p75']:.2f} p90={all_stats['p90']:.2f}"
    )
    visit_counts = data.groupby("subject_id")["hadm_id"].nunique()
    print("Visits per patient:")
    print(
        f"  min={visit_counts.min():.0f} max={visit_counts.max():.0f} "
        f"median={visit_counts.median():.2f} mean={visit_counts.mean():.2f} "
        f"p25={visit_counts.quantile(0.25):.2f} p75={visit_counts.quantile(0.75):.2f} "
        f"p90={visit_counts.quantile(0.90):.2f}"
    )
    summary_payload["visits_per_patient"] = {
        "count": int(visit_counts.shape[0]),
        "min": float(visit_counts.min()),
        "max": float(visit_counts.max()),
        "median": float(visit_counts.median()),
        "mean": float(visit_counts.mean()),
        "std": float(visit_counts.std(ddof=0)),
        "p25": float(visit_counts.quantile(0.25)),
        "p75": float(visit_counts.quantile(0.75)),
        "p90": float(visit_counts.quantile(0.90)),
    }
    print("Top codes per modality:")
    summary_payload["top_codes"] = {}
    for name, col in [("dx", "icd_code"), ("proc", "pro_code"), ("med", "ndc"), ("lab", "lab_test")]:
        top_codes = pd.Series([c for codes in data[col] for c in codes]).value_counts().head(5)
        summary_payload["top_codes"][name] = {str(k): int(v) for k, v in top_codes.to_dict().items()}
        print(f"  {name}: {top_codes.to_dict()}")
    print("Samples (first 3 visits):")
    print(data[['subject_id', 'hadm_id', 'admittime', 'icd_code', 'pro_code', 'ndc', 'lab_test']].head(3).to_string())
    print("Lab items sample (first 3):")
    print(lab_pd.head(3).to_string())
    print("Output files:")
    print(table_path)
    print(lab_pickle_path)
    print(lab_group_path)
    print(label_map_path)
    with open(stats_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(stats_path)

_print_summary()

max_rows = 50
max_visits_per_patient = 5
patient_order = (
    data[["subject_id"]]
    .drop_duplicates()
    .sample(frac=1, random_state=13)["subject_id"]
    .tolist()
)
chunks = []
rows = 0
for pid in patient_order:
    part = data[data["subject_id"] == pid].sort_values("admittime").head(max_visits_per_patient)
    if part.empty:
        continue
    chunks.append(part)
    rows += len(part)
    if rows >= max_rows:
        break
if chunks:
    data_sample = pd.concat(chunks, ignore_index=True).head(max_rows).copy()
else:
    data_sample = data.head(0).copy()
data_sample = data_sample.sort_values(["subject_id", "admittime"]).copy()
data_sample["patient_visit_rank"] = data_sample.groupby("subject_id").cumcount() + 1

def _render_label(code, modality, mode):
    code = str(code)
    desc = label_map.get(modality, {}).get(code, "")
    if mode == "code":
        return code
    if mode == "text":
        return desc if desc else code
    # hybrid
    return f"{desc} ({code})" if desc else code

def _render_codes(codes, modality, mode):
    if not isinstance(codes, list):
        return []
    return [_render_label(c, modality, mode) for c in codes]

for modality, col in [("dx", "icd_code"), ("proc", "pro_code"), ("med", "ndc"), ("lab", "lab_test")]:
    data_sample[f"{modality}_code_view"] = data_sample[col].map(lambda x: _render_codes(x, modality, "code"))
    data_sample[f"{modality}_text_view"] = data_sample[col].map(lambda x: _render_codes(x, modality, "text"))
    data_sample[f"{modality}_hybrid_view"] = data_sample[col].map(lambda x: _render_codes(x, modality, "hybrid"))

data_sample.to_csv(sample_path, index=False)
print(f"Wrote sample file: {sample_path}")


# ds = pd.read_pickle('./Processed/discharge_separated.pkl') 
# merged = pd.merge(data, ds, on=["subject_id", "hadm_id"],   how="inner", suffixes=("_df1", "_df2"))
# print(merged.head(10))
# print(merged.head(5).to_string())

# merged.to_pickle('./Processed/mimic_combined_withlab.pkl')
