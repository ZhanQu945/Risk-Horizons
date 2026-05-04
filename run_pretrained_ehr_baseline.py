import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretrained BEHRT/MedBERT baseline on GRAIL data.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mimic", "eicu"])
    parser.add_argument("--model", type=str, required=True, choices=["behrt", "medbert"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-mode", type=str, default="hybrid", choices=["hybrid", "code", "text"])
    args, passthrough = parser.parse_known_args()

    base = os.environ.get("GRAIL_BASE", ".")
    if args.dataset == "mimic":
        data_path = f"{base}/Processed/mimiciv/ehr_tables_full_withlab.pkl"
        label_map = f"{base}/Processed/mimiciv/ehr_label_map.json"
        lab_groups = f"{base}/Processed/mimiciv/ehr_lab_groups.csv"
        split_default = os.environ.get("GRAIL_SPLIT_FILE_MIMIC", "")
    else:
        data_path = f"{base}/Processed/eicu/ehr_tables_full_withlab.pkl"
        label_map = f"{base}/Processed/eicu/ehr_label_map.json"
        lab_groups = f"{base}/Processed/eicu/ehr_lab_groups.csv"
        split_default = os.environ.get("GRAIL_SPLIT_FILE_EICU", "")

    split_file = args.split_file or split_default
    run_name = args.run_name or f"{args.dataset}_{args.model}_pretrained"

    cmd = [
        sys.executable,
        "-u",
        "run_experiment.py",
        "--data-path",
        data_path,
        "--format",
        "pickle",
        "--label-map",
        label_map,
        "--lab-groups",
        lab_groups,
        "--text-mode",
        args.text_mode,
        "--disable-lab-tests",
        "--baseline",
        args.model,
        "--llm-backend",
        "none",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--run-name",
        run_name,
        "--baseline-pretrained-checkpoint",
        args.checkpoint,
        "--baseline-skip-mlm-pretrain",
    ]
    if split_file:
        cmd.extend(["--split-file", split_file])
    if args.freeze_backbone:
        cmd.append("--baseline-freeze-backbone")
    cmd.extend(passthrough)

    print("Executing:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

