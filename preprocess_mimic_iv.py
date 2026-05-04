import argparse
import os
import runpy


DEFAULT_DATA_DIR = "./Data"
DEFAULT_OUT_DIR = "./Processed/mimiciv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIMIC-IV preprocessing into GRAIL visit table format.")
    parser.add_argument("--mimic-root", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["GRAIL_MIMIC_DATA_PATH"] = args.mimic_root
    os.environ["GRAIL_PROCESSED_DIR"] = args.output_dir

    runpy.run_path("ehr_tables.py", run_name="__main__")


if __name__ == "__main__":
    main()
