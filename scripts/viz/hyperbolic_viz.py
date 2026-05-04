"""
Canonical hyperbolic visualization entrypoint.

Merged behavior:
- Keeps legacy CLI flags used by existing Slurm scripts.
- Delegates plotting to scripts/visualize_hyperbolic_geometry.py clean figure suite.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Reviewer-friendly hyperbolic embedding visualization tool (merged).")
    # legacy names (kept for backward compatibility)
    ap.add_argument("--node-metadata", type=str, required=True)
    ap.add_argument("--edges", type=str, required=True)
    ap.add_argument("--embedding-full", type=str, required=True)
    ap.add_argument("--embedding-hier", type=str, required=True)
    ap.add_argument("--paths-json", type=str, default=None)
    ap.add_argument("--projection", type=str, default="pca", choices=["first2", "pca"])
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--paths-per-modality", type=int, default=3)
    ap.add_argument("--max-background-nodes", type=int, default=3000)
    ap.add_argument("--background-alpha", type=float, default=0.10)  # accepted but clean suite uses internal alpha
    ap.add_argument("--show-labels", action="store_true")  # accepted for compatibility
    # new optional controls
    ap.add_argument("--dataset", type=str, default="mimiciv", choices=["mimiciv", "eicu"])
    ap.add_argument("--color-by", type=str, default="modality", choices=["modality", "depth"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hide-axes", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exclude-roots-for-radial-corr", action="store_true")
    ap.add_argument("--include-root-edges", type=str, default="both", choices=["true", "false", "both"])
    ap.add_argument("--max-edges-for-plot", type=int, default=50000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cmd = [
        sys.executable,
        "scripts/visualize_hyperbolic_geometry.py",
        "--dataset",
        args.dataset,
        "--embedding_full",
        args.embedding_full,
        "--embedding_hierarchy_only",
        args.embedding_hier,
        "--node_metadata",
        args.node_metadata,
        "--edge_file",
        args.edges,
        "--output_dir",
        args.out_dir,
        "--curvature",
        str(args.curvature),
        "--paths_per_modality",
        str(args.paths_per_modality),
        "--max_background_nodes",
        str(args.max_background_nodes),
        "--max_edges_for_plot",
        str(args.max_edges_for_plot),
        "--seed",
        str(args.seed),
        "--projection",
        args.projection,
    ]
    if args.overwrite:
        cmd += ["--overwrite"]
    if args.exclude_roots_for_radial_corr:
        cmd += ["--exclude_roots_for_radial_corr"]
    if args.show_labels:
        cmd += ["--label_path_nodes", "all"]
    if args.debug:
        cmd += ["--debug"]

    print("Running merged visualization command:")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
