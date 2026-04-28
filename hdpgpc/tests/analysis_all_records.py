#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_analysis_all_records.py

Wrapper to run analysis_one_record.run_one_record over all MIT-BIH records, with optional MDS
and optional warp computation/plotting disable.

Examples
--------
# Run all records, MDS off, warps ON (default)
python run_analysis_all_records.py --pred_dir results/cluster_labels/v2_UCR_ver --out_dir results/eval_all

# Enable MDS
python run_analysis_all_records.py --mds

# Disable warps (no warp computation and no warp plots)
python run_analysis_all_records.py --warp_off

# Subset of records + MDS + warps off
python run_analysis_all_records.py --records 100 101 102 --mds --warp_off
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path


def _list_records(data_dir: Path):
    """Discover records as '*.npy' excluding '*_labels.npy' (same style as your scripts)."""
    recs = []
    for f in sorted(data_dir.glob("*.npy")):
        if f.name.endswith("_labels.npy"):
            continue
        if "labels" in f.stem:
            continue
        rec = f.stem
        if (data_dir / f"{rec}_labels.npy").exists():
            recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="*", default=None,
                    help="Explicit record IDs (e.g. 100 101 102). If omitted, runs all.")
    ap.add_argument("--pred_dir", type=str, default=None,
                    help="Directory containing cluster_labels_<rec>_offline.npy etc.")
    ap.add_argument("--out_dir", type=str, default="results/eval_final_ver",
                    help="Output directory root.")
    ap.add_argument("--label_map_json", type=str, default=None,
                    help="Optional JSON list mapping integer label->symbol.")
    ap.add_argument("--max_warps_per_cluster", type=int, default=30,
                    help="Max beats per cluster for warp plotting (ignored if --warp_off).")
    ap.add_argument("--warp_lead", type=int, default=0,
                    help="Which lead to use for warp plots (ignored if --warp_off).")
    ap.add_argument("--mapping", type=str, default="majority",
                    choices=["majority", "hungarian"],
                    help="Cluster->label mapping for confusion/macro-F1.")
    ap.add_argument("--mds", action="store_true",
                    help="Compute MDS region plot (slow). If not set, MDS is skipped.")
    ap.add_argument("--warp_off", action="store_true",
                    help="Disable ANY warp computation and plotting for each record.")
    args = ap.parse_args()

    # Ensure we can import analysis_one_record if this wrapper is placed next to it
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    import analysis_one_record as one  # your script

    repo_root = one.find_project_root()
    data_dir = one.find_data_dir(repo_root)

    # pred_dir default: follow the same spirit as analysis_one_record.py
    if args.pred_dir is not None:
        pred_dir = Path(args.pred_dir)
    else:
        cand = repo_root / "results" / "cluster_labels" / "final_ver"
        pred_dir = cand if cand.exists() else (repo_root / "results" / "cluster_labels")

    if not pred_dir.exists():
        raise FileNotFoundError(f"Could not find pred_dir at: {pred_dir}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    label_map_json = Path(args.label_map_json) if args.label_map_json else None

    # ---- Toggle MDS without editing analysis_one_record.py
    # run_one_record() calls the MDS plot function unconditionally in your setup,
    # so replace it with a no-op when --mds is not set.
    if not args.mds:
        if hasattr(one, "plot_MDS_regions_transitions"):
            one.plot_MDS_regions_transitions = lambda *a, **k: None

    # ---- Toggle warps without editing analysis_one_record.py
    # run_one_record() calls warp plotting unconditionally (plot_warps_plotly_per_cluster
    # in the newer version, compute_and_plot_warps in older variants).
    if args.warp_off:
        if hasattr(one, "plot_warps_plotly_per_cluster"):
            one.plot_warps_plotly_per_cluster = lambda *a, **k: None
        if hasattr(one, "compute_and_plot_warps"):
            one.compute_and_plot_warps = lambda *a, **k: None

    records = args.records if (args.records and len(args.records) > 0) else _list_records(data_dir)
    if not records:
        raise RuntimeError(f"No records found in {data_dir}")

    print(f"[INFO] data_dir:  {data_dir}")
    print(f"[INFO] pred_dir:  {pred_dir}")
    print(f"[INFO] out_dir:   {out_root}")
    print(f"[INFO] records:   {len(records)}")
    print(f"[INFO] MDS:       {'ON' if args.mds else 'OFF'}")
    print(f"[INFO] Warps:     {'OFF' if args.warp_off else 'ON'}")

    failures = []

    for i, rec in enumerate(records, 1):
        rec_out = out_root / f"Rec{rec}"
        try:
            print(f"\n[{i}/{len(records)}] Processing record {rec} ...")
            one.run_one_record(
                rec=str(rec),
                pred_dir=pred_dir,
                out_dir=rec_out,
                label_map_json=label_map_json,
                max_warps_per_cluster=int(args.max_warps_per_cluster),
                warp_lead=int(args.warp_lead),
                mapping=str(args.mapping),
            )
        except Exception as e:
            tb = traceback.format_exc()
            rec_out.mkdir(parents=True, exist_ok=True)
            (rec_out / "FAILED.txt").write_text(tb, encoding="utf-8")
            failures.append({"record": str(rec), "error": repr(e)})
            print(f"[FAIL] {rec}: {e}")

    if failures:
        fail_path = out_root / "failures.csv"
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["record", "error"])
            w.writeheader()
            w.writerows(failures)
        print(f"\n[WARN] {len(failures)} records failed. See: {fail_path}")

    print(f"\n[DONE] Finished. Outputs in: {out_root}")


if __name__ == "__main__":
    main()
