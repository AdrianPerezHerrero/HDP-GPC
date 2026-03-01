#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_json_summaries.py

Reads summary_*.json files (one per record) and builds a table with:
  n_samples, n_clusters, purity, nmi, ari, homogeneity, completeness, macro_f1

Example:
  python collect_json_summaries.py --in_dir results/eval_record_checks_all --out_csv summary_table.csv
  python collect_json_summaries.py --in_dir results/eval_record_checks_all --out_csv summary_table.csv --out_xlsx summary_table.xlsx
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def extract_record_id(path: Path) -> str:
    """
    Extract record id from filename 'summary_XXX.json'. If not matching, use stem.
    """
    m = re.search(r"summary_(.+)\.json$", path.name)
    return m.group(1) if m else path.stem


def flatten_json(obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flatten nested dicts/lists into a flat dict with dotted keys.
    """
    out: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            out.update(flatten_json(v, key, sep=sep))
    elif isinstance(obj, list):
        # store list as-is and also flatten list-of-dicts if present
        out[prefix] = obj
        for i, v in enumerate(obj):
            key = f"{prefix}{sep}{i}" if prefix else str(i)
            out.update(flatten_json(v, key, sep=sep))
    else:
        out[prefix] = obj

    return out


def get_first_match(flat: Dict[str, Any], candidates: Iterable[str]) -> Optional[Any]:
    """
    Find the first matching key in 'flat' using case-insensitive comparison,
    and also allowing keys to appear as suffixes (e.g., 'metrics.purity').
    """
    # Build lowercase map
    lower_map = {k.lower(): k for k in flat.keys()}

    for cand in candidates:
        c = cand.lower()

        # exact match
        if c in lower_map:
            return flat[lower_map[c]]

        # suffix match: "*.cand"
        suffix = "." + c
        for lk, orig in lower_map.items():
            if lk.endswith(suffix):
                return flat[orig]

        # also allow "cand" contained in key token-wise
        for lk, orig in lower_map.items():
            if re.search(rf"(^|[._-]){re.escape(c)}($|[._-])", lk):
                return flat[orig]

    return None


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(round(float(x)))
    except Exception:
        return None


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing summary_*.json files.")
    ap.add_argument("--pattern", default="summary_*.json", help="Glob pattern (default: summary_*.json)")
    ap.add_argument("--out_csv", default="summary_table.csv", help="Output CSV path.")
    ap.add_argument("--out_xlsx", default=None, help="Optional output XLSX path.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {in_dir}")

    # Candidate keys (robust to naming differences)
    KEY_SAMPLES = ["n_samples", "num_samples", "samples", "n_beats", "beats", "T"]
    KEY_CLUSTERS = ["n_clusters", "num_clusters", "clusters", "clusters_unique", "clusters_nonempty", "K", "M"]
    KEY_PURITY = ["purity"]
    KEY_NMI = ["nmi", "normalized_mutual_info", "normalized_mutual_info_score"]
    KEY_ARI = ["ari", "adjusted_rand", "adjusted_rand_score"]
    KEY_HOM = ["homogeneity", "homogeneity_score"]
    KEY_COMP = ["completeness", "completeness_score"]
    KEY_F1 = ["macro_f1", "macro-f1", "macro_f1_score", "f1_macro", "macroF1", "macro f1"]

    rows = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        flat = flatten_json(data)

        rec = extract_record_id(fp)
        n_samples = to_int(get_first_match(flat, KEY_SAMPLES))
        n_clusters = to_int(get_first_match(flat, KEY_CLUSTERS))

        row = {
            "record": rec,
            "n_samples": n_samples,
            "n_clusters": n_clusters,
            "purity": to_float(get_first_match(flat, KEY_PURITY)),
            "nmi": to_float(get_first_match(flat, KEY_NMI)),
            "ari": to_float(get_first_match(flat, KEY_ARI)),
            "homogeneity": to_float(get_first_match(flat, KEY_HOM)),
            "completeness": to_float(get_first_match(flat, KEY_COMP)),
            "macro_f1": to_float(get_first_match(flat, KEY_F1)),
            "source_file": fp.name,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # sort by numeric record id when possible
    def _rec_sort_key(r: str):
        m = re.match(r"^\d+$", str(r))
        return (0, int(r)) if m else (1, str(r))

    df = df.sort_values(by="record", key=lambda s: s.map(_rec_sort_key)).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if args.out_xlsx:
        out_xlsx = Path(args.out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out_xlsx, index=False)

    print(f"[OK] Parsed {len(df)} summaries from: {in_dir}")
    print(f"[OK] Wrote CSV: {out_csv}")
    if args.out_xlsx:
        print(f"[OK] Wrote XLSX: {args.out_xlsx}")


if __name__ == "__main__":
    main()