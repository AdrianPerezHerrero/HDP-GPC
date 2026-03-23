#!/usr/bin/env python3
"""
Find cases where HDP-GPC "merges" (explains) two CLARA clusters into one HDP cluster.

We look for an HDP cluster h such that:
  - The two most frequent CLARA clusters inside h cover >= cover_thr of h
  - Each of those CLARA clusters maps back to h with purity >= purity_thr
  - And each has at least min_count points in that overlap

This is detected via a contingency table between labelings (standard practice). :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def load_labels(path: str) -> np.ndarray:
    y = np.load(path)
    y = np.asarray(y).reshape(-1)
    return y


def contingency_table(labels_a: np.ndarray, labels_b: np.ndarray, ignore_neg: bool = True):
    """
    Build contingency matrix M where rows=unique labels_a, cols=unique labels_b.
    Returns: M, ua (row label ids), ub (col label ids), mask_used (bool mask over original indices)
    """
    a = np.asarray(labels_a).reshape(-1)
    b = np.asarray(labels_b).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Label vectors must have same length. Got {a.shape[0]} vs {b.shape[0]}")

    mask = np.ones(a.shape[0], dtype=bool)
    # ignore NaNs if any
    mask &= np.isfinite(a) & np.isfinite(b)

    if ignore_neg:
        mask &= (a >= 0) & (b >= 0)

    a_m = a[mask].astype(np.int64)
    b_m = b[mask].astype(np.int64)

    ua, ia = np.unique(a_m, return_inverse=True)
    ub, ib = np.unique(b_m, return_inverse=True)

    M = np.zeros((ua.size, ub.size), dtype=np.int64)
    np.add.at(M, (ia, ib), 1)
    return M, ua, ub, mask


def find_merge_case(
    M: np.ndarray,
    ua: np.ndarray,  # HDP label ids
    ub: np.ndarray,  # CLARA label ids
    cover_thr: float,
    purity_thr: float,
    min_count: int,
):
    """
    Search for HDP cluster that is mostly composed of two CLARA clusters
    AND those CLARA clusters mostly map back to that HDP cluster.

    Returns: dict with best candidate or None.
    """
    if M.shape[1] < 2:
        return None

    row_sums = M.sum(axis=1)  # size per HDP cluster
    col_sums = M.sum(axis=0)  # size per CLARA cluster

    best = None
    for r in range(M.shape[0]):
        n_h = int(row_sums[r])
        if n_h == 0:
            continue

        row = M[r]
        top2 = np.argsort(row)[::-1][:2]
        c1, c2 = int(top2[0]), int(top2[1])
        cnt1, cnt2 = int(row[c1]), int(row[c2])

        if cnt2 < min_count or cnt1 < min_count:
            continue

        cover = (cnt1 + cnt2) / n_h

        # "purity": how much of each CLARA cluster falls into this HDP cluster
        purity1 = cnt1 / (col_sums[c1] + 1e-12)
        purity2 = cnt2 / (col_sums[c2] + 1e-12)

        if cover < cover_thr:
            continue
        if (purity1 < purity_thr) or (purity2 < purity_thr):
            continue

        # score candidate (prefer high cover and high minimum purity)
        score = cover * min(purity1, purity2)

        # optional: see if there is a strong 3rd CLARA component (should be small if it's a "2-to-1")
        third = np.argsort(row)[::-1][2] if M.shape[1] > 2 else None
        cnt3 = int(row[third]) if third is not None else 0
        frac3 = cnt3 / n_h if n_h else 0.0

        cand = {
            "score": float(score),
            "hdp_row_index": r,
            "hdp_label": int(ua[r]),
            "hdp_size": n_h,
            "clara_label_1": int(ub[c1]),
            "clara_label_2": int(ub[c2]),
            "overlap_1": cnt1,
            "overlap_2": cnt2,
            "cover": float(cover),
            "purity_1": float(purity1),
            "purity_2": float(purity2),
            "third_overlap": cnt3,
            "third_frac_in_hdp": float(frac3),
        }

        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="K for CLARA labels_K{K}.npy")
    ap.add_argument("--clara-out-dir", type=str, default=None,
                    help="Folder containing labels_K{K}.npy (used if --clara-labels not set)")
    ap.add_argument("--clara-labels", type=str, default=None,
                    help="Direct path to CLARA labels .npy (overrides --clara-out-dir/--k)")
    ap.add_argument("--hdp-labels", type=str,
                    default="/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/cluster_labels/resampled_low_freq/cluster_labels_hillary_202407_dynamic_8.npy",
                    help="Path to HDP-GPC labels .npy")
    ap.add_argument("--cover-thr", type=float, default=0.80,
                    help="Min fraction of an HDP cluster explained by its top-2 CLARA clusters")
    ap.add_argument("--purity-thr", type=float, default=0.60,
                    help="Min fraction of each CLARA cluster that maps into the HDP cluster")
    ap.add_argument("--min-count", type=int, default=50,
                    help="Min overlap count for each of the two CLARA clusters")
    ap.add_argument("--ignore-neg", action="store_true",
                    help="Ignore negative labels (e.g., -1) if present")
    ap.add_argument("--out-csv", type=str, default=None,
                    help="If set, write a CSV of sample indices for the found merge case")
    args = ap.parse_args()

    # Resolve CLARA labels path
    if args.clara_labels is not None:
        clara_path = args.clara_labels
    else:
        if args.clara_out_dir is None or args.k is None:
            raise SystemExit("Provide either --clara-labels OR both --clara-out-dir and --k.")
        clara_path = os.path.join(args.clara_out_dir, f"labels_K{args.k}.npy")

    if not os.path.exists(clara_path):
        raise FileNotFoundError(f"CLARA labels not found: {clara_path}")
    if not os.path.exists(args.hdp_labels):
        raise FileNotFoundError(f"HDP labels not found: {args.hdp_labels}")

    clara = load_labels(clara_path)
    hdp = load_labels(args.hdp_labels)

    print(f"Loaded CLARA labels: {clara_path} (n={clara.size})")
    print(f"Loaded HDP  labels: {args.hdp_labels} (n={hdp.size})")

    # Build contingency table: rows=HDP, cols=CLARA :contentReference[oaicite:2]{index=2}
    M, ua, ub, mask = contingency_table(hdp, clara, ignore_neg=args.ignore_neg)

    print(f"Contingency table size: HDP clusters={len(ua)}, CLARA clusters={len(ub)}")
    best = find_merge_case(
        M, ua, ub,
        cover_thr=args.cover_thr,
        purity_thr=args.purity_thr,
        min_count=args.min_count
    )

    if best is None:
        print("\nNo clear '2 CLARA -> 1 HDP' reduction found under thresholds:")
        print(f"  cover_thr={args.cover_thr}, purity_thr={args.purity_thr}, min_count={args.min_count}")
        print("Try relaxing thresholds (e.g. --cover-thr 0.7 --purity-thr 0.5 --min-count 25).")
        return

    print("\nFound candidate HDP merge case:")
    print(f"  HDP cluster: label={best['hdp_label']}  size={best['hdp_size']}")
    print(f"  Dominant CLARA clusters inside it:")
    print(f"    CLARA {best['clara_label_1']}: overlap={best['overlap_1']}  purity={best['purity_1']:.3f}")
    print(f"    CLARA {best['clara_label_2']}: overlap={best['overlap_2']}  purity={best['purity_2']:.3f}")
    print(f"  Coverage by top-2 CLARA inside HDP cluster: {best['cover']:.3f}")
    print(f"  Third component inside HDP cluster: overlap={best['third_overlap']}  frac={best['third_frac_in_hdp']:.3f}")
    print(f"  Score: {best['score']:.4f}")

    # Optionally export indices for inspection
    if args.out_csv:
        # indices in original array where mask is True
        idx_all = np.arange(hdp.size)
        idx_used = idx_all[mask]

        hdp_used = hdp[mask].astype(int)
        clara_used = clara[mask].astype(int)

        # Keep only points in the selected HDP cluster
        target = best["hdp_label"]
        keep = (hdp_used == target)

        df = pd.DataFrame({
            "index": idx_used[keep],
            "hdp_label": hdp_used[keep],
            "clara_label": clara_used[keep],
        }).sort_values("index")

        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote indices for this HDP cluster to: {args.out_csv}")


if __name__ == "__main__":
    main()
