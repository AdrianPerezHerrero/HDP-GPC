# evaluate_offline_cluster_labels_v2.py
# -*- coding: utf-8 -*-

import argparse
import json
import gc
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.cluster import contingency_matrix

# Optional plotting
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

# Optional Hungarian/linear assignment
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# Helpers: paths / discovery
# -----------------------------
def find_repo_root() -> Path:
    candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for c in candidates:
        if (c / "hdpgpc").exists():
            return c
    return Path.cwd()


def find_data_dir(repo_root: Path) -> Path:
    pkg_dir = repo_root / "hdpgpc"
    candidates = [
        pkg_dir / "data" / "mitdb",
        pkg_dir / "data" / "mitbih",
        repo_root / "data" / "mitdb",
        repo_root / "data" / "mitbih",
    ]
    for d in candidates:
        if d.exists():
            return d
    raise FileNotFoundError(
        "Could not find data directory. Looked for data/mitdb or data/mitbih "
        "under both repo root and hdpgpc/."
    )


def list_records(data_dir: Path) -> List[str]:
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers: label loading / shaping
# -----------------------------
def squeeze_to_1d(arr: np.ndarray) -> np.ndarray:
    """Best-effort squeeze to a 1D vector."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    # If it's (n,1) or (1,n)
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return arr.reshape(-1)
    # If it's higher-dim, last resort: flatten
    return arr.reshape(-1)


def load_true_labels(path: Path) -> np.ndarray:
    y = np.load(path, allow_pickle=True)
    return squeeze_to_1d(y)


def load_pred_clusters(path: Path) -> np.ndarray:
    y = np.load(path, allow_pickle=True)
    y = squeeze_to_1d(y)
    # Predicted clusters should be integers; coerce robustly
    # (handles float tensors saved to numpy, etc.)
    if y.dtype.kind in ("f", "c"):
        y = np.rint(y).astype(np.int64)
    elif y.dtype.kind in ("U", "S", "O"):
        # If somehow string/object clusters appear, try coercing
        try:
            y = y.astype(np.int64)
        except Exception as e:
            raise ValueError(f"Pred cluster labels are not numeric in {path.name}: dtype={y.dtype}") from e
    else:
        y = y.astype(np.int64)
    return y


def align_lengths(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rec: str,
    mode: str = "trim",
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    Ensure both arrays have the same length.

    mode:
      - "strict": raise if mismatch
      - "trim": trim both to min length (warn)
    """
    n_true = int(len(y_true))
    n_pred = int(len(y_pred))
    if n_true == n_pred:
        return y_true, y_pred, None

    msg = f"[WARN] Record {rec}: length mismatch y_true={n_true}, y_pred={n_pred}"
    if mode == "strict":
        raise ValueError(msg + " (strict mode)")
    n = min(n_true, n_pred)
    return y_true[:n], y_pred[:n], msg + f" -> trimming to {n}"


# -----------------------------
# Metrics
# -----------------------------
def purity_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Purity = (1/N) * sum_k max_j |C_k ∩ T_j|
    """
    cm = contingency_matrix(labels_true, labels_pred)  # requires same length, 1D
    return float(np.sum(np.max(cm, axis=0)) / np.sum(cm))


def map_clusters_to_labels_majority(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, Any]]:
    """
    Many-to-one mapping: each cluster -> most frequent true label in that cluster.
    Works with string true labels (e.g., "N", "V", ...).
    """
    labels_true = squeeze_to_1d(labels_true)
    labels_pred = squeeze_to_1d(labels_pred).astype(np.int64)

    uniq_true = np.unique(labels_true)       # sorted
    uniq_pred = np.unique(labels_pred)       # sorted

    cm = contingency_matrix(labels_true, labels_pred)  # rows=true, cols=pred

    mapping: Dict[int, Any] = {}
    for j, pred_c in enumerate(uniq_pred):
        i_best = int(np.argmax(cm[:, j]))
        mapping[int(pred_c)] = uniq_true[i_best]  # <-- no int() cast on true label

    mapped = np.array([mapping[int(c)] for c in labels_pred], dtype=labels_true.dtype)
    return mapped, mapping


def map_clusters_to_labels_hungarian_1to1(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, Any]]:
    """
    One-to-one mapping using linear sum assignment.
    Falls back to majority mapping if SciPy is unavailable.
    """
    if not HAS_SCIPY:
        return map_clusters_to_labels_majority(labels_true, labels_pred)

    labels_true = squeeze_to_1d(labels_true)
    labels_pred = squeeze_to_1d(labels_pred).astype(np.int64)

    uniq_true = np.unique(labels_true)
    uniq_pred = np.unique(labels_pred)

    cm = contingency_matrix(labels_true, labels_pred)  # rows=true, cols=pred
    maxv = cm.max() if cm.size else 0
    cost = maxv - cm

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: Dict[int, Any] = {}
    assigned_pred = set()
    for r, c in zip(row_ind, col_ind):
        mapping[int(uniq_pred[c])] = uniq_true[r]
        assigned_pred.add(int(uniq_pred[c]))

    # fallback for unassigned clusters
    if len(assigned_pred) < len(uniq_pred):
        _, maj_map = map_clusters_to_labels_majority(labels_true, labels_pred)
        for p in uniq_pred:
            p = int(p)
            if p not in mapping:
                mapping[p] = maj_map[p]

    mapped = np.array([mapping[int(c)] for c in labels_pred], dtype=labels_true.dtype)
    return mapped, mapping


def maybe_plot_cm(cm: np.ndarray, labels: List[Any], out_path: Path, title: str) -> None:
    if not HAS_PLOT:
        return
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def eval_record(
    rec: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mapping_mode: str,
    align_mode: str,
    out_dir: Path,
    make_plots: bool,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Returns:
      summary (dict),
      y_true_used,
      y_pred_mapped_used
    """
    y_true = squeeze_to_1d(y_true)
    y_pred = squeeze_to_1d(y_pred).astype(np.int64)

    y_true, y_pred, warn = align_lengths(y_true, y_pred, rec, mode=align_mode)

    # permutation-invariant clustering metrics
    purity = purity_score(y_true, y_pred)
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ari = float(adjusted_rand_score(y_true, y_pred))
    ami = float(adjusted_mutual_info_score(y_true, y_pred))
    hom, comp, vmeas = homogeneity_completeness_v_measure(y_true, y_pred)
    fms = float(fowlkes_mallows_score(y_true, y_pred))

    # mapping for confusion matrix + classification-style metrics
    if mapping_mode == "hungarian":
        y_pred_mapped, mapping = map_clusters_to_labels_hungarian_1to1(y_true, y_pred)
    else:
        y_pred_mapped, mapping = map_clusters_to_labels_majority(y_true, y_pred)

    # Use sorted unique true labels (strings are fine)
    labels_sorted = list(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred_mapped, labels=labels_sorted)

    acc = float(accuracy_score(y_true, y_pred_mapped))
    f1_macro = float(f1_score(y_true, y_pred_mapped, labels=labels_sorted, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred_mapped, labels=labels_sorted, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred_mapped, labels=labels_sorted, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred_mapped, labels=labels_sorted, average="macro", zero_division=0))

    ensure_dir(out_dir)
    np.save(out_dir / f"confusion_matrix_{rec}.npy", cm)
    if make_plots:
        maybe_plot_cm(
            cm,
            labels_sorted,
            out_dir / f"confusion_matrix_{rec}.png",
            title=f"Record {rec} confusion matrix ({mapping_mode} mapping)",
        )
    print("n_true_classes:", int(len(np.unique(y_true))))
    print("n_pred_clusters:", int(len(np.unique(y_pred))))
    print([str(x) for x in labels_sorted])
    print(cm)
    summary = {
        "record": rec,
        "warning": warn,
        "n_samples_used": int(len(y_true)),
        "n_true_classes": int(len(np.unique(y_true))),
        "n_pred_clusters": int(len(np.unique(y_pred))),
        "mapping_mode": mapping_mode,
        "align_mode": align_mode,
        "cluster_to_label_mapping": {str(k): str(v) for k, v in mapping.items()},
        "metrics": {
            "purity": purity,
            "nmi": nmi,
            "ari": ari,
            "ami": ami,
            "homogeneity": float(hom),
            "completeness": float(comp),
            "v_measure": float(vmeas),
            "fowlkes_mallows": fms,
            "accuracy_mapped": acc,
            "f1_macro_mapped": f1_macro,
            "f1_weighted_mapped": f1_weighted,
            "precision_macro_mapped": prec_macro,
            "recall_macro_mapped": rec_macro,
        },
        "confusion_matrix_labels": [str(x) for x in labels_sorted],
        "confusion_matrix": cm.tolist(),
    }

    with open(out_dir / f"summary_{rec}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary, y_true, y_pred_mapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--pred-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--mapping", type=str, default="majority", choices=["majority", "hungarian"])
    ap.add_argument("--align", type=str, default="trim", choices=["trim", "strict"],
                    help="What to do when y_true and y_pred length mismatch.")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    repo_root = find_repo_root()
    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir(repo_root)
    pred_dir = Path(args.pred_dir) if args.pred_dir else (repo_root / "results" / "cluster_labels" / "final_ver")
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "results" / "eval_final_ver")

    ensure_dir(out_dir)
    per_rec_dir = out_dir / "per_record"
    ensure_dir(per_rec_dir)

    recs = list_records(data_dir)
    if not recs:
        raise RuntimeError(f"No records found in {data_dir}.")

    print(f"[INFO] data_dir: {data_dir}")
    print(f"[INFO] pred_dir: {pred_dir}")
    print(f"[INFO] out_dir : {out_dir}")
    print(f"[INFO] mapping : {args.mapping} (SciPy available: {HAS_SCIPY})")
    print(f"[INFO] align   : {args.align}")
    print(f"[INFO] plotting: {args.plot} (matplotlib available: {HAS_PLOT})")

    per_record_summaries: List[Dict[str, Any]] = []
    failures = []

    # For global confusion matrix / mapped classification metrics
    y_true_all = []
    y_pred_all = []

    # For global clustering metrics (offset cluster ids per record)
    y_true_all_cluster = []
    y_pred_all_cluster = []
    offset = 0

    for i, rec in enumerate(recs, 1):
        true_path = data_dir / f"{rec}_labels.npy"
        pred_path = pred_dir / f"cluster_labels_{rec}_offline.npy"
        if not pred_path.exists():
            print(f"[WARN] Missing predictions for {rec}: {pred_path.name}. Skipping.")
            continue

        print(f"\n[{i}/{len(recs)}] Evaluating record {rec} ...")
        try:
            y_true = load_true_labels(true_path)
            y_pred = load_pred_clusters(pred_path)

            summary, y_true_used, y_pred_mapped_used = eval_record(
                rec=rec,
                y_true=y_true,
                y_pred=y_pred,
                mapping_mode=args.mapping,
                align_mode=args.align,
                out_dir=per_rec_dir,
                make_plots=args.plot,
            )
            per_record_summaries.append(summary)

            # Accumulate mapped labels for global confusion matrix
            y_true_all.append(y_true_used)
            y_pred_all.append(y_pred_mapped_used)

            # Accumulate for global clustering metrics using offset clusters
            y_true_all_cluster.append(y_true_used)
            y_pred_all_cluster.append(y_pred[: len(y_true_used)] + offset)
            offset += int(np.max(y_pred)) + 1 if len(y_pred) else 1

            print(f"[OK] {rec} purity={summary['metrics']['purity']:.4f}, "
                  f"NMI={summary['metrics']['nmi']:.4f}, ARI={summary['metrics']['ari']:.4f}, "
                  f"acc(mapped)={summary['metrics']['accuracy_mapped']:.4f}")
            if summary["warning"]:
                print(summary["warning"])

        except Exception as e:
            print(f"[FAIL] {rec}: {repr(e)}")
            failures.append((rec, repr(e)))
        finally:
            gc.collect()

    if not per_record_summaries:
        raise RuntimeError("No records evaluated (missing preds or all failed).")

    # Global summaries
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    labels_sorted = list(np.unique(y_true_all))
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=labels_sorted)
    np.save(out_dir / "confusion_matrix_all_records.npy", cm_all)

    if args.plot:
        maybe_plot_cm(
            cm_all,
            labels_sorted,
            out_dir / "confusion_matrix_all_records.png",
            title=f"All records confusion matrix ({args.mapping} mapping)",
        )

    # Global metrics
    y_true_all_cluster = np.concatenate(y_true_all_cluster)
    y_pred_all_cluster = np.concatenate(y_pred_all_cluster)

    purity_all = purity_score(y_true_all_cluster, y_pred_all_cluster)
    nmi_all = float(normalized_mutual_info_score(y_true_all_cluster, y_pred_all_cluster))
    ari_all = float(adjusted_rand_score(y_true_all_cluster, y_pred_all_cluster))
    ami_all = float(adjusted_mutual_info_score(y_true_all_cluster, y_pred_all_cluster))
    hom_all, comp_all, vmeas_all = homogeneity_completeness_v_measure(y_true_all_cluster, y_pred_all_cluster)
    fms_all = float(fowlkes_mallows_score(y_true_all_cluster, y_pred_all_cluster))

    acc_all = float(accuracy_score(y_true_all, y_pred_all))
    f1_macro_all = float(f1_score(y_true_all, y_pred_all, labels=labels_sorted, average="macro", zero_division=0))
    f1_weighted_all = float(f1_score(y_true_all, y_pred_all, labels=labels_sorted, average="weighted", zero_division=0))
    prec_macro_all = float(precision_score(y_true_all, y_pred_all, labels=labels_sorted, average="macro", zero_division=0))
    rec_macro_all = float(recall_score(y_true_all, y_pred_all, labels=labels_sorted, average="macro", zero_division=0))

    final_summary = {
        "n_records_evaluated": int(len(per_record_summaries)),
        "n_samples_total": int(len(y_true_all)),
        "mapping_mode": args.mapping,
        "align_mode": args.align,
        "global_confusion_matrix_labels": [str(x) for x in labels_sorted],
        "global_confusion_matrix": cm_all.tolist(),
        "global_metrics": {
            "purity": purity_all,
            "nmi": nmi_all,
            "ari": ari_all,
            "ami": ami_all,
            "homogeneity": float(hom_all),
            "completeness": float(comp_all),
            "v_measure": float(vmeas_all),
            "fowlkes_mallows": fms_all,
            "accuracy_mapped": acc_all,
            "f1_macro_mapped": f1_macro_all,
            "f1_weighted_mapped": f1_weighted_all,
            "precision_macro_mapped": prec_macro_all,
            "recall_macro_mapped": rec_macro_all,
        },
        "failures": [{"record": r, "error": e} for r, e in failures],
        "per_record_summaries_dir": str(per_rec_dir.resolve()),
    }

    with open(out_dir / "summary_all_records.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] Global acc(mapped)={acc_all:.4f}, purity={purity_all:.4f}, NMI={nmi_all:.4f}, ARI={ari_all:.4f}")
    if failures:
        print("[WARN] Some records failed:")
        for r, e in failures:
            print(f"  - {r}: {e}")


if __name__ == "__main__":
    main()
