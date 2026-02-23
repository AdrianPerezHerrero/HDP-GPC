#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_record_offline.py

Usage (from repo root):
  python evaluate_record_:contentReference[oaicite:6]{index=6}--pred_dir results/cluster_labels/v1_UCR_ver \
    --out_dir  results/eval_record_checks \
    --max_warps_per_cluster 30

Notes
-----
- Reconstructs the model from saved labels exactly like reload_and_plot.py:
    sw_gp.reload_model_from_labels(x_trains, data, cluster_labels, M)
  See reload_and_plot.py and GPI_HDP.py.
- Produces:
    * cluster plots (plot_models_plotly)
    * evaluation_report.txt with metrics + confusion matrices
    * confusion_full.csv, confusion_aami.csv, A_C_taxonomy.csv
    * eigenvalue plot (A) and warp plots
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    f1_score,
)
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt

import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.get_data import compute_estimators_LDS
from hdpgpc.util_plots import plot_models_plotly, print_results


dtype = torch.float64
torch.set_default_dtype(dtype)


# ------------------------
# Path helpers (same spirit as reload_and_plot.py)
# ------------------------
def find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here, here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent]
    for c in candidates:
        if (c / "hdpgpc").exists():
            return c
    return Path.cwd()


def find_data_dir(repo_root: Path) -> Path:
    cand1 = repo_root / "data" / "mitdb"
    cand2 = repo_root / "data" / "mitbih"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Could not find data dir under {repo_root / 'data'}")


def find_cluster_label_file(pred_dir: Path, rec: str) -> Path:
    candidates = [
        pred_dir / f"cluster_labels_{rec}_offline.npy",
        pred_dir / f"cluster_labels_{rec}.npy",
        pred_dir / f"cluster_labels_{rec}_offline_labels.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(pred_dir.glob(f"*{rec}*_offline*.npy"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(f"Multiple cluster-label files for {rec} in {pred_dir}: {[m.name for m in matches]}")
    raise FileNotFoundError(f"No cluster-label file found for record {rec} in {pred_dir}")


# ------------------------
# Model builder (same hyperparameters as reload_and_plot.py)
# ------------------------
def build_sw_gp(
    x_basis,
    x_basis_warp,
    num_outputs,
    sigma,
    gamma,
    outputscale_,
    ini_lengthscale,
    bound_lengthscale,
    noise_warp,
    bound_sigma,
    bound_gamma,
    bound_noise_warp,
):
    sw_gp = hdpgp.GPI_HDP(
        x_basis,
        x_basis_warp=x_basis_warp,
        n_outputs=num_outputs,
        kernels=None,
        model_type="dynamic",
        ini_lengthscale=ini_lengthscale,
        bound_lengthscale=bound_lengthscale,
        ini_gamma=gamma,
        ini_sigma=sigma,
        ini_outputscale=outputscale_,
        noise_warp=noise_warp,
        bound_sigma=bound_sigma,
        bound_gamma=bound_gamma,
        bound_noise_warp=bound_noise_warp,
        warp_updating=False,
        method_compute_warp="greedy",
        verbose=True,
        hmm_switch=True,
        max_models=100,
        mode_warp="rough",
        bayesian_params=True,
        inducing_points=False,
        reestimate_initial_params=True,
        n_explore_steps=15,
        free_deg_MNIV=3,
    )
    return sw_gp


# ------------------------
# Label normalization + AAMI mapping
# ------------------------
DEFAULT_INDEX_TO_SYMBOL = [
    # Common MIT-BIH symbol set; adjust if your stored labels are integers with a different order.
    "N", "L", "R", "A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"
]


AAMI_GROUPS = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced / fusion paced often treated as Q in AAMI-style 5-class reports
    "/": "Q", "f": "Q", "Q": "Q",
}


def labels_to_symbols(y_true: np.ndarray, index_to_symbol=None) -> np.ndarray:
    """
    Returns array of strings.
    - If y_true already contains strings/bytes/object strings -> cast to str.
    - If y_true contains ints -> map via index_to_symbol (fallback to 'class_<id>').
    """
    y = np.asarray(y_true)

    if y.dtype.kind in ("U", "S"):
        return y.astype(str)

    if y.dtype.kind in ("i", "u", "f"):
        idx_to_sym = index_to_symbol or DEFAULT_INDEX_TO_SYMBOL
        out = []
        for v in y.astype(int):
            if 0 <= v < len(idx_to_sym):
                out.append(idx_to_sym[v])
            else:
                out.append(f"class_{v}")
        return np.asarray(out, dtype=str)

    # object / other
    return y.astype(str)


def to_aami(symbols: np.ndarray) -> np.ndarray:
    out = []
    for s in symbols:
        out.append(AAMI_MAP.get(str(s), "Q"))
    return np.asarray(out, dtype=str)


# ------------------------
# Clustering metrics + mapping
# ------------------------
def purity_score(y_true_int: np.ndarray, clusters: np.ndarray) -> float:
    """
    Purity = sum_k max_j |C_k ∩ T_j| / N
    """
    n = len(y_true_int)
    if n == 0:
        return 0.0
    K = int(np.max(clusters)) + 1
    J = int(np.max(y_true_int)) + 1
    cont = np.zeros((K, J), dtype=np.int64)
    for c, t in zip(clusters, y_true_int):
        cont[int(c), int(t)] += 1
    return float(np.sum(np.max(cont, axis=1)) / n)


def hungarian_cluster_to_class_map(y_true_int: np.ndarray, clusters: np.ndarray):
    """
    Returns:
      cluster_to_class: dict {cluster_id: class_id} (after 1-1 Hungarian mapping where possible)
      y_pred_mapped:    predicted class per sample (uses Hungarian where assigned; fallback to majority label)
    """
    clusters = clusters.astype(int)
    K = int(np.max(clusters)) + 1
    J = int(np.max(y_true_int)) + 1

    cont = np.zeros((K, J), dtype=np.int64)
    for c, t in zip(clusters, y_true_int):
        cont[c, t] += 1

    # Pad to square for Hungarian
    n = max(K, J)
    cost = np.zeros((n, n), dtype=np.int64)
    cost[:K, :J] = cont

    # Maximize counts -> minimize negative
    row_ind, col_ind = linear_sum_assignment(-cost)

    cluster_to_class = {}
    assigned_clusters = set()
    assigned_classes = set()
    for r, c in zip(row_ind, col_ind):
        if r < K and c < J:
            cluster_to_class[r] = c
            assigned_clusters.add(r)
            assigned_classes.add(c)

    # For clusters not assigned (e.g., K > J), fallback to majority class
    for k in range(K):
        if k not in cluster_to_class:
            cluster_to_class[k] = int(np.argmax(cont[k])) if cont[k].sum() > 0 else 0

    y_pred_mapped = np.asarray([cluster_to_class[int(c)] for c in clusters], dtype=int)
    return cluster_to_class, y_pred_mapped, cont


# ------------------------
# A/C eigen taxonomy
# ------------------------
def _to_numpy(mat):
    if isinstance(mat, torch.Tensor):
        return mat.detach().cpu().numpy()
    return np.asarray(mat)


def classify_A(A: np.ndarray):
    """
    Taxonomy:
      - Stable–near identity: slow drift (high stability)
      - Stable–contractive: quick convergence to a prototype
      - Marginal/oscillatory: quasi-periodic micro-variations
      - Ill-conditioned / unstable estimates
    """
    n = A.shape[0]
    I = np.eye(n)

    # Spectral radius and eigen structure
    try:
        eig = np.linalg.eigvals(A)
        rho = float(np.max(np.abs(eig)))
        frac_complex = float(np.mean(np.abs(np.imag(eig)) > 1e-8))
    except Exception:
        eig = np.array([])
        rho = float("nan")
        frac_complex = float("nan")

    # Conditioning + distance to identity
    try:
        cond = float(np.linalg.cond(A))
    except Exception:
        cond = float("inf")
    try:
        id_dist = float(np.linalg.norm(A - I, ord="fro") / np.linalg.norm(I, ord="fro"))
    except Exception:
        id_dist = float("inf")

    # Heuristics
    if (not np.isfinite(cond)) or cond > 1e8 or (np.isfinite(rho) and rho > 1.05):
        tag = "Ill-conditioned / unstable"
    elif np.isfinite(rho) and 0.95 <= rho <= 1.05 and frac_complex > 0.02:
        tag = "Marginal/oscillatory"
    elif np.isfinite(rho) and rho < 0.95:
        tag = "Stable–contractive"
    else:
        # Default stable bucket: near identity if close enough
        tag = "Stable–near identity" if id_dist < 0.25 and (np.isfinite(rho) and rho <= 1.05) else "Marginal/oscillatory"

    return {
        "spectral_radius": rho,
        "cond": cond,
        "id_dist_fro": id_dist,
        "frac_complex": frac_complex,
        "taxonomy": tag,
        "eigvals": eig,
    }


def analyze_A_C(sw_gp, out_dir: Path):
    """
    Analyze last A and C for each cluster per lead.
    Saves:
      - A_C_taxonomy.csv
      - eig_A_complex_plane.png (lead 0; all clusters)
    """
    rows = []
    eig_plot_done = False

    for ld in range(sw_gp.n_outputs):
        for m in range(sw_gp.M):
            gp = sw_gp.gpmodels[ld][m]
            if len(gp.indexes) == 0:
                continue

            A = _to_numpy(gp.A[-1])
            C = _to_numpy(gp.C[-1])

            a_info = classify_A(A)
            c_info = classify_A(C)

            rows.append({
                "lead": ld,
                "cluster": m,
                "n_samples_in_cluster": int(len(gp.indexes)),
                "A_spectral_radius": a_info["spectral_radius"],
                "A_cond": a_info["cond"],
                "A_id_dist_fro": a_info["id_dist_fro"],
                "A_frac_complex": a_info["frac_complex"],
                "A_taxonomy": a_info["taxonomy"],
                "C_spectral_radius": c_info["spectral_radius"],
                "C_cond": c_info["cond"],
                "C_id_dist_fro": c_info["id_dist_fro"],
                "C_frac_complex": c_info["frac_complex"],
                "C_taxonomy": c_info["taxonomy"],
            })

            # One combined eigenvalue plot (A) for lead 0
            if (ld == 0) and (not eig_plot_done):
                # We'll build after loop (need all clusters)
                pass

    df = pd.DataFrame(rows).sort_values(["lead", "cluster"])
    df.to_csv(out_dir / "A_C_taxonomy.csv", index=False)

    # Eigenvalue plot for A (lead 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta))
    ax.set_aspect("equal", "box")
    ax.set_title("Eigenvalues of A (lead 0), unit circle shown")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")

    plotted_any = False
    for r in rows:
        if r["lead"] != 0:
            continue
        gp = sw_gp.gpmodels[0][r["cluster"]]
        A = _to_numpy(gp.A[-1])
        eig = np.linalg.eigvals(A)
        ax.scatter(np.real(eig), np.imag(eig), s=6, alpha=0.35, label=f"cl{r['cluster']}")
        plotted_any = True

    if plotted_any:
        # keep legend small-ish
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "eig_A_complex_plane_lead0.png", dpi=200)
    plt.close(fig)

    return df


# ------------------------
# Warp plotting
# ------------------------
def compute_and_plot_warps(
    sw_gp,
    x_train: np.ndarray,
    data: np.ndarray,
    cluster_labels: np.ndarray,
    out_dir: Path,
    lead: int = 0,
    max_per_cluster: int = 30,
):
    """
    Computes warps post-hoc per cluster using sw_gp.compute_warp_y(..., force_model=m).
    Saves:
      - warps_lead{lead}.png
      - warps_summary.csv (per cluster: median |warp| stats)
    """
    x_t = torch.from_numpy(np.asarray(x_train, dtype=np.float64))
    cluster_labels = cluster_labels.astype(int)

    clusters = sorted(list(set(cluster_labels.tolist())))
    warp_rows = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Warp functions (lead {lead}) - up to {max_per_cluster} beats/cluster")
    ax.set_xlabel("t")
    ax.set_ylabel("warp(t)  (as returned by compute_warp_y)")

    for m in clusters:
        idxs = np.where(cluster_labels == m)[0][:max_per_cluster]
        if idxs.size == 0:
            continue

        warps = []
        for i in idxs:
            y = torch.from_numpy(np.asarray(data[i, :, lead], dtype=np.float64)).reshape(-1, 1)
            # compute warp only against cluster m
            y_w, x_w, liks = sw_gp.compute_warp_y(
                x_t, y,
                strategie="standard",
                force_model=m,
                gpmodel=sw_gp.gpmodels[lead][m],
                ld=lead,
            )
            w = x_w[m].detach().cpu().numpy().reshape(-1)
            warps.append(w)

        W = np.vstack(warps)  # (nbeats, T)
        w_med = np.median(W, axis=0)
        w_abs = np.abs(W)

        # plot individual + median
        for k in range(W.shape[0]):
            ax.plot(x_train.reshape(-1), W[k], alpha=0.12)
        ax.plot(x_train.reshape(-1), w_med, linewidth=2.0, label=f"cl{m} median")

        warp_rows.append({
            "cluster": m,
            "n_warps": int(W.shape[0]),
            "median_abs_warp_mean": float(np.mean(np.median(w_abs, axis=0))),
            "median_abs_warp_max": float(np.max(np.median(w_abs, axis=0))),
            "median_warp_range": float(np.max(w_med) - np.min(w_med)),
        })

    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"warps_lead{lead}.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(warp_rows).to_csv(out_dir / f"warps_summary_lead{lead}.csv", index=False)


# ------------------------
# Main per-record routine
# ------------------------
def run_one_record(rec: str, pred_dir: Path, out_dir: Path, label_map_json: Path | None,
                   max_warps_per_cluster: int, warp_lead: int):
    repo_root = find_project_root()
    data_dir = find_data_dir(repo_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data + true labels
    data = np.load(data_dir / f"{rec}.npy")
    labels_true_raw = np.load(data_dir / f"{rec}_labels.npy", allow_pickle=True)

    num_samples, num_obs_per_sample, num_outputs = data.shape

    # Load predicted cluster labels
    pred_path = find_cluster_label_file(pred_dir, rec)
    cluster_labels = np.load(pred_path, allow_pickle=True).reshape(-1)

    # Align lengths
    if len(cluster_labels) != num_samples:
        n = min(len(cluster_labels), num_samples)
        print(f"[WARN] {rec}: cluster_labels length={len(cluster_labels)} vs num_samples={num_samples}. Trimming to {n}.")
        cluster_labels = cluster_labels[:n]
        data = data[:n]
        labels_true_raw = labels_true_raw[:n]
        num_samples = n

    # Ensure integer cluster ids
    if cluster_labels.dtype.kind in ("f", "c"):
        cluster_labels = np.rint(cluster_labels).astype(np.int64)
    else:
        cluster_labels = cluster_labels.astype(np.int64)

    if np.min(cluster_labels) < 0:
        raise ValueError(f"{rec}: cluster_labels contains negative values (min={np.min(cluster_labels)})")

    M = int(np.max(cluster_labels)) + 1
    if M <= 0:
        raise ValueError(f"{rec}: computed M={M} from cluster_labels")

    # Label index-to-symbol override (optional)
    index_to_symbol = None
    if label_map_json is not None:
        with open(label_map_json, "r", encoding="utf-8") as f:
            index_to_symbol = json.load(f)

    y_true_sym = labels_to_symbols(labels_true_raw, index_to_symbol=index_to_symbol)

    # Estimate priors (same as reload_and_plot.py)
    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data)

    # Hyperparameters (same as reload_and_plot.py)
    sigma = std * 1.0
    gamma = std_dif * 1.0
    outputscale_ = 300.0
    ini_lengthscale = 3.0
    bound_lengthscale = (1.0, 20.0)

    noise_warp = std * 0.1
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

    # Time supports
    l, L = 0, num_obs_per_sample
    x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(l, L, 2, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T
    x_trains = np.array([x_train] * num_samples)

    # Build + reload model
    sw_gp = build_sw_gp(
        x_basis=x_basis,
        x_basis_warp=x_basis_warp,
        num_outputs=num_outputs,
        sigma=sigma,
        gamma=gamma,
        outputscale_=outputscale_,
        ini_lengthscale=ini_lengthscale,
        bound_lengthscale=bound_lengthscale,
        noise_warp=noise_warp,
        bound_sigma=bound_sigma,
        bound_gamma=bound_gamma,
        bound_noise_warp=bound_noise_warp,
    )

    t0 = time.time()
    sw_gp.reload_model_from_labels(x_trains, data, cluster_labels, M)
    dt_min = (time.time() - t0) / 60.0
    print(f"[OK] {rec}: reload_model_from_labels done in {dt_min:.2f} min (M={M}).")

    # ------------------------
    # 1) Cluster plots (same as reload_and_plot.py)
    # ------------------------
    out_prefix = str(out_dir / f"Rec{rec}_")

    main_model = print_results(sw_gp, y_true_sym, 0, error=False)
    selected_gpmodels = sw_gp.selected_gpmodels()

    plot_models_plotly(
        sw_gp, selected_gpmodels, main_model, y_true_sym, 0,
        lead=0, save=out_prefix + "Offline_Clusters_Lead_1.png",
        step=0.5, plot_latent=True
    )
    if num_outputs > 1:
        plot_models_plotly(
            sw_gp, selected_gpmodels, main_model, y_true_sym, 0,
            lead=1, save=out_prefix + "Offline_Clusters_Lead_2.png",
            step=0.5, plot_latent=True
        )

    # ------------------------
    # 2) Metrics + full confusion (after mapping)
    # ------------------------
    le = LabelEncoder()
    y_true_int = le.fit_transform(y_true_sym)
    class_names = le.classes_.tolist()

    # Metrics on partitions
    purity = purity_score(y_true_int, cluster_labels)
    nmi = normalized_mutual_info_score(y_true_int, cluster_labels)
    ari = adjusted_rand_score(y_true_int, cluster_labels)
    hom = homogeneity_score(y_true_int, cluster_labels)
    comp = completeness_score(y_true_int, cluster_labels)

    # Cluster -> class mapping + macro-F1 on mapped labels
    cl2c, y_pred_mapped_int, contingency = hungarian_cluster_to_class_map(y_true_int, cluster_labels)
    macro_f1 = f1_score(y_true_int, y_pred_mapped_int, average="macro")

    # Full label confusion
    cm_full = confusion_matrix(y_true_int, y_pred_mapped_int, labels=np.arange(len(class_names)))
    df_cm_full = pd.DataFrame(cm_full, index=class_names, columns=class_names)
    df_cm_full.to_csv(out_dir / "confusion_full.csv")

    # ------------------------
    # 3) AAMI confusion
    # ------------------------
    y_true_aami = to_aami(y_true_sym)

    # Map predicted class -> symbol -> AAMI
    pred_sym = le.inverse_transform(y_pred_mapped_int)
    y_pred_aami = to_aami(pred_sym)

    cm_aami = confusion_matrix(y_true_aami, y_pred_aami, labels=AAMI_GROUPS)
    df_cm_aami = pd.DataFrame(cm_aami, index=AAMI_GROUPS, columns=AAMI_GROUPS)
    df_cm_aami.to_csv(out_dir / "confusion_aami.csv")

    # ------------------------
    # 4) A/C eigen taxonomy (+ plot)
    # ------------------------
    df_tax = analyze_A_C(sw_gp, out_dir)

    # ------------------------
    # 5) Warp plots
    # ------------------------
    try:
        compute_and_plot_warps(
            sw_gp,
            x_train=x_train.reshape(-1),
            data=data,
            cluster_labels=cluster_labels,
            out_dir=out_dir,
            lead=int(warp_lead),
            max_per_cluster=int(max_warps_per_cluster),
        )
    except Exception as e:
        print(f"[WARN] Warp plotting failed: {repr(e)}")

    # ------------------------
    # Write TXT report
    # ------------------------
    K_unique = int(np.unique(cluster_labels).size)
    K_nonempty = int(len(sw_gp.selected_gpmodels()))
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

    # Make a readable mapping table (cluster -> label)
    cl_map_rows = []
    for cl_id in sorted(cl2c.keys()):
        cl_map_rows.append({
            "cluster": cl_id,
            "mapped_label": class_names[cl2c[cl_id]] if cl2c[cl_id] < len(class_names) else "UNK",
            "cluster_size": int(cluster_sizes.get(cl_id, 0)),
        })
    df_map = pd.DataFrame(cl_map_rows)

    report = []
    report.append(f"Record: {rec}")
    report.append(f"Samples (beats): {num_samples}")
    report.append(f"Clusters (unique in labels): {K_unique}")
    report.append(f"Clusters (non-empty in reloaded model): {K_nonempty}")
    report.append("")
    report.append("Metrics (partition-based):")
    report.append(f"  Purity:        {purity:.6f}")
    report.append(f"  NMI:           {nmi:.6f}")
    report.append(f"  ARI:           {ari:.6f}")
    report.append(f"  Homogeneity:   {hom:.6f}")
    report.append(f"  Completeness:  {comp:.6f}")
    report.append(f"  Macro-F1 (after mapping): {macro_f1:.6f}")
    report.append("")
    report.append("Cluster -> Label mapping (after Hungarian + majority fallback):")
    report.append(df_map.to_string(index=False))
    report.append("")
    report.append("Full confusion matrix (rows=true, cols=pred after mapping):")
    report.append(df_cm_full.to_string())
    report.append("")
    report.append("AAMI confusion matrix (rows=true, cols=pred) [N,S,V,F,Q]:")
    report.append(df_cm_aami.to_string())
    report.append("")
    report.append("A/C taxonomy summary (per cluster, per lead; see A_C_taxonomy.csv for full table):")
    if not df_tax.empty:
        report.append(df_tax[["lead","cluster","n_samples_in_cluster","A_spectral_radius","A_cond","A_id_dist_fro","A_taxonomy",
                              "C_spectral_radius","C_cond","C_id_dist_fro","C_taxonomy"]].to_string(index=False))
    else:
        report.append("  (No non-empty clusters found?)")
    report.append("")
    report.append("State-space stability note:")
    report.append("  For a discrete-time linear system x_{t+1} = A x_t + w_t, asymptotic stability is linked to eigenvalues of A:")
    report.append("  |λ_i(A)| < 1 for all i -> stable; eigenvalues near/on the unit circle -> marginal/slow decay; complex pairs -> oscillatory modes.")
    report.append("")

    (out_dir / "evaluation_report.txt").write_text("\n".join(report), encoding="utf-8")

    print(f"[DONE] Outputs written to: {out_dir}")

    # Cleanup
    del sw_gp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=str, help="MIT-BIH record id (e.g., 100)")
    ap.add_argument("--pred_dir", type=str, default=None,
                    help="Directory containing cluster_labels_<rec>_offline.npy etc.")
    ap.add_argument("--out_dir", type=str, default="results/eval_record_checks",
                    help="Output directory root.")
    ap.add_argument("--label_map_json", type=str, default=None,
                    help="Optional JSON file mapping integer label->symbol (list of strings).")
    ap.add_argument("--max_warps_per_cluster", type=int, default=30,
                    help="Max beats per cluster for warp plotting.")
    ap.add_argument("--warp_lead", type=int, default=0,
                    help="Which lead to use for warp plots.")
    args = ap.parse_args()

    repo_root = find_project_root()
    pred_dir = Path(args.pred_dir) if args.pred_dir is not None else (repo_root / "results" / "cluster_labels" / "v1_UCR_ver")
    if not pred_dir.exists():
        # fallback
        pred_dir = repo_root / "results" / "cluster_labels"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Could not find pred_dir at: {pred_dir}")

    out_dir = Path(args.out_dir) / f"Rec{args.record}"
    label_map_json = Path(args.label_map_json) if args.label_map_json else None

    run_one_record(
        rec=args.record,
        pred_dir=pred_dir,
        out_dir=out_dir,
        label_map_json=label_map_json,
        max_warps_per_cluster=args.max_warps_per_cluster,
        warp_lead=args.warp_lead,
    )


if __name__ == "__main__":
    main()
