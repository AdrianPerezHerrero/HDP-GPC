#!/usr/bin/env python3
"""
CLARA baseline for WA Waves spectra (Hillarys / drifting buoys) — comparable to the notebook pipeline.

- Loads MATLAB v7.3 .mat files containing SpotData/* using h5py
- Reconstructs directional spectra S(f,theta) from varianceDensity + (a1,b1,a2,b2)
- (Optionally) interpolates frequency to a dense grid (like your notebook)
- Selects the same frequency window used in your HDP-GPC notebook by default:
      ini_lim_freq = 0.040 Hz
      lim_freq     = startFreq_sea (read from file)
- Flattens each (freq,dir) spectrum into a feature vector and clusters with CLARA (k-medoids subsampling)
- Grid-searches K and selects K via elbow (knee on CLARA cost) and/or silhouette;
  also computes heuristic AIC/BIC using a Laplace (L1) likelihood.

Outputs:
- out_dir/clara_grid.csv
- out_dir/labels_K{K}.npy, medoids_K{K}.npy
- out_dir/elbow_cost.png, out_dir/silhouette.png (if matplotlib available)
"""

from __future__ import annotations

import argparse
import os
import math
import numpy as np
import pandas as pd
import h5py

from scipy.interpolate import interp1d
from sklearn.metrics import pairwise_distances, silhouette_score

# ----------------------------
# Data loading (matches notebook)
# ----------------------------

SPOT_KEYS = [
    "varianceDensity", "frequency", "a1", "b1", "a2", "b2",
    "spec_time", "time", "direction", "dm", "dp"
]

def _read_spot_dataset(f: h5py.File, key: str) -> np.ndarray:
    path = f"SpotData/{key}"
    if path not in f:
        raise KeyError(f"Missing {path} in file.")
    return np.array(f[path])

def load_spotdata_concat(mat_paths: list[str]) -> dict[str, np.ndarray]:
    """Concatenate SpotData arrays along time axis (axis=1) like the notebook."""

    def read_first_scalar(f, path: str) -> float:
        arr = np.array(f[path])
        return float(arr.reshape(-1)[0])  # always takes the first value

    accum = {k: [] for k in SPOT_KEYS}

    startFreq_sea = None
    startFreq_swell = None

    for p in mat_paths:
        with h5py.File(p, "r") as f:
            for k in SPOT_KEYS:
                accum[k].append(_read_spot_dataset(f, k))

            # scalar thresholds (take from first file)
            if startFreq_sea is None and "SpotData/startFreq_sea" in f:
                startFreq_sea = read_first_scalar(f, "SpotData/startFreq_sea")
            if startFreq_swell is None and "SpotData/startFreq_swell" in f:
                startFreq_swell = read_first_scalar(f, "SpotData/startFreq_swell")

    out = {}
    for k, parts in accum.items():
        # All SpotData arrays are (n_freq, n_time) (or compatible); notebook concatenates axis=1
        out[k] = np.concatenate(parts, axis=1)

    out["startFreq_sea"] = startFreq_sea
    out["startFreq_swell"] = startFreq_swell
    return out


def interpolate_freq_dense(freq: np.ndarray,
                           arrays: dict[str, np.ndarray],
                           n_points: int = 200) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Interpolate along frequency axis to a dense grid.
    Assumes freq is (n_freq, n_time) and uses freq[:,0] as the base grid (as in notebook).
    Arrays in `arrays` are interpolated along axis=0 (frequency).
    """
    f_old = np.asarray(freq[:, 0]).squeeze()
    f_dense = np.linspace(f_old.min(), f_old.max(), n_points)

    out = {}
    for name, A in arrays.items():
        if A.shape[0] != f_old.shape[0]:
            raise ValueError(f"{name}: expected first dim {f_old.shape[0]}, got {A.shape[0]}")
        f = interp1d(f_old, A, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=True)
        out[name] = f(f_dense)

    # Soft constraints (optional but usually sensible)
    out["varianceDensity"] = np.clip(out["varianceDensity"], 0.0, None)
    for m in ["a1", "b1", "a2", "b2"]:
        if m in out:
            out[m] = np.clip(out[m], -1.0, 1.0)

    return f_dense, out

def save_clusters_grid_pdf(
    out_dir: str,
    K: int,
    freq_cut: np.ndarray,
    S_use: np.ndarray,          # (time,freq) OR (time,freq,dir)
    labels: np.ndarray,         # (time,)
    medoids: np.ndarray,        # (K,) time indices
    grid_cols: int = 4,
    rows_per_page: int | None = None,
    plot_max_members: int = 150,
    plot_seed: int = 0,
    logy: bool = False,
    pdf_name: str | None = None,
    delta_theta: float | None = None,
) -> str:
    """
    Saves a PDF with a grid of clusters (4 columns by default).
    Each cluster panel shows: member examples (subsampled), cluster mean, and cluster medoid.

    If S_use is directional (3D), panels plot the omnidirectional spectrum S(f)
    computed by integrating over direction with delta_theta.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import math
    import os
    import numpy as np

    rng = np.random.default_rng(plot_seed)

    is_dir = (S_use.ndim == 3)
    if is_dir:
        if delta_theta is None:
            raise ValueError("delta_theta must be provided when S_use is directional (3D).")
        S_omni = S_use.sum(axis=2) * float(delta_theta)  # (time,freq)
    else:
        S_omni = S_use  # (time,freq)

    K_eff = int(labels.max()) + 1
    cluster_ids = list(range(K_eff))

    if pdf_name is None:
        pdf_name = f"clusters_grid_K{K}.pdf"
    pdf_path = os.path.join(out_dir, pdf_name)

    # Paging
    if rows_per_page is None:
        # single page with "necessary" rows for all clusters
        page_size = len(cluster_ids)
    else:
        page_size = rows_per_page * grid_cols

    # Use A4 landscape-like proportions by default (inches)
    # (still fine on US Letter; it just scales)
    n_panels = len(cluster_ids)
    n_rows = int(math.ceil(n_panels / grid_cols))

    base_figsize = (8.27 / 2 * grid_cols, 11.69 / 2 * n_rows)

    with PdfPages(pdf_path) as pdf:  # PdfPages writes each figure as a PDF page :contentReference[oaicite:1]{index=1}
        fig, axes = plt.subplots(
            n_rows, grid_cols,
            figsize=base_figsize,
            squeeze=False,
            sharey=True,
            sharex=True
        )  # subplots grid creation :contentReference[oaicite:2]{index=2}
        axes_flat = axes.ravel()

        for ax in axes_flat[n_panels:]:
            ax.axis("off")

        for i, c in enumerate(range(n_panels)):
            ax = axes_flat[i]
            members = np.where(labels == c)[0]
            n_mem = members.size
            if n_mem == 0:
                ax.axis("off")
                continue

            # Subsample member curves to avoid overplotting
            if plot_max_members and plot_max_members > 0 and n_mem > plot_max_members:
                members_plot = rng.choice(members, size=plot_max_members, replace=False)
            else:
                members_plot = members

            mean_omni = S_omni[members].mean(axis=0)

            medoid_idx = int(medoids[c]) if c < len(medoids) else int(members[0])
            medoid_omni = S_omni[medoid_idx]

            for j in members_plot:
                ax.plot(freq_cut, S_omni[j]/6.0, alpha=0.4, linewidth=0.8, c='lightblue')

            ax.plot(freq_cut, mean_omni/6.0, linewidth=1.8, label="mean", c='b')
            ax.plot(freq_cut, medoid_omni/6.0, linestyle="--", linewidth=1.4, label="medoid", c='red')

            ax.set_title(f"C{c} (N={n_mem})", fontsize=28)
            if logy:
                ax.set_yscale("log")

            # Cleaner labels: only left column + bottom row
            if (i % grid_cols) == 0:
                ax.set_ylabel("S(f)", fontsize=25)
            if i >= (n_rows - 1) * grid_cols:
                ax.set_xlabel("f (Hz)", fontsize=25)

            ax.tick_params(labelsize=20)

            # One legend for the page
        handles, labels_ = axes_flat[0].get_legend_handles_labels()
        # if handles:
        #     fig.legend(handles, labels_, loc="lower center", ncol=2, frameon=True, fontsize=12)

        #fig.suptitle(f"CLARA clusters — K={K_eff}", fontsize=12)
        fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.94])

        pdf.savefig(fig)  # writes current fig as a PDF page :contentReference[oaicite:3]{index=3}
        plt.close(fig)

    return pdf_path


def reconstruct_directional_spectrum(variance_density: np.ndarray,
                                     a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray,
                                     n_dir: int = 37) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Matches notebook cell that builds S_theta[t,f,dir] using:
      D = (1/pi) * (1/2 + a1 cosθ + b1 sinθ + a2 cos2θ + b2 sin2θ), clamped >=0,
      S(f,θ) = S(f) * D, then renormalize so integral over θ ~= S(f)
    Returns:
      S_theta: (n_time, n_freq, n_dir)
      dirs_deg: (n_dir,)
      delta_theta_rad: scalar
    """
    # Shapes: all (n_freq, n_time)
    n_freq, n_time = variance_density.shape
    dirs_deg = np.linspace(0.0, 360.0, n_dir)
    theta = np.deg2rad(dirs_deg)[None, None, :]  # (1,1,n_dir)
    delta_theta = np.deg2rad(10.0)  # same as notebook

    # Transpose to (n_time, n_freq)
    S = variance_density.T
    A1 = a1.T
    B1 = b1.T
    A2 = a2.T
    B2 = b2.T

    cos1, sin1 = np.cos(theta), np.sin(theta)
    cos2, sin2 = np.cos(2.0 * theta), np.sin(2.0 * theta)

    D = (1.0 / np.pi) * (0.5
                         + A1[..., None] * cos1 + B1[..., None] * sin1
                         + A2[..., None] * cos2 + B2[..., None] * sin2)
    D = np.maximum(D, 0.0)

    S_theta = S[..., None] * D  # (time,freq,dir)

    # Renormalize to preserve S(f) when integrating over theta
    integral = S_theta.sum(axis=2) * delta_theta  # (time,freq)
    integral = np.where(integral > 0, integral, 1.0)
    S_theta *= (S / integral)[..., None]

    return S_theta, dirs_deg, delta_theta


def plot_cluster_mean_and_examples(
    out_dir: str,
    K: int,
    freq_cut: np.ndarray,
    S_use: np.ndarray,          # (time,freq) OR (time,freq,dir)
    labels: np.ndarray,         # (time,)
    medoids: np.ndarray,        # indices into time axis
    dirs_deg: np.ndarray | None = None,
    delta_theta: float | None = None,
    plot_max_members: int = 250,
    plot_seed: int = 0,
    logy: bool = False
) -> None:
    """
    If S_use is 2D -> plots member S(f) + mean + medoid
    If S_use is 3D -> plots S(f) overlays + mean directional heatmap (as before)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available; skipping plots. Error: {e}")
        return

    rng = np.random.default_rng(plot_seed)

    plot_dir = os.path.join(out_dir, f"cluster_plots_K{K}")
    os.makedirs(plot_dir, exist_ok=True)

    is_dir = (S_use.ndim == 3)

    # compute omnidirectional spectra for overlay plot
    if is_dir:
        if delta_theta is None:
            raise ValueError("delta_theta must be provided for directional plotting.")
        S_omni = S_use.sum(axis=2) * float(delta_theta)  # (time,freq)
    else:
        S_omni = S_use  # already (time,freq)

    K_eff = int(labels.max()) + 1

    for c in range(K_eff):
        members = np.where(labels == c)[0]
        n_mem = members.size
        if n_mem == 0:
            continue

        if plot_max_members and plot_max_members > 0 and n_mem > plot_max_members:
            members_plot = rng.choice(members, size=plot_max_members, replace=False)
        else:
            members_plot = members

        mean_omni = S_omni[members].mean(axis=0)

        medoid_idx = int(medoids[c]) if c < len(medoids) else int(members[0])
        medoid_omni = S_omni[medoid_idx]

        if is_dir:
            mean_dir = S_use[members].mean(axis=0)  # (freq,dir)
            fig = plt.figure(figsize=(12, 4.2))
            ax1 = fig.add_subplot(1, 2, 1)
        else:
            fig = plt.figure(figsize=(7.2, 4.2))
            ax1 = fig.add_subplot(1, 1, 1)

        # overlays
        for i in members_plot:
            ax1.plot(freq_cut, S_omni[i], alpha=0.08, linewidth=0.8)
        ax1.plot(freq_cut, mean_omni, linewidth=2.5, label="Cluster mean")
        ax1.plot(freq_cut, medoid_omni, linestyle="--", linewidth=2.0, label="Cluster medoid")
        ax1.set_title(f"Cluster {c} — S(f) overlays (N={n_mem})")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Spectral density")
        if logy:
            ax1.set_yscale("log")
        ax1.legend(loc="best", frameon=True)

        if is_dir:
            ax2 = fig.add_subplot(1, 2, 2)
            im = ax2.imshow(
                mean_dir,
                origin="lower",
                aspect="auto",
                extent=[float(dirs_deg[0]), float(dirs_deg[-1]), float(freq_cut[0]), float(freq_cut[-1])]
            )
            ax2.set_title("Mean directional spectrum  $\u0304S(f,\\theta)$")
            ax2.set_xlabel("Direction (deg)")
            ax2.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"cluster_{c:02d}_mean_examples.png"), dpi=200)
        plt.close(fig)


def rotate_dirs_180(S_theta: np.ndarray) -> np.ndarray:
    """
    Notebook uses wavespectra interpolation to (180 + dir) mod 360.
    With 10-deg bins including 360, this is effectively a roll by 18 bins.
    """
    return np.roll(S_theta, shift=18, axis=2)


def select_freq_window(freq_vec: np.ndarray,
                       ini_lim_freq: float,
                       lim_freq: float) -> tuple[int, int]:
    """
    Mimic notebook indices:
      index_ini  = first index where freq < ini_lim_freq, +1
      index_freq = first index where freq > lim_freq, +1
    """
    freq_vec = np.asarray(freq_vec).squeeze()
    idx_start = np.where(freq_vec < ini_lim_freq)[0]
    idx_start = int(idx_start[0] + 1) if idx_start.size else 0

    idx_end = np.where(freq_vec > lim_freq)[0]
    idx_end = int(idx_end[0] + 1) if idx_end.size else len(freq_vec)

    idx_start = max(0, min(idx_start, len(freq_vec)))
    idx_end = max(idx_start + 1, min(idx_end, len(freq_vec)))
    return idx_start, idx_end


# ----------------------------
# CLARA (k-medoids subsampling)
# ----------------------------

def pam_build(D: np.ndarray, k: int) -> np.ndarray:
    n = D.shape[0]
    medoids = [int(np.argmin(D.sum(axis=1)))]
    nearest = D[:, medoids[0]].copy()
    for _ in range(1, k):
        candidates = [i for i in range(n) if i not in medoids]
        gains = [np.sum(np.maximum(0.0, nearest - D[:, h])) for h in candidates]
        h_best = candidates[int(np.argmax(gains))]
        medoids.append(h_best)
        nearest = np.minimum(nearest, D[:, h_best])
    return np.array(medoids, dtype=int)

def pam_swap(D: np.ndarray, medoids: np.ndarray) -> tuple[np.ndarray, float, bool]:
    n = D.shape[0]
    k = len(medoids)
    medoid_set = set(medoids.tolist())
    non_medoids = [i for i in range(n) if i not in medoid_set]

    dist_to_medoids = D[:, medoids]
    cur_d = np.min(dist_to_medoids, axis=1)
    cur_cost = float(cur_d.sum())

    best_change = 0.0
    best_pair = None

    for m_pos, m in enumerate(medoids):
        for h in non_medoids:
            new_meds = medoids.copy()
            new_meds[m_pos] = h
            new_cost = float(np.min(D[:, new_meds], axis=1).sum())
            change = new_cost - cur_cost
            if change < best_change:
                best_change = change
                best_pair = (m_pos, h, new_cost)

    if best_pair is None:
        return medoids, cur_cost, False

    m_pos, h, new_cost = best_pair
    new_medoids = medoids.copy()
    new_medoids[m_pos] = h
    return new_medoids, new_cost, True

def pam(D: np.ndarray, k: int, max_iter: int = 100) -> tuple[np.ndarray, float]:
    medoids = pam_build(D, k)
    cur_cost = float(np.min(D[:, medoids], axis=1).sum())
    for _ in range(max_iter):
        medoids2, cost2, improved = pam_swap(D, medoids)
        if (not improved) or (cost2 >= cur_cost - 1e-12):
            break
        medoids, cur_cost = medoids2, cost2
    return medoids, cur_cost

def clara(X: np.ndarray,
          k: int,
          metric: str = "manhattan",
          samples: int = 5,
          sampsize: int | None = None,
          seed: int = 0,
          max_pam_iter: int = 100,
          verbose: bool = False) -> dict:
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    if sampsize is None:
        sampsize = min(n, 40 + 2 * k)  # matches standard clara() default

    best_cost = np.inf
    best_medoids = None

    for s in range(samples):
        idx = rng.choice(n, size=sampsize, replace=False)
        Xs = X[idx]
        D = pairwise_distances(Xs, metric=metric)

        med_sub, _ = pam(D, k, max_iter=max_pam_iter)
        medoids = idx[med_sub]  # map back to original indices

        dist_full = pairwise_distances(X, X[medoids], metric=metric)
        cost = float(np.min(dist_full, axis=1).sum())

        if verbose:
            print(f"[CLARA] k={k:2d} sample {s+1}/{samples}: cost={cost:.6g}")

        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids

    dist_full = pairwise_distances(X, X[best_medoids], metric=metric)
    labels = np.argmin(dist_full, axis=1).astype(int)

    return {"labels": labels, "medoids": best_medoids, "cost": best_cost}

def _svd_pca_basis(X: np.ndarray):
    """
    Returns Vt from SVD so that scores Z = X @ Vt.T
    """
    # full_matrices=False is important for speed
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt


def sample_reference_H0(X: np.ndarray, rng: np.random.Generator, space: str) -> np.ndarray:
    """
    Generate a single reference dataset X* under H0 as in R cluster::clusGap:
      - "scaledPCA": center, scale, PCA-rotate, sample uniform in bounding box in PCA space, rotate back
      - "pca":       center only, PCA-rotate, sample uniform in bounding box in PCA space, rotate back
      - "box":       sample uniform in axis-aligned box in original space

    This matches the clusGap description: uniform on hypercube determined by ranges of x,
    after centering and PCA-rotating for spaceH0='scaledPCA'. :contentReference[oaicite:2]{index=2}
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    if space == "box":
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        return rng.uniform(lo, hi, size=(n, d))

    # PCA-based spaces
    mu = X.mean(axis=0)
    Xc = X - mu

    if space == "scaledPCA":
        sd = Xc.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        Xcs = Xc / sd
        Vt = _svd_pca_basis(Xcs)
        Z = Xcs @ Vt.T
        zmin = Z.min(axis=0)
        zmax = Z.max(axis=0)
        Zb = rng.uniform(zmin, zmax, size=Z.shape)
        Xb = (Zb @ Vt) * sd + mu
        return Xb

    elif space == "pca":
        Vt = _svd_pca_basis(Xc)
        Z = Xc @ Vt.T
        zmin = Z.min(axis=0)
        zmax = Z.max(axis=0)
        Zb = rng.uniform(zmin, zmax, size=Z.shape)
        Xb = (Zb @ Vt) + mu
        return Xb

    else:
        raise ValueError(f"Unknown space={space}")


def gap_statistic_for_clara(
    X: np.ndarray,
    ks: list[int],
    metric: str,
    samples: int,
    seed: int,
    B: int = 20,
    space: str = "scaledPCA",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute Gap(k) = E*[log Wk*] - log Wk with Wk from CLARA objective (sum distance to medoid),
    using B reference datasets.

    Also computes sk = sqrt(1+1/B) * sd( log Wk* ), as in the standard gap implementation. :contentReference[oaicite:3]{index=3}
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Observed logW(k)
    logW = {}
    medoids_obs = {}
    labels_obs = {}
    for k in ks:
        res = clara(X, k, metric=metric, samples=samples, seed=seed, verbose=False)
        Wk = float(res["cost"])
        logW[k] = np.log(Wk + 1e-300)
        medoids_obs[k] = res["medoids"]
        labels_obs[k] = res["labels"]
        if verbose:
            print(f"[GAP] observed k={k} logW={logW[k]:.6g}")

    # Reference logW*(k) across B
    logW_star = {k: np.empty(B, dtype=np.float64) for k in ks}

    for b in range(B):
        Xb = sample_reference_H0(X, rng, space=space)
        for k in ks:
            resb = clara(Xb, k, metric=metric, samples=samples, seed=seed + 10_000 + b, verbose=False)
            Wkb = float(resb["cost"])
            logW_star[k][b] = np.log(Wkb + 1e-300)
        if verbose:
            print(f"[GAP] bootstrap {b+1}/{B} done")

    rows = []
    for k in ks:
        mu_star = float(np.mean(logW_star[k]))
        sd_star = float(np.std(logW_star[k], ddof=1)) if B > 1 else 0.0
        sk = np.sqrt(1.0 + 1.0 / B) * sd_star  # standard gap SE factor :contentReference[oaicite:4]{index=4}
        gap = mu_star - float(logW[k])
        rows.append({
            "K": k,
            "logW": float(logW[k]),
            "E_logW_star": mu_star,
            "sd_logW_star": sd_star,
            "s_k": sk,
            "Gap": gap,
        })

    df_gap = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    return df_gap


def choose_k_by_gap(df_gap: pd.DataFrame, rule: str = "1se") -> int:
    """
    Tibshirani 1-SE rule:
      choose smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}. :contentReference[oaicite:5]{index=5}
    """
    Ks = df_gap["K"].to_numpy()
    G = df_gap["Gap"].to_numpy()
    s = df_gap["s_k"].to_numpy()

    if rule == "max":
        return int(Ks[int(np.argmax(G))])

    # 1se
    for i in range(len(Ks) - 1):
        if G[i] >= (G[i + 1] - s[i + 1]):
            return int(Ks[i])
    return int(Ks[int(np.argmax(G))])  # fallback


def laplace_aic_bic(X: np.ndarray, labels: np.ndarray, medoids_idx: np.ndarray) -> tuple[float, float, float, float]:
    """
    Heuristic AIC/BIC: treat each cluster as independent-dim Laplace around the medoid.
    This makes the k-medoids L1 objective correspond to (negative) log-likelihood.
    """
    n, d = X.shape
    k = len(medoids_idx)
    medoids = X[medoids_idx]
    l1 = float(np.abs(X - medoids[labels]).sum())
    b = l1 / (n * d)  # MLE for Laplace scale (per-dimension)
    logL = -n * d * math.log(2.0 * b) - (1.0 / b) * l1
    p = k * d + 1 + (k - 1)  # locations + scale + mixture weights (approx)
    aic = 2.0 * p - 2.0 * logL
    bic = p * math.log(n) - 2.0 * logL
    return aic, bic, logL, b


def knee_point(ks: list[int], vals: list[float]) -> int:
    """
    Simple "max distance to line" knee detector for a decreasing curve (elbow on cost).
    """
    x = np.array(ks, dtype=float)
    y = np.array(vals, dtype=float)
    # normalize to [0,1]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    # line from first to last
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    v = p2 - p1
    v_norm = v / (np.linalg.norm(v) + 1e-12)
    # distance to line
    dists = []
    for xi, yi in zip(x_n, y_n):
        p = np.array([xi, yi])
        proj = p1 + np.dot(p - p1, v_norm) * v_norm
        dists.append(np.linalg.norm(p - proj))
    return int(ks[int(np.argmax(dists))])


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True,
                    help="Folder containing the .mat files (e.g., ../data/ocean/wawaves)")
    ap.add_argument("--files", type=str, nargs="+", required=True,
                    help="List of .mat files to concatenate (e.g., Hillarys_202407.mat Hillarys_202408.mat)")
    ap.add_argument("--out-dir", type=str, default="clara_out")
    ap.add_argument("--metric", type=str, default="manhattan", choices=["manhattan", "euclidean"])
    ap.add_argument("--interpolate-dense", action="store_true",
                    help="Interpolate to dense frequency grid (default off; notebook had True)")
    ap.add_argument("--n-freq", type=int, default=200,
                    help="Dense frequency points if --interpolate-dense")
    ap.add_argument("--ini-lim-freq", type=float, default=0.040,
                    help="Lower cutoff used in notebook (Hz)")
    ap.add_argument("--use-startFreq-sea", action="store_true",
                    help="Use startFreq_sea as upper cutoff (default True if available)")
    ap.add_argument("--lim-freq", type=float, default=None,
                    help="Upper cutoff (Hz). If not set and --use-startFreq-sea, uses startFreq_sea from file.")
    ap.add_argument("--rotate-180", action="store_true",
                    help="Apply 180-deg direction roll like notebook's rotated_dirs step")
    ap.add_argument("--log1p", action="store_true",
                    help="Use log(1+x) transform before clustering")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=25)
    ap.add_argument("--samples", type=int, default=5, help="CLARA subsamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sil-sample", type=int, default=2000,
                    help="Approximate silhouette using this many points (0=skip)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--omni-only", action="store_true",
                    help="Cluster using non-directional spectra S(f) (varianceDensity) only")
    ap.add_argument("--shape-normalize", action="store_true",
                    help="Normalize each spectrum by its total energy (area) to cluster shape rather than energy")
    ap.add_argument("--make-plots", action="store_true",
                    help="Save per-cluster plots: mean + member examples")
    ap.add_argument("--plot-k", type=int, default=None,
                    help="Which K to plot (default: elbow K)")
    ap.add_argument("--plot-max-members", type=int, default=250,
                    help="Max member curves to overlay per cluster (0 = plot all)")
    ap.add_argument("--plot-seed", type=int, default=0,
                    help="Seed for sampling members to plot")
    ap.add_argument("--plot-logy", action="store_true",
                    help="Log-scale y-axis for 1D spectra overlay plots")
    ap.add_argument("--grid-pdf", action="store_true",
                    help="Save all clusters into a single PDF (4 columns, auto rows)")
    ap.add_argument("--grid-cols", type=int, default=4,
                    help="Number of columns in the grid (default 4)")
    ap.add_argument("--rows-per-page", type=int, default=None,
                    help="If set, split PDF into multiple pages with this many rows per page")
    ap.add_argument("--pdf-name", type=str, default=None,
                    help="PDF filename (default: clusters_grid_K{K}.pdf)")
    # ---- GAP statistic options ----
    ap.add_argument("--gap", action="store_true",
                    help="Compute Tibshirani gap statistic over K grid and choose K by 1-SE rule")
    ap.add_argument("--gap-b", type=int, default=20,
                    help="Number of reference bootstraps B for gap statistic (default 20)")
    ap.add_argument("--gap-space", type=str, default="scaledPCA",
                    choices=["scaledPCA", "pca", "box"],
                    help="Reference H0 space: 'scaledPCA' (like R cluster::clusGap), 'pca', or 'box'")
    ap.add_argument("--gap-rule", type=str, default="1se",
                    choices=["1se", "max"],
                    help="K selection: '1se' (Tibshirani rule) or 'max' (argmax Gap)")
    ap.add_argument("--gap-seed", type=int, default=0,
                    help="Random seed for gap reference sampling")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mat_paths = [os.path.join(args.data_root, f) for f in args.files]
    spot = load_spotdata_concat(mat_paths)

    arrays = {k: spot[k] for k in ["varianceDensity", "frequency", "a1", "b1", "a2", "b2"]}
    freq = arrays["frequency"]

    if args.interpolate_dense:
        fvec, dense = interpolate_freq_dense(freq, {k: arrays[k] for k in ["varianceDensity", "a1", "b1", "a2", "b2"]},
                                             n_points=args.n_freq)
        variance_density = dense["varianceDensity"]
        a1, b1, a2, b2 = dense["a1"], dense["b1"], dense["a2"], dense["b2"]
        freq_vec = fvec
    else:
        variance_density = arrays["varianceDensity"]
        a1, b1, a2, b2 = arrays["a1"], arrays["b1"], arrays["a2"], arrays["b2"]
        freq_vec = np.asarray(freq[:, 0]).squeeze()

    lim_freq = args.lim_freq
    if lim_freq is None and args.use_startFreq_sea and spot.get("startFreq_sea") is not None:
        # IMPORTANT: use robust scalar read (first element)
        lim_freq = float(np.array(spot["startFreq_sea"]).reshape(-1)[0])
    if lim_freq is None:
        raise ValueError("Upper cutoff lim_freq not set and startFreq_sea not found. Provide --lim-freq.")

    idx0, idx1 = select_freq_window(freq_vec, args.ini_lim_freq, lim_freq)
    freq_cut = np.asarray(freq_vec[idx0:idx1]).squeeze()

    if args.omni_only:
        # varianceDensity is (n_freq, n_time) -> transpose to (time, freq)
        S_omni = variance_density.T[:, idx0:idx1].astype(np.float64)  # (time, freq)

        if args.shape_normalize:
            area = S_omni.sum(axis=1, keepdims=True) + 1e-15
            S_omni = S_omni / area

        X = S_omni.copy()
        if args.log1p:
            X = np.log1p(X)

        # keep for plotting later
        S_use = S_omni # 2D

    else:
        # directional pipeline (your existing code)
        S_theta, dirs_deg, delta_theta = reconstruct_directional_spectrum(
            variance_density, a1, b1, a2, b2, n_dir=37
        )
        if args.rotate_180:
            S_theta = rotate_dirs_180(S_theta)

        S_use = S_theta[:, idx0:idx1, :]  # (time, freq, dir)
        X = S_use.reshape(S_use.shape[0], -1).astype(np.float64)
        if args.log1p:
            X = np.log1p(X)

    # Flatten to vectors
    X = S_use.reshape(S_use.shape[0], -1).astype(np.float64)
    if args.log1p:
        X = np.log1p(X)

    ks = list(range(args.k_min, args.k_max + 1))
    rows = []

    for k in ks:
        res = clara(X, k, metric=args.metric, samples=args.samples, seed=args.seed, verbose=args.verbose)
        labels = res["labels"]
        medoids = res["medoids"]
        cost = res["cost"]

        sil = np.nan
        if args.sil_sample and args.sil_sample > 0 and X.shape[0] > 2 and k > 1:
            ss = min(args.sil_sample, X.shape[0])
            sil = float(silhouette_score(X, labels, metric=args.metric, sample_size=ss, random_state=args.seed))

        aic, bic, logL, b = laplace_aic_bic(X, labels, medoids)

        np.save(os.path.join(args.out_dir, f"labels_K{k}.npy"), labels)
        np.save(os.path.join(args.out_dir, f"medoids_K{k}.npy"), medoids)

        rows.append({
            "K": k,
            "cost_L1": cost,
            "silhouette": sil,
            "AIC_laplace": aic,
            "BIC_laplace": bic,
            "logL_laplace": logL,
            "b_laplace": b
        })

        print(f"K={k:2d}  cost={cost:.6g}  silhouette={sil:.4f}  BIC={bic:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "clara_grid.csv"), index=False)

    # ---- GAP selection (optional) ----
    if args.gap:
        df_gap = gap_statistic_for_clara(
            X=X,
            ks=ks,
            metric=args.metric,
            samples=args.samples,
            seed=args.gap_seed,
            B=args.gap_b,
            space=args.gap_space,
            verbose=args.verbose,
        )
        df_gap.to_csv(os.path.join(args.out_dir, "gap_grid.csv"), index=False)

        k_gap = choose_k_by_gap(df_gap, rule=args.gap_rule)
        print(f"\n[GAP] Selected K by Gap ({args.gap_rule}, space={args.gap_space}, B={args.gap_b}): K = {k_gap}")

        # Plot Gap curve with error bars
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.errorbar(df_gap["K"], df_gap["Gap"], yerr=df_gap["s_k"], fmt="o-")
            plt.axvline(k_gap, linestyle="--")
            plt.xlabel("K")
            plt.ylabel("Gap(K)")
            plt.title(f"Gap statistic (space={args.gap_space}, B={args.gap_b})")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "gap_statistic.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[WARN] Gap plot skipped: {e}")

    # Choose K by elbow and by best silhouette / best BIC
    k_elbow = knee_point(df["K"].tolist(), df["cost_L1"].tolist())
    k_sil = int(df.loc[df["silhouette"].idxmax(), "K"]) if df["silhouette"].notna().any() else None
    k_bic = int(df.loc[df["BIC_laplace"].idxmin(), "K"])

    print("\n--- Suggested K ---")
    print(f"Elbow (knee on cost): K = {k_elbow}")
    if k_sil is not None:
        print(f"Max silhouette:       K = {k_sil}")
    print(f"Min BIC (heuristic):  K = {k_bic}")
    # ----- Plot cluster mean + examples for a selected K
    if args.make_plots:
        k_plot = int(args.plot_k) if args.plot_k is not None else int(k_elbow)

        labels_plot = np.load(os.path.join(args.out_dir, f"labels_K{k_plot}.npy"))
        medoids_plot = np.load(os.path.join(args.out_dir, f"medoids_K{k_plot}.npy"))

        plot_cluster_mean_and_examples(
            out_dir=args.out_dir,
            K=k_plot,
            freq_cut=freq_cut,
            S_use=S_use,
            labels=labels_plot,
            medoids=medoids_plot,
            dirs_deg=(dirs_deg if (S_use.ndim == 3) else None),
            delta_theta=(delta_theta if (S_use.ndim == 3) else None),
            plot_max_members=args.plot_max_members,
            plot_seed=args.plot_seed,
            logy=args.plot_logy
        )

        print(f"[OK] Saved plots to {os.path.join(args.out_dir, f'cluster_plots_K{k_plot}')}")

    if args.grid_pdf:
        k_plot = int(args.plot_k) if args.plot_k is not None else int(k_elbow)

        labels_plot = np.load(os.path.join(args.out_dir, f"labels_K{k_plot}.npy"))
        medoids_plot = np.load(os.path.join(args.out_dir, f"medoids_K{k_plot}.npy"))

        #freq_cut = np.asarray(freq_vec[idx0:idx1]).squeeze()
        freq_cut = np.asarray(freq_vec[idx0:idx1+20]).squeeze()

        pdf_path = save_clusters_grid_pdf(
            out_dir=args.out_dir,
            K=k_plot,
            freq_cut=freq_cut,
            S_use=variance_density.T[:, idx0:idx1+20].astype(np.float64),  # 2D omni OR 3D directional slice
            labels=labels_plot,
            medoids=medoids_plot,
            grid_cols=args.grid_cols,
            rows_per_page=args.rows_per_page,
            plot_max_members=args.plot_max_members,
            plot_seed=args.plot_seed,
            logy=args.plot_logy,
            pdf_name=(args.pdf_name or None),
            delta_theta=(delta_theta if (S_use.ndim == 3) else None),
        )
        print(f"[OK] Wrote cluster grid PDF: {pdf_path}")

    # Plots (optional)
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(df["K"], df["cost_L1"], marker="o")
        plt.xlabel("K")
        plt.ylabel("CLARA cost (sum L1 to medoid)")
        plt.title("Elbow on CLARA cost")
        plt.axvline(k_elbow, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "elbow_cost.png"), dpi=200)
        plt.close()

        if df["silhouette"].notna().any():
            plt.figure()
            plt.plot(df["K"], df["silhouette"], marker="o")
            plt.xlabel("K")
            plt.ylabel("Silhouette (approx)")
            plt.title("Silhouette vs K")
            if k_sil is not None:
                plt.axvline(k_sil, linestyle="--")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "silhouette.png"), dpi=200)
            plt.close()

    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")


if __name__ == "__main__":
    main()
