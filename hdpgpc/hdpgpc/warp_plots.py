#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
naive_cluster_basewarp_refit.py

Pipeline (no full-model reload from clustering labels):
  1) Select beats by predicted cluster_labels
  2) Fit naive model on selected raw beats (M=1) -> GP_raw
  3) Fit a *separate* base model on only the first beat (M=1) -> GP_base
  4) Compute warps + warped beats w.r.t GP_base
  5) Fit a third model on warped beats (M=1) -> GP_aligned
  6) Plot: raw+GP overlays, warps (with y=0), warped+GP overlays

Example:
  python naive_cluster_basewarp_refit.py 100 \
    --pred_dir results/cluster_labels/v1_UCR_ver \
    --cluster 3 \
    --labels N L R \
    --n_samples 20 \
    --lead 0 \
    --out_dir results/naive_warp_checks \
    --legend
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib as mpl
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14

import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.get_data import compute_estimators_LDS, included_labels as MIT_INCLUDED_LABELS
import hdpgpc.util_plots as uplt  # labels_trans + color

torch.set_default_dtype(torch.float64)


# ------------------------
# Paths
# ------------------------
def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for c in [here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent]:
        if (c / "hdpgpc").exists():
            return c
    return Path.cwd()


def find_data_dir(repo_root: Path) -> Path:
    for d in [repo_root / "data" / "mitdb", repo_root / "data" / "mitbih",
              repo_root / "hdpgpc" / "data" / "mitdb", repo_root / "hdpgpc" / "data" / "mitbih"]:
        if d.exists():
            return d
    raise FileNotFoundError("Could not find data directory (data/mitdb or data/mitbih).")


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
        raise FileExistsError(f"Multiple cluster-label files match record {rec}: {[m.name for m in matches]}")
    raise FileNotFoundError(f"No cluster-label file found for record {rec} in {pred_dir}")


# ------------------------
# Basic alignment/filtering
# ------------------------
def align_like_pipeline(data, y_true, cluster_labels,
                        included=MIT_INCLUDED_LABELS, drop_zero=True, eps=1e-12):
    data = np.asarray(data)
    y_true = np.asarray(y_true, dtype=object).astype(str).reshape(-1)
    cluster_labels = np.asarray(cluster_labels).reshape(-1)

    n = min(data.shape[0], y_true.shape[0], cluster_labels.shape[0])
    data, y_true, cluster_labels = data[:n], y_true[:n], cluster_labels[:n]

    inc = np.isin(y_true, np.asarray(included, dtype=str))
    data, y_true, cluster_labels = data[inc], y_true[inc], cluster_labels[inc]

    if drop_zero:
        energy = np.sum(np.abs(data), axis=(1, 2))
        nz = energy > eps
        data, y_true, cluster_labels = data[nz], y_true[nz], cluster_labels[nz]

    n = min(data.shape[0], y_true.shape[0], cluster_labels.shape[0])
    return data[:n], y_true[:n], np.asarray(cluster_labels[:n]).reshape(-1)


# ------------------------
# Model builder (same config as your scripts)
# ------------------------
def build_sw_gp(x_basis, x_basis_warp, num_outputs,
                sigma, gamma, outputscale_, ini_lengthscale, bound_lengthscale,
                noise_warp, bound_sigma, bound_gamma, bound_noise_warp):
    return hdpgp.GPI_HDP(
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


def col_fun(label_symbol: str) -> str:
    k = uplt.labels_trans.get(str(label_symbol), 0)
    return to_hex(uplt.color.get(k, "b"))


# ------------------------
# GP helpers
# ------------------------
def gp_mean_on_grid(gp, x_grid: torch.Tensor) -> np.ndarray:
    mean_, _ = gp.observe_last(x_grid)
    return mean_.detach().cpu().numpy().reshape(-1)


def gamma_band(gp):
    """
    Returns (x_basis_1d, mean_on_basis, sd_gamma_on_basis),
    where sd_gamma_on_basis = sqrt(diag(Gamma[-1])).
    """
    x_b_t = gp.x_basis  # (L,1)
    x_b = x_b_t.detach().cpu().numpy().reshape(-1)
    mean_b = gp_mean_on_grid(gp, x_b_t)

    G = gp.Gamma[-1]
    if isinstance(G, torch.Tensor):
        G = G.detach().cpu().numpy()
    sd_g = np.sqrt(np.clip(np.diag(G), 0.0, None))
    return x_b, mean_b, sd_g

def sigma_band(gp):
    """
    Returns (x_basis_1d, mean_on_basis, sd_gamma_on_basis),
    where sd_gamma_on_basis = sqrt(diag(Gamma[-1])).
    """
    x_b_t = gp.x_basis  # (L,1)
    x_b = x_b_t.detach().cpu().numpy().reshape(-1)
    mean_b = gp_mean_on_grid(gp, x_b_t)

    G = gp.Sigma[-1]
    if isinstance(G, torch.Tensor):
        G = G.detach().cpu().numpy()
    sd_g = np.sqrt(np.clip(np.diag(G), 0.0, None))
    return x_b, mean_b, sd_g


def full_box(ax, square_box=False):
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(False)
    if square_box:
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=str)
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--cluster", type=int, required=True, help="Cluster id (1-based by default).")
    ap.add_argument("--zero_based", action="store_true", help="Treat --cluster as 0-based.")
    ap.add_argument("--labels", nargs="*", default=None, help="Filter selected beats by TRUE labels (e.g. N L R).")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lead", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="results/naive_warp_checks")

    # opacity defaults reduced
    ap.add_argument("--raw_alpha", type=float, default=0.10)
    ap.add_argument("--warp_alpha", type=float, default=0.10)
    ap.add_argument("--warped_alpha", type=float, default=0.10)

    # Gamma band (darkblue)
    ap.add_argument("--gamma_band", action="store_true", help="Add darkblue Gamma shading around means.")
    ap.add_argument("--gamma_mult", type=float, default=1.96)
    ap.add_argument("--gamma_alpha", type=float, default=0.30)

    ap.add_argument("--square_box", action="store_true", help="Square axes boxes (set_box_aspect(1) if available).")
    ap.add_argument("--legend", action="store_true")
    args = ap.parse_args()

    rec = args.record
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cl = int(args.cluster)
    if not args.zero_based:
        cl -= 1
    if cl < 0:
        raise ValueError("Cluster must be >= 1 (or >= 0 with --zero_based).")

    repo_root = find_project_root()
    data_dir = find_data_dir(repo_root)

    # load
    data = np.load(data_dir / f"{rec}.npy")  # (T, L, D)
    y_true_raw = np.load(data_dir / f"{rec}_labels.npy", allow_pickle=True)
    pred_path = find_cluster_label_file(pred_dir, rec)
    cluster_labels = np.load(pred_path, allow_pickle=True)

    data, y_true, cluster_labels = align_like_pipeline(data, y_true_raw, cluster_labels)
    cluster_labels = np.rint(np.asarray(cluster_labels).reshape(-1)).astype(np.int64)

    T, L, D = data.shape
    if not (0 <= args.lead < D):
        raise ValueError(f"--lead must be in [0, {D-1}]")

    # ------------------------
    # Select beats by predicted cluster + optional true-label filter
    # ------------------------
    idxs = np.where(cluster_labels == cl)[0].tolist()
    if args.labels:
        allowed = set(map(str, args.labels))
        idxs = [i for i in idxs if str(y_true[i]) in allowed]
    if len(idxs) == 0:
        raise RuntimeError("No beats after filtering by cluster and/or true labels.")

    random.seed(args.seed)
    if args.n_samples > 0 and len(idxs) > args.n_samples:
        idxs = sorted(random.sample(idxs, k=args.n_samples))
    else:
        idxs = sorted(idxs)

    data_sel = data[idxs]    # (N, L, D)
    y_sel = y_true[idxs]     # (N,)
    N = data_sel.shape[0]

    # time supports
    x_basis = np.atleast_2d(np.arange(0, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(0, L, 2, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(0, L, 1, dtype=np.float64)).T
    x_trains = np.array([x_train] * N)

    # Estimate hypers ONCE from selected raw beats, reuse for all 3 models for comparability
    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data_sel)
    sigma = std * 3.0
    gamma = std_dif * 1.0
    outputscale_ = 300.0
    ini_lengthscale = 3.0
    bound_lengthscale = (1.0, 20.0)
    noise_warp = std * 0.1
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

    labels_one = np.zeros((N,), dtype=np.int64)

    # ------------------------
    # Model 1: naive raw-fit on selected beats (M=1) -> GP_raw
    # ------------------------
    sw_gp_raw = build_sw_gp(
        x_basis, x_basis_warp, D,
        sigma, gamma, outputscale_, ini_lengthscale, bound_lengthscale,
        noise_warp, bound_sigma, bound_gamma, bound_noise_warp
    )
    sw_gp_raw.reload_model_from_labels(x_trains, data_sel, labels_one, M=1)
    gp_raw = sw_gp_raw.gpmodels[args.lead][0]

    # ------------------------
    # Model 2: base GP trained ONLY on first selected beat (M=1) -> GP_base
    # ------------------------
    data_first = data_sel[:1].copy()     # (1, L, D)
    x_trains_first = np.array([x_train]) # (1, L, 1)
    labels_first = np.zeros((1,), dtype=np.int64)

    sw_gp_base = build_sw_gp(
        x_basis, x_basis_warp, D,
        sigma, gamma, outputscale_, ini_lengthscale, bound_lengthscale,
        noise_warp, bound_sigma, bound_gamma, bound_noise_warp
    )
    sw_gp_base.reload_model_from_labels(x_trains_first, data_first, labels_first, M=1)
    gp_base = sw_gp_base.gpmodels[args.lead][0]

    # ------------------------
    # Compute warps + warped observations w.r.t GP_base (NOT GP_raw)
    # ------------------------
    x_torch = torch.from_numpy(np.asarray(x_train, dtype=np.float64))
    warps = np.zeros((N, L), dtype=np.float64)
    warped_y = np.zeros((N, L), dtype=np.float64)

    print("Computing warp")
    for i in range(N):
        y_i = torch.from_numpy(np.asarray(data_sel[i, :, args.lead], dtype=np.float64)).reshape(-1, 1)
        y_w, x_w, _ = sw_gp_base.compute_warp_y(
            x_torch, y_i,
            strategie="standard",
            force_model=0,
            gpmodel=gp_base,
            ld=args.lead,
        )
        warps[i] = x_w[0].detach().cpu().numpy().reshape(-1)
        warped_y[i] = y_w[0].detach().cpu().numpy().reshape(-1)

    # ------------------------
    # Model 3: aligned-fit on warped beats (M=1) -> GP_aligned
    # ------------------------
    data_al = data_sel.copy()
    data_al[:, :, args.lead] = warped_y

    sw_gp_aligned = build_sw_gp(
        x_basis, x_basis_warp, D,
        sigma, gamma, outputscale_, ini_lengthscale, bound_lengthscale,
        noise_warp, bound_sigma, bound_gamma, bound_noise_warp
    )
    sw_gp_aligned.reload_model_from_labels(x_trains, data_al, labels_one, M=1)
    gp_aligned = sw_gp_aligned.gpmodels[args.lead][0]

    # ------------------------
    # Plot (no titles)
    # ------------------------
    # Dense grid for mean lines
    xb = gp_raw.x_basis.detach().cpu().numpy().reshape(-1)
    x_dense = torch.arange(float(np.min(xb)), float(np.max(xb)), args.step, dtype=torch.float64)
    x_dense_np = x_dense.detach().cpu().numpy()
    x_dense_t = torch.atleast_2d(x_dense).T

    m_raw = gp_mean_on_grid(gp_raw, x_dense_t)
    m_base = gp_mean_on_grid(gp_base, x_dense_t)
    m_al = gp_mean_on_grid(gp_aligned, x_dense_t)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax0, ax1, ax2 = axes

    for ax in axes:
        full_box(ax, square_box=args.square_box)

    # Axis 0: raw beats + GP overlays (raw/base/aligned)
    for i in range(N):
        ax0.plot(
            x_train.reshape(-1),
            data_sel[i, :, args.lead],
            color=col_fun(y_sel[i]),
            linewidth=1.2,
            alpha=float(args.raw_alpha),
        )

    ax0.plot(x_dense_np, m_raw, color="black", linewidth=2.0, label="GP_raw (fit on selected)")
    #ax0.plot(x_dense_np, m_base, color="gray", linewidth=2.0, linestyle=":", label="GP_base (fit on first beat)")
    #ax0.plot(x_dense_np, m_al, color="black", linewidth=2.0, linestyle="--", label="GP_aligned (fit on warped)")
    xb_r, mb_r_s, sds_r = sigma_band(gp_raw)
    galpha = float(args.gamma_alpha)
    ax0.fill_between(xb_r, mb_r_s - 1.96*sds_r, mb_r_s + 1.96*sds_r, color="lightblue", alpha=galpha, linewidth=0)

    if args.gamma_band:
        gmult = float(args.gamma_mult)
        galpha = float(args.gamma_alpha)

        xb_r, mb_r, sdg_r = gamma_band(gp_raw)
        #xb_b, mb_b, sdg_b = gamma_band(gp_base)
        #xb_a, mb_a, sdg_a = gamma_band(gp_aligned)

        ax0.fill_between(xb_r, mb_r - gmult*sdg_r, mb_r + gmult*sdg_r, color="darkblue", alpha=galpha, linewidth=0)
        #ax0.fill_between(xb_b, mb_b - gmult*sdg_b, mb_b + gmult*sdg_b, color="darkblue", alpha=galpha*0.85, linewidth=0)
        #ax0.fill_between(xb_a, mb_a - gmult*sdg_a, mb_a + gmult*sdg_a, color="darkblue", alpha=galpha*0.85, linewidth=0)

    if args.legend:
        ax0.legend(frameon=False, fontsize=9, loc="best")

    # Axis 1: warps + y=0 reference
    ax1.hlines(0.0, 0.0, 90.0, color="black", linewidth=1.0, alpha=0.35, linestyle="--")
    for i in range(N):
        ax1.plot(
            x_train.reshape(-1),
            warps[i],
            color=col_fun(y_sel[i]),
            linewidth=1.2,
            alpha=float(args.warp_alpha),
        )

    # Axis 2: warped beats + same GP overlays
    for i in range(N):
        ax2.plot(
            x_train.reshape(-1),
            warped_y[i],
            color=col_fun(y_sel[i]),
            linewidth=1.2,
            alpha=float(args.warped_alpha),
        )

    #ax2.plot(x_dense_np, m_raw, color="black", linewidth=2.0, label="GP_raw")
    #ax2.plot(x_dense_np, m_base, color="gray", linewidth=2.0, linestyle=":", label="GP_base")
    ax2.plot(x_dense_np, m_al, color="black", linewidth=2.0, label="GP_aligned")
    xb_r, mb_r_s, sds_r = sigma_band(gp_aligned)
    galpha = float(args.gamma_alpha)
    ax2.fill_between(xb_r, mb_r_s - 1.96 * sds_r, mb_r_s + 1.96 * sds_r, color="lightblue", alpha=galpha, linewidth=0)

    if args.gamma_band:
        gmult = float(args.gamma_mult)
        galpha = float(args.gamma_alpha)
        #xb_r, mb_r, sdg_r = gamma_band(gp_raw)
        #xb_b, mb_b, sdg_b = gamma_band(gp_base)
        xb_a, mb_a, sdg_a = gamma_band(gp_aligned)
        #ax2.fill_between(xb_r, mb_r - gmult*sdg_r, mb_r + gmult*sdg_r, color="darkblue", alpha=galpha, linewidth=0)
        #ax2.fill_between(xb_b, mb_b - gmult*sdg_b, mb_b + gmult*sdg_b, color="darkblue", alpha=galpha*0.85, linewidth=0)
        ax2.fill_between(xb_a, mb_a - gmult*sdg_a, mb_a + gmult*sdg_a, color="darkblue", alpha=galpha*0.85, linewidth=0)

    if args.legend:
        ax2.legend(frameon=False, fontsize=9, loc="best")

    # Only bottom x-label
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    ax2.set_xlabel("t")

    fig.tight_layout()

    base = out_dir / f"Rec{rec}_cl{cl+1}_baseWarp_refit_lead{args.lead}"
    fig.savefig(str(base) + ".png", dpi=220)
    fig.savefig(str(base) + ".pdf")
    plt.close(fig)

    print(f"[OK] Saved:\n  {base}.png\n  {base}.pdf")


if __name__ == "__main__":
    main()