#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_record_snr_leads_with_interval_overlay.py

Reload HDP-GPC from cluster labels, then:
  - Plot per-observation SNR score psi[n,lead] across whole record (no artificial noise)
  - Optionally plot weights w[n,:] = softmax(psi[n,:])
  - If an interval is specified (mark_start/mark_end), also plot an overlay of the raw beat
    waveforms in that interval for each lead to visually identify noisy segments.

Outputs:
  - snr_over_record_<rec>.png/.pdf
  - snr_over_record_<rec>.csv
  - interval_overlay_<rec>_<start>-<end>.png/.pdf  (if interval provided)

Example:
  python plot_record_snr_leads_with_interval_overlay.py 100 \
      --pred_dir results/cluster_labels/v1_UCR_ver \
      --out_dir results/snr_balance \
      --plot_weights \
      --mark_start 800 --mark_end 950
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14

import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.get_data import compute_estimators_LDS

torch.set_default_dtype(torch.float64)


# ------------------------
# Path helpers
# ------------------------
def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for c in [here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent]:
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
        raise FileExistsError(f"Multiple cluster-label files for {rec}: {[m.name for m in matches]}")
    raise FileNotFoundError(f"No cluster-label file found for record {rec} in {pred_dir}")


# ------------------------
# Model builder (same style as your reload scripts)
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
        use_snr=True,
    )


# ------------------------
# SNR -> psi (max over clusters)
# ------------------------
@torch.no_grad()
def compute_snr_tensor(sw_gp, data_np: np.ndarray) -> torch.Tensor:
    """
    Compute snr[n, m, ld] for all observations n, clusters m, and leads ld.
    data_np: (T, L, D)
    Returns torch.Tensor (T, M, D)
    """
    y = torch.from_numpy(np.asarray(data_np, dtype=np.float64))  # (T, L, D)
    T, L, D = y.shape
    M = int(sw_gp.M)

    snr = torch.zeros((T, M, D), dtype=torch.float64)
    for ld in range(D):
        y_ld = y[:, :, ld]  # (T, L)
        for m in range(M):
            gp = sw_gp.gpmodels[ld][m]
            snr[:, m, ld] = sw_gp.compute_snr(y_ld, gp)
    return snr


@torch.no_grad()
def get_psi_and_weights(sw_gp, data_np: np.ndarray):
    """
    psi[n,ld] = max_m snr[n,m,ld]
    w[n,:] = softmax(psi[n,:]) across leads
    """
    # Prefer model-computed snr_last if present and compatible, else recompute.
    snr = None
    if hasattr(sw_gp, "snr_last") and sw_gp.snr_last is not None:
        snr = sw_gp.snr_last
        if not isinstance(snr, torch.Tensor):
            snr = torch.as_tensor(snr, dtype=torch.float64)
        snr = snr.detach().cpu()

    if snr is None or snr.ndim != 3:
        snr = compute_snr_tensor(sw_gp, data_np).detach().cpu()

    psi = torch.max(snr, dim=1).values  # (T, D)

    # stable softmax across leads
    psi_shift = psi - torch.max(psi, dim=1, keepdim=True).values
    w = torch.softmax(psi_shift, dim=1)

    return psi.detach().cpu().numpy(), w.detach().cpu().numpy()


# ------------------------
# Interval overlay plot
# ------------------------
def plot_interval_overlay(
    data_np: np.ndarray,
    s0: int,
    s1: int,
    out_dir: Path,
    rec: str,
    interval_alpha: float = 0.20,
    show_mean_band: bool = True,
    band_mult: float = 2.0,
):
    """
    Overlays all beats in [s0, s1) for each lead as semi-transparent curves.
    Also plots the mean waveform (thicker) and optional mean ± band_mult*std band.
    """
    seg = data_np[s0:s1]  # (K, L, D)
    K, L, D = seg.shape
    x = np.arange(L)

    fig, axes = plt.subplots(1, D, figsize=(12, 3.6), sharex=True, sharey=False)
    if D == 1:
        axes = [axes]
    ymin = np.min(seg) * 1.05
    ymax = np.max(seg) * 1.05
    for ld, ax in enumerate(axes):
        for k in range(K):
            ax.plot(x, seg[k, :, ld], linewidth=1.0, alpha=interval_alpha, color='b')

        mu = np.mean(seg[:, :, ld], axis=0)
        sd = np.std(seg[:, :, ld], axis=0)

        ax.plot(x, mu, linewidth=2.2, color="black")
        if show_mean_band:
            ax.fill_between(x, mu - band_mult * sd, mu + band_mult * sd, alpha=0.10, linewidth=0, color='lightblue')

        # Minimal but readable
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([])
        #ax.set_xlabel("t")
        #ax.set_ylabel(f"Lead {ld+1}")
        ax.grid(alpha=0.15)

    fig.tight_layout()
    base = out_dir / f"interval_overlay_{rec}_{s0}-{s1}"
    fig.savefig(str(base) + ".png", dpi=220, bbox_inches='tight')
    fig.savefig(str(base) + ".pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved interval overlay:\n  {base}.png\n  {base}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=str, help="MIT-BIH record id (e.g. 100)")
    ap.add_argument("--pred_dir", type=str, required=True, help="Folder containing cluster_labels_<rec>*.npy")
    ap.add_argument("--out_dir", type=str, default="../results/eval_final_ver/snr_balance", help="Output folder")
    ap.add_argument("--labels", nargs="*", default=None, help="Filter selected beats by TRUE labels (e.g. N L R).")
    ap.add_argument("--plot_weights", action="store_true",
                    help="Also plot weights w=softmax(psi) in a second panel.")
    ap.add_argument("--mark_start", type=int, default=None, help="Optional interval start index (beat index).")
    ap.add_argument("--mark_end", type=int, default=None, help="Optional interval end index (beat index).")
    ap.add_argument("--ylim", type=float, default=None, help="Optional y-limit for psi plot (upper limit).")
    ap.add_argument("--save_prefix", type=str, default=None, help="Optional filename prefix.")

    # overlay settings
    ap.add_argument("--interval_alpha", type=float, default=0.18,
                    help="Opacity for overlaid beat waveforms in interval overlay plot.")
    ap.add_argument("--no_mean_band", action="store_true",
                    help="Disable mean ± std band in interval overlay plot.")
    ap.add_argument("--band_mult", type=float, default=2.0,
                    help="Multiplier for std band: mean ± band_mult*std")

    args = ap.parse_args()

    rec = args.record
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = find_project_root()
    data_dir = find_data_dir(repo_root)

    # Load record
    data = np.load(data_dir / f"{rec}.npy")  # (T, L, D)
    y_true_raw = np.load(data_dir / f"{rec}_labels.npy", allow_pickle=True)
    T, L, D = data.shape

    # Load cluster labels
    cl_path = find_cluster_label_file(pred_dir, rec)
    cluster_labels = np.load(cl_path, allow_pickle=True).reshape(-1)
    if cluster_labels.dtype.kind in ("f", "c"):
        cluster_labels = np.rint(cluster_labels).astype(np.int64)
    else:
        cluster_labels = cluster_labels.astype(np.int64)

    # Align lengths (no additional filtering)
    n = min(T, len(cluster_labels))
    data = data[:n]
    cluster_labels = cluster_labels[:n]
    y_true = y_true_raw[:n]
    T = n

    M = int(np.max(cluster_labels)) + 1
    if M <= 0:
        raise ValueError("Computed M<=0 from cluster labels")

    # Time supports
    x_basis = np.atleast_2d(np.arange(0, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(0, L, 2, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(0, L, 1, dtype=np.float64)).T
    x_trains = np.array([x_train] * T)

    # Hyperparameters
    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data)
    sigma = std * 1.0
    gamma = std_dif * 1.0
    outputscale_ = 300.0
    ini_lengthscale = 3.0
    bound_lengthscale = (1.0, 20.0)
    noise_warp = std * 0.1
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

    # Reload model from labels
    sw_gp = build_sw_gp(
        x_basis=x_basis,
        x_basis_warp=x_basis_warp,
        num_outputs=D,
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
    sw_gp.reload_model_from_labels(x_trains, data, cluster_labels, M)

    # Compute psi + weights
    psi_np, w_np = get_psi_and_weights(sw_gp, data)

    # Save CSV
    csv_path = out_dir / f"snr_over_record_{rec}.csv"
    header = (
        "beat_index,"
        + ",".join([f"psi_lead{ld+1}" for ld in range(D)])
        + ","
        + ",".join([f"w_lead{ld+1}" for ld in range(D)])
    )
    rows = np.column_stack([np.arange(T), psi_np, w_np])
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")
    print(f"[OK] Wrote: {csv_path}")

    # Determine optional marked interval
    s0 = args.mark_start
    s1 = args.mark_end
    shade_ok = (s0 is not None) and (s1 is not None)
    if shade_ok:
        s0 = int(np.clip(s0, 0, T))
        s1 = int(np.clip(s1, 0, T))
        shade_ok = (s1 > s0)

    # Plot SNR / weights
    nrows = 2 if args.plot_weights else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 4.5 if nrows == 1 else 7.2), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax0 = axes[0]
    if shade_ok:
        ax0.axvspan(s0, s1, alpha=0.15)

    ax0.plot(np.arange(T), psi_np[:, 0], linewidth=1.5, label=f"Lead 1", color='#67a9cf')
    ax0.plot(np.arange(T), psi_np[:, 1], linewidth=1.5, label=f"Lead 2", color='#ef8a62')
    #ax0.set_ylabel("SNR score")
    if args.ylim is not None:
        ax0.set_ylim(None, float(args.ylim))
    ax0.grid(alpha=0.25)
    #ax0.legend(frameon=False, ncol=2, fontsize=9)

    if args.plot_weights:
        ax1 = axes[1]
        if shade_ok:
            ax1.axvspan(s0, s1, alpha=0.15)

        ax1.plot(np.arange(T), w_np[:, 0], linewidth=1.5, label=f"Lead 1 weight", color='#67a9cf')
        ax1.plot(np.arange(T), w_np[:, 1], linewidth=1.5, label=f"Lead 2 weight", color='#ef8a62')
        #ax1.set_ylabel("w (softmax)")
        ax1.set_ylim(-0.02, 1.02)
        ax1.grid(alpha=0.25)
        #ax1.legend(frameon=False, ncol=2, fontsize=9)
        #ax1.set_xlabel("Beat index n")
    #else:
        #ax0.set_xlabel("Beat index n")
    fig.tight_layout()

    prefix = args.save_prefix or f"snr_over_record_{rec}"
    png_path = out_dir / f"{prefix}.png"
    pdf_path = out_dir / f"{prefix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved:\n  {png_path}\n  {pdf_path}")

    # ------------------------
    # Select beats by predicted cluster + optional true-label filter
    # ------------------------
    idxs = np.arange(data.shape[0]).tolist()
    if args.labels:
        allowed = set(map(str, args.labels))
        idxs = [i for i in idxs if str(y_true[i]) in allowed]
    if len(idxs) == 0:
        raise RuntimeError("No beats after filtering by cluster and/or true labels.")

    # Interval overlay waveforms (new)
    if shade_ok:
        plot_interval_overlay(
            data_np=data[idxs],
            s0=s0,
            s1=s1,
            out_dir=out_dir,
            rec=rec,
            interval_alpha=float(args.interval_alpha),
            show_mean_band=(not args.no_mean_band),
            band_mult=float(args.band_mult),
        )


if __name__ == "__main__":
    main()