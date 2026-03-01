#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, TwoSlopeNorm
import matplotlib as mpl
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.get_data import compute_estimators_LDS
import hdpgpc.util_plots as uplt

torch.set_default_dtype(torch.float64)


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
    )


def col_fun(lab):
    if isinstance(lab, (int, np.integer)):
        return to_hex(uplt.color.get(int(lab), "b"))
    s = str(lab)
    key = uplt.labels_trans.get(s, 0)
    return to_hex(uplt.color.get(key, "b"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=str)
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--zero_based", action="store_true")
    ap.add_argument("--lead", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="../results/eval_final_ver/cluster_A_plots")
    ap.add_argument("--dpi", type=int, default=220)
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

    data = np.load(data_dir / f"{rec}.npy")  # (T, L, D)
    labels_true = np.load(data_dir / f"{rec}_labels.npy", allow_pickle=True).reshape(-1)
    cl_path = find_cluster_label_file(pred_dir, rec)
    cluster_labels = np.load(cl_path, allow_pickle=True).reshape(-1)
    cluster_labels = np.rint(np.asarray(cluster_labels).reshape(-1)).astype(np.int64)

    n = min(data.shape[0], labels_true.shape[0], cluster_labels.shape[0])
    data = data[:n]
    labels_true = labels_true[:n]
    cluster_labels = cluster_labels[:n]

    T, L, D = data.shape
    if not (0 <= args.lead < D):
        raise ValueError(f"--lead must be in [0, {D-1}] (got {args.lead})")

    M = int(np.max(cluster_labels)) + 1
    if cl >= M:
        raise ValueError(f"Requested cluster {cl} but labels imply M={M} (0-based).")

    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data)
    sigma = std * 1.0
    gamma = std_dif * 1.4
    outputscale_ = 300.0
    ini_lengthscale = 3.0
    bound_lengthscale = (1.0, 20.0)

    noise_warp = std * 0.1
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

    x_basis = np.atleast_2d(np.arange(0, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(0, L, 2, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(0, L, 1, dtype=np.float64)).T
    x_trains = np.array([x_train] * T)

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

    idxs = sw_gp.gpmodels[0][cl].indexes
    if len(idxs) == 0:
        raise RuntimeError(f"Cluster {cl} is empty in the reloaded model.")
    vals, cnts = np.unique([labels_true[int(i)] for i in idxs], return_counts=True)
    maj_label = vals[int(np.argmax(cnts))] if len(cnts) else "None"

    gp = sw_gp.gpmodels[args.lead][cl]

    # ---- Layout: 2 columns only (plot | matrix). No colorbar axis at all.
    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.15)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # ---- Left: cluster plot (util_plots style)
    for j_, _ in enumerate(gp.y_train):
        j = int(gp.indexes[j_])
        x_t = gp.x_train[j_].T[0]
        d = sw_gp.y_train[j, :, [args.lead]]

        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
            x_t = x_t.detach().cpu().numpy()

        alpha = max(0.07, 0.5 / (np.log(len(gp.y_train) - j_ + 1) + 1))
        ax0.plot(
            x_t,
            d.T[0],
            alpha=alpha,
            color=col_fun(labels_true[j]),
            linewidth=1.2
        )

    x_b = gp.x_basis.T[0]
    x_ = torch.arange(float(torch.min(x_b)), float(torch.max(x_b)), args.step, dtype=torch.float64).cpu()
    mean_, Sig_ = gp.observe_last(torch.atleast_2d(x_).T)

    noise_ob = np.sqrt(np.diag(Sig_.detach().cpu().numpy()))
    mean = mean_.detach().cpu().numpy().reshape(-1)

    ax0.plot(x_.detach().cpu().numpy(), mean, color="black", linewidth=2)
    ax0.fill_between(
        x_.detach().cpu().numpy(),
        mean - 1.9 * noise_ob,
        mean + 1.9 * noise_ob,
        color=col_fun(maj_label),
        alpha=0.30,
        linewidth=0
    )

    mean_lat = gp.f_star_sm[-1].detach().cpu().numpy().reshape(-1)
    noise_lat = 1.9 * np.sqrt(np.diag(gp.Gamma[-1].detach().cpu().numpy()))
    xb_np = x_b.detach().cpu().numpy().reshape(-1)

    ax0.fill_between(
        xb_np,
        mean_lat - noise_lat,
        mean_lat + noise_lat,
        color=col_fun(maj_label),
        alpha=0.22,
        linewidth=0
    )


    # ---- Right: A - I heatmap (RdBu), square axes, NO colorbar
    A = gp.A[-1]
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    else:
        A = np.asarray(A)

    A_minus_I = A - np.eye(A.shape[0], dtype=A.dtype)

    # Center colormap at 0 so positive/negative deviations are balanced
    norm = TwoSlopeNorm(vcenter=0.0)

    ax1.imshow(
        A_minus_I,
        cmap="coolwarm",
        norm=norm,
        aspect="equal",          # square pixels
        interpolation=None
    )

    fig.tight_layout()      # square axes box

    base = out_dir / f"Rec{rec}_Cluster{cl+1}_Lead{args.lead}_AminusI"
    fig.savefig(str(base) + ".png", dpi=args.dpi, bbox_inches='tight')
    fig.savefig(str(base) + ".pdf", bbox_inches='tight')
    plt.close(fig)

    print(f"[OK] Saved:\n  {base}.png\n  {base}.pdf")


if __name__ == "__main__":
    main()