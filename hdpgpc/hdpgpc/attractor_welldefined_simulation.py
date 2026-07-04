# -*- coding: utf-8 -*-
"""
Observation-index transient simulation for static DP-GMM vs HDP-GPC.

This script generates 2 ordered groups of time-series segments.

IMPORTANT DISTINCTION
---------------------
The transient is NOT inside the time axis of an individual segment.

Each sample is a full time-series segment:

    data[i].shape = [T, D]

The transient happens along the OBSERVATION/SAMPLE INDEX inside cluster 1:

    cluster 0:
        similar morphology + higher random variability

    cluster 1:
        different morphology, and local samples 20,...,35 inside this cluster
        are progressively transformed and then return to the normal morphology.

The final dataset can be interleaved as:
    cluster0 sample 0, cluster1 sample 0, cluster0 sample 1, cluster1 sample 1, ...

This keeps the relative order within each cluster but mixes the observations globally.

Therefore, for cluster 1:

    sample local index r = 0,...,49

    Y_r(t) = base_segment(t)                         for most r
    Y_r(t) = base_segment(t) + a(r) * transform(...) for r = 20,...,35

where a(r) is a smooth bump over the observation index r, not over t.

Expected behaviour
------------------
Static clustering of segments may produce 3 or more clusters:
    - cluster 0 normal noisy morphology
    - cluster 1 normal morphology
    - transformed/excursion samples from cluster 1

HDP-GPC, if used in its dynamic/HMM mode over the ordered batch of samples,
should ideally keep the transformed samples as part of the same evolving group
rather than treating them as a third static morphology.

Run
---
From the parent folder containing hdpgpc/:

    python simulate_observation_index_transient_hdpgpc.py

Static baselines only:

    python simulate_observation_index_transient_hdpgpc.py --skip-hdpgpc
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in {"true", "1", "yes", "y"}:
        return True
    if str(v).lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected boolean.")


def ensure_repo_on_path(repo_root):
    repo_root = Path(repo_root).expanduser().resolve()
    candidates = [repo_root, repo_root.parent, Path.cwd().resolve(), Path.cwd().resolve().parent]
    for c in candidates:
        if (c / "hdpgpc").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    raise FileNotFoundError("Could not find hdpgpc/. Run from its parent folder or pass --repo-root.")


def compact_labels(labels):
    labels = np.asarray(labels)
    values = []
    for x in labels:
        if x not in values:
            values.append(x)
    mp = {v: i for i, v in enumerate(values)}
    return np.array([mp[x] for x in labels], dtype=int)


def count_active(weights, threshold=0.01):
    return int(np.sum(np.asarray(weights) > threshold))


def count_assigned(labels):
    return int(len(np.unique(labels)))


def to_numpy_safe(x):
    try:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def try_extract_hdpgpc_labels(sw_gp, n):
    """
    Your local implementation stores labels as:
        self.resp_assigned.append(torch.argmax(resp, axis=1))
    """
    if hasattr(sw_gp, "resp_assigned"):
        ra = sw_gp.resp_assigned
        last = ra[-1] if isinstance(ra, (list, tuple)) else ra
        arr = to_numpy_safe(last)
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == n:
                return compact_labels(arr.astype(int)), "sw_gp.resp_assigned[-1]"
            if arr.ndim == 2 and arr.shape[-1] == n:
                return compact_labels(arr[-1].astype(int)), "sw_gp.resp_assigned[-1][-1]"
            if arr.ndim == 2 and arr.shape[0] == n:
                return compact_labels(arr[:, -1].astype(int)), "sw_gp.resp_assigned[-1][:,-1]"
    return None, None


def smooth_bump_over_observations(n, start, end):
    """
    Smooth bump over OBSERVATION INDEX, not time.
    """
    a = np.zeros(n)
    idx = np.arange(start, end + 1)
    phase = (idx - start) / max(1, end - start)
    a[idx] = np.sin(np.pi * phase) ** 2
    return a


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
def make_base_morphologies(T, D):
    """
    Two related but distinguishable segment morphologies.
    """
    t = np.linspace(0, 1, T)

    # Shared components.
    p1 = np.exp(-0.5 * ((t - 0.28) / 0.08) ** 2)
    p2 = -0.65 * np.exp(-0.5 * ((t - 0.55) / 0.11) ** 2)
    p3 = 0.45 * np.exp(-0.5 * ((t - 0.78) / 0.07) ** 2)

    base_scalar_0 = 1.05 * p1 + 0.95 * p2 + 0.35 * p3

    # More clearly different morphology for cluster 1:
    # delayed first positive peak, deeper middle negative phase, and stronger late rebound.
    p1_shift = np.exp(-0.5 * ((t - 0.36) / 0.09) ** 2)
    p2_shift = -0.95 * np.exp(-0.5 * ((t - 0.60) / 0.10) ** 2)
    p3_shift = 0.80 * np.exp(-0.5 * ((t - 0.83) / 0.08) ** 2)
    base_scalar_1 = 0.72 * p1_shift + 1.10 * p2_shift + 0.95 * p3_shift

    # Smooth output loadings, made more distinct between groups.
    d = np.linspace(0, 2 * np.pi, D)
    load0 = 1.0 + 0.36 * np.sin(d) + 0.20 * np.cos(2 * d)
    load1 = 0.85 + 0.42 * np.cos(d + 0.45) - 0.22 * np.sin(2 * d) + 0.16 * np.cos(3 * d)

    # Add a weak cross-output temporal pattern to cluster 1 so PCA separates groups more clearly.
    cross_pattern = 0.18 * np.sin(2 * np.pi * t)[:, None] * np.sin(d)[None, :]

    base0 = base_scalar_0[:, None] * load0[None, :]
    base1 = base_scalar_1[:, None] * load1[None, :] + cross_pattern

    return base0, base1


def make_transformation_matrix(D, seed=123, nonlinear=False):
    rng = np.random.default_rng(seed)
    M = rng.normal(0, 1, size=(D, D))
    M = 0.7 * M + 0.3 * np.roll(M, shift=1, axis=0)
    M = M / (np.linalg.norm(M, ord=2) + 1e-12)
    return M


def simulate_observation_index_transient(
    seed=42,
    n_per_cluster=50,
    T=80,
    D=8,
    transient_start=20,
    transient_end=35,
    transient_amplitude=2.10,
    noise_cluster0=0.14,
    noise_cluster1=0.055,
    amplitude_jitter=0.07,
    offset_jitter=0.04,
    transformation_type="linear",
    interleave_clusters=True,
):
    """
    Returns
    -------
    data : [100, T, D]
    labels : [100]
        True group labels: first 50 are 0, second 50 are 1.
    transient_indicator : [100]
        1 for the transformed samples in cluster 1, else 0.
    severity : [100]
        Smooth transformation strength over the ordered observation index.
    """
    rng = np.random.default_rng(seed)
    base0, base1 = make_base_morphologies(T, D)
    M = make_transformation_matrix(D, seed=seed + 10)

    bump_local = smooth_bump_over_observations(
        n_per_cluster,
        start=transient_start,
        end=transient_end,
    )

    records = []

    for k in [0, 1]:
        base = base0 if k == 0 else base1
        sigma = noise_cluster0 if k == 0 else noise_cluster1

        for r in range(n_per_cluster):
            amp = np.exp(rng.normal(0, amplitude_jitter))
            offset = rng.normal(0, offset_jitter, size=D)

            Y = amp * base + offset[None, :]

            if k == 1:
                a = transient_amplitude * bump_local[r]

                if transformation_type == "linear":
                    transformed_direction = Y @ M.T
                elif transformation_type == "nonlinear":
                    transformed_direction = np.tanh(Y @ M.T)
                else:
                    raise ValueError("transformation_type must be 'linear' or 'nonlinear'.")

                Y = Y + a * transformed_direction
            else:
                a = 0.0

            # Cluster 0 has no structured excursion but more random variability.
            Y = Y + rng.normal(0, sigma, size=Y.shape)

            records.append(
                {
                    "Y": Y,
                    "true_group": k,
                    "local_index_in_group": r,
                    "transient_severity": a,
                    "is_transient_sample": int(k == 1 and bump_local[r] > 0.05),
                }
            )

    if interleave_clusters:
        ordered_records = []
        for r in range(n_per_cluster):
            ordered_records.extend(
                [rec for rec in records if rec["local_index_in_group"] == r and rec["true_group"] == 0]
            )
            ordered_records.extend(
                [rec for rec in records if rec["local_index_in_group"] == r and rec["true_group"] == 1]
            )
    else:
        ordered_records = records

    data = []
    labels = []
    transient_indicator = []
    severity = []
    meta = []

    for global_index, rec in enumerate(ordered_records):
        data.append(rec["Y"])
        labels.append(rec["true_group"])
        transient_indicator.append(rec["is_transient_sample"])
        severity.append(rec["transient_severity"])
        meta.append(
            {
                "true_group": rec["true_group"],
                "local_index_in_group": rec["local_index_in_group"],
                "global_index": global_index,
                "transient_severity": rec["transient_severity"],
                "is_transient_sample": rec["is_transient_sample"],
            }
        )

    return (
        np.asarray(data, dtype=np.float64),
        np.asarray(labels, dtype=int),
        np.asarray(transient_indicator, dtype=int),
        np.asarray(severity, dtype=np.float64),
        pd.DataFrame(meta),
        {"M": M, "bump_local": bump_local, "base0": base0, "base1": base1},
    )


# ---------------------------------------------------------------------
# Static baselines
# ---------------------------------------------------------------------
def fit_dpgmm(X, max_components, alpha, max_iter, reg_covar, covariance_type, seed):
    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=alpha,
        max_iter=max_iter,
        reg_covar=reg_covar,
        random_state=seed,
    )
    pred = compact_labels(model.fit_predict(X))
    return pred, count_active(model.weights_), model


def run_static_dpgmm_on_segments(data, labels, transient_indicator, out_dir, args):
    """
    Static clustering where each sample/segment is one flattened vector.
    """
    X = data.reshape(data.shape[0], -1)

    pred, active, _ = fit_dpgmm(
        X,
        max_components=args.dpgmm_max_components,
        alpha=args.dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.dpgmm_reg_covar,
        covariance_type=args.dpgmm_covariance_type,
        seed=args.seed,
    )

    ari = adjusted_rand_score(labels, pred)

    pd.DataFrame(
        {
            "sample_index": np.arange(data.shape[0]),
            "true_group": labels,
            "is_transient_sample": transient_indicator,
            "static_dpgmm_cluster": pred,
        }
    ).to_csv(out_dir / "static_dpgmm_segment_labels.csv", index=False)

    pd.crosstab(
        pd.Series(pred, name="static_cluster"),
        pd.Series(transient_indicator, name="is_transient_sample"),
    ).to_csv(out_dir / "static_dpgmm_transient_crosstab.csv")

    return {
        "method": "static_dp_gmm_on_flattened_segments",
        "assigned_clusters": count_assigned(pred),
        "active_components_by_weight": active,
        "ARI_vs_true_2_groups": float(ari),
    }, pred


# ---------------------------------------------------------------------
# HDP-GPC
# ---------------------------------------------------------------------
def run_hdpgpc(data, labels, out_dir, args):
    import hdpgpc.GPI_HDP as hdpgp
    from hdpgpc.get_data import compute_estimators_LDS
    from hdpgpc.util_plots import plot_models_plotly, print_results

    torch.set_default_dtype(torch.float64)

    n, T, D = data.shape
    std, std_dif, _, bound_gamma = compute_estimators_LDS(data)

    sigma = std * args.sigma_multiplier
    gamma = std_dif * args.gamma_multiplier

    bound_sigma = (
        np.maximum(sigma * args.bound_sigma_low_multiplier, args.sigma_floor),
        np.maximum(sigma * args.bound_sigma_high_multiplier, args.sigma_floor * 10.0),
    )

    outputscale = (
        float(np.max(np.abs(data)) * args.outputscale_multiplier + 1e-8)
        if args.outputscale is None
        else float(args.outputscale)
    )

    noise_warp = std * args.noise_warp_multiplier
    bound_noise_warp = (
        np.maximum(noise_warp * args.bound_noise_warp_low_multiplier, args.sigma_floor),
        np.maximum(noise_warp * args.bound_noise_warp_high_multiplier, args.sigma_floor * 10.0),
    )

    x_basis = np.atleast_2d(np.arange(T, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(0, T, args.warp_basis_step, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(T, dtype=np.float64)).T
    x_trains = np.array([x_train] * n)

    sw_gp = hdpgp.GPI_HDP(
        x_basis,
        x_basis_warp=x_basis_warp,
        n_outputs=D,
        kernels=None,
        model_type=args.hdpgpc_model_type,
        ini_lengthscale=args.ini_lengthscale,
        bound_lengthscale=(args.bound_lengthscale_low, args.bound_lengthscale_high),
        ini_gamma=gamma,
        ini_sigma=sigma,
        ini_outputscale=outputscale,
        noise_warp=noise_warp,
        bound_sigma=bound_sigma,
        bound_gamma=bound_gamma,
        bound_noise_warp=bound_noise_warp,
        warp_updating=args.warp_updating,
        method_compute_warp=args.method_compute_warp,
        verbose=args.verbose_hdpgpc,
        hmm_switch=args.hmm_switch,
        max_models=args.max_models,
        use_snr=False,
        mode_warp=args.mode_warp,
        bayesian_params=args.bayesian_params,
        inducing_points=args.inducing_points,
        reestimate_initial_params=args.reestimate_initial_params,
        n_explore_steps=args.n_explore_steps,
        free_deg_MNIV=args.free_deg_MNIV,
        share_gp=args.share_gp,
        hdp_hyp='less'
    )

    start = time.time()
    sw_gp.include_batch(x_trains, data, warp=args.warp)
    elapsed_min = (time.time() - start) / 60.0

    main_model = print_results(sw_gp, labels, 0, error=False)
    selected_gpmodels = sw_gp.selected_gpmodels()

    for lead in range(min(D, args.max_plot_outputs)):
        try:
            plot_models_plotly(
                sw_gp,
                selected_gpmodels,
                main_model,
                labels,
                0,
                lead=lead,
                save=str(out_dir / f"hdpgpc_offline_clusters_output{lead}.png"),
                step=0.5,
                plot_latent=True,
            )
        except Exception as exc:
            warnings.warn(f"plot_models_plotly failed for output {lead}: {repr(exc)}")

    pred, source = try_extract_hdpgpc_labels(sw_gp, n)

    if pred is None:
        return {
            "method": "actual_hdpgpc",
            "assigned_clusters": np.nan,
            "ARI_vs_true_2_groups": np.nan,
            "elapsed_min": elapsed_min,
            "assignment_source": "not_found",
        }, None

    ari = adjusted_rand_score(labels, pred)

    pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "true_group": labels,
            "hdpgpc_cluster": pred,
        }
    ).to_csv(out_dir / "hdpgpc_segment_labels.csv", index=False)

    return {
        "method": "actual_hdpgpc",
        "assigned_clusters": count_assigned(pred),
        "active_components_by_weight": np.nan,
        "ARI_vs_true_2_groups": float(ari),
        "elapsed_min": float(elapsed_min),
        "assignment_source": source,
    }, pred


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_sample_index_pca(data, labels, transient_indicator, severity, out_path):
    """
    PCA over flattened segments. This is the most important plot:
    it shows the transient happening over OBSERVATION INDEX.
    """
    X = data.reshape(data.shape[0], -1)
    Z = PCA(n_components=2, random_state=0).fit_transform(X)

    plt.figure(figsize=(9.0, 6.0))

    # Draw ordered paths within each true group, preserving the relative order
    # after global interleaving.
    for k in [0, 1]:
        idx = np.where(labels == k)[0]
        idx = idx[np.argsort(idx)]
        plt.plot(Z[idx, 0], Z[idx, 1], alpha=0.35, linewidth=1.5)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=35, alpha=0.75, label=f"true group {k}")

    # Highlight transformed samples.
    trans = transient_indicator == 1
    plt.scatter(
        Z[trans, 0],
        Z[trans, 1],
        s=90,
        marker="x",
        linewidths=2.0,
        label="group 1 transformed samples",
    )

    # Annotate every 5th sample index for readability.
    for i in range(data.shape[0]):
        local_i = i if labels[i] == 0 else i - np.sum(labels == 0)
        if local_i % 5 == 0:
            plt.text(Z[i, 0], Z[i, 1], str(local_i), fontsize=8, alpha=0.7)

    plt.title("PCA of full time-series segments: transient occurs over sample index")
    plt.xlabel("segment PCA 1")
    plt.ylabel("segment PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_clustered_sample_index_pca(data, cluster_labels, title, out_path):
    X = data.reshape(data.shape[0], -1)
    Z = PCA(n_components=2, random_state=0).fit_transform(X)

    plt.figure(figsize=(8.5, 5.8))
    for k in sorted(np.unique(cluster_labels)):
        idx = np.where(cluster_labels == k)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=45, alpha=0.8, label=f"cluster {k}")

    # Preserve relative ordering path within each true group if available
    # from the interleaved sequence.

    plt.title(title)
    plt.xlabel("segment PCA 1")
    plt.ylabel("segment PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_severity(severity, labels, out_path):
    plt.figure(figsize=(8.5, 3.2))
    plt.plot(severity, linewidth=2.5)
    plt.axvline(50, linestyle="--", linewidth=1.0)
    plt.title("Transformation severity over observation/sample index")
    plt.xlabel("global sample index")
    plt.ylabel("transformation strength")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_example_segments(data, labels, transient_indicator, out_dir, max_outputs=3):
    """
    Shows time-series segment morphology, but transient is between samples,
    not within time support.
    """
    for d in range(min(max_outputs, data.shape[2])):
        plt.figure(figsize=(9.0, 5.0))

        # Group 0 noisy examples.
        idx0 = np.where(labels == 0)[0]
        for i in idx0[:12]:
            plt.plot(data[i, :, d], color="gray", alpha=0.25, linewidth=1.0)
        plt.plot(data[idx0, :, d].mean(axis=0), linewidth=3.0, label="group 0 mean")

        # Group 1 normal and transformed examples.
        idx1_normal = np.where((labels == 1) & (transient_indicator == 0))[0]
        idx1_trans = np.where((labels == 1) & (transient_indicator == 1))[0]

        for i in idx1_normal[:8]:
            plt.plot(data[i, :, d], alpha=0.20, linewidth=1.0)
        for i in idx1_trans[:8]:
            plt.plot(data[i, :, d], alpha=0.45, linewidth=1.4, linestyle="--")

        plt.plot(data[idx1_normal, :, d].mean(axis=0), linewidth=3.0, label="group 1 normal mean")
        plt.plot(data[idx1_trans, :, d].mean(axis=0), linewidth=3.0, linestyle="--", label="group 1 transformed mean")

        plt.title(f"Output {d}: segment shapes; transformation affects selected samples, not time axis")
        plt.xlabel("within-segment support index")
        plt.ylabel(f"y(t, output {d})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"segment_examples_output{d}.png", dpi=180)
        plt.close()


# ---------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="results_observation_index_transient")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-cluster", type=int, default=80)
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--n-outputs", type=int, default=2)
    p.add_argument("--transient-start", type=int, default=30)
    p.add_argument("--transient-end", type=int, default=50)
    p.add_argument("--transient-amplitude", type=float, default=0.90)
    p.add_argument("--noise-cluster0", type=float, default=0.14)
    p.add_argument("--noise-cluster1", type=float, default=0.025)
    p.add_argument("--amplitude-jitter", type=float, default=0.07)
    p.add_argument("--offset-jitter", type=float, default=0.04)
    p.add_argument("--transformation-type", type=str, default="linear", choices=["linear", "nonlinear"])
    p.add_argument(
        "--interleave-clusters",
        type=str2bool,
        default=True,
        help=(
            "If true, output order is c0_0, c1_0, c0_1, c1_1, ... . "
            "Relative order within each cluster is preserved while the two clusters are mixed."
        ),
    )

    p.add_argument("--skip-hdpgpc", action="store_true")

    p.add_argument("--dpgmm-max-components", type=int, default=8)
    p.add_argument("--dpgmm-alpha", type=float, default=0.05)
    p.add_argument("--dpgmm-max-iter", type=int, default=800)
    p.add_argument("--dpgmm-reg-covar", type=float, default=1e-5)
    p.add_argument("--dpgmm-covariance-type", type=str, default="diag", choices=["full", "tied", "diag", "spherical"])

    # HDP-GPC parameters
    p.add_argument("--warp", type=str2bool, default=False)
    p.add_argument("--warp-updating", type=str2bool, default=False)
    p.add_argument("--method-compute-warp", type=str, default="greedy")
    p.add_argument("--mode-warp", type=str, default="rough")
    p.add_argument("--warp-basis-step", type=int, default=2)
    p.add_argument("--hdpgpc-model-type", type=str, default="dynamic")
    p.add_argument("--hmm-switch", type=str2bool, default=True)
    p.add_argument("--max-models", type=int, default=100)
    p.add_argument("--bayesian-params", type=str2bool, default=True)
    p.add_argument("--inducing-points", type=str2bool, default=False)
    p.add_argument("--reestimate-initial-params", type=str2bool, default=False)
    p.add_argument("--n-explore-steps", type=int, default=5)
    p.add_argument("--free-deg-MNIV", type=int, default=3)
    p.add_argument("--share-gp", type=str2bool, default=True)
    p.add_argument("--verbose-hdpgpc", action="store_true")
    p.add_argument("--max-plot-outputs", type=int, default=4)

    p.add_argument("--sigma-multiplier", type=float, default=1.0)
    p.add_argument("--gamma-multiplier", type=float, default=1.0)
    p.add_argument("--sigma-floor", type=float, default=1e-8)
    p.add_argument("--bound-sigma-low-multiplier", type=float, default=1e-6)
    p.add_argument("--bound-sigma-high-multiplier", type=float, default=1e1)
    p.add_argument("--outputscale", type=float, default=None)
    p.add_argument("--outputscale-multiplier", type=float, default=1.2)
    p.add_argument("--ini-lengthscale", type=float, default=8.0)
    p.add_argument("--bound-lengthscale-low", type=float, default=1.0)
    p.add_argument("--bound-lengthscale-high", type=float, default=45.0)
    p.add_argument("--noise-warp-multiplier", type=float, default=60.0)
    p.add_argument("--bound-noise-warp-low-multiplier", type=float, default=0.01)
    p.add_argument("--bound-noise-warp-high-multiplier", type=float, default=60.0)

    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    data, labels, transient_indicator, severity, meta, params = simulate_observation_index_transient(
        seed=args.seed,
        n_per_cluster=args.n_per_cluster,
        T=args.T,
        D=args.n_outputs,
        transient_start=args.transient_start,
        transient_end=args.transient_end,
        transient_amplitude=args.transient_amplitude,
        noise_cluster0=args.noise_cluster0,
        noise_cluster1=args.noise_cluster1,
        amplitude_jitter=args.amplitude_jitter,
        offset_jitter=args.offset_jitter,
        transformation_type=args.transformation_type,
        interleave_clusters=args.interleave_clusters,
    )

    np.save(out_dir / "data.npy", data)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "transient_indicator.npy", transient_indicator)
    np.save(out_dir / "transient_severity.npy", severity)
    np.save(out_dir / "transformation_matrix_M.npy", params["M"])
    np.save(out_dir / "base_morphology_cluster0.npy", params["base0"])
    np.save(out_dir / "base_morphology_cluster1.npy", params["base1"])
    meta.to_csv(out_dir / "metadata.csv", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"data.shape = {data.shape}")
    print(f"labels = {np.bincount(labels)}")
    print(f"transient samples in group 1 local index {args.transient_start}:{args.transient_end}")
    print(f"interleave_clusters = {args.interleave_clusters}")
    print(f"out_dir = {out_dir}")

    plot_sample_index_pca(data, labels, transient_indicator, severity, out_dir / "true_sample_index_transient_pca.png")
    plot_severity(severity, labels, out_dir / "transient_severity_over_sample_index.png")
    plot_example_segments(data, labels, transient_indicator, out_dir, max_outputs=args.max_plot_outputs)

    results = []

    static_res, static_pred = run_static_dpgmm_on_segments(data, labels, transient_indicator, out_dir, args)
    results.append(static_res)
    print(pd.DataFrame([static_res]).to_string(index=False))
    plot_clustered_sample_index_pca(
        data,
        static_pred,
        f"Static DP-GMM on full segments ({static_res['assigned_clusters']} clusters)",
        out_dir / "static_dpgmm_segment_pca_clusters.png",
    )

    if not args.skip_hdpgpc:
        try:
            ensure_repo_on_path(args.repo_root)
            hdpgpc_res, hdpgpc_pred = run_hdpgpc(data, labels, out_dir, args)
            results.append(hdpgpc_res)
            print(pd.DataFrame([hdpgpc_res]).to_string(index=False))

            if hdpgpc_pred is not None:
                plot_clustered_sample_index_pca(
                    data,
                    hdpgpc_pred,
                    f"HDP-GPC on ordered segments ({hdpgpc_res['assigned_clusters']} clusters)",
                    out_dir / "hdpgpc_segment_pca_clusters.png",
                )
        except Exception as exc:
            warnings.warn(f"HDP-GPC failed: {repr(exc)}")
            results.append(
                {
                    "method": "actual_hdpgpc",
                    "assigned_clusters": np.nan,
                    "active_components_by_weight": np.nan,
                    "ARI_vs_true_2_groups": np.nan,
                    "elapsed_min": np.nan,
                    "assignment_source": f"failed: {repr(exc)}",
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "comparison_results.csv", index=False)

    print("\nComparison")
    print(results_df.to_string(index=False))

    print(
        "\nInterpretation:\n"
        "  The transient is over observation/sample index, not over the segment support axis.\n"
        "  The global order is interleaved by default as c0_0, c1_0, c0_1, c1_1, ... ,\n"
        "  so the relative order inside each cluster is preserved while observations are mixed.\n"
        "  Static DP-GMM sees the transformed samples as a different morphology and may split\n"
        "  them as an extra cluster. HDP-GPC is given the ordered segment sequence; if its\n"
        "  dynamic/HMM machinery is active, it may retain the transient as part of the same\n"
        "  evolving group rather than creating a third static morphology.\n"
    )


if __name__ == '__main__':
    main()