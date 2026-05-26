# -*- coding: utf-8 -*-
"""
Non-rotational dynamic-system simulation using the actual hdpgpc.GPI_HDP API.

Goal
----
Construct two dynamic systems that occupy the same 2D curve but evolve in opposite
temporal directions. A static nonparametric clustering method sees only spatial
locations and tends to split the curve into several spatial components. HDP-GPC,
when applied to complete trajectories, should recover the two original dynamic
systems.

This script compares:

1. Static nonparametric clustering:
   BayesianGaussianMixture with a Dirichlet-process prior, applied to pooled
   2D points (x_t, y_t).

2. Actual HDP-GPC:
   hdpgpc.GPI_HDP, applied to full trajectories with shape:
       [num_samples, num_obs_per_sample, num_outputs]

Expected data shape
-------------------
data.shape = [N, T, 2]
labels.shape = [N]

Run from the parent folder containing the `hdpgpc` package, for example:

    python simulate_nonrotation_actual_hdpgpc.py

or specify the repository root explicitly:

    python simulate_nonrotation_actual_hdpgpc.py --repo-root C:/Users/Adrian/Projects/YourProject

Useful options:

    python simulate_nonrotation_actual_hdpgpc.py --warp
    python simulate_nonrotation_actual_hdpgpc.py --n-traj-per-system 15 --T 45
    python simulate_nonrotation_actual_hdpgpc.py --standardize
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from scipy import stats
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def ensure_repo_on_path(repo_root):
    """
    Make imports work whether the script is run from:
        - the project root containing hdpgpc/
        - inside hdpgpc/
        - another directory with --repo-root specified
    """
    repo_root = Path(repo_root).expanduser().resolve()

    candidates = [
        repo_root,
        repo_root.parent,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    ]

    for c in candidates:
        if (c / "hdpgpc").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c

    raise FileNotFoundError(
        "Could not find an `hdpgpc` folder. Run from the parent folder "
        "containing `hdpgpc`, or pass --repo-root."
    )


def compact_labels(labels):
    labels = np.asarray(labels)
    unique = []
    for value in labels:
        if value not in unique:
            unique.append(value)
    mapping = {value: i for i, value in enumerate(unique)}
    return np.array([mapping[value] for value in labels], dtype=int)


def majority_vote(values):
    values = np.asarray(values)
    mode = stats.mode(values, keepdims=False)
    return int(mode.mode)


def infer_trajectory_labels_from_point_clusters(point_clusters, n_traj, T):
    point_clusters = np.asarray(point_clusters).reshape(n_traj, T)
    return np.array([majority_vote(point_clusters[i]) for i in range(n_traj)])


def count_active_mixture_components(weights, threshold=0.01):
    return int(np.sum(np.asarray(weights) > threshold))


def to_numpy_safe(value):
    """
    Convert Torch tensors, NumPy arrays, lists, or tuples to NumPy when possible.
    """
    try:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass

    try:
        return np.asarray(value)
    except Exception:
        return None


def try_extract_hdpgpc_labels(sw_gp, num_samples):
    """
    Best-effort extractor.

    In the local HDP-GPC implementation, labels are computed as:

        self.resp_assigned.append(torch.argmax(resp, axis=1))

    Therefore the preferred extraction is:

        sw_gp.resp_assigned[-1]

    which should contain one final assigned model/cluster index per trajectory.

    If that attribute is unavailable, this function falls back to other common
    names so the script remains robust to older package versions.
    """

    # ------------------------------------------------------------------
    # Preferred local HDP-GPC label source:
    #     self.resp_assigned.append(torch.argmax(resp, axis=1))
    # ------------------------------------------------------------------
    if hasattr(sw_gp, "resp_assigned"):
        try:
            resp_assigned = getattr(sw_gp, "resp_assigned")

            # Usually a list where each entry is a torch tensor of shape [N].
            if isinstance(resp_assigned, (list, tuple)) and len(resp_assigned) > 0:
                final_assigned = resp_assigned[-1]
            else:
                final_assigned = resp_assigned

            arr = to_numpy_safe(final_assigned)

            if arr is not None:
                arr = np.asarray(arr)

                # Expected case: [num_samples]
                if arr.ndim == 1 and arr.shape[0] == num_samples:
                    return compact_labels(arr.astype(int)), "sw_gp.resp_assigned[-1]"

                # Possible case: [iterations, num_samples] or [num_samples, iterations]
                if arr.ndim == 2:
                    if arr.shape[-1] == num_samples:
                        return compact_labels(arr[-1, :].astype(int)), "sw_gp.resp_assigned[-1][-1, :]"
                    if arr.shape[0] == num_samples:
                        return compact_labels(arr[:, -1].astype(int)), "sw_gp.resp_assigned[-1][:, -1]"

        except Exception as exc:
            warnings.warn(f"Could not extract labels from sw_gp.resp_assigned: {repr(exc)}")

    # ------------------------------------------------------------------
    # Fallbacks for alternative package versions.
    # ------------------------------------------------------------------
    candidate_attr_names = [
        "labels",
        "label",
        "z",
        "Z",
        "assignments",
        "assignment",
        "cluster_assignments",
        "model_assignments",
        "selected_models",
        "selected_model",
        "model_index",
        "model_indices",
        "main_model",
        "model_id",
        "model_ids",
        "d",
        "s",
    ]

    for name in candidate_attr_names:
        if hasattr(sw_gp, name):
            value = getattr(sw_gp, name)
            arr = to_numpy_safe(value)
            if arr is None:
                continue

            try:
                arr = np.asarray(arr)
                if arr.ndim == 1 and arr.shape[0] == num_samples:
                    return compact_labels(arr.astype(int)), f"sw_gp.{name}"
                if arr.ndim == 2 and arr.shape[0] == num_samples:
                    return compact_labels(arr[:, -1].astype(int)), f"sw_gp.{name}[:, -1]"
                if arr.ndim == 2 and arr.shape[-1] == num_samples:
                    return compact_labels(arr[-1, :].astype(int)), f"sw_gp.{name}[-1, :]"
            except Exception:
                pass

    # Try parameter/state dictionaries if present.
    candidate_dict_attr_names = [
        "params",
        "parameters",
        "state",
        "states",
        "model_state",
        "__dict__",
    ]

    for dict_name in candidate_dict_attr_names:
        if not hasattr(sw_gp, dict_name):
            continue

        try:
            dct = getattr(sw_gp, dict_name)
        except Exception:
            continue

        if not isinstance(dct, dict):
            continue

        for key, value in dct.items():
            key_lower = str(key).lower()
            if not any(token in key_lower for token in ["resp_assigned", "label", "assign", "cluster", "model", "z"]):
                continue

            arr = to_numpy_safe(value)
            if arr is None:
                continue

            try:
                arr = np.asarray(arr)

                # Special handling if the dictionary stores a list of tensors.
                if arr.dtype == object and len(arr) > 0:
                    arr = to_numpy_safe(value[-1])
                    if arr is None:
                        continue
                    arr = np.asarray(arr)

                if arr.ndim == 1 and arr.shape[0] == num_samples:
                    return compact_labels(arr.astype(int)), f"sw_gp.{dict_name}['{key}']"
                if arr.ndim == 2 and arr.shape[0] == num_samples:
                    return compact_labels(arr[:, -1].astype(int)), f"sw_gp.{dict_name}['{key}'][:, -1]"
                if arr.ndim == 2 and arr.shape[-1] == num_samples:
                    return compact_labels(arr[-1, :].astype(int)), f"sw_gp.{dict_name}['{key}'][-1, :]"
            except Exception:
                pass

    return None, None


def save_true_trajectory_plot(data, labels, out_path):
    plt.figure(figsize=(7.5, 5.2))
    for i in range(data.shape[0]):
        linestyle = "-" if labels[i] == 0 else "--"
        plt.plot(
            data[i, :, 0],
            data[i, :, 1],
            linestyle=linestyle,
            alpha=0.65,
            linewidth=1.6,
        )
        # Direction arrow near the end of the trajectory.
        plt.annotate(
            "",
            xy=data[i, -1, :2],
            xytext=data[i, -4, :2],
            arrowprops=dict(arrowstyle="->", linewidth=0.8),
        )

    plt.title("True dynamic systems: same curve, opposite temporal direction")
    plt.xlabel("output 1: x(t)")
    plt.ylabel("output 2: y(t)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_static_dpgmm_plot(points, point_clusters, out_path, title):
    plt.figure(figsize=(7.5, 5.2))
    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=point_clusters,
        s=15,
        alpha=0.75,
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_hdpgpc_assignment_plot(data, labels_pred, out_path, title):
    plt.figure(figsize=(7.5, 5.2))
    for k in sorted(set(labels_pred)):
        idx = np.where(labels_pred == k)[0]
        for i in idx:
            plt.plot(
                data[i, :, 0],
                data[i, :, 1],
                alpha=0.70,
                linewidth=1.6,
            )
            plt.annotate(
                "",
                xy=data[i, -1, :2],
                xytext=data[i, -4, :2],
                arrowprops=dict(arrowstyle="->", linewidth=0.8),
            )
    plt.title(title)
    plt.xlabel("output 1: x(t)")
    plt.ylabel("output 2: y(t)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def save_hdpgpc_points_colored_plot(data, trajectory_labels, out_path, title):
    """
    Plot all 2D observations, coloring each point by the HDP-GPC cluster assigned
    to its parent trajectory.
    """
    n_traj, T, n_outputs = data.shape
    if n_outputs < 2:
        raise ValueError("This plot expects at least 2 output dimensions.")

    plt.figure(figsize=(7.5, 5.2))
    for k in sorted(set(np.asarray(trajectory_labels).tolist())):
        idx = np.where(np.asarray(trajectory_labels) == k)[0]
        pts = data[idx, :, :2].reshape(-1, 2)
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=16,
            alpha=0.75,
            label=f"HDP-GPC cluster {k}",
        )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
def simulate_same_curve_opposite_direction(
    seed=42,
    n_traj_per_system=10,
    T=35,
    obs_noise=0.035,
    amp_jitter=0.02,
    phase_jitter=0.012,
):
    """
    Non-rotational dynamic example.

    System 0 moves left-to-right along a nonlinear curve:
        x(t) = -3 + 6t
        y(t) = f(x(t))

    System 1 moves right-to-left along the same curve:
        x(t) = 3 - 6t
        y(t) = f(x(t))

    The two systems have nearly identical static support in the 2D observation
    space, but opposite temporal evolution.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)

    trajectories = []
    labels = []

    for system in [0, 1]:
        for _ in range(n_traj_per_system):
            phase = rng.normal(0.0, phase_jitter)
            amp = 1.0 + rng.normal(0.0, amp_jitter)

            if system == 0:
                s = np.clip(t + phase, 0.0, 1.0)
            else:
                s = np.clip(1.0 - t + phase, 0.0, 1.0)

            x_clean = -3.0 + 6.0 * s
            y_clean = amp * (
                1.05 * np.sin(1.35 * x_clean)
                + 0.30 * np.sin(2.6 * x_clean)
            )

            x = x_clean + rng.normal(0.0, obs_noise, T)
            y = y_clean + rng.normal(0.0, obs_noise, T)

            trajectories.append(np.stack([x, y], axis=1))
            labels.append(system)

    data = np.asarray(trajectories, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    return data, labels


def maybe_standardize(data, enabled):
    if not enabled:
        return data, None

    mean = data.reshape(-1, data.shape[-1]).mean(axis=0)
    std = data.reshape(-1, data.shape[-1]).std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)

    data_std = (data - mean[None, None, :]) / std[None, None, :]

    return data_std, {"mean": mean.tolist(), "std": std.tolist()}



# ---------------------------------------------------------------------
# Ordered pooled-point construction
# ---------------------------------------------------------------------
def build_ordered_pooled_windows(data, labels, window_length=12, stride=6, order_mode="time_then_system"):
    """
    Build short ordered sequences from pooled 2D points.

    This intentionally discards the original trajectory identity but keeps a
    chosen ordering of the pooled points. HDP-GPC can then be run on these
    short windows.

    Parameters
    ----------
    data : array, shape [N, T, D]
        Original trajectories.
    labels : array, shape [N]
        True dynamic-system labels at trajectory level.
    window_length : int
        Number of ordered points per HDP-GPC sample/window.
    stride : int
        Step between consecutive windows.
    order_mode : str
        - "time_then_system":
            order all points first by time index, then by true system, then by
            trajectory index. This keeps global temporal ordering while pooling.
        - "time":
            order by time index, then trajectory index.
        - "x":
            order by x coordinate.
        - "curve":
            order by x coordinate and then y coordinate.
        - "original":
            flatten trajectories in their original storage order.

    Returns
    -------
    pooled_windows : array, shape [M, window_length, D]
    window_labels : array, shape [M]
        Majority true dynamic-system label in each window. This is used only
        for evaluation, not for fitting.
    ordered_points : array, shape [N*T, D]
    ordered_point_labels : array, shape [N*T]
    """
    data = np.asarray(data)
    labels = np.asarray(labels)

    n_traj, T, n_outputs = data.shape

    points = data.reshape(-1, n_outputs)
    point_labels = np.repeat(labels, T)

    traj_ids = np.repeat(np.arange(n_traj), T)
    time_ids = np.tile(np.arange(T), n_traj)

    if order_mode == "time_then_system":
        order = np.lexsort((traj_ids, point_labels, time_ids))
    elif order_mode == "time":
        order = np.lexsort((traj_ids, time_ids))
    elif order_mode == "x":
        order = np.argsort(points[:, 0])
    elif order_mode == "curve":
        order = np.lexsort((points[:, 1], points[:, 0]))
    elif order_mode == "original":
        order = np.arange(points.shape[0])
    else:
        raise ValueError(
            "Unknown order_mode. Use one of: "
            "'time_then_system', 'time', 'x', 'curve', 'original'."
        )

    ordered_points = points[order]
    ordered_point_labels = point_labels[order]

    if window_length < 2:
        raise ValueError("window_length should be at least 2 for a dynamic HDP-GPC run.")

    windows = []
    window_labels = []

    for start in range(0, len(ordered_points) - window_length + 1, stride):
        stop = start + window_length
        windows.append(ordered_points[start:stop])
        window_labels.append(majority_vote(ordered_point_labels[start:stop]))

    if len(windows) == 0:
        raise ValueError(
            "No ordered pooled windows were created. Reduce --pooled-window-length "
            "or --pooled-window-stride."
        )

    return (
        np.asarray(windows, dtype=np.float64),
        np.asarray(window_labels, dtype=int),
        ordered_points,
        ordered_point_labels,
    )


def save_ordered_pooled_windows_plot(ordered_points, ordered_point_labels, out_path, title):
    """
    Plot the ordered pooled point cloud colored by the true generating system.
    This is only a diagnostic plot to show what information remains after pooling.
    """
    plt.figure(figsize=(7.5, 5.2))
    plt.scatter(
        ordered_points[:, 0],
        ordered_points[:, 1],
        c=ordered_point_labels,
        s=15,
        alpha=0.70,
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------
# Static nonparametric baseline
# ---------------------------------------------------------------------
def run_static_dpgmm(data, labels, out_dir, args):
    n_traj, T, n_outputs = data.shape
    if n_outputs < 2:
        raise ValueError("This static baseline expects 2D observations.")

    points = data.reshape(-1, n_outputs)[:, :2]
    repeated_labels = np.repeat(labels, T)

    static_model = BayesianGaussianMixture(
        n_components=args.dpgmm_max_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=args.dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        random_state=args.seed,
        reg_covar=args.dpgmm_reg_covar,
    )

    point_clusters = static_model.fit_predict(points)
    active_components = count_active_mixture_components(
        static_model.weights_,
        threshold=args.active_weight_threshold,
    )

    traj_majority_labels = infer_trajectory_labels_from_point_clusters(
        point_clusters,
        n_traj=n_traj,
        T=T,
    )

    point_ari = adjusted_rand_score(repeated_labels, point_clusters)
    traj_majority_ari = adjusted_rand_score(labels, traj_majority_labels)

    point_clusters_by_traj = point_clusters.reshape(n_traj, T)
    components_per_traj = [
        len(np.unique(point_clusters_by_traj[i]))
        for i in range(n_traj)
    ]

    save_static_dpgmm_plot(
        points,
        point_clusters,
        out_dir / "static_dpgmm_pooled_points.png",
        (
            "Static DP-GMM on pooled 2D points "
            f"({active_components} active spatial components)"
        ),
    )

    return {
        "method": "static_dp_gmm_pooled_points",
        "input_used": "pooled (x_t, y_t) locations",
        "active_or_inferred_clusters": active_components,
        "ARI_point_clusters_vs_repeated_system_labels": float(point_ari),
        "ARI_trajectory_majority_label_vs_system": float(traj_majority_ari),
        "mean_static_components_per_trajectory": float(np.mean(components_per_traj)),
        "median_static_components_per_trajectory": float(np.median(components_per_traj)),
    }


# ---------------------------------------------------------------------
# Actual HDP-GPC
# ---------------------------------------------------------------------
def run_actual_hdpgpc(data, labels, out_dir, args, repo_root, tag="trajectory", method_name="actual_hdpgpc_gpi_hdp"):
    """
    Run the actual hdpgpc.GPI_HDP API using a configuration close to the ECG
    example supplied by the user.
    """

    # Imports are intentionally inside the function so that the static baseline
    # can still be tested even if hdpgpc is not importable.
    import hdpgpc.GPI_HDP as hdpgp
    from hdpgpc.get_data import compute_estimators_LDS
    from hdpgpc.util_plots import plot_models_plotly, print_results

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    prefix = f"{tag}_"

    num_samples, num_obs_per_sample, num_outputs = data.shape

    std, std_dif, bound_sigma_from_data, bound_gamma = compute_estimators_LDS(data)

    # ECG-inspired priors, scaled to the simulated data.
    sigma = std * args.sigma_multiplier
    gamma = std_dif * args.gamma_multiplier

    # In the ECG example:
    #     bound_sigma = (sigma*1e-6, sigma*1e-4)
    # For very small synthetic data, the lower and upper bounds can become tiny.
    # The floor avoids numerical degeneracy while preserving the same relative
    # structure.
    sigma_floor = args.sigma_floor
    bound_sigma = (
        np.maximum(sigma * args.bound_sigma_low_multiplier, sigma_floor),
        np.maximum(sigma * args.bound_sigma_high_multiplier, sigma_floor * 10.0),
    )

    # The output scale should match the data amplitude. For ECG, the user used
    # 300.0. For this synthetic example, using data amplitude is safer.
    if args.outputscale is None:
        outputscale_ = float(np.max(np.abs(data)) * args.outputscale_multiplier + 1e-8)
    else:
        outputscale_ = float(args.outputscale)

    ini_lengthscale = args.ini_lengthscale
    bound_lengthscale = (args.bound_lengthscale_low, args.bound_lengthscale_high)

    noise_warp = std * args.noise_warp_multiplier
    bound_noise_warp = (
        np.maximum(noise_warp * args.bound_noise_warp_low_multiplier, sigma_floor),
        np.maximum(noise_warp * args.bound_noise_warp_high_multiplier, sigma_floor * 10.0),
    )

    samples = [0, num_obs_per_sample]
    l, L = samples[0], samples[1]

    x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(l, L, args.warp_basis_step, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T
    x_trains = np.array([x_train] * num_samples)

    sw_gp = hdpgp.GPI_HDP(
        x_basis,
        x_basis_warp=x_basis_warp,
        n_outputs=num_outputs,
        kernels=None,
        model_type=args.hdpgpc_model_type,
        ini_lengthscale=ini_lengthscale,
        bound_lengthscale=bound_lengthscale,
        ini_gamma=gamma,
        ini_sigma=sigma,
        ini_outputscale=outputscale_,
        noise_warp=noise_warp,
        bound_sigma=bound_sigma,
        bound_gamma=bound_gamma,
        bound_noise_warp=bound_noise_warp,
        warp_updating=args.warp_updating,
        method_compute_warp=args.method_compute_warp,
        verbose=args.verbose_hdpgpc,
        hmm_switch=args.hmm_switch,
        max_models=args.max_models,
        mode_warp=args.mode_warp,
        bayesian_params=args.bayesian_params,
        inducing_points=args.inducing_points,
        reestimate_initial_params=args.reestimate_initial_params,
        n_explore_steps=args.n_explore_steps,
        free_deg_MNIV=args.free_deg_MNIV,
        share_gp=args.share_gp,
        hdp_hyp='less',
    )

    start = time.time()
    sw_gp.include_batch(x_trains, data, warp=args.warp)
    elapsed_min = (time.time() - start) / 60.0

    # Use package utilities, as in the ECG example.
    print("\nHDP-GPC package results:")
    main_model = print_results(sw_gp, labels, 0, error=False)
    selected_gpmodels = sw_gp.selected_gpmodels()

    # Try to save the official HDP-GPC plots for each output.
    for lead in range(num_outputs):
        try:
            plot_models_plotly(
                sw_gp,
                selected_gpmodels,
                main_model,
                labels,
                0,
                lead=lead,
                save=str(out_dir / f"{prefix}hdpgpc_offline_clusters_output{lead}.png"),
                step=0.5,
                plot_latent=True,
            )
        except Exception as exc:
            warnings.warn(
                f"plot_models_plotly failed for output {lead}: {repr(exc)}"
            )

    labels_pred, source = try_extract_hdpgpc_labels(sw_gp, num_samples)

    if labels_pred is not None:
        hdpgpc_ari = adjusted_rand_score(labels, labels_pred)
        n_clusters = len(np.unique(labels_pred))

        # Save final recovered HDP-GPC labels for direct inspection/reuse.
        np.save(out_dir / f"{prefix}hdpgpc_extracted_trajectory_labels.npy", labels_pred)
        pd.DataFrame(
            {
                "trajectory_index": np.arange(num_samples),
                "true_system": labels,
                "hdpgpc_cluster": labels_pred,
            }
        ).to_csv(out_dir / f"{prefix}hdpgpc_extracted_trajectory_labels.csv", index=False)

        save_hdpgpc_assignment_plot(
            data,
            labels_pred,
            out_dir / f"{prefix}hdpgpc_extracted_assignments_2d.png",
            (
                "HDP-GPC extracted trajectory assignments "
                f"({n_clusters} clusters, ARI={hdpgpc_ari:.3f})"
            ),
        )
        save_hdpgpc_points_colored_plot(
            data,
            labels_pred,
            out_dir / f"{prefix}hdpgpc_points_colored_by_cluster.png",
            (
                "2D points colored by HDP-GPC trajectory cluster "
                f"({n_clusters} clusters, ARI={hdpgpc_ari:.3f})"
            ),
        )
    else:
        hdpgpc_ari = np.nan
        n_clusters = np.nan
        source = "not_found"

    # Save a small diagnostics file with relevant hyperparameters.
    diagnostics = {
        "elapsed_min": elapsed_min,
        "num_samples": int(num_samples),
        "num_obs_per_sample": int(num_obs_per_sample),
        "num_outputs": int(num_outputs),
        "warp": bool(args.warp),
        "outputscale": outputscale_,
        "ini_lengthscale": ini_lengthscale,
        "bound_lengthscale": list(bound_lengthscale),
        "sigma": np.asarray(sigma).tolist(),
        "gamma": np.asarray(gamma).tolist(),
        "bound_sigma_low": np.asarray(bound_sigma[0]).tolist(),
        "bound_sigma_high": np.asarray(bound_sigma[1]).tolist(),
        "bound_gamma": [
            np.asarray(bound_gamma[0]).tolist(),
            np.asarray(bound_gamma[1]).tolist(),
        ] if isinstance(bound_gamma, tuple) and len(bound_gamma) == 2 else str(bound_gamma),
        "noise_warp": np.asarray(noise_warp).tolist(),
        "bound_noise_warp_low": np.asarray(bound_noise_warp[0]).tolist(),
        "bound_noise_warp_high": np.asarray(bound_noise_warp[1]).tolist(),
        "assignment_source": source,
    }

    with open(out_dir / f"{prefix}hdpgpc_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    return {
        "method": method_name,
        "input_used": "full trajectories x(t), y(t)",
        "active_or_inferred_clusters": n_clusters,
        "ARI_point_clusters_vs_repeated_system_labels": np.nan,
        "ARI_trajectory_majority_label_vs_system": float(hdpgpc_ari) if not np.isnan(hdpgpc_ari) else np.nan,
        "mean_static_components_per_trajectory": np.nan,
        "median_static_components_per_trajectory": np.nan,
        "elapsed_min": float(elapsed_min),
        "assignment_source": source,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Non-rotational dynamic simulation with actual HDP-GPC."
    )

    parser.add_argument("--repo-root", type=str, default=".", help="Path to project root containing hdpgpc/.")
    parser.add_argument("--out-dir", type=str, default="results_nonrotation_hdpgpc", help="Output directory.")

    # Simulation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-traj-per-system", type=int, default=10)
    parser.add_argument("--T", type=int, default=35)
    parser.add_argument("--obs-noise", type=float, default=0.035)
    parser.add_argument("--amp-jitter", type=float, default=0.02)
    parser.add_argument("--phase-jitter", type=float, default=0.012)
    parser.add_argument("--standardize", action="store_true")

    # Static DP-GMM
    parser.add_argument("--dpgmm-max-components", type=int, default=12)
    parser.add_argument("--dpgmm-alpha", type=float, default=0.03)
    parser.add_argument("--dpgmm-max-iter", type=int, default=800)
    parser.add_argument("--dpgmm-reg-covar", type=float, default=1e-5)
    parser.add_argument("--active-weight-threshold", type=float, default=0.01)

    # HDP-GPC control
    parser.add_argument("--skip-hdpgpc", action="store_true", help="Only run the static baseline.")
    parser.add_argument(
        "--skip-ordered-pooled-hdpgpc",
        action="store_true",
        help="Skip the additional HDP-GPC run over ordered pooled 2D point windows.",
    )
    parser.add_argument(
        "--pooled-order-mode",
        type=str,
        default="time_then_system",
        choices=["time_then_system", "time", "x", "curve", "original"],
        help=(
            "Ordering used before forming pooled point windows. "
            "'time_then_system' preserves time and groups the two generating systems; "
            "'time' preserves only global time; 'x'/'curve' sort spatially."
        ),
    )
    parser.add_argument("--pooled-window-length", type=int, default=12)
    parser.add_argument("--pooled-window-stride", type=int, default=6)
    parser.add_argument("--warp", action="store_true", help="Use HDP-GPC warping in include_batch.")
    parser.add_argument("--warp-updating", action="store_true")
    parser.add_argument("--method-compute-warp", type=str, default="greedy")
    parser.add_argument("--mode-warp", type=str, default="rough")
    parser.add_argument("--warp-basis-step", type=int, default=2)

    parser.add_argument("--hdpgpc-model-type", type=str, default="dynamic")
    parser.add_argument("--hmm-switch", type=str2bool, default=True)
    parser.add_argument("--max-models", type=int, default=100)
    parser.add_argument("--bayesian-params", type=str2bool, default=True)
    parser.add_argument("--inducing-points", type=str2bool, default=False)
    parser.add_argument("--reestimate-initial-params", type=str2bool, default=False)
    parser.add_argument("--n-explore-steps", type=int, default=5)
    parser.add_argument("--free-deg-MNIV", type=int, default=3)
    parser.add_argument("--share-gp", type=str2bool, default=True)
    parser.add_argument("--verbose-hdpgpc", action="store_true")

    # HDP-GPC priors, close to user's ECG example but scale-safe
    parser.add_argument("--sigma-multiplier", type=float, default=0.5)
    parser.add_argument("--gamma-multiplier", type=float, default=1.5)
    parser.add_argument("--sigma-floor", type=float, default=1e-8)
    parser.add_argument("--bound-sigma-low-multiplier", type=float, default=1e-6)
    parser.add_argument("--bound-sigma-high-multiplier", type=float, default=1e-4)

    parser.add_argument("--outputscale", type=float, default=None)
    parser.add_argument("--outputscale-multiplier", type=float, default=1.2)
    parser.add_argument("--ini-lengthscale", type=float, default=3.0)
    parser.add_argument("--bound-lengthscale-low", type=float, default=1.0)
    parser.add_argument("--bound-lengthscale-high", type=float, default=20.0)

    parser.add_argument("--noise-warp-multiplier", type=float, default=60.0)
    parser.add_argument("--bound-noise-warp-low-multiplier", type=float, default=0.01)
    parser.add_argument("--bound-noise-warp-high-multiplier", type=float, default=60.0)

    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = ensure_repo_on_path(args.repo_root)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    data_raw, labels = simulate_same_curve_opposite_direction(
        seed=args.seed,
        n_traj_per_system=args.n_traj_per_system,
        T=args.T,
        obs_noise=args.obs_noise,
        amp_jitter=args.amp_jitter,
        phase_jitter=args.phase_jitter,
    )

    data, standardization = maybe_standardize(data_raw, args.standardize)

    # Persist data so you can re-run/inspect it exactly.
    np.save(out_dir / "synthetic_nonrotation_data.npy", data)
    np.save(out_dir / "synthetic_nonrotation_labels.npy", labels)

    if standardization is not None:
        with open(out_dir / "standardization.json", "w", encoding="utf-8") as f:
            json.dump(standardization, f, indent=2)

    # Save true geometry plot using unstandardized data for interpretation.
    save_true_trajectory_plot(
        data_raw,
        labels,
        out_dir / "true_dynamic_systems_same_curve_opposite_direction.png",
    )

    print("\nData:")
    print(f"  data.shape   = {data.shape}")
    print(f"  labels.shape = {labels.shape}")
    print(f"  labels       = {np.bincount(labels)}")
    print(f"  out_dir      = {out_dir}")

    # Static nonparametric baseline.
    print("\nRunning static nonparametric baseline: DP-GMM on pooled 2D points...")
    static_result = run_static_dpgmm(data, labels, out_dir, args)
    print(pd.DataFrame([static_result]).to_string(index=False))

    results = [static_result]

    if not args.skip_hdpgpc:
        print("\nRunning actual HDP-GPC...")
        try:
            hdpgpc_result = run_actual_hdpgpc(data, labels, out_dir, args, repo_root, tag="trajectory", method_name="actual_hdpgpc_trajectory_level")
            print(pd.DataFrame([hdpgpc_result]).to_string(index=False))
            results.append(hdpgpc_result)
        except Exception as exc:
            warnings.warn(
                "Actual HDP-GPC run failed. The static baseline and simulated data "
                f"were still saved. Error was:\n{repr(exc)}"
            )
            results.append({
                "method": "actual_hdpgpc_gpi_hdp",
                "input_used": "full trajectories x(t), y(t)",
                "active_or_inferred_clusters": np.nan,
                "ARI_point_clusters_vs_repeated_system_labels": np.nan,
                "ARI_trajectory_majority_label_vs_system": np.nan,
                "mean_static_components_per_trajectory": np.nan,
                "median_static_components_per_trajectory": np.nan,
                "elapsed_min": np.nan,
                "assignment_source": f"failed: {repr(exc)}",
            })


    if (not args.skip_hdpgpc) and (not args.skip_ordered_pooled_hdpgpc):
        print("\nBuilding ordered pooled 2D point windows...")
        try:
            (
                pooled_data,
                pooled_labels,
                ordered_points,
                ordered_point_labels,
            ) = build_ordered_pooled_windows(
                data,
                labels,
                window_length=args.pooled_window_length,
                stride=args.pooled_window_stride,
                order_mode=args.pooled_order_mode,
            )

            np.save(out_dir / "ordered_pooled_windows_data.npy", pooled_data)
            np.save(out_dir / "ordered_pooled_windows_labels.npy", pooled_labels)
            np.save(out_dir / "ordered_pooled_points.npy", ordered_points)
            np.save(out_dir / "ordered_pooled_point_labels.npy", ordered_point_labels)

            save_ordered_pooled_windows_plot(
                ordered_points,
                ordered_point_labels,
                out_dir / "ordered_pooled_points_true_system.png",
                (
                    "Ordered pooled 2D points colored by true system "
                    f"(mode={args.pooled_order_mode})"
                ),
            )

            print(f"  pooled_data.shape   = {pooled_data.shape}")
            print(f"  pooled_labels.shape = {pooled_labels.shape}")
            print(f"  order_mode          = {args.pooled_order_mode}")
            print(f"  window_length       = {args.pooled_window_length}")
            print(f"  stride              = {args.pooled_window_stride}")

            print("\nRunning actual HDP-GPC on ordered pooled 2D point windows...")
            ordered_pooled_result = run_actual_hdpgpc(
                pooled_data,
                pooled_labels,
                out_dir,
                args,
                repo_root,
                tag="ordered_pooled",
                method_name="actual_hdpgpc_ordered_pooled_point_windows",
            )

            # Add metadata specific to this comparison.
            ordered_pooled_result["pooled_order_mode"] = args.pooled_order_mode
            ordered_pooled_result["pooled_window_length"] = args.pooled_window_length
            ordered_pooled_result["pooled_window_stride"] = args.pooled_window_stride
            ordered_pooled_result["num_ordered_pooled_windows"] = int(pooled_data.shape[0])

            print(pd.DataFrame([ordered_pooled_result]).to_string(index=False))
            results.append(ordered_pooled_result)

        except Exception as exc:
            warnings.warn(
                "The ordered pooled HDP-GPC run failed. The trajectory-level run "
                f"and static baseline were still saved. Error was:\n{repr(exc)}"
            )
            results.append({
                "method": "actual_hdpgpc_ordered_pooled_point_windows",
                "input_used": "ordered pooled 2D point windows",
                "active_or_inferred_clusters": np.nan,
                "ARI_point_clusters_vs_repeated_system_labels": np.nan,
                "ARI_trajectory_majority_label_vs_system": np.nan,
                "mean_static_components_per_trajectory": np.nan,
                "median_static_components_per_trajectory": np.nan,
                "elapsed_min": np.nan,
                "assignment_source": f"failed: {repr(exc)}",
                "pooled_order_mode": args.pooled_order_mode,
                "pooled_window_length": args.pooled_window_length,
                "pooled_window_stride": args.pooled_window_stride,
                "num_ordered_pooled_windows": np.nan,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "comparison_results.csv", index=False)

    print("\nSaved files:")
    for path in sorted(out_dir.iterdir()):
        print(" ", path)

    print("\nComparison:")
    print(results_df.to_string(index=False))

    print(
        "\nInterpretation:\n"
        "  - The static DP-GMM sees only pooled spatial samples and usually splits the\n"
        "    nonlinear curve into several spatial components.\n"
        "  - HDP-GPC sees complete trajectories. If the actual package exposes final\n"
        "    trajectory assignments, the script reports ARI against the two true\n"
        "    dynamic systems. In any case, it also calls print_results and\n"
        "    plot_models_plotly exactly as in your ECG example.\n"
        "  - The ordered-pooled HDP-GPC run discards original trajectory identity,\n"
        "    keeps an imposed ordering of pooled 2D points, creates short windows,\n"
        "    and tests whether HDP-GPC can recover the generating systems from\n"
        "    ordered point sequences rather than full original trajectories."
    )


if __name__ == "__main__":
    main()
