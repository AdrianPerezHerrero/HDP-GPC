# -*- coding: utf-8 -*-

import os
import sys
import gc
import time
from pathlib import Path

import numpy as np
import torch

import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.get_data import compute_estimators_LDS
from hdpgpc.util_plots import plot_models_plotly, print_results


dtype = torch.float64
torch.set_default_dtype(dtype)


def find_project_root() -> Path:
    """
    Assumes you run from the repo root (same pattern as your test script),
    but makes it robust if you run from somewhere else.
    """
    here = Path(__file__).resolve()
    candidates = [
        here,
        here.parent,
        here.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for c in candidates:
        if (c / "hdpgpc").exists():
            return c
    return Path.cwd()


def find_data_dir(hdpgpc_dir: Path) -> Path:
    """
    Prefer mitdb if it exists, otherwise mitbih (your current test script uses mitbih).
    """
    cand1 = hdpgpc_dir / "data" / "mitdb"
    cand2 = hdpgpc_dir / "data" / "mitbih"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Could not find data dir under: {hdpgpc_dir / 'data'}")


def list_records(data_dir: Path):
    """
    Records inferred from '*.npy' excluding '*_labels.npy'.
    """
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


def find_cluster_label_file(pred_dir: Path, rec: str) -> Path:
    """
    Supports:
      - cluster_labels_{rec}_offline.npy
      - cluster_labels/cluster_labels_{rec}_offline.npy (if pred_dir is already .../cluster_labels)
      - any '*{rec}*_offline*.npy' unique match as fallback
    """
    candidates = [
        pred_dir / f"cluster_labels_{rec}_offline.npy",
        pred_dir / f"cluster_labels_{rec}_offline.npy",
        pred_dir / f"cluster_labels_{rec}.npy",
        pred_dir / f"cluster_labels_{rec}_offline_labels.npy",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback glob
    matches = list(pred_dir.glob(f"*{rec}*_offline*.npy"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple cluster-label files match record {rec} in {pred_dir}: {[m.name for m in matches]}"
        )
    raise FileNotFoundError(f"No cluster-label file found for record {rec} in {pred_dir}")


def build_sw_gp(x_basis, x_basis_warp, num_outputs, sigma, gamma, outputscale_,
                ini_lengthscale, bound_lengthscale, noise_warp,
                bound_sigma, bound_gamma, bound_noise_warp):
    """
    Mirrors your test_offline_multi_output_local.py config.
    """
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


def run_one_record(hdpgpc_dir: Path, data_dir: Path, pred_dir: Path, rec: str):
    # Load data
    data = np.load(data_dir / f"{rec}.npy")
    labels_true = np.load(data_dir / f"{rec}_labels.npy", allow_pickle=True)

    num_samples, num_obs_per_sample, num_outputs = data.shape

    # Load predicted cluster labels
    pred_path = find_cluster_label_file(pred_dir, rec)
    cluster_labels = np.load(pred_path, allow_pickle=True).reshape(-1)

    # Align lengths if needed (avoid index errors in reload_model_from_labels)
    if len(cluster_labels) != num_samples:
        n = min(len(cluster_labels), num_samples)
        print(f"[WARN] {rec}: cluster_labels length={len(cluster_labels)} vs num_samples={num_samples}. "
              f"Trimming to {n}.")
        cluster_labels = cluster_labels[:n]
        data = data[:n]
        labels_true = labels_true[:n]
        num_samples = n

    # Ensure integer cluster ids (0..M-1)
    if cluster_labels.dtype.kind in ("f", "c"):
        cluster_labels = np.rint(cluster_labels).astype(np.int64)
    else:
        cluster_labels = cluster_labels.astype(np.int64)

    if np.min(cluster_labels) < 0:
        raise ValueError(f"{rec}: cluster_labels contains negative values (min={np.min(cluster_labels)})")

    M = int(np.max(cluster_labels)) + 1
    if M <= 0:
        raise ValueError(f"{rec}: computed M={M} from cluster_labels")

    # Estimate priors from full batch (same as test script)
    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data)

    # Hyperparameters (same as test script)
    sigma = std * 1.0
    gamma = std_dif * 1.0
    outputscale_ = 300.0
    ini_lengthscale = 3.0
    bound_lengthscale = (1.0, 20.0)

    noise_warp = std * 0.1
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

    # Time index support
    l, L = 0, num_obs_per_sample
    x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(l, L, 2, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T
    x_trains = np.array([x_train] * num_samples)

    # Build model
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

    # Reload from labels
    t0 = time.time()
    sw_gp.reload_model_from_labels(x_trains, data, cluster_labels, M)
    dt_min = (time.time() - t0) / 60.0
    print(f"[OK] {rec}: reload_model_from_labels done in {dt_min:.2f} min (M={M}).")

    # Plot results (same as your snippet)
    results_dir = hdpgpc_dir / "results" / "eval_final_ver" / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = str(results_dir / f"Rec{rec}_")

    main_model = print_results(sw_gp, labels_true, 0, error=False)
    selected_gpmodels = sw_gp.selected_gpmodels()

    plot_models_plotly(
        sw_gp, selected_gpmodels, main_model, labels_true, 0,
        lead=0, save=out_prefix + "Offline_Clusters_Lead_1.pdf",
        step=0.5, plot_latent=True
    )
    if num_outputs > 1:
        plot_models_plotly(
            sw_gp, selected_gpmodels, main_model, labels_true, 0,
            lead=1, save=out_prefix + "Offline_Clusters_Lead_2.pdf",
            step=0.5, plot_latent=True
        )

    # Cleanup to avoid memory buildup
    del sw_gp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    project_root = find_project_root()
    hdpgpc_dir = project_root
    data_dir = find_data_dir(hdpgpc_dir)

    # Where you saved cluster_labels_...npy
    pred_dir = project_root / "results" / "cluster_labels" / "final_ver"
    if not pred_dir.exists():
        # fallback: maybe user saved under hdpgpc/results/cluster_labels
        pred_dir = hdpgpc_dir / "results" / "cluster_labels"

    if not pred_dir.exists():
        raise FileNotFoundError(f"Could not find cluster_labels directory at {pred_dir}")

    recs = list_records(data_dir)
    if not recs:
        raise RuntimeError(f"No records found in {data_dir} (expected *.npy with matching *_labels.npy).")

    # Optional CLI: process single record
    if len(sys.argv) > 1:
        rec = sys.argv[1]
        if rec not in recs:
            raise ValueError(f"Record {rec} not found. Example: {recs[:10]}")
        recs = [rec]

    print(f"[INFO] data_dir: {data_dir}")
    print(f"[INFO] pred_dir: {pred_dir}")
    print(f"[INFO] records: {len(recs)}")

    failures = []
    for i, rec in enumerate(recs, 1):
        print(f"\n[{i}/{len(recs)}] Processing record {rec} ...")
        try:
            run_one_record(hdpgpc_dir, data_dir, pred_dir, rec)
        except Exception as e:
            print(f"[FAIL] {rec}: {repr(e)}")
            failures.append((rec, repr(e)))

    if failures:
        print("\n[WARN] Some records failed:")
        for r, err in failures:
            print(f"  - {r}: {err}")


if __name__ == "__main__":
    main()
