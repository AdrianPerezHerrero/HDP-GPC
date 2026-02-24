# run_offline_all_records.py
# -*- coding: utf-8 -*-

import os
import sys
import time
import gc
from pathlib import Path

import numpy as np
import torch

import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.get_data import compute_estimators_LDS

# Optional (only if you want to keep the per-record summary/plots)
from hdpgpc.util_plots import plot_models_plotly, print_results

dtype = torch.float64
torch.set_default_dtype(dtype)


def find_repo_root() -> Path:
    """Try to find a root that contains the 'hdpgpc' package directory."""
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
    """
    Prefer data/mitdb (as requested), but support existing layout used
    by your test script: hdpgpc/data/mitbih.
    """
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
        "Could not find a data directory. Looked for data/mitdb or data/mitbih "
        "under both repo root and hdpgpc/."
    )


def list_records(data_dir: Path):
    """
    Records are inferred from '*.npy' files excluding '*_labels.npy'.
    """
    recs = []
    for f in sorted(data_dir.glob("*.npy")):
        if f.name.endswith("_labels.npy"):
            continue
        # Skip any other non-record artifacts if needed
        if "labels" in f.stem:
            continue
        rec = f.stem
        labels_f = data_dir / f"{rec}_labels.npy"
        if not labels_f.exists():
            print(f"[WARN] Missing labels for record {rec}: expected {labels_f.name}. Skipping.")
            continue
        recs.append(rec)
    return recs


def run_one_record(data_dir: Path, rec: str, out_dir: Path, warp: bool = False):
    data_path = data_dir / f"{rec}.npy"
    labels_path = data_dir / f"{rec}_labels.npy"

    data = np.load(data_path)
    labels = np.load(labels_path)

    num_samples, num_obs_per_sample, num_outputs = data.shape

    # Estimate priors from full batch (same as your test script)
    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data)

    # Hyperparameters (copied from test_offline_multi_output_local.py)
    M = 2
    sigma = std * 2.0
    gamma = std_dif * 1.0
    outputscale_ = 300.0
    ini_lengthscale = 3.0
    bound_lengthscale = (1.0, 20.0)

    bound_sigma = (std * 1e-7, std * 1e-5)

    # Warp priors
    noise_warp = std * 0.1
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

    # Time index support
    l, L = 0, num_obs_per_sample
    x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(l, L, 2, dtype=np.float64)).T

    x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T
    x_trains = np.array([x_train] * num_samples)

    # Define the model (same config)
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
        verbose=False,
        hmm_switch=True,
        max_models=100,
        mode_warp="rough",
        bayesian_params=True,
        inducing_points=False,
        reestimate_initial_params=False,
        n_explore_steps=16,
        free_deg_MNIV=3,
        share_gp=True
    )

    start_t = time.time()
    sw_gp.include_batch(x_trains, data, warp=warp)
    elapsed_min = (time.time() - start_t) / 60.0

    # Extract and save cluster labels
    # resp_assigned[-1] is what you requested; detach/cpu for safety.
    cluster_labels = sw_gp.resp_assigned[-1].detach().cpu().numpy()
    main_model = print_results(sw_gp, labels, 0, error=False)
    out_path = out_dir / f"cluster_labels_{rec}_offline.npy"
    np.save(out_path, cluster_labels)

    # Optional: keep the original reporting/plotting per record
    #main_model = print_results(sw_gp, labels, 0, error=False)
    # selected_gpmodels = sw_gp.selected_gpmodels()
    # plot_models_plotly(sw_gp, selected_gpmodels, main_model, labels, 0, lead=0,
    #                    save=str(out_dir / f"Rec{rec}_Offline_Clusters_Lead_1.png"),
    #                    step=0.5, plot_latent=True)
    # plot_models_plotly(sw_gp, selected_gpmodels, main_model, labels, 0, lead=1,
    #                    save=str(out_dir / f"Rec{rec}_Offline_Clusters_Lead_2.png"),
    #                    step=0.5, plot_latent=True)

    # Cleanup to avoid GPU/CPU memory buildup across records
    del sw_gp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return elapsed_min, out_path


def main():
    repo_root = find_repo_root()
    data_dir = find_data_dir(repo_root)

    # Save under results/cluster_labels by default (adjust to your preference)
    out_dir = repo_root / "results" / "cluster_labels" / "v3_UCR_ver"
    out_dir.mkdir(parents=True, exist_ok=True)

    #recs = list_records(data_dir)
    recs = ["104", "207", "105", "203", "217", "223", "213", "208"]
    if not recs:
        raise RuntimeError(f"No records found in {data_dir}. (Expected *.npy plus *_labels.npy)")

    print(f"[INFO] Using data dir: {data_dir}")
    print(f"[INFO] Found {len(recs)} records. (MIT-BIH standard set is 48 records.)")
    print(f"[INFO] Output dir: {out_dir}")

    # Optional: allow processing a single record via CLI: python run_offline_all_records.py 231
    if len(sys.argv) > 1:
        rec = sys.argv[1]
        if rec not in recs:
            raise ValueError(f"Record {rec} not found in {data_dir}. Available example: {recs[:10]}")
        recs = [rec]

    total_start = time.time()
    failures = []

    for i, rec in enumerate(recs, 1):
        try:
            print(f"\n[{i}/{len(recs)}] Processing record {rec} ...")
            mins, saved_path = run_one_record(data_dir, rec, out_dir, warp=False)
            print(f"[OK] {rec} done in {mins:.2f} min. Saved: {saved_path}")
        except Exception as e:
            print(f"[FAIL] {rec}: {repr(e)}")
            failures.append((rec, repr(e)))
            # continue to next record

    total_min = (time.time() - total_start) / 60.0
    print(f"\n[INFO] Finished in {total_min:.2f} min.")
    if failures:
        print("[WARN] Failures:")
        for rec, err in failures:
            print(f"  - {rec}: {err}")


if __name__ == "__main__":
    main()
