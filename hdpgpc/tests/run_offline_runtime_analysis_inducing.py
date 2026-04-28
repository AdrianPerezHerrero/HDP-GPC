# -*- coding: utf-8 -*-
"""
Runtime analysis runner for the offline HDP-GPC ECG experiment.

Features
- per-record runtime logging
- fit-only and end-to-end timing
- inferred cluster count K for each record
- warp OFF / ON benchmarking
- single-output (lead 1) / multi-output (all leads) benchmarking
- CPU peak RSS tracking (psutil when available, with a resource fallback)
- GPU peak memory tracking when CUDA is available
- incremental CSV logging with resume/skip support
- CSV summaries and plots refreshed after each successful run

Example
    python run_offline_runtime_analysis.py --records 100 102 --warp both --output-mode both
    python run_offline_runtime_analysis.py 100 --warp off --output-mode single-output
    python run_offline_runtime_analysis.py --warp both --output-mode both --rerun-existing
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.get_data import compute_estimators_LDS

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False

try:
    import resource
    HAS_RESOURCE = True
except Exception:
    resource = None
    HAS_RESOURCE = False


dtype = torch.float64
torch.set_default_dtype(dtype)


RECORD_FIELDNAMES = [
    "method",
    "record",
    "output_mode",
    "warp",
    "num_samples",
    "num_obs_per_sample",
    "num_outputs",
    "num_input_leads",
    "num_classes_true",
    "num_clusters_pred",
    "fit_runtime_sec",
    "fit_runtime_min",
    "total_runtime_sec",
    "total_runtime_min",
    "cpu_start_rss_mb",
    "cpu_end_rss_mb",
    "cpu_peak_rss_mb",
    "cpu_peak_delta_mb",
    "ru_maxrss_start_mb",
    "ru_maxrss_end_mb",
    "gpu_peak_allocated_mb",
    "cluster_labels_path",
]

FAILURE_FIELDNAMES = [
    "record",
    "output_mode",
    "warp",
    "error",
]

NUMERIC_FIELDS_INT = {
    "num_samples",
    "num_obs_per_sample",
    "num_outputs",
    "num_input_leads",
    "num_classes_true",
    "num_clusters_pred",
}

NUMERIC_FIELDS_FLOAT = {
    "fit_runtime_sec",
    "fit_runtime_min",
    "total_runtime_sec",
    "total_runtime_min",
    "cpu_start_rss_mb",
    "cpu_end_rss_mb",
    "cpu_peak_rss_mb",
    "cpu_peak_delta_mb",
    "ru_maxrss_start_mb",
    "ru_maxrss_end_mb",
    "gpu_peak_allocated_mb",
    "mean_total_runtime_sec",
    "std_total_runtime_sec",
    "mean_fit_runtime_sec",
    "std_fit_runtime_sec",
    "mean_runtime_min",
    "mean_num_samples",
    "mean_num_clusters_pred",
    "mean_cpu_peak_rss_mb",
    "mean_cpu_peak_delta_mb",
    "mean_gpu_peak_allocated_mb",
    "benchmark_wall_clock_sec",
    "benchmark_wall_clock_min",
}


# -------------------------------
# Path and data helpers
# -------------------------------

def find_repo_root(explicit_repo_root: Optional[str] = None) -> Path:
    if explicit_repo_root is not None:
        repo_root = Path(explicit_repo_root).expanduser().resolve()
        if not (repo_root / "hdpgpc").exists():
            raise FileNotFoundError(f"Expected 'hdpgpc' under repo root: {repo_root}")
        return repo_root

    candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "hdpgpc").exists():
            return candidate
    return Path.cwd()


def find_data_dir(repo_root: Path, explicit_data_dir: Optional[str] = None) -> Path:
    if explicit_data_dir is not None:
        data_dir = Path(explicit_data_dir).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        return data_dir

    pkg_dir = repo_root / "hdpgpc"
    candidates = [
        pkg_dir / "data" / "mitdb",
        pkg_dir / "data" / "mitbih",
        repo_root / "data" / "mitdb",
        repo_root / "data" / "mitbih",
    ]
    for data_dir in candidates:
        if data_dir.exists():
            return data_dir
    raise FileNotFoundError(
        "Could not find a data directory. Looked for data/mitdb or data/mitbih under both repo root and hdpgpc/."
    )


def list_records(data_dir: Path) -> List[str]:
    records: List[str] = []
    for file_path in sorted(data_dir.glob("*.npy")):
        if file_path.name.endswith("_labels.npy"):
            continue
        if "labels" in file_path.stem:
            continue
        rec = file_path.stem
        labels_path = data_dir / f"{rec}_labels.npy"
        if not labels_path.exists():
            print(f"[WARN] Missing labels for record {rec}: expected {labels_path.name}. Skipping.")
            continue
        records.append(rec)
    return records


# -------------------------------
# Resource helpers
# -------------------------------

def get_ru_maxrss_mb() -> float:
    if not HAS_RESOURCE:
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 ** 2)
    return float(rss) / 1024.0


def get_current_rss_mb() -> float:
    if not HAS_PSUTIL:
        return float("nan")
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss) / (1024.0 ** 2)


class MemorySampler:
    def __init__(self, interval_sec: float = 0.2) -> None:
        self.interval_sec = max(0.05, float(interval_sec))
        self.peak_rss_mb = float("nan")
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid()) if HAS_PSUTIL else None

    @property
    def enabled(self) -> bool:
        return self._process is not None

    def start(self) -> None:
        if not self.enabled:
            return
        self.peak_rss_mb = float(self._process.memory_info().rss) / (1024.0 ** 2)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._process is not None
        while not self._stop_event.is_set():
            try:
                rss_mb = float(self._process.memory_info().rss) / (1024.0 ** 2)
                if math.isnan(self.peak_rss_mb) or rss_mb > self.peak_rss_mb:
                    self.peak_rss_mb = rss_mb
            except Exception:
                pass
            self._stop_event.wait(self.interval_sec)

    def stop(self) -> float:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        return self.peak_rss_mb


# -------------------------------
# CSV helpers
# -------------------------------

def save_csv(rows: List[Dict], csv_path: Path, fieldnames: Optional[List[str]] = None) -> None:
    if not rows and not fieldnames:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    effective_fieldnames = fieldnames or list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=effective_fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def append_csv_row(row: Dict, csv_path: Path, fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _parse_int(value) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None:
        return 0
    text = str(value).strip()
    if text == "":
        return 0
    return int(float(text))


def _parse_float(value) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    lowered = text.lower()
    if lowered in {"nan", "none"}:
        return float("nan")
    return float(text)


def normalize_loaded_rows(rows: List[Dict], is_failure: bool = False) -> List[Dict]:
    normalized: List[Dict] = []
    for row in rows:
        new_row = dict(row)
        new_row.setdefault("output_mode", "single-output")
        if not is_failure:
            # backfill old logs
            if "num_input_leads" not in new_row or str(new_row.get("num_input_leads", "")).strip() == "":
                fallback = new_row.get("num_outputs", "1")
                new_row["num_input_leads"] = fallback
            for key in list(new_row.keys()):
                if key in NUMERIC_FIELDS_INT:
                    try:
                        new_row[key] = _parse_int(new_row[key])
                    except Exception:
                        new_row[key] = 0
                elif key in NUMERIC_FIELDS_FLOAT:
                    try:
                        new_row[key] = _parse_float(new_row[key])
                    except Exception:
                        new_row[key] = float("nan")
        normalized.append(new_row)
    return normalized


def run_key(row: Dict) -> Tuple[str, str, str]:
    return str(row.get("record", "")), str(row.get("output_mode", "single-output")), str(row.get("warp", "off"))


def deduplicate_latest_rows(rows: List[Dict]) -> List[Dict]:
    latest: Dict[Tuple[str, str, str], Dict] = {}
    for row in rows:
        latest[run_key(row)] = row
    return sorted(latest.values(), key=lambda r: (str(r["record"]), str(r["output_mode"]), str(r["warp"])))


# -------------------------------
# Numeric summaries
# -------------------------------

def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def summarize_rows(rows: List[Dict], benchmark_wall_clock_sec: float) -> List[Dict]:
    summary_rows: List[Dict] = []

    def build_summary(label: str, subset: List[Dict]) -> Dict:
        return {
            "scope": label,
            "num_runs": len(subset),
            "num_unique_records": len(sorted({r["record"] for r in subset})),
            "mean_total_runtime_sec": safe_mean(r["total_runtime_sec"] for r in subset),
            "std_total_runtime_sec": safe_std(r["total_runtime_sec"] for r in subset),
            "mean_fit_runtime_sec": safe_mean(r["fit_runtime_sec"] for r in subset),
            "std_fit_runtime_sec": safe_std(r["fit_runtime_sec"] for r in subset),
            "mean_runtime_min": safe_mean(r["total_runtime_min"] for r in subset),
            "mean_num_samples": safe_mean(r["num_samples"] for r in subset),
            "mean_num_clusters_pred": safe_mean(r["num_clusters_pred"] for r in subset),
            "mean_cpu_peak_rss_mb": safe_mean(r["cpu_peak_rss_mb"] for r in subset),
            "mean_cpu_peak_delta_mb": safe_mean(r["cpu_peak_delta_mb"] for r in subset),
            "mean_gpu_peak_allocated_mb": safe_mean(r["gpu_peak_allocated_mb"] for r in subset),
            "benchmark_wall_clock_sec": float(benchmark_wall_clock_sec),
            "benchmark_wall_clock_min": float(benchmark_wall_clock_sec / 60.0),
        }

    if not rows:
        return summary_rows

    summary_rows.append(build_summary("overall", rows))

    for output_mode in sorted({row["output_mode"] for row in rows}):
        subset = [row for row in rows if row["output_mode"] == output_mode]
        summary_rows.append(build_summary(f"output_mode_{output_mode}", subset))

    for warp_label in sorted({row["warp"] for row in rows}):
        subset = [row for row in rows if row["warp"] == warp_label]
        summary_rows.append(build_summary(f"warp_{warp_label}", subset))

    for output_mode in sorted({row["output_mode"] for row in rows}):
        for warp_label in sorted({row["warp"] for row in rows}):
            subset = [
                row for row in rows
                if row["output_mode"] == output_mode and row["warp"] == warp_label
            ]
            if subset:
                summary_rows.append(build_summary(f"{output_mode}_warp_{warp_label}", subset))

    return summary_rows


# -------------------------------
# Plotting helpers
# -------------------------------

def _group_styles(rows: List[Dict]) -> List[Tuple[str, str, str]]:
    groups = sorted({(row["output_mode"], row["warp"]) for row in rows})
    result = []
    for output_mode, warp in groups:
        label = f"{output_mode}, warp={warp}"
        result.append((output_mode, warp, label))
    return result


def plot_runtime_vs_variable(rows: List[Dict], x_key: str, x_label: str, out_path: Path) -> None:
    if not HAS_MATPLOTLIB or not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {
        ("single-output", "off"): "o",
        ("single-output", "on"): "s",
        ("multi-output", "off"): "^",
        ("multi-output", "on"): "D",
    }

    for output_mode, warp, label in _group_styles(rows):
        subset = [row for row in rows if row["output_mode"] == output_mode and row["warp"] == warp]
        if not subset:
            continue
        xs = np.asarray([row[x_key] for row in subset], dtype=float)
        ys = np.asarray([row["total_runtime_sec"] for row in subset], dtype=float)
        finite = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[finite]
        ys = ys[finite]
        if xs.size == 0:
            continue
        ax.scatter(xs, ys, label=label, marker=markers.get((output_mode, warp), "o"), alpha=0.85)
        if xs.size >= 2 and np.unique(xs).size >= 2:
            coeffs = np.polyfit(xs, ys, deg=1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, linewidth=1.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtime vs {x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_warp_on_off_runtime(rows: List[Dict], out_path: Path) -> None:
    if not HAS_MATPLOTLIB or not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    data: List[np.ndarray] = []
    tick_labels: List[str] = []

    for output_mode, warp, label in _group_styles(rows):
        subset_vals = np.asarray(
            [row["total_runtime_sec"] for row in rows if row["output_mode"] == output_mode and row["warp"] == warp],
            dtype=float,
        )
        subset_vals = subset_vals[np.isfinite(subset_vals)]
        if subset_vals.size == 0:
            continue
        data.append(subset_vals)
        tick_labels.append(label)

    if not data:
        plt.close(fig)
        return

    ax.boxplot(data, tick_labels=tick_labels, showmeans=True)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime distribution by output mode and warp")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_memory_summary(rows: List[Dict], out_path: Path) -> None:
    if not HAS_MATPLOTLIB or not rows:
        return

    groups = _group_styles(rows)
    labels = [label for _, _, label in groups]
    cpu_means = []
    cpu_delta_means = []
    gpu_means = []
    for output_mode, warp, _ in groups:
        subset = [row for row in rows if row["output_mode"] == output_mode and row["warp"] == warp]
        cpu_means.append(safe_mean(row["cpu_peak_rss_mb"] for row in subset))
        cpu_delta_means.append(safe_mean(row["cpu_peak_delta_mb"] for row in subset))
        gpu_means.append(safe_mean(row["gpu_peak_allocated_mb"] for row in subset))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(labels))

    axes[0].bar(x, np.asarray(cpu_means, dtype=float))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Peak CPU RSS (MB)")
    axes[0].set_title("CPU memory footprint summary")
    axes[0].grid(True, axis="y", alpha=0.3)

    width = 0.35
    axes[1].bar(x - width / 2, np.asarray(cpu_delta_means, dtype=float), width=width, label="CPU peak delta (MB)")
    gpu_arr = np.asarray(gpu_means, dtype=float)
    if np.any(np.isfinite(gpu_arr)) and np.nanmax(gpu_arr) > 0:
        axes[1].bar(x + width / 2, gpu_arr, width=width, label="GPU peak allocated (MB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].set_title("Peak memory by output mode and warp")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -------------------------------
# Model runner
# -------------------------------

def build_model(num_obs_per_sample: int, num_outputs: int, data: np.ndarray):
    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data)

    sigma = std * 1.0
    gamma = std_dif * 1.0
    #gamma = 10.0
    outputscale_ = 300.0
    ini_lengthscale = 3.0
    bound_lengthscale = (1.0, 20.0)

    bound_sigma = (std * 1e-7, sigma * 0.1)
    noise_warp = max(std * 3.0, 20.0)
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 1.0)

    l, L = 0, num_obs_per_sample
    x_basis = np.atleast_2d(np.arange(l, L, 20, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(l, L, 30, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T

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
        inducing_points=True,
        reestimate_initial_params=False,
        n_explore_steps=3,
        free_deg_MNIV=3,
        share_gp=True,
        hdp_hyp='less'
    )
    return sw_gp, x_train


def select_record_data(raw_data: np.ndarray, output_mode: str) -> Tuple[np.ndarray, int]:
    data = np.asarray(raw_data)
    if data.ndim == 2:
        data = data[:, :, None]

    if data.ndim != 3:
        raise ValueError(f"Expected ECG data with shape (N, T, C). Got shape {data.shape}.")

    num_input_leads = int(data.shape[2])
    if output_mode == "single-output":
        return data[:, :, [0]], num_input_leads
    if output_mode == "multi-output":
        return data, num_input_leads
    raise ValueError(f"Unsupported output_mode: {output_mode}")


def run_one_record(
    data_dir: Path,
    rec: str,
    out_dir: Path,
    warp: bool,
    output_mode: str,
    memory_interval_sec: float,
    verbose_samples: bool,
    save_labels: bool,
) -> Dict:
    data_path = data_dir / f"{rec}.npy"
    labels_path = data_dir / f"{rec}_labels.npy"

    raw_data = np.load(data_path)
    data, num_input_leads = select_record_data(raw_data, output_mode=output_mode)
    labels = np.load(labels_path)

    num_samples, num_obs_per_sample, num_outputs = data.shape
    warp_label = "on" if warp else "off"

    labels_out_dir = out_dir / "cluster_labels" / output_mode / f"warp_{warp_label}"
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_path = labels_out_dir / f"cluster_labels_{rec}_offline.npy"

    cpu_start_rss_mb = get_current_rss_mb()
    ru_maxrss_start_mb = get_ru_maxrss_mb()
    sampler = MemorySampler(interval_sec=memory_interval_sec)

    sw_gp = None
    fit_runtime_sec = float("nan")
    total_runtime_sec = float("nan")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        sampler.start()
        total_start = time.perf_counter()

        sw_gp, x_train = build_model(num_obs_per_sample=num_obs_per_sample, num_outputs=num_outputs, data=data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fit_start = time.perf_counter()

        x_trains = np.array([x_train] * num_samples)
        sw_gp.include_batch(x_trains, data, warp=warp)


        # for i in range(num_samples):
        #     print(f"Sample: {i}/{num_samples - 1} label: {labels[i]}")
        #     sw_gp.include_sample_fast(x_train, data[i], with_warp=warp)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fit_runtime_sec = time.perf_counter() - fit_start

        cluster_labels = sw_gp.resp_assigned[-1].detach().cpu().numpy()
        if save_labels:
            np.save(labels_out_path, cluster_labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_runtime_sec = time.perf_counter() - total_start

        num_clusters_pred = int(np.unique(cluster_labels).size)
        num_classes_true = int(np.unique(labels).size)

        cpu_peak_rss_mb = sampler.stop()
        cpu_end_rss_mb = get_current_rss_mb()
        ru_maxrss_end_mb = get_ru_maxrss_mb()
        if not np.isfinite(cpu_peak_rss_mb):
            cpu_peak_rss_mb = ru_maxrss_end_mb

        gpu_peak_allocated_mb = (
            float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )

        return {
            "method": "offline_induced",
            "record": str(rec),
            "output_mode": str(output_mode),
            "warp": str(warp_label),
            "num_samples": int(num_samples),
            "num_obs_per_sample": int(num_obs_per_sample),
            "num_outputs": int(num_outputs),
            "num_input_leads": int(num_input_leads),
            "num_classes_true": int(num_classes_true),
            "num_clusters_pred": int(num_clusters_pred),
            "fit_runtime_sec": float(fit_runtime_sec),
            "fit_runtime_min": float(fit_runtime_sec / 60.0),
            "total_runtime_sec": float(total_runtime_sec),
            "total_runtime_min": float(total_runtime_sec / 60.0),
            "cpu_start_rss_mb": float(cpu_start_rss_mb),
            "cpu_end_rss_mb": float(cpu_end_rss_mb),
            "cpu_peak_rss_mb": float(cpu_peak_rss_mb),
            "cpu_peak_delta_mb": float(cpu_peak_rss_mb - cpu_start_rss_mb)
            if np.isfinite(cpu_peak_rss_mb) and np.isfinite(cpu_start_rss_mb)
            else float("nan"),
            "ru_maxrss_start_mb": float(ru_maxrss_start_mb),
            "ru_maxrss_end_mb": float(ru_maxrss_end_mb),
            "gpu_peak_allocated_mb": float(gpu_peak_allocated_mb),
            "cluster_labels_path": str(labels_out_path) if save_labels else "",
        }
    finally:
        try:
            sampler.stop()
        except Exception:
            pass
        if sw_gp is not None:
            del sw_gp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -------------------------------
# Incremental output refresh
# -------------------------------

def refresh_outputs(
    output_root: Path,
    record_log_rows: List[Dict],
    failure_log_rows: List[Dict],
    benchmark_wall_clock_sec: float,
    skip_plots: bool,
) -> Tuple[List[Dict], List[Dict]]:
    latest_rows = deduplicate_latest_rows(normalize_loaded_rows(record_log_rows, is_failure=False))
    latest_failures = normalize_loaded_rows(failure_log_rows, is_failure=True)

    records_csv = output_root / "runtime_records.csv"
    save_csv(latest_rows, records_csv, fieldnames=RECORD_FIELDNAMES)

    summary_rows = summarize_rows(latest_rows, benchmark_wall_clock_sec)
    summary_csv = output_root / "runtime_summary.csv"
    save_csv(summary_rows, summary_csv)

    failures_csv = output_root / "runtime_failures.csv"
    save_csv(latest_failures, failures_csv, fieldnames=FAILURE_FIELDNAMES)

    if not skip_plots and HAS_MATPLOTLIB and latest_rows:
        plots_dir = output_root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_runtime_vs_variable(latest_rows, x_key="num_samples", x_label="N (number of samples)", out_path=plots_dir / "runtime_vs_N.png")
        plot_runtime_vs_variable(latest_rows, x_key="num_clusters_pred", x_label="K (predicted number of clusters)", out_path=plots_dir / "runtime_vs_K.png")
        plot_warp_on_off_runtime(latest_rows, out_path=plots_dir / "runtime_warp_on_off.png")
        plot_memory_summary(latest_rows, out_path=plots_dir / "memory_footprint_summary.png")

    return latest_rows, summary_rows


# -------------------------------
# CLI helpers
# -------------------------------

def resolve_records(args: argparse.Namespace, data_dir: Path) -> List[str]:
    available_records = list_records(data_dir)
    if not available_records:
        raise RuntimeError(f"No records found in {data_dir}. Expected *.npy plus *_labels.npy.")

    if args.record is not None:
        records = [args.record]
    elif args.records:
        records = list(args.records)
    else:
        records = available_records

    missing = [rec for rec in records if rec not in available_records]
    if missing:
        raise ValueError(
            f"Requested record(s) not found in {data_dir}: {missing}. Available examples: {available_records[:10]}"
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime analysis for offline HDP-GPC ECG clustering.")
    parser.add_argument("record", nargs="?", help="Optional single record id, e.g. 100")
    parser.add_argument("--records", nargs="*", help="Optional explicit record list, e.g. --records 100 102 104")
    parser.add_argument("--warp", choices=["off", "on", "both"], default="both", help="Benchmark warp OFF, ON, or both")
    parser.add_argument(
        "--output-mode",
        choices=["single-output", "multi-output", "both"],
        default="both",
        help="Use only lead 1, all available leads, or both comparisons",
    )
    parser.add_argument("--repo-root", type=str, default=None, help="Explicit repository root containing the hdpgpc package")
    parser.add_argument("--data-dir", type=str, default=None, help="Explicit data directory with <rec>.npy and <rec>_labels.npy")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="v1_offline_runtime_analysis",
        help="Subdirectory under results/runtime_analysis/",
    )
    parser.add_argument(
        "--memory-interval-sec",
        type=float,
        default=0.2,
        help="CPU RSS sampling interval for approximate peak-memory tracking",
    )
    parser.add_argument(
        "--verbose-samples",
        action="store_true",
        help="Print one line per sample (disabled by default because it distorts timing)",
    )
    parser.add_argument("--no-save-labels", action="store_true", help="Do not save per-record cluster label arrays")
    parser.add_argument("--skip-plots", action="store_true", help="Skip PNG plot generation")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Force recomputation even if an entry for (record, output_mode, warp) already exists in the log",
    )
    return parser.parse_args()


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(args.repo_root)
    data_dir = find_data_dir(repo_root, args.data_dir)
    output_root = repo_root / "results" / "runtime_analysis" / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    records = resolve_records(args, data_dir)
    warp_values = [False, True] if args.warp == "both" else [args.warp == "on"]
    output_modes = ["single-output", "multi-output"] if args.output_mode == "both" else [args.output_mode]

    print(f"[INFO] Using data dir: {data_dir}")
    print(f"[INFO] Found {len(list_records(data_dir))} available records.")
    print(f"[INFO] Running {len(records)} record(s): {records}")
    print(f"[INFO] Warp modes: {[('on' if w else 'off') for w in warp_values]}")
    print(f"[INFO] Output modes: {output_modes}")
    print(f"[INFO] Output dir: {output_root}")
    if not HAS_PSUTIL:
        print("[WARN] psutil not available. CPU peak RSS will fall back to ru_maxrss-based reporting.")
    if not HAS_MATPLOTLIB and not args.skip_plots:
        print("[WARN] matplotlib not available. Plot generation will be skipped.")

    config = {
        "records": records,
        "warp": args.warp,
        "output_mode": args.output_mode,
        "repo_root": str(repo_root),
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "memory_interval_sec": args.memory_interval_sec,
        "verbose_samples": bool(args.verbose_samples),
        "save_labels": not args.no_save_labels,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "rerun_existing": bool(args.rerun_existing),
    }
    with (output_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    records_log_csv = output_root / "runtime_records_log_nowarp.csv"
    failures_log_csv = output_root / "runtime_failures_log.csv"
    record_log_rows = normalize_loaded_rows(load_csv_rows(records_log_csv), is_failure=False)
    failure_log_rows = normalize_loaded_rows(load_csv_rows(failures_log_csv), is_failure=True)
    completed_keys = {run_key(row) for row in deduplicate_latest_rows(record_log_rows)}

    if record_log_rows:
        print(f"[INFO] Loaded {len(completed_keys)} completed run(s) from the incremental log.")

    benchmark_start = time.perf_counter()
    latest_rows, summary_rows = refresh_outputs(
        output_root=output_root,
        record_log_rows=record_log_rows,
        failure_log_rows=failure_log_rows,
        benchmark_wall_clock_sec=0.0,
        skip_plots=args.skip_plots,
    )

    total_runs = len(records) * len(warp_values) * len(output_modes)
    run_index = 0

    for rec in records:
        for output_mode in output_modes:
            for warp in warp_values:
                run_index += 1
                warp_label = "on" if warp else "off"
                key = (str(rec), str(output_mode), str(warp_label))

                if key in completed_keys and not args.rerun_existing:
                    print(
                        f"\n[{run_index}/{total_runs}] Skipping record {rec} output_mode={output_mode} warp={warp_label} "
                        f"(already in incremental log)."
                    )
                    continue

                try:
                    print(f"\n[{run_index}/{total_runs}] Processing record {rec} output_mode={output_mode} warp={warp_label} ...")
                    row = run_one_record(
                        data_dir=data_dir,
                        rec=rec,
                        out_dir=output_root,
                        warp=warp,
                        output_mode=output_mode,
                        memory_interval_sec=args.memory_interval_sec,
                        verbose_samples=args.verbose_samples,
                        save_labels=not args.no_save_labels,
                    )
                    record_log_rows.append(row)
                    append_csv_row(row, records_log_csv, RECORD_FIELDNAMES)
                    completed_keys.add(run_key(row))

                    latest_rows, summary_rows = refresh_outputs(
                        output_root=output_root,
                        record_log_rows=record_log_rows,
                        failure_log_rows=failure_log_rows,
                        benchmark_wall_clock_sec=time.perf_counter() - benchmark_start,
                        skip_plots=args.skip_plots,
                    )

                    print(
                        f"[OK] rec={rec} output_mode={output_mode} warp={warp_label} "
                        f"runtime={row['total_runtime_sec']:.2f}s "
                        f"fit={row['fit_runtime_sec']:.2f}s "
                        f"K={row['num_clusters_pred']} "
                        f"CPU_peak={row['cpu_peak_rss_mb']:.1f}MB "
                        f"GPU_peak={row['gpu_peak_allocated_mb']:.1f}MB"
                    )
                except Exception as exc:
                    failure_row = {
                        "record": str(rec),
                        "output_mode": str(output_mode),
                        "warp": str(warp_label),
                        "error": repr(exc),
                    }
                    failure_log_rows.append(failure_row)
                    append_csv_row(failure_row, failures_log_csv, FAILURE_FIELDNAMES)
                    latest_rows, summary_rows = refresh_outputs(
                        output_root=output_root,
                        record_log_rows=record_log_rows,
                        failure_log_rows=failure_log_rows,
                        benchmark_wall_clock_sec=time.perf_counter() - benchmark_start,
                        skip_plots=args.skip_plots,
                    )
                    print(f"[FAIL] rec={rec} output_mode={output_mode} warp={warp_label}: {repr(exc)}")

    benchmark_wall_clock_sec = time.perf_counter() - benchmark_start
    latest_rows, summary_rows = refresh_outputs(
        output_root=output_root,
        record_log_rows=record_log_rows,
        failure_log_rows=failure_log_rows,
        benchmark_wall_clock_sec=benchmark_wall_clock_sec,
        skip_plots=args.skip_plots,
    )

    records_csv = output_root / "runtime_records.csv"
    summary_csv = output_root / "runtime_summary.csv"

    print("\n[INFO] Benchmark finished.")
    print(f"[INFO] Total benchmark wall clock: {benchmark_wall_clock_sec:.2f}s ({benchmark_wall_clock_sec / 60.0:.2f} min)")
    print(f"[INFO] Per-run results: {records_csv}")
    print(f"[INFO] Summary results: {summary_csv}")
    if summary_rows:
        overall = next((row for row in summary_rows if row["scope"] == "overall"), None)
        if overall is not None:
            print(
                "[INFO] Mean total runtime per run: "
                f"{overall['mean_total_runtime_sec']:.2f}s ({overall['mean_runtime_min']:.2f} min)"
            )
            print(f"[INFO] Mean fit-only runtime per run: {overall['mean_fit_runtime_sec']:.2f}s")

    latest_failures = normalize_loaded_rows(failure_log_rows, is_failure=True)
    if latest_failures:
        print("[WARN] Failure log contains entries:")
        for failure in latest_failures[-10:]:
            print(
                f"  - rec={failure['record']} output_mode={failure.get('output_mode', 'single-output')} "
                f"warp={failure['warp']}: {failure['error']}"
            )


if __name__ == "__main__":
    main()
