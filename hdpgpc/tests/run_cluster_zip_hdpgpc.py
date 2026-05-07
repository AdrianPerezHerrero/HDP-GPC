#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cluster_zip_hdpgpc.py

Adapter for clustering records/leads packaged as:
    <record>-ECG<lead>-50Hz.dat
    <record>-ECG<lead>-50Hz.hea
    <record>-ECG<lead>-50Hz.iatr
    <record>-ECG<lead>-50Hz.cluster   # optional/example labels

The script:
  1. Opens a cluster.zip or an extracted cluster directory.
  2. Reads one-channel WFDB-like 16-bit .dat/.hea signals without requiring wfdb.
  3. Parses WFDB/MIT-format binary annotation files with extension .iatr.
  4. Keeps annotations whose symbol is N, extracts a fixed ECG beat window around each N.
  5. Runs HDP-GPC separately per lead/file, saves final .cluster files, and plots clusters.

Typical usage from the root of your HDP-GPC repository, with analysis_one_record.py next to this file:

    python run_cluster_zip_hdpgpc.py \
        --input_zip cluster.zip \
        --records n17 n17c t08 t65 t66 \
        --out_dir results/cluster_zip_hdpgpc

The generated .cluster file has the same two-column style as the t01 example:
    <annotation_sample_or_ms>,<cluster_label>

By default the first column is the annotation SAMPLE index because the provided t01 .cluster
values match the N annotation samples. Use --cluster_time_units ms if you really want ms.

Notes on HDP-GPC fitting
------------------------
The uploaded analysis_one_record.py only reconstructs/plots a model from existing labels via
reload_model_from_labels(...), while this script needs to infer labels. The function
_fit_hdpgpc_auto(...) tries common fitting method names on GPI_HDP objects and then extracts
labels from the return value or from gpmodel.indexes. If your local GPI_HDP exposes a different
training method, run with --hdpgpc_fit_method METHOD_NAME or edit _fit_hdpgpc_auto.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


# -------------------------
# WFDB annotation constants
# -------------------------
# Standard WFDB/MIT annotation code table. Codes 39 and 40 appear in the provided .iatr files
# as delineation markers; they are ignored here because beats are taken from code 1 / symbol N.
ANN_CODE_TO_SYMBOL = {
    0: "NOTQRS",
    1: "N",
    2: "L",
    3: "R",
    4: "a",
    5: "V",
    6: "F",
    7: "J",
    8: "A",
    9: "S",
    10: "E",
    11: "j",
    12: "/",
    13: "Q",
    14: "~",
    16: "|",
    18: "s",
    19: "T",
    20: "*",
    21: "D",
    22: '"',
    23: "=",
    24: "p",
    25: "B",
    26: "^",
    27: "t",
    28: "+",
    29: "u",
    30: "?",
    31: "!",
    32: "[",
    33: "]",
    34: "e",
    35: "n",
    36: "@",
    39: "CODE39",
    40: "CODE40",
}


@dataclass(frozen=True)
class LeadRecord:
    base: str
    record: str
    lead_name: str
    hea_path: Path
    dat_path: Path
    iatr_path: Optional[Path]
    cluster_path: Optional[Path]


@dataclass(frozen=True)
class HeaderInfo:
    record_name: str
    n_sig: int
    fs: float
    n_samples: int
    dat_file: str
    fmt: str
    adc_gain: float
    adc_units: str
    adc_zero: int
    signal_name: str


@dataclass(frozen=True)
class Annotation:
    sample: int
    code: int
    symbol: str
    aux: str = ""
    num: int = 0
    sub: int = 0
    chan: int = 0


# -------------------------
# Input discovery
# -------------------------
def _prepare_input_dir(input_zip: Optional[str], input_dir: Optional[str]) -> tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if input_zip:
        tmp = tempfile.TemporaryDirectory(prefix="cluster_zip_")
        with zipfile.ZipFile(input_zip, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        # The supplied zip contains a top-level folder named cluster/.
        if (root / "cluster").exists():
            root = root / "cluster"
        return root, tmp

    if input_dir:
        root = Path(input_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {root}")
        return root, None

    raise ValueError("Provide either --input_zip or --input_dir")


def _parse_record_from_base(base: str) -> str:
    # Examples: t08-ECG1-50Hz -> t08, n17c-ECG1-50Hz -> n17c
    return base.split("-", 1)[0]


def discover_lead_records(root: Path, records: Optional[list[str]]) -> list[LeadRecord]:
    wanted = set(records) if records else None
    out: list[LeadRecord] = []
    for hea in sorted(root.glob("*.hea")):
        base = hea.stem
        rec = _parse_record_from_base(base)
        if wanted is not None and rec not in wanted:
            continue
        dat = hea.with_suffix(".dat")
        iatr = hea.with_suffix(".iatr")
        cluster = hea.with_suffix(".cluster")
        if not dat.exists():
            print(f"[WARN] Missing .dat for {base}; skipping", file=sys.stderr)
            continue
        # Lead name is taken from header when available; keep a safe base fallback.
        try:
            h = read_header(hea)
            lead_name = h.signal_name
        except Exception:
            lead_name = base.split("-")[1] if "-" in base else base
        out.append(
            LeadRecord(
                base=base,
                record=rec,
                lead_name=lead_name,
                hea_path=hea,
                dat_path=dat,
                iatr_path=iatr if iatr.exists() else None,
                cluster_path=cluster if cluster.exists() else None,
            )
        )
    return out


# -------------------------
# WFDB-like .hea/.dat reader
# -------------------------
def _parse_gain_units(token: str) -> tuple[float, str]:
    # WFDB token examples: "200/mV" or "200".
    if "/" in token:
        gain_s, units = token.split("/", 1)
        return float(gain_s), units
    return float(token), "adu"


def read_header(path: Path) -> HeaderInfo:
    lines = [ln.strip() for ln in path.read_text(encoding="latin1").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Header {path} has fewer than two non-empty lines")

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Cannot parse first header line in {path}: {lines[0]!r}")
    record_name = first[0]
    n_sig = int(first[1])
    fs_token = first[2]
    fs = float(fs_token.split("/", 1)[0])
    n_samples = int(first[3])

    sig = lines[1].split()
    if len(sig) < 9:
        raise ValueError(f"Cannot parse signal header line in {path}: {lines[1]!r}")
    dat_file = sig[0]
    fmt = sig[1]
    gain, units = _parse_gain_units(sig[2])
    adc_zero = int(sig[4])
    signal_name = sig[8]
    return HeaderInfo(
        record_name=record_name,
        n_sig=n_sig,
        fs=fs,
        n_samples=n_samples,
        dat_file=dat_file,
        fmt=fmt,
        adc_gain=gain,
        adc_units=units,
        adc_zero=adc_zero,
        signal_name=signal_name,
    )


def read_signal(hea_path: Path, dat_path: Path, physical: bool = True) -> tuple[np.ndarray, HeaderInfo]:
    header = read_header(hea_path)
    if header.n_sig != 1:
        raise NotImplementedError(f"This adapter expects one signal per file; {hea_path.name} has n_sig={header.n_sig}")
    if header.fmt != "16":
        raise NotImplementedError(f"Only WFDB format 16 is implemented; {hea_path.name} has format={header.fmt}")
    x = np.fromfile(dat_path, dtype="<i2", count=header.n_samples)
    if x.size != header.n_samples:
        raise ValueError(f"{dat_path.name}: expected {header.n_samples} samples, found {x.size}")
    if physical:
        x = (x.astype(np.float64) - header.adc_zero) / header.adc_gain
    else:
        x = x.astype(np.float64)
    return x, header


# -------------------------
# WFDB/MIT binary annotations
# -------------------------
def _wfdb_skip_value(raw4: bytes) -> int:
    """Decode a WFDB SKIP sample interval.

    In the supplied .iatr files, the four bytes after a SKIP annotation are stored as two
    byte-swapped 16-bit words. This is the same byte order used by WFDB's MIT annotation
    reader/writer for long time skips. For example:
        00 ec 00 00 d8 0f -> skip 0x00000fd8 = 4056
        00 ec 01 00 23 70 -> skip 0x00017023 = 94243
    """
    if len(raw4) != 4:
        raise ValueError("SKIP value requires exactly 4 bytes")
    return int.from_bytes(bytes([raw4[1], raw4[0], raw4[3], raw4[2]]), byteorder="big", signed=True)


def read_iatr_annotations(path: Path) -> list[Annotation]:
    b = path.read_bytes()
    i = 0
    t = 0
    anns: list[Annotation] = []
    num = 0
    sub = 0
    chan = 0

    while i + 1 < len(b):
        word = b[i] + (b[i + 1] << 8)
        i += 2
        code = word >> 10
        dt = word & 0x03FF

        if code == 0 and dt == 0:
            break
        if code == 59:  # SKIP
            if i + 4 > len(b):
                break
            t += _wfdb_skip_value(b[i:i + 4])
            i += 4
            continue
        if code == 60:  # NUM
            num = dt
            continue
        if code == 61:  # SUB
            sub = dt
            continue
        if code == 62:  # CHN
            chan = dt
            continue
        if code == 63:  # AUX string attached to previous annotation
            n = int(dt)
            aux_raw = b[i:i + n]
            i += n
            if n % 2:
                i += 1  # aux strings are padded to an even byte count
            if anns:
                aux = aux_raw.rstrip(b"\x00").decode("latin1", errors="replace")
                prev = anns[-1]
                anns[-1] = Annotation(prev.sample, prev.code, prev.symbol, aux, prev.num, prev.sub, prev.chan)
            continue

        t += int(dt)
        anns.append(
            Annotation(
                sample=int(t),
                code=int(code),
                symbol=ANN_CODE_TO_SYMBOL.get(int(code), f"CODE{code}"),
                num=int(num),
                sub=int(sub),
                chan=int(chan),
            )
        )

    return anns


def select_beat_annotations(annotations: Iterable[Annotation], beat_symbol: str = "N") -> list[Annotation]:
    return [a for a in annotations if a.symbol == beat_symbol]


# -------------------------
# Beat extraction and plotting
# -------------------------
def extract_beats(
    signal: np.ndarray,
    beat_annotations: list[Annotation],
    fs: float,
    pre_ms: float,
    post_ms: float,
    drop_first_beat: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    pre = int(round(pre_ms * fs / 1000.0))
    post = int(round(post_ms * fs / 1000.0))
    if pre <= 0 or post <= 0:
        raise ValueError(f"pre/post windows must be positive; got pre={pre}, post={post} samples")

    anns = beat_annotations[1:] if drop_first_beat and len(beat_annotations) > 0 else beat_annotations
    beats = []
    samples = []
    skipped_left = 0
    skipped_right = 0
    for ann in anns:
        lo = ann.sample - pre
        hi = ann.sample + post
        if lo < 0:
            skipped_left += 1
            continue
        if hi > len(signal):
            skipped_right += 1
            continue
        beats.append(signal[lo:hi].astype(np.float64))
        samples.append(ann.sample)

    if not beats:
        return np.empty((0, pre + post), dtype=np.float64), np.empty((0,), dtype=np.int64), {
            "pre_samples": pre,
            "post_samples": post,
            "skipped_left": skipped_left,
            "skipped_right": skipped_right,
        }

    return np.vstack(beats), np.asarray(samples, dtype=np.int64), {
        "pre_samples": pre,
        "post_samples": post,
        "skipped_left": skipped_left,
        "skipped_right": skipped_right,
    }


def normalize_beats(beats: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return beats.astype(np.float64, copy=False)
    if beats.size == 0:
        return beats.astype(np.float64, copy=False)
    x = beats.astype(np.float64, copy=True)
    if mode == "per_beat_zscore":
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return (x - mu) / sd
    if mode == "per_beat_center":
        return x - x.mean(axis=1, keepdims=True)
    raise ValueError(f"Unknown normalization mode: {mode}")


def _save_cluster_file(path: Path, beat_samples: np.ndarray, labels: np.ndarray, fs: float, units: str) -> None:
    if units == "sample":
        locs = beat_samples.astype(int)
    elif units == "ms":
        locs = np.rint(beat_samples.astype(float) / float(fs) * 1000.0).astype(int)
    else:
        raise ValueError(f"Unknown cluster_time_units: {units}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for loc, lab in zip(locs, labels.astype(int)):
            w.writerow([int(loc), int(lab)])


def _plot_cluster_overview_png(beats: np.ndarray, labels: np.ndarray, out_png: Path, title: str, max_traces_per_cluster: int = 80) -> None:
    import matplotlib.pyplot as plt

    if beats.ndim == 3:
        y = beats[:, :, 0]
    else:
        y = beats
    labels = labels.astype(int)
    clusters = sorted(np.unique(labels).tolist())

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    x = np.arange(y.shape[1])
    for cl in clusters:
        idx = np.where(labels == cl)[0]
        if len(idx) == 0:
            continue
        shown = idx[:max_traces_per_cluster]
        for j in shown:
            ax.plot(x, y[j], alpha=0.12, linewidth=0.8)
        ax.plot(x, np.median(y[idx], axis=0), linewidth=2.2, label=f"cluster {cl} (n={len(idx)})")
    ax.set_title(title)
    ax.set_xlabel("samples in extracted beat window")
    ax.set_ylabel("amplitude")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# -------------------------
# HDP-GPC model helpers
# -------------------------
def _import_analysis_module(script_dir: Path):
    sys.path.insert(0, str(script_dir.resolve()))
    try:
        return importlib.import_module("analysis_one_record")
    finally:
        try:
            sys.path.remove(str(script_dir.resolve()))
        except ValueError:
            pass


def _build_hdpgpc_model(data_3d: np.ndarray, analysis_module: Any):
    """Build GPI_HDP with the same hyperparameters used in analysis_one_record.py."""
    import torch
    from hdpgpc.get_data import compute_estimators_LDS

    torch.set_default_dtype(torch.float64)

    num_samples, num_obs_per_sample, num_outputs = data_3d.shape
    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data_3d)

    sigma = std * 1.0
    gamma = std_dif * 1.0
    outputscale_ = 300.0
    ini_lengthscale = 3.0
    bound_lengthscale = (1.0, 20.0)
    bound_sigma = (std * 1e-5, std * 1e-3)
    noise_warp = std * 0.1
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

    x_basis = np.atleast_2d(np.arange(0, num_obs_per_sample, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(0, num_obs_per_sample, 2, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(0, num_obs_per_sample, dtype=np.float64)).T
    x_trains = np.array([x_train] * num_samples)

    if hasattr(analysis_module, "build_sw_gp"):
        sw_gp = analysis_module.build_sw_gp(
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
    else:
        import hdpgpc.GPI_HDP as hdpgp
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
            reestimate_initial_params=False,
            n_explore_steps=15,
            free_deg_MNIV=3,
        )
    return sw_gp, x_train, x_trains


def _candidate_label_array(obj: Any, n: int) -> Optional[np.ndarray]:
    """Try to find a 1-D label vector of length n inside a return value or attribute."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if arr.ndim == 1 and arr.shape[0] == n:
            return arr.astype(np.int64)
        if arr.ndim == 2 and 1 in arr.shape and arr.size == n:
            return arr.reshape(-1).astype(np.int64)
    if isinstance(obj, (list, tuple)):
        # A direct list of numeric labels
        if len(obj) == n and all(isinstance(v, (int, np.integer, float, np.floating)) for v in obj):
            return np.asarray(obj, dtype=np.int64)
        # A tuple/list return that includes labels somewhere inside
        for item in obj:
            lab = _candidate_label_array(item, n)
            if lab is not None:
                return lab
    if isinstance(obj, dict):
        for key in ("labels", "cluster_labels", "assignments", "z", "clusters", "y_pred"):
            if key in obj:
                lab = _candidate_label_array(obj[key], n)
                if lab is not None:
                    return lab
    return None


def _labels_from_model_indexes(sw_gp: Any, n: int) -> Optional[np.ndarray]:
    """Reconstruct labels from gpmodel.indexes after fitting, if available."""
    if not hasattr(sw_gp, "gpmodels"):
        return None
    labels = np.full(n, -1, dtype=np.int64)
    try:
        # In one-lead runs, gpmodels[0][m].indexes stores beat indexes in model m.
        gp_list = sw_gp.gpmodels[0]
        for m, gp in enumerate(gp_list):
            idxs = getattr(gp, "indexes", [])
            for idx in idxs:
                if 0 <= int(idx) < n:
                    labels[int(idx)] = int(m)
    except Exception:
        return None
    if np.all(labels >= 0):
        return labels
    return None


def _extract_labels_from_sw_gp(sw_gp: Any, n: int) -> Optional[np.ndarray]:
    for attr in ("cluster_labels", "labels", "assignments", "z", "c", "C", "clusters"):
        if hasattr(sw_gp, attr):
            lab = _candidate_label_array(getattr(sw_gp, attr), n)
            if lab is not None:
                return lab
    return _labels_from_model_indexes(sw_gp, n)


def _call_fit_method(sw_gp: Any, method_name: str, x_trains: np.ndarray, data_3d: np.ndarray) -> Any:
    method = getattr(sw_gp, method_name)
    attempts = [
        (x_trains, data_3d),
        (x_trains, data_3d, True),
        (x_trains, data_3d),
        (data_3d,),
        (data_3d, x_trains),
    ]
    kw_attempts = [
        {"warp": True},
        {"compute_warp": True},
        {},
    ]

    last_error: Optional[BaseException] = None
    for args in attempts:
        for kwargs in kw_attempts:
            try:
                return method(*args, **kwargs)
            except TypeError as e:
                last_error = e
                continue
    raise TypeError(f"Could not call HDP-GPC method {method_name!r}; last error: {last_error}")


def _fit_hdpgpc_auto(
    sw_gp: Any,
    x_trains: np.ndarray,
    data_3d: np.ndarray,
    method_name: Optional[str] = None,
) -> np.ndarray:
    """Fit HDP-GPC and return cluster labels.

    This intentionally supports several method names because local HDP-GPC forks often expose
    different training entry points. If none work, the raised error names the place to edit.
    """
    n = int(data_3d.shape[0])
    if n == 0:
        raise ValueError("No beats to cluster")

    candidates = [method_name] if method_name else [
        "fit_predict",
        "fit",
        "train",
        "run",
        "cluster",
        "clustering",
        "online_clustering",
        "sequential_clustering",
        "infer",
        "inference",
    ]
    candidates = [m for m in candidates if m]

    errors: list[str] = []
    for name in candidates:
        if not hasattr(sw_gp, name):
            continue
        try:
            ret = _call_fit_method(sw_gp, name, x_trains, data_3d)
            labels = _candidate_label_array(ret, n)
            if labels is None:
                labels = _extract_labels_from_sw_gp(sw_gp, n)
            if labels is not None:
                labels = np.asarray(labels).reshape(-1).astype(np.int64)
                # Normalize to 1-based labels to match the t01 .cluster example style.
                uniq = sorted(np.unique(labels).tolist())
                remap = {old: new for new, old in enumerate(uniq, start=1)}
                return np.asarray([remap[int(v)] for v in labels], dtype=np.int64)
            errors.append(f"{name}: ran, but no label vector of length {n} was found")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    msg = [
        "Could not infer cluster labels from the local HDP-GPC API.",
        "The uploaded analysis_one_record.py reconstructs plots from existing labels but does not expose the original fitting call.",
        "Set --hdpgpc_fit_method to your GPI_HDP training method name, or edit _fit_hdpgpc_auto(...).",
        "Tried:",
    ]
    msg.extend(f"  - {e}" for e in errors)
    raise RuntimeError("\n".join(msg))


def run_hdpgpc_for_lead(
    beats_2d: np.ndarray,
    analysis_module: Any,
    hdpgpc_fit_method: Optional[str],
) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    """Return labels, fitted model, x_train, and 3-D data array."""
    # HDP-GPC expects (num_beats, num_time_points, num_outputs). One lead => num_outputs=1.
    data_3d = np.asarray(beats_2d, dtype=np.float64)[:, :, None]
    sw_gp, x_train, x_trains = _build_hdpgpc_model(data_3d, analysis_module)
    labels = _fit_hdpgpc_auto(sw_gp, x_trains, data_3d, method_name=hdpgpc_fit_method)
    return labels, sw_gp, x_train, data_3d


def plot_hdpgpc_overview(
    sw_gp: Any,
    data_3d: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    base: str,
    analysis_module: Any,
) -> None:
    """Make the classic HDP-GPC plot if util_plots are available; otherwise do nothing."""
    try:
        from hdpgpc.util_plots import plot_models_plotly, print_results

        selected_gpmodels = sw_gp.selected_gpmodels() if hasattr(sw_gp, "selected_gpmodels") else sorted(np.unique(labels).tolist())
        y_names = np.asarray(["N"] * len(labels), dtype=str)
        main_model = print_results(sw_gp, y_names, 0, error=False)
        plot_models_plotly(
            sw_gp,
            selected_gpmodels,
            main_model,
            y_names,
            0,
            lead=0,
            save=str(out_dir / f"{base}_HDPGPC_clusters.pdf"),
            step=0.5,
            plot_latent=True,
        )
    except Exception as e:
        (out_dir / f"{base}_HDPGPC_plot_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[WARN] HDP-GPC plot failed for {base}: {e}", file=sys.stderr)


# -------------------------
# Existing .cluster validation/demo
# -------------------------
def read_existing_cluster_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    samples = []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        samples.append(int(float(parts[0])))
        labels.append(int(float(parts[1])))
    return np.asarray(samples, dtype=np.int64), np.asarray(labels, dtype=np.int64)


# -------------------------
# Main per-lead workflow
# -------------------------
def process_one_lead(
    lr: LeadRecord,
    out_root: Path,
    args: argparse.Namespace,
    analysis_module: Optional[Any],
) -> dict[str, Any]:
    out_dir = out_root / lr.record / lr.base
    out_dir.mkdir(parents=True, exist_ok=True)

    signal, header = read_signal(lr.hea_path, lr.dat_path, physical=not args.raw_adc)
    annotations = read_iatr_annotations(lr.iatr_path) if lr.iatr_path and lr.iatr_path.exists() else []
    ann_counts = Counter(a.symbol for a in annotations)
    beats_ann = select_beat_annotations(annotations, beat_symbol=args.beat_symbol)

    row: dict[str, Any] = {
        "base": lr.base,
        "record": lr.record,
        "lead": header.signal_name,
        "fs": header.fs,
        "signal_samples": len(signal),
        "annotations_total": len(annotations),
        "beat_symbol": args.beat_symbol,
        "beat_annotations": len(beats_ann),
        "status": "started",
        "error": "",
        "cluster_file": "",
        "plot_png": "",
    }

    # Save an annotation summary for debugging.
    (out_dir / "annotation_counts.json").write_text(json.dumps(dict(ann_counts), indent=2), encoding="utf-8")

    if len(beats_ann) == 0:
        row["status"] = "skipped_no_beat_annotations"
        row["error"] = f"No {args.beat_symbol!r} beat annotations in {lr.iatr_path.name if lr.iatr_path else 'missing .iatr'}"
        print(f"[SKIP] {lr.base}: {row['error']}")
        return row

    beats, beat_samples, ext_info = extract_beats(
        signal,
        beats_ann,
        fs=header.fs,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        drop_first_beat=args.drop_first_beat,
    )
    row.update(ext_info)
    row["beats_extracted"] = int(beats.shape[0])
    if beats.shape[0] == 0:
        row["status"] = "skipped_no_valid_windows"
        row["error"] = "All beat windows would cross signal boundaries"
        print(f"[SKIP] {lr.base}: {row['error']}")
        return row

    if args.max_beats and beats.shape[0] > args.max_beats:
        beats = beats[:args.max_beats]
        beat_samples = beat_samples[:args.max_beats]
        row["beats_extracted"] = int(beats.shape[0])
        row["truncated_to_max_beats"] = int(args.max_beats)

    beats_for_model = normalize_beats(beats, args.normalize)
    np.save(out_dir / f"{lr.base}_beats.npy", beats_for_model)
    np.save(out_dir / f"{lr.base}_beat_samples.npy", beat_samples)

    if args.use_existing_clusters and lr.cluster_path:
        existing_samples, labels = read_existing_cluster_file(lr.cluster_path)
        # Align existing cluster labels to extracted windows by sample index.
        label_by_sample = {int(s): int(l) for s, l in zip(existing_samples, labels)}
        keep = [i for i, s in enumerate(beat_samples) if int(s) in label_by_sample]
        if not keep:
            raise ValueError(f"{lr.base}: existing .cluster labels do not overlap extracted beat samples")
        beat_samples = beat_samples[keep]
        beats_for_model = beats_for_model[keep]
        labels = np.asarray([label_by_sample[int(s)] for s in beat_samples], dtype=np.int64)
        row["status"] = "used_existing_cluster_labels"
        sw_gp = None
        data_3d = beats_for_model[:, :, None]
    else:
        if analysis_module is None:
            raise ImportError(
                "analysis_one_record.py / hdpgpc could not be imported. Put this script next to analysis_one_record.py "
                "inside your HDP-GPC repository, or run with --use_existing_clusters for t01-only validation."
            )
        labels, sw_gp, x_train, data_3d = run_hdpgpc_for_lead(
            beats_for_model,
            analysis_module=analysis_module,
            hdpgpc_fit_method=args.hdpgpc_fit_method,
        )
        row["status"] = "clustered_hdpgpc"
        np.save(out_dir / f"cluster_labels_{lr.base}_offline.npy", labels)
        if sw_gp is not None and not args.no_hdpgpc_plot:
            plot_hdpgpc_overview(sw_gp, data_3d, labels, out_dir, lr.base, analysis_module)

    cluster_out = out_root / f"{lr.base}.cluster"
    _save_cluster_file(cluster_out, beat_samples, labels, fs=header.fs, units=args.cluster_time_units)
    row["cluster_file"] = str(cluster_out)
    row["n_clusters"] = int(np.unique(labels).size)
    row["cluster_sizes"] = json.dumps({int(k): int(v) for k, v in Counter(labels.astype(int)).items()}, sort_keys=True)

    plot_png = out_dir / f"{lr.base}_cluster_overview.png"
    _plot_cluster_overview_png(
        beats_for_model,
        labels,
        plot_png,
        title=f"{lr.base}: extracted {args.beat_symbol} beats clustered per lead",
        max_traces_per_cluster=args.max_plot_traces_per_cluster,
    )
    row["plot_png"] = str(plot_png)

    # Save an aligned CSV for auditability.
    with (out_dir / f"{lr.base}_beats_and_clusters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["beat_index", "sample", "ms", "cluster"])
        w.writeheader()
        for i, (s, lab) in enumerate(zip(beat_samples, labels.astype(int))):
            w.writerow({"beat_index": i, "sample": int(s), "ms": float(s) / header.fs * 1000.0, "cluster": int(lab)})

    print(f"[OK] {lr.base}: {len(labels)} beats, {row['n_clusters']} clusters -> {cluster_out}")
    return row


# -------------------------
# CLI
# -------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cluster ZIP ECG beat records with HDP-GPC, one lead at a time.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_zip", type=str, help="Path to cluster.zip")
    src.add_argument("--input_dir", type=str, help="Path to extracted cluster/ directory")
    ap.add_argument("--records", nargs="*", default=["n17", "n17c", "t08", "t65", "t66"],
                    help="Record ids to process. Default: n17 n17c t08 t65 t66")
    ap.add_argument("--out_dir", type=str, default="results/cluster_zip_hdpgpc", help="Output directory")
    ap.add_argument("--analysis_script_dir", type=str, default=".",
                    help="Directory containing analysis_one_record.py and the hdpgpc package import path")
    ap.add_argument("--beat_symbol", type=str, default="N", help="Annotation symbol used as heartbeat location")
    ap.add_argument("--pre_ms", type=float, default=250.0, help="Milliseconds before annotation to include in beat window")
    ap.add_argument("--post_ms", type=float, default=450.0, help="Milliseconds after annotation to include in beat window")
    ap.add_argument("--drop_first_beat", action="store_true",
                    help="Drop the first N annotation. Useful if you want to mirror the provided t01 .cluster files.")
    ap.add_argument("--cluster_time_units", choices=["sample", "ms"], default="sample",
                    help="First column in output .cluster files. Default sample matches the provided t01 example values.")
    ap.add_argument("--normalize", choices=["none", "per_beat_center", "per_beat_zscore"], default="none",
                    help="Optional beat normalization before clustering")
    ap.add_argument("--raw_adc", action="store_true", help="Use raw ADC counts instead of physical units")
    ap.add_argument("--hdpgpc_fit_method", type=str, default=None,
                    help="Name of the local GPI_HDP method used to fit/infer clusters. Default: auto-detect.")
    ap.add_argument("--use_existing_clusters", action="store_true",
                    help="Use existing .cluster labels instead of HDP-GPC. Mainly for validating t01 extraction/plots.")
    ap.add_argument("--no_hdpgpc_plot", action="store_true", help="Disable HDP-GPC plot_models_plotly output")
    ap.add_argument("--max_plot_traces_per_cluster", type=int, default=80)
    ap.add_argument("--max_beats", type=int, default=0,
                    help="Debug option: limit beats per lead before clustering. 0 means no limit.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    input_root, tmp = _prepare_input_dir(args.input_zip, args.input_dir)
    try:
        lead_records = discover_lead_records(input_root, args.records)
        if not lead_records:
            raise RuntimeError(f"No matching .hea/.dat files found under {input_root} for records={args.records}")

        analysis_module = None
        if not args.use_existing_clusters:
            analysis_module = _import_analysis_module(Path(args.analysis_script_dir))

        print(f"[INFO] input_root: {input_root}")
        print(f"[INFO] out_dir:    {out_root}")
        print(f"[INFO] leads:      {len(lead_records)}")
        print(f"[INFO] records:    {sorted(set(lr.record for lr in lead_records))}")

        rows: list[dict[str, Any]] = []
        for lr in lead_records:
            try:
                rows.append(process_one_lead(lr, out_root, args, analysis_module))
            except Exception as e:
                fail_dir = out_root / lr.record / lr.base
                fail_dir.mkdir(parents=True, exist_ok=True)
                tb = traceback.format_exc()
                (fail_dir / "FAILED.txt").write_text(tb, encoding="utf-8")
                print(f"[FAIL] {lr.base}: {e}", file=sys.stderr)
                rows.append({
                    "base": lr.base,
                    "record": lr.record,
                    "lead": lr.lead_name,
                    "status": "failed",
                    "error": repr(e),
                })

        summary_path = out_root / "summary.csv"
        fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[DONE] Summary written to: {summary_path}")
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
