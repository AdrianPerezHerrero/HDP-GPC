#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cluster_zip_hdpgpc.py

Record-based adapter for clustering ECG records packaged as one file per lead:
    <record>-ECG0-50Hz.dat/.hea/.iatr/.cluster
    <record>-ECG1-50Hz.dat/.hea/.iatr/.cluster

Default workflow is multi-output:
  1. Open a cluster.zip or extracted cluster directory.
  2. For each requested record, read ECG0 and ECG1.
  3. Apply optional record-level filtering to all used signals.
  4. Detect/choose beat locations on ECG0 by default.
  5. Extract identical beat windows from the output lead(s) at those locations.
  6. Build data with shape (num_beats, window_samples, num_outputs).
  7. Run HDP-GPC once per record.
  8. Save cluster labels and overview plots.

Single-lead mode is available with:
    --lead_mode single --selected_lead 0
or:
    --lead_mode single --selected_lead 1

In single-lead mode, the detector still uses ECG0 by default because it is usually the
best detection lead. Use --detection_lead selected to detect on the selected lead.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import tempfile
import time
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
ANN_CODE_TO_SYMBOL = {
    0: "NOTQRS", 1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S",
    10: "E", 11: "j", 12: "/", 13: "Q", 14: "~", 16: "|", 18: "s", 19: "T", 20: "*",
    21: "D", 22: '"', 23: "=", 24: "p", 25: "B", 26: "^", 27: "t", 28: "+", 29: "u",
    30: "?", 31: "!", 32: "[", 33: "]", 34: "e", 35: "n", 36: "@", 39: "CODE39", 40: "CODE40",
}


@dataclass(frozen=True)
class LeadRecord:
    base: str
    record: str
    lead_name: str
    lead_index: int
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
    return base.split("-", 1)[0]


def _parse_ecg_index_from_base(base: str) -> Optional[int]:
    # Examples: t08-ECG1-50Hz -> 1, n17c-ECG0-50Hz -> 0
    for part in base.split("-"):
        if part.upper().startswith("ECG"):
            suffix = part[3:]
            if suffix.isdigit():
                return int(suffix)
    return None


def discover_lead_records(root: Path, records: Optional[list[str]]) -> list[LeadRecord]:
    wanted = set(records) if records else None
    out: list[LeadRecord] = []
    for hea in sorted(root.glob("*.hea")):
        base = hea.stem
        rec = _parse_record_from_base(base)
        lead_index = _parse_ecg_index_from_base(base)
        if lead_index is None:
            print(f"[WARN] Could not infer ECG lead index from {base}; skipping", file=sys.stderr)
            continue
        if wanted is not None and rec not in wanted:
            continue
        dat = hea.with_suffix(".dat")
        iatr = hea.with_suffix(".iatr")
        cluster = hea.with_suffix(".cluster")
        if not dat.exists():
            print(f"[WARN] Missing .dat for {base}; skipping", file=sys.stderr)
            continue
        try:
            h = read_header(hea)
            lead_name = h.signal_name
        except Exception:
            lead_name = f"ECG{lead_index}"
        out.append(
            LeadRecord(
                base=base,
                record=rec,
                lead_name=lead_name,
                lead_index=lead_index,
                hea_path=hea,
                dat_path=dat,
                iatr_path=iatr if iatr.exists() else None,
                cluster_path=cluster if cluster.exists() else None,
            )
        )
    return out


def _lead_map_for_record(record: str, lead_records: list[LeadRecord]) -> dict[int, LeadRecord]:
    by_idx: dict[int, list[LeadRecord]] = {}
    for lr in lead_records:
        by_idx.setdefault(lr.lead_index, []).append(lr)

    out: dict[int, LeadRecord] = {}
    for idx, vals in by_idx.items():
        if len(vals) != 1:
            raise RuntimeError(f"Record {record}: expected exactly one ECG{idx} file, found {[x.base for x in vals]}")
        out[idx] = vals[0]
    return out


def _require_leads(record: str, lead_map: dict[int, LeadRecord], required: Iterable[int]) -> None:
    missing = sorted({int(idx) for idx in required if int(idx) not in lead_map})
    if missing:
        raise FileNotFoundError(f"Record {record}: missing required ECG lead(s): {missing}")


def _output_and_detection_indices(args: argparse.Namespace) -> tuple[list[int], int]:
    selected = int(args.selected_lead)
    if args.lead_mode == "multi":
        output_indices = [0, 1]
    else:
        output_indices = [selected]

    if str(args.detection_lead) == "selected":
        detection_idx = selected
    else:
        detection_idx = int(args.detection_lead)
    return output_indices, detection_idx


# -------------------------
# WFDB-like .hea/.dat reader
# -------------------------
def _parse_gain_units(token: str) -> tuple[float, str]:
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
    fs = float(first[2].split("/", 1)[0])
    n_samples = int(first[3])

    sig = lines[1].split()
    if len(sig) < 9:
        raise ValueError(f"Cannot parse signal header line in {path}: {lines[1]!r}")
    dat_file = sig[0]
    fmt = sig[1]
    gain, units = _parse_gain_units(sig[2])
    adc_zero = int(sig[4])
    signal_name = sig[8]
    return HeaderInfo(record_name, n_sig, fs, n_samples, dat_file, fmt, gain, units, adc_zero, signal_name)


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
# Signal preprocessing
# -------------------------
def _apply_sos_filter(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import sosfilt, sosfiltfilt
    except Exception as e:
        raise ImportError("SciPy is required for signal filtering. Install it with: pip install scipy") from e
    try:
        return sosfiltfilt(sos, x)
    except ValueError:
        return sosfilt(sos, x)


def highpass_filter_signal(signal: np.ndarray, fs: float, cutoff_hz: float = 0.7, order: int = 2) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    fs = float(fs)
    cutoff_hz = float(cutoff_hz)
    order = int(order)
    nyq = fs / 2.0
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive; got fs={fs}")
    if order < 1:
        raise ValueError(f"High-pass filter order must be >= 1; got order={order}")
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"High-pass cutoff must be in (0, {nyq}) Hz; got {cutoff_hz}")
    try:
        from scipy.signal import butter
    except Exception as e:
        raise ImportError("SciPy is required for --signal_filter highpass. Install it with: pip install scipy") from e
    sos = butter(order, cutoff_hz / nyq, btype="highpass", output="sos")
    return _apply_sos_filter(sos, x)


def bandpass_filter_signal(signal: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 45.0, order: int = 2) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    fs = float(fs)
    low_hz = float(low_hz)
    high_hz = float(high_hz)
    order = int(order)
    nyq = fs / 2.0
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive; got fs={fs}")
    if order < 1:
        raise ValueError(f"Band-pass filter order must be >= 1; got order={order}")
    if low_hz <= 0 or high_hz <= 0:
        raise ValueError(f"Band-pass cutoffs must be positive; got low={low_hz}, high={high_hz}")
    if high_hz >= nyq:
        high_hz = 0.99 * nyq
    if low_hz >= high_hz:
        raise ValueError(f"Band-pass low cutoff must be below high cutoff; got {low_hz} >= {high_hz}")
    try:
        from scipy.signal import butter
    except Exception as e:
        raise ImportError("SciPy is required for --signal_filter bandpass. Install it with: pip install scipy") from e
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return _apply_sos_filter(sos, x)


def preprocess_signal(signal: np.ndarray, header: HeaderInfo, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(getattr(args, "signal_filter", "highpass"))
    info: dict[str, Any] = {"signal_filter": mode}
    if mode == "none":
        return np.asarray(signal, dtype=np.float64), info
    if mode == "highpass":
        cutoff = float(getattr(args, "highpass_cutoff_hz", 0.7))
        order = int(getattr(args, "highpass_order", 2))
        y = highpass_filter_signal(signal, fs=header.fs, cutoff_hz=cutoff, order=order)
        info.update({"highpass_cutoff_hz": cutoff, "highpass_order": order})
        return y, info
    if mode == "bandpass":
        low = float(getattr(args, "bandpass_low_hz", 0.5))
        high = float(getattr(args, "bandpass_high_hz", 45.0))
        order = int(getattr(args, "bandpass_order", 2))
        y = bandpass_filter_signal(signal, fs=header.fs, low_hz=low, high_hz=high, order=order)
        info.update({"bandpass_low_hz": low, "bandpass_high_hz": min(high, 0.99 * header.fs / 2.0), "bandpass_order": order})
        return y, info
    raise ValueError(f"Unknown signal_filter mode: {mode!r}")


# -------------------------
# WFDB/MIT binary annotations
# -------------------------
def _wfdb_skip_value(raw4: bytes) -> int:
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
        if code == 60:
            num = dt
            continue
        if code == 61:
            sub = dt
            continue
        if code == 62:
            chan = dt
            continue
        if code == 63:
            n = int(dt)
            aux_raw = b[i:i + n]
            i += n
            if n % 2:
                i += 1
            if anns:
                aux = aux_raw.rstrip(b"\x00").decode("latin1", errors="replace")
                prev = anns[-1]
                anns[-1] = Annotation(prev.sample, prev.code, prev.symbol, aux, prev.num, prev.sub, prev.chan)
            continue
        t += int(dt)
        anns.append(Annotation(int(t), int(code), ANN_CODE_TO_SYMBOL.get(int(code), f"CODE{code}"), num=int(num), sub=int(sub), chan=int(chan)))
    return anns


def select_beat_annotations(annotations: Iterable[Annotation], beat_symbol: str = "N") -> list[Annotation]:
    return [a for a in annotations if a.symbol == beat_symbol]


# -------------------------
# Optional WFDB XQRS detector
# -------------------------
def _install_scipy_ricker_compat() -> None:
    try:
        import scipy.signal as scipy_signal
    except Exception:
        return
    if hasattr(scipy_signal, "ricker"):
        return

    def _ricker(points, a):
        points = int(points)
        a = float(a)
        if points < 1:
            return np.asarray([], dtype=float)
        if a <= 0:
            raise ValueError("a must be positive")
        vec = np.arange(points, dtype=float) - (points - 1.0) / 2.0
        xsq = (vec / a) ** 2
        A = 2.0 / (math.sqrt(3.0 * a) * (math.pi ** 0.25))
        return A * (1.0 - xsq) * np.exp(-0.5 * xsq)

    scipy_signal.ricker = _ricker  # type: ignore[attr-defined]


def detect_beats_xqrs(signal: np.ndarray, fs: float, sampfrom: int = 0, sampto: Optional[int] = None, learn: bool = True, verbose: bool = False) -> list[Annotation]:
    _install_scipy_ricker_compat()
    try:
        from wfdb import processing
    except Exception as e:
        raise ImportError("WFDB is required for XQRS. Install or update it with: pip install -U wfdb") from e

    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    start = max(0, int(sampfrom or 0))
    stop = len(sig) if sampto is None or int(sampto) <= 0 else min(len(sig), int(sampto))
    if stop <= start:
        raise ValueError(f"Invalid XQRS interval: sampfrom={start}, sampto={stop}, signal length={len(sig)}")

    try:
        qrs = processing.xqrs_detect(sig=sig, fs=float(fs), sampfrom=start, sampto=stop, learn=bool(learn), verbose=bool(verbose))
    except AttributeError as e:
        if "ricker" not in str(e):
            raise
        _install_scipy_ricker_compat()
        qrs = processing.xqrs_detect(sig=sig, fs=float(fs), sampfrom=start, sampto=stop, learn=bool(learn), verbose=bool(verbose))
    return [Annotation(sample=int(s), code=1, symbol="N") for s in np.asarray(qrs, dtype=int).reshape(-1)]


def choose_beat_locations(signal: np.ndarray, header: HeaderInfo, annotation_beats: list[Annotation], args: argparse.Namespace, detector_label: str) -> tuple[list[Annotation], str]:
    source = str(getattr(args, "beat_source", "xqrs"))
    if getattr(args, "use_xqrs", False):
        source = "xqrs"
    if getattr(args, "xqrs_if_missing", False):
        source = "auto"
    if source == "annotations":
        return annotation_beats, f"annotations_{detector_label}"
    if source == "auto" and len(annotation_beats) > 0:
        return annotation_beats, f"annotations_{detector_label}"
    if source in {"auto", "xqrs"}:
        beats = detect_beats_xqrs(
            signal=signal,
            fs=header.fs,
            sampfrom=int(getattr(args, "xqrs_sampfrom", 0) or 0),
            sampto=(None if int(getattr(args, "xqrs_sampto", 0) or 0) <= 0 else int(getattr(args, "xqrs_sampto"))),
            learn=not bool(getattr(args, "xqrs_no_learn", False)),
            verbose=bool(getattr(args, "xqrs_verbose", False)),
        )
        return beats, f"xqrs_{detector_label}"
    raise ValueError(f"Unknown beat_source: {source!r}")


# -------------------------
# Beat extraction and plotting
# -------------------------
def extract_beats_outputs(signals: list[np.ndarray], beat_annotations: list[Annotation], fs: float, pre_ms: float, post_ms: float, drop_first_beat: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    pre = int(round(pre_ms * fs / 1000.0))
    post = int(round(post_ms * fs / 1000.0))
    if pre <= 0 or post <= 0:
        raise ValueError(f"pre/post windows must be positive; got pre={pre}, post={post} samples")

    anns = beat_annotations[1:] if drop_first_beat and len(beat_annotations) > 0 else beat_annotations
    all_beats: list[np.ndarray] = []
    samples: list[int] = []
    skipped_left = 0
    skipped_right = 0
    for ann in anns:
        lo = ann.sample - pre
        hi = ann.sample + post
        if lo < 0:
            skipped_left += 1
            continue
        if any(hi > len(sig) for sig in signals):
            skipped_right += 1
            continue
        all_beats.append(np.column_stack([sig[lo:hi].astype(np.float64) for sig in signals]))
        samples.append(int(ann.sample))

    win_len = pre + post
    n_outputs = len(signals)
    if not all_beats:
        return np.empty((0, win_len, n_outputs), dtype=np.float64), np.empty((0,), dtype=np.int64), {
            "pre_samples": pre, "post_samples": post, "skipped_left": skipped_left, "skipped_right": skipped_right,
        }
    return np.stack(all_beats, axis=0), np.asarray(samples, dtype=np.int64), {
        "pre_samples": pre, "post_samples": post, "skipped_left": skipped_left, "skipped_right": skipped_right,
    }


def normalize_beats(data: np.ndarray, mode: str) -> np.ndarray:
    """Normalize a beat tensor (N, L, D). Default mode 'none' returns data unchanged."""
    if mode == "none":
        return np.asarray(data, dtype=np.float64)
    if data.size == 0:
        return np.asarray(data, dtype=np.float64)
    x = np.asarray(data, dtype=np.float64).copy()
    if mode == "per_beat_center":
        return x - x.mean(axis=1, keepdims=True)
    if mode == "per_beat_zscore":
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return (x - mu) / sd
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


def _plot_cluster_overview_png(data: np.ndarray, labels: np.ndarray, out_png: Path, title: str, model_lead: int, max_traces_per_cluster: int = 80) -> None:
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    if data.ndim == 2:
        y = data
    elif data.ndim == 3:
        y = data[:, :, int(model_lead)]
    else:
        raise ValueError(f"Expected data with ndim 2 or 3, got shape {data.shape}")

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
    script_path = str(script_dir.resolve())
    if script_path not in sys.path:
        sys.path.insert(0, script_path)
    try:
        return importlib.import_module("analysis_one_record")
    except ModuleNotFoundError:
        return None


def _build_hdpgpc_model(data_3d: np.ndarray, args: argparse.Namespace):
    """Build GPI_HDP for one-output or multi-output data."""
    import torch
    import hdpgpc.GPI_HDP as hdpgp
    from hdpgpc.get_data import compute_estimators_LDS

    torch.set_default_dtype(torch.float64)

    data_3d = np.asarray(data_3d, dtype=np.float64)
    num_samples, num_obs_per_sample, num_outputs = data_3d.shape
    std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data_3d)

    sigma = std * float(getattr(args, "sigma_scale", 1.0))
    gamma = std_dif * float(getattr(args, "gamma_scale", 1.2))
    outputscale_ = float(getattr(args, "outputscale", 300.0))
    ini_lengthscale = float(getattr(args, "ini_lengthscale", 3.0))
    bound_lengthscale = (float(getattr(args, "bound_lengthscale_low", 1.0)), float(getattr(args, "bound_lengthscale_high", 20.0)))

    noise_warp = std * float(getattr(args, "noise_warp_scale", 20.0))
    bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

    l, L = 0, num_obs_per_sample
    x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(l, L, 5, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T
    x_trains = np.array([x_train] * num_samples)

    print(f"[HDPGPC] data shape={data_3d.shape}, sigma={sigma}, gamma={gamma}, with_warp={bool(args.with_warp)}")
    kwargs = dict(
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
        max_models=int(getattr(args, "max_models", 100)),
        mode_warp="rough",
        bayesian_params=True,
        inducing_points=False,
        reestimate_initial_params=False,
        n_explore_steps=int(getattr(args, "n_explore_steps", 5)),
        free_deg_MNIV=int(getattr(args, "free_deg_MNIV", 5)),
        share_gp=True,
    )
    # Some local GPI_HDP versions accept these kwargs; older ones may not.
    for optional_key, optional_value in {"use_snr": True, "hdp_hyp": "less"}.items():
        kwargs[optional_key] = optional_value
    try:
        sw_gp = hdpgp.GPI_HDP(x_basis, **kwargs)
    except TypeError:
        kwargs.pop("use_snr", None)
        kwargs.pop("hdp_hyp", None)
        sw_gp = hdpgp.GPI_HDP(x_basis, **kwargs)
    return sw_gp, x_train, x_trains


def _candidate_label_array(obj: Any, n: int) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if arr.ndim == 1 and arr.shape[0] == n:
            return arr.astype(np.int64)
        if arr.ndim == 2 and 1 in arr.shape and arr.size == n:
            return arr.reshape(-1).astype(np.int64)
    if isinstance(obj, (list, tuple)):
        if len(obj) == n and all(isinstance(v, (int, np.integer, float, np.floating)) for v in obj):
            return np.asarray(obj, dtype=np.int64)
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
    if not hasattr(sw_gp, "gpmodels"):
        return None
    labels = np.full(n, -1, dtype=np.int64)
    try:
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


def _include_batch(sw_gp: Any, x_trains: np.ndarray, data_3d: np.ndarray, with_warp: bool) -> None:
    """Call include_batch robustly across HDP-GPC variants."""
    try:
        sw_gp.include_batch(x_trains, data_3d, with_warp=bool(with_warp))
        return
    except TypeError as first_error:
        try:
            sw_gp.include_batch(x_trains, data_3d, warp=bool(with_warp))
            return
        except TypeError:
            try:
                sw_gp.include_batch(x_trains, data_3d)
                return
            except TypeError:
                raise first_error


def fit_hdpgpc(data_3d: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, Any, np.ndarray, str, float]:
    n = int(data_3d.shape[0])
    if n == 0:
        raise ValueError("No beats to cluster")
    sw_gp, x_train, x_trains = _build_hdpgpc_model(data_3d, args)
    t0 = time.time()
    _include_batch(sw_gp, x_trains, data_3d, with_warp=bool(args.with_warp))
    dt_min = (time.time() - t0) / 60.0
    print(f"[HDPGPC] include_batch finished in {dt_min:.2f} min")
    labels = _extract_labels_from_sw_gp(sw_gp, n)
    if labels is None:
        raise RuntimeError("include_batch(...) finished, but cluster labels could not be reconstructed from the fitted model")
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    uniq = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(uniq, start=1)}
    return np.asarray([remap[int(v)] for v in labels], dtype=np.int64), sw_gp, x_train, "include_batch", dt_min


def plot_hdpgpc_overview(sw_gp: Any, labels: np.ndarray, out_dir: Path, record: str, output_lead_labels: list[str]) -> None:
    try:
        from hdpgpc.util_plots import plot_models_plotly, print_results

        selected_gpmodels = sw_gp.selected_gpmodels() if hasattr(sw_gp, "selected_gpmodels") else sorted(np.unique(labels).tolist())
        y_names = np.asarray(["N"] * len(labels), dtype=str)
        main_model = print_results(sw_gp, y_names, 0, error=False)
        for model_lead, label in enumerate(output_lead_labels):
            plot_models_plotly(
                sw_gp,
                selected_gpmodels,
                main_model,
                y_names,
                0,
                lead=model_lead,
                save=str(out_dir / f"{record}_HDPGPC_clusters_{label}.pdf"),
                step=0.5,
                plot_latent=True,
                y_share=True
            )
    except Exception as e:
        (out_dir / f"{record}_HDPGPC_plot_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[WARN] HDP-GPC plot failed for {record}: {e}", file=sys.stderr)


# -------------------------
# Existing .cluster utility
# -------------------------
def read_existing_cluster_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    samples: list[int] = []
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        samples.append(int(float(parts[0])))
        labels.append(int(float(parts[1])))
    return np.asarray(samples, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def _cluster_reference_lead(lead_map: dict[int, LeadRecord], detection_idx: int, output_indices: list[int]) -> LeadRecord:
    candidates = [detection_idx] + output_indices
    for idx in candidates:
        lr = lead_map.get(idx)
        if lr is not None and lr.cluster_path is not None:
            return lr
    return lead_map[output_indices[0]]


# -------------------------
# Main per-record workflow
# -------------------------
def _base_row(record: str, output_lrs: list[LeadRecord], detection_lr: LeadRecord, headers: dict[int, HeaderInfo], preproc: dict[int, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    output_indices = [lr.lead_index for lr in output_lrs]
    row: dict[str, Any] = {
        "record": record,
        "lead_mode": args.lead_mode,
        "selected_lead": int(args.selected_lead),
        "output_leads": ",".join(f"ECG{i}" for i in output_indices),
        "num_outputs": len(output_indices),
        "detection_lead": f"ECG{detection_lr.lead_index}",
        "fs": headers[detection_lr.lead_index].fs,
        "signal_filter": preproc[detection_lr.lead_index].get("signal_filter", "none"),
        "normalization": args.normalize,
        "beat_symbol": args.beat_symbol,
        "beat_source_requested": ("xqrs" if getattr(args, "use_xqrs", False) else ("auto" if getattr(args, "xqrs_if_missing", False) else args.beat_source)),
        "beat_source_used": "",
        "status": "started",
        "error": "",
        "cluster_file": "",
    }
    for lr in output_lrs:
        idx = lr.lead_index
        row[f"base_ecg{idx}"] = lr.base
        row[f"lead{idx}"] = headers[idx].signal_name
        row[f"signal_samples_ecg{idx}"] = headers[idx].n_samples
        row[f"cluster_file_ecg{idx}"] = ""
        row[f"plot_png_ecg{idx}"] = ""
    for idx, info in preproc.items():
        row[f"signal_filter_ecg{idx}"] = info.get("signal_filter", "none")
        row[f"highpass_cutoff_hz_ecg{idx}"] = info.get("highpass_cutoff_hz", "")
        row[f"highpass_order_ecg{idx}"] = info.get("highpass_order", "")
        row[f"bandpass_low_hz_ecg{idx}"] = info.get("bandpass_low_hz", "")
        row[f"bandpass_high_hz_ecg{idx}"] = info.get("bandpass_high_hz", "")
        row[f"bandpass_order_ecg{idx}"] = info.get("bandpass_order", "")
    return row


def process_one_record(record: str, lead_records: list[LeadRecord], out_root: Path, args: argparse.Namespace, analysis_module: Optional[Any]) -> dict[str, Any]:
    rec_out = out_root / record
    rec_out.mkdir(parents=True, exist_ok=True)

    lead_map = _lead_map_for_record(record, lead_records)
    output_indices, detection_idx = _output_and_detection_indices(args)
    _require_leads(record, lead_map, set(output_indices + [detection_idx]))

    output_lrs = [lead_map[idx] for idx in output_indices]
    detection_lr = lead_map[detection_idx]
    needed_indices = sorted(set(output_indices + [detection_idx]))

    signals: dict[int, np.ndarray] = {}
    headers: dict[int, HeaderInfo] = {}
    preproc: dict[int, dict[str, Any]] = {}
    for idx in needed_indices:
        lr = lead_map[idx]
        raw_signal, header = read_signal(lr.hea_path, lr.dat_path, physical=not args.raw_adc)
        filtered_signal, info = preprocess_signal(raw_signal, header, args)
        signals[idx] = filtered_signal
        headers[idx] = header
        preproc[idx] = info

    fs0 = headers[detection_idx].fs
    for idx in needed_indices:
        if abs(headers[idx].fs - fs0) > 1e-9:
            raise ValueError(f"Record {record}: ECG{idx} fs={headers[idx].fs} differs from detection ECG{detection_idx} fs={fs0}")

    row = _base_row(record, output_lrs, detection_lr, headers, preproc, args)

    annotations = read_iatr_annotations(detection_lr.iatr_path) if detection_lr.iatr_path and detection_lr.iatr_path.exists() else []
    ann_counts = Counter(a.symbol for a in annotations)
    beats_ann = select_beat_annotations(annotations, beat_symbol=args.beat_symbol)
    row[f"annotations_total_ecg{detection_idx}"] = len(annotations)
    row[f"beat_annotations_ecg{detection_idx}"] = len(beats_ann)
    (rec_out / f"annotation_counts_ecg{detection_idx}.json").write_text(json.dumps(dict(ann_counts), indent=2), encoding="utf-8")

    beats_selected, beat_source_used = choose_beat_locations(
        signals[detection_idx],
        headers[detection_idx],
        beats_ann,
        args,
        detector_label=f"ecg{detection_idx}",
    )
    row["beat_source_used"] = beat_source_used
    row["beats_before_windowing"] = len(beats_selected)
    if len(beats_selected) == 0:
        row["status"] = "skipped_no_beats"
        row["error"] = f"No beats from ECG{detection_idx} XQRS/annotations"
        print(f"[SKIP] {record}: {row['error']}")
        return row

    output_signals = [signals[idx] for idx in output_indices]
    data, beat_samples, ext_info = extract_beats_outputs(
        output_signals,
        beats_selected,
        fs=fs0,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        drop_first_beat=args.drop_first_beat,
    )
    row.update(ext_info)
    row["beats_extracted"] = int(data.shape[0])
    row["num_outputs"] = int(data.shape[2]) if data.ndim == 3 else 0
    if data.shape[0] == 0:
        row["status"] = "skipped_no_valid_windows"
        row["error"] = "All beat windows would cross signal boundaries"
        print(f"[SKIP] {record}: {row['error']}")
        return row

    if args.max_beats and data.shape[0] > args.max_beats:
        data = data[:args.max_beats]
        beat_samples = beat_samples[:args.max_beats]
        row["beats_extracted"] = int(data.shape[0])
        row["truncated_to_max_beats"] = int(args.max_beats)

    data_for_model = normalize_beats(data, args.normalize)
    output_tag = "multi_output" if len(output_indices) > 1 else f"ECG{output_indices[0]}"
    np.save(rec_out / f"{record}_beats_{output_tag}.npy", data_for_model)
    np.save(rec_out / f"{record}_beat_samples.npy", beat_samples)

    if args.use_existing_clusters:
        ref_lr = _cluster_reference_lead(lead_map, detection_idx, output_indices)
        if ref_lr.cluster_path is None:
            raise ValueError(f"{record}: --use_existing_clusters requested but no .cluster file is available for detection/output leads")
        existing_samples, existing_labels = read_existing_cluster_file(ref_lr.cluster_path)
        label_by_sample = {int(s): int(l) for s, l in zip(existing_samples, existing_labels)}
        keep = [i for i, s in enumerate(beat_samples) if int(s) in label_by_sample]
        if not keep:
            raise ValueError(f"{record}: existing .cluster labels in {ref_lr.cluster_path.name} do not overlap extracted beat samples")
        beat_samples = beat_samples[keep]
        data_for_model = data_for_model[keep]
        labels = np.asarray([label_by_sample[int(s)] for s in beat_samples], dtype=np.int64)
        sw_gp = None
        row["status"] = f"used_existing_ecg{ref_lr.lead_index}_cluster_labels"
        row["beats_extracted"] = int(data_for_model.shape[0])
    else:
        labels, sw_gp, x_train, method_used, fit_time_min = fit_hdpgpc(data_for_model, args)
        row["status"] = "clustered_hdpgpc_multi_output" if len(output_indices) > 1 else "clustered_hdpgpc_single_lead"
        row["hdpgpc_fit_method_used"] = method_used
        row["hdpgpc_fit_time_min"] = float(fit_time_min)
        np.save(rec_out / f"cluster_labels_{record}_{output_tag}.npy", labels)
        if sw_gp is not None and not args.no_hdpgpc_plot:
            plot_hdpgpc_overview(sw_gp, labels, rec_out, record, output_lead_labels=[f"ECG{idx}" for idx in output_indices])

    cluster_record = out_root / f"{record}.cluster"
    _save_cluster_file(cluster_record, beat_samples, labels, fs=fs0, units=args.cluster_time_units)
    row["cluster_file"] = str(cluster_record)
    for lr in output_lrs:
        p = out_root / f"{lr.base}.cluster"
        _save_cluster_file(p, beat_samples, labels, fs=fs0, units=args.cluster_time_units)
        row[f"cluster_file_ecg{lr.lead_index}"] = str(p)

    row["n_clusters"] = int(np.unique(labels).size)
    row["cluster_sizes"] = json.dumps({int(k): int(v) for k, v in Counter(labels.astype(int)).items()}, sort_keys=True)

    for model_lead, actual_idx in enumerate(output_indices):
        plot_png = rec_out / f"{record}_cluster_overview_ECG{actual_idx}.png"
        title = f"{record}: {'multi-output' if len(output_indices) > 1 else 'single-lead'} clusters, ECG{actual_idx}"
        _plot_cluster_overview_png(data_for_model, labels, plot_png, title=title, model_lead=model_lead, max_traces_per_cluster=args.max_plot_traces_per_cluster)
        row[f"plot_png_ecg{actual_idx}"] = str(plot_png)

    with (rec_out / f"{record}_beats_and_clusters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["beat_index", "sample", "ms", "cluster"])
        w.writeheader()
        for i, (s, lab) in enumerate(zip(beat_samples, labels.astype(int))):
            w.writerow({"beat_index": i, "sample": int(s), "ms": float(s) / fs0 * 1000.0, "cluster": int(lab)})

    mode_text = " + ".join(f"ECG{i}" for i in output_indices)
    print(f"[OK] {record}: {len(labels)} beats, {row['n_clusters']} clusters, outputs={mode_text}, detection=ECG{detection_idx} -> {cluster_record}")
    return row


def run_one_record(record: str, lead_records: list[LeadRecord], out_root: Path, args: argparse.Namespace, analysis_module: Optional[Any]) -> list[dict[str, Any]]:
    output_indices, detection_idx = _output_and_detection_indices(args)
    print(f"\n[RECORD] {record}: lead_mode={args.lead_mode}, outputs={output_indices}, detection=ECG{detection_idx}; found {len(lead_records)} lead file(s)")
    try:
        return [process_one_record(record, lead_records, out_root, args, analysis_module)]
    except Exception as e:
        fail_dir = out_root / record
        fail_dir.mkdir(parents=True, exist_ok=True)
        (fail_dir / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[FAIL] {record}: {e}", file=sys.stderr)
        return [{"record": record, "status": "failed", "error": repr(e), "lead_mode": args.lead_mode}]


# -------------------------
# CLI
# -------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cluster ZIP ECG records with HDP-GPC in multi-output or single-lead mode.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_zip", type=str, help="Path to cluster.zip")
    src.add_argument("--input_dir", type=str, help="Path to extracted cluster/ directory")
    ap.add_argument("--records", nargs="*", default=["n17", "n17c", "t08", "t65", "t66"], help="Record ids to process. Default: n17 n17c t08 t65 t66")
    ap.add_argument("--out_dir", type=str, default="results/cluster_zip_hdpgpc", help="Output directory")
    ap.add_argument("--analysis_script_dir", type=str, default=".", help="Directory containing the hdpgpc package import path; analysis_one_record.py is optional")

    ap.add_argument("--lead_mode", choices=["multi", "single"], default="multi", help="multi=use ECG0+ECG1 jointly; single=use only --selected_lead as one-output data. Default: multi")
    ap.add_argument("--selected_lead", type=int, choices=[0, 1], default=0, help="Lead used as model output when --lead_mode single. Default: 0")
    ap.add_argument("--detection_lead", choices=["0", "1", "selected"], default="0", help="Lead used for QRS/annotation beat locations. Default: 0. In single mode, use 'selected' to detect on the selected lead")

    ap.add_argument("--beat_symbol", type=str, default="N", help="Annotation symbol used as heartbeat location when --beat_source annotations/auto uses the detection lead annotations")
    ap.add_argument("--pre_ms", type=float, default=350.0, help="Milliseconds before detected/annotated beat location to include in beat windows")
    ap.add_argument("--post_ms", type=float, default=350.0, help="Milliseconds after detected/annotated beat location to include in beat windows")
    ap.add_argument("--drop_first_beat", action="store_true", help="Drop the first beat location")
    ap.add_argument("--cluster_time_units", choices=["sample", "ms"], default="sample", help="First column in output .cluster files")
    ap.add_argument("--normalize", choices=["none", "per_beat_center", "per_beat_zscore"], default="none", help="Optional per-beat normalization before clustering. Default: none/off")

    ap.add_argument("--signal_filter", choices=["none", "highpass", "bandpass"], default="highpass", help="Preprocessing for all used signals. Default: highpass baseline removal")
    ap.add_argument("--highpass_cutoff_hz", type=float, default=0.7, help="Cutoff for --signal_filter highpass, in Hz")
    ap.add_argument("--highpass_order", type=int, default=2, help="Butterworth order for --signal_filter highpass")
    ap.add_argument("--bandpass_low_hz", type=float, default=0.5, help="Low cutoff for --signal_filter bandpass, in Hz")
    ap.add_argument("--bandpass_high_hz", type=float, default=45.0, help="High cutoff for --signal_filter bandpass, in Hz; values above Nyquist are clamped")
    ap.add_argument("--bandpass_order", type=int, default=2, help="Butterworth order for --signal_filter bandpass")
    ap.add_argument("--raw_adc", action="store_true", help="Use raw ADC counts instead of physical units")
    ap.add_argument("--use_existing_clusters", action="store_true", help="Use an existing .cluster file instead of HDP-GPC, mainly for extraction validation")
    ap.add_argument("--no_hdpgpc_plot", action="store_true", help="Disable plot_models_plotly output")
    ap.add_argument("--max_plot_traces_per_cluster", type=int, default=1000)
    ap.add_argument("--max_beats", type=int, default=0, help="Debug option: limit beats per record before clustering. 0 means no limit")

    ap.add_argument("--beat_source", choices=["annotations", "auto", "xqrs"], default="xqrs", help="Beat locations from detection lead: xqrs=run XQRS; annotations=.iatr; auto=.iatr if available otherwise XQRS. Default: xqrs")
    ap.add_argument("--use_xqrs", action="store_true", help="Alias for --beat_source xqrs")
    ap.add_argument("--xqrs_if_missing", action="store_true", help="Alias for --beat_source auto")
    ap.add_argument("--xqrs_no_learn", action="store_true", help="Disable XQRS learning initialization")
    ap.add_argument("--xqrs_verbose", action="store_true", help="Print WFDB XQRS detector progress")
    ap.add_argument("--xqrs_sampfrom", type=int, default=0, help="Start sample for XQRS detection")
    ap.add_argument("--xqrs_sampto", type=int, default=0, help="Stop sample for XQRS detection. 0 means end of signal")

    ap.add_argument("--with_warp", action="store_true", help="Pass with_warp=True/warp=True to include_batch. Default false")
    ap.add_argument("--n_explore_steps", type=int, default=5, help="GPI_HDP n_explore_steps")
    ap.add_argument("--free_deg_MNIV", type=int, default=5, help="GPI_HDP free_deg_MNIV")
    ap.add_argument("--max_models", type=int, default=100, help="GPI_HDP max_models")
    ap.add_argument("--sigma_scale", type=float, default=0.04, help="Scale for LDS sigma prior")
    ap.add_argument("--gamma_scale", type=float, default=0.02, help="Scale for LDS gamma prior")
    ap.add_argument("--noise_warp_scale", type=float, default=20.0, help="Scale for warp-noise prior")
    ap.add_argument("--outputscale", type=float, default=300.0, help="Initial GP outputscale")
    ap.add_argument("--ini_lengthscale", type=float, default=3.0, help="Initial GP lengthscale")
    ap.add_argument("--bound_lengthscale_low", type=float, default=1.0)
    ap.add_argument("--bound_lengthscale_high", type=float, default=20.0)
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

        output_indices, detection_idx = _output_and_detection_indices(args)
        print(f"[INFO] input_root: {input_root}")
        print(f"[INFO] out_dir:    {out_root}")
        print(f"[INFO] lead files: {len(lead_records)}")
        print(f"[INFO] records:    {sorted(set(lr.record for lr in lead_records))}")
        print(f"[INFO] lead_mode:  {args.lead_mode}; outputs={output_indices}; detection=ECG{detection_idx}")

        by_record: dict[str, list[LeadRecord]] = {}
        for lr in lead_records:
            by_record.setdefault(lr.record, []).append(lr)

        rows: list[dict[str, Any]] = []
        requested_order = args.records or sorted(by_record)
        for rec in requested_order:
            rec_leads = by_record.get(rec, [])
            if not rec_leads:
                continue
            rows.extend(run_one_record(rec, rec_leads, out_root, args, analysis_module))

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
