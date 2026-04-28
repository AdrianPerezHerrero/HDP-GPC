# -*- coding: utf-8 -*-
"""
Replot runtime-analysis results with a style closer to the attached clinical-paper figures:
- light grey figure/axes background
- boxed axes, no grid
- muted pastel fills with thin darker outlines
- frameless legends
- compact typography and wider margins
- 3-line x tick labels for grouped comparison plots
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


INT_FIELDS = {
    "num_samples",
    "num_obs_per_sample",
    "num_outputs",
    "num_input_leads",
    "num_classes_true",
    "num_clusters_pred",
}
FLOAT_FIELDS = {
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
}

GROUP_ORDER = [
    ("online",  "single-output", "off"),
    ("offline", "single-output", "off"),
    ("offline", "multi-output", "off"),
    ("offline", "single-output", "on"),
    ("offline", "multi-output", "on"),
]
GROUP_LABELS = {
    ("online",  "single-output", "off"): "Online, SO, warp off",
    ("offline", "single-output", "off"): "Offline, SO, warp off",
    ("offline", "multi-output", "off"): "Offline, MO, warp off",
    ("offline", "single-output", "on"): "Offline, SO, warp on",
    ("offline", "multi-output", "on"): "Offline, MO, warp on",
}
GROUP_TICK_LABELS = {
    ("online",  "single-output", "off"): "Online\nSO\nwarp off",
    ("offline", "single-output", "off"): "Offline\nSO\nwarp off",
    ("offline", "multi-output", "off"): "Offline\nMO\nwarp off",
    ("offline", "single-output", "on"): "Offline\nSO\nwarp on",
    ("offline", "multi-output", "on"): "Offline\nMO\nwarp on",
}

# Muted, clinical-paper-like palette
GROUP_COLORS = {
    ("online",  "single-output", "off"): "#6B5FB5",   # muted violet
    ("offline", "single-output", "off"): "#4C92B8",   # muted blue
    ("offline", "multi-output", "off"): "#7BAF9E",    # muted sage/teal
    ("offline", "single-output", "on"): "#C96F5A",    # muted terracotta
    ("offline", "multi-output", "on"): "#E7B78C",     # pale sand
}
GROUP_MARKERS = {
    ("online",  "single-output", "off"): "o",
    ("offline", "single-output", "off"): "s",
    ("offline", "multi-output", "off"): "^",
    ("offline", "single-output", "on"): "X",
    ("offline", "multi-output", "on"): "P",
}

BG = "#FFFFFF"
AX_BG = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clinical-style runtime plots.")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fit", choices=["linear", "none"], default="linear")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--legend-outside", action="store_true")
    return parser.parse_args()


def _parse_int(value) -> int:
    if value is None:
        return 0
    txt = str(value).strip()
    if txt == "":
        return 0
    return int(float(txt))


def _parse_float(value) -> float:
    if value is None:
        return float("nan")
    txt = str(value).strip()
    if txt == "":
        return float("nan")
    low = txt.lower()
    if low in {"nan", "none"}:
        return float("nan")
    return float(txt)


def load_rows(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_row = dict(row)
            new_row.setdefault("method", "offline")
            new_row.setdefault("output_mode", "single-output")
            if "num_input_leads" not in new_row or str(new_row.get("num_input_leads", "")).strip() == "":
                fallback = new_row.get("num_outputs", "1")
                new_row["num_input_leads"] = fallback
            for key in list(new_row.keys()):
                if key in INT_FIELDS:
                    try:
                        new_row[key] = _parse_int(new_row[key])
                    except Exception:
                        new_row[key] = 0
                elif key in FLOAT_FIELDS:
                    try:
                        new_row[key] = _parse_float(new_row[key])
                    except Exception:
                        new_row[key] = float("nan")
            rows.append(new_row)
    return rows


def run_key(row: Dict) -> Tuple[str, str, str, str]:
    return (
        str(row.get("method", "offline")),
        str(row.get("record", "")),
        str(row.get("output_mode", "single-output")),
        str(row.get("warp", "off")),
    )


def deduplicate_latest(rows: List[Dict]) -> List[Dict]:
    latest: Dict[Tuple[str, str, str, str], Dict] = {}
    for row in rows:
        latest[run_key(row)] = row
    return sorted(
        latest.values(),
        key=lambda r: (
            str(r.get("method", "offline")),
            str(r["record"]),
            str(r.get("output_mode", "single-output")),
            str(r["warp"]),
        ),
    )


def resolve_input_csv(results_dir: Optional[Path], csv_path: Optional[Path]) -> Path:
    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return csv_path
    if results_dir is None:
        raise ValueError("Please provide --results-dir or --csv.")
    candidates = [
        results_dir / "runtime_records_v1.csv",
        results_dir / "runtime_records_log_nowarp.csv",
        results_dir / "runtime_records_log_interpolated_table_for_plots_with_nowarp.csv",
        results_dir / "runtime_records_log_real_plus_interpolated_table_with_nowarp.csv",
        results_dir / "runtime_records_log_interpolated_table_for_plots.csv",
        results_dir / "runtime_records_log_real_plus_interpolated_table.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a supported runtime CSV in "
        f"{results_dir}. Checked: {[p.name for p in candidates]}"
    )


def resolve_output_dir(results_dir: Optional[Path], input_csv: Path, out_dir: Optional[Path]) -> Path:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    out = (results_dir / "plots_clinical_style") if results_dir is not None else (input_csv.parent / "plots_clinical_style")
    out.mkdir(parents=True, exist_ok=True)
    return out


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 19,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.facecolor": AX_BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.15,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": False,
        "axes.grid": False,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
    })


def to_group(row: Dict) -> Tuple[str, str, str]:
    return (
        str(row.get("method", "offline")),
        str(row.get("output_mode", "single-output")),
        str(row.get("warp", "off")),
    )


def compact_output_mode(output_mode: str) -> str:
    if output_mode == "single-output":
        return "SO"
    if output_mode == "multi-output":
        return "MO"
    return output_mode


def warp_text(warp: str) -> str:
    return "warp off" if warp == "off" else "warp on"


def group_label(group: Tuple[str, str, str]) -> str:
    return GROUP_LABELS.get(
        group,
        f"{group[0].title()}, {compact_output_mode(group[1])}, {warp_text(group[2])}",
    )


def group_tick_label(group: Tuple[str, str, str]) -> str:
    return GROUP_TICK_LABELS.get(
        group,
        f"{group[0].title()}\n{compact_output_mode(group[1])}\n{warp_text(group[2])}",
    )


def group_color(group: Tuple[str, str, str]) -> str:
    default_colors = ["#4C92B8", "#7BAF9E", "#6B5FB5", "#C96F5A", "#E7B78C", "#909090"]
    if group in GROUP_COLORS:
        return GROUP_COLORS[group]
    return default_colors[abs(hash(group)) % len(default_colors)]


def group_marker(group: Tuple[str, str, str]) -> str:
    default_markers = ["o", "s", "^", "D", "P", "X"]
    if group in GROUP_MARKERS:
        return GROUP_MARKERS[group]
    return default_markers[abs(hash(group)) % len(default_markers)]


def present_groups(rows: List[Dict]) -> List[Tuple[str, str, str]]:
    available = {to_group(r) for r in rows}
    ordered = [g for g in GROUP_ORDER if g in available]
    extras = sorted(available - set(GROUP_ORDER))
    return ordered + extras


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def save_group_summary(rows: List[Dict], out_path: Path) -> None:
    summary = []
    for group in present_groups(rows):
        subset = [r for r in rows if to_group(r) == group]
        runtime = finite_array(r["total_runtime_sec"] for r in subset)
        cpu_peak = finite_array(r["cpu_peak_rss_mb"] for r in subset)
        cpu_delta = finite_array(r["cpu_peak_delta_mb"] for r in subset)
        gpu_peak = finite_array(r["gpu_peak_allocated_mb"] for r in subset)
        summary.append({
            "method": group[0],
            "output_mode": group[1],
            "warp": group[2],
            "label": group_label(group),
            "n_runs": len(subset),
            "mean_runtime_sec": float(np.mean(runtime)) if runtime.size else float("nan"),
            "std_runtime_sec": float(np.std(runtime)) if runtime.size else float("nan"),
            "mean_cpu_peak_rss_mb": float(np.mean(cpu_peak)) if cpu_peak.size else float("nan"),
            "std_cpu_peak_rss_mb": float(np.std(cpu_peak)) if cpu_peak.size else float("nan"),
            "mean_cpu_peak_delta_mb": float(np.mean(cpu_delta)) if cpu_delta.size else float("nan"),
            "std_cpu_peak_delta_mb": float(np.std(cpu_delta)) if cpu_delta.size else float("nan"),
            "mean_gpu_peak_allocated_mb": float(np.mean(gpu_peak)) if gpu_peak.size else float("nan"),
            "std_gpu_peak_allocated_mb": float(np.std(gpu_peak)) if gpu_peak.size else float("nan"),
        })
    fieldnames = list(summary[0].keys()) if summary else [
        "method","output_mode","warp","label","n_runs","mean_runtime_sec","std_runtime_sec",
        "mean_cpu_peak_rss_mb","std_cpu_peak_rss_mb","mean_cpu_peak_delta_mb","std_cpu_peak_delta_mb",
        "mean_gpu_peak_allocated_mb","std_gpu_peak_allocated_mb"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if summary:
            writer.writerows(summary)


def linfit_with_band(x: np.ndarray, y: np.ndarray, grid: np.ndarray, n_bootstrap: int, rng: np.random.Generator):
    coeffs = np.polyfit(x, y, deg=1)
    mean_line = np.polyval(coeffs, grid)
    if x.size < 3 or n_bootstrap <= 1:
        return mean_line, np.full_like(grid, np.nan), np.full_like(grid, np.nan)

    preds = []
    n = x.size
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        if np.unique(xb).size < 2:
            continue
        try:
            cb = np.polyfit(xb, yb, deg=1)
            preds.append(np.polyval(cb, grid))
        except Exception:
            continue
    if not preds:
        return mean_line, np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    boot = np.vstack(preds)
    return mean_line, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def format_ax(ax):
    ax.grid(False)
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
    ax.tick_params(direction="out", length=5, width=1.1)


def scatter_runtime(rows: List[Dict], x_key: str, x_label: str, y_key: str, y_label: str, title: str,
                    out_dir: Path, base_name: str, formats: Sequence[str], dpi: int,
                    fit_mode: str, bootstrap: int, seed: int, legend_outside: bool) -> None:
    groups = present_groups(rows)
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    rng = np.random.default_rng(seed)
    format_ax(ax)

    for group in groups:
        subset = [r for r in rows if to_group(r) == group]
        x = np.asarray([r[x_key] for r in subset], dtype=float)
        y = np.asarray([r[y_key] for r in subset], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue

        color = group_color(group)
        marker = group_marker(group)
        label = group_label(group)

        ax.scatter(
            x, y, s=58, marker=marker, facecolor=color, edgecolor="white",
            linewidth=0.65, alpha=0.88, label=label, zorder=3
        )

        if fit_mode == "linear" and x.size >= 2 and np.unique(x).size >= 2:
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            y_line, y_lo, y_hi = linfit_with_band(x, y, grid, bootstrap, rng)
            ax.plot(grid, y_line, color=color, linewidth=2.2, zorder=4)
            if np.all(np.isfinite(y_lo)) and np.all(np.isfinite(y_hi)):
                ax.fill_between(grid, y_lo, y_hi, color=color, alpha=0.12, zorder=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(title, pad=10)
    ax.margins(x=0.03, y=0.08)

    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    else:
        ax.legend(loc="best", frameon=False)

    for ext in formats:
        fig.savefig(out_dir / f"{base_name}.{ext}", dpi=dpi if ext == "png" else None)
    plt.close(fig)


def draw_distribution_plot(rows: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int, seed: int) -> None:
    groups = present_groups(rows)
    labels = [group_tick_label(g) for g in groups]
    data = [finite_array(r["total_runtime_sec"] for r in rows if to_group(r) == g) for g in groups]
    positions = np.arange(1, len(groups) + 1)
    fig, ax = plt.subplots(figsize=(max(9.2, 1.65 * len(groups)), 4.8))
    rng = np.random.default_rng(seed)
    format_ax(ax)

    vp = ax.violinplot(data, positions=positions, widths=0.78, showmeans=False, showextrema=False, showmedians=False)
    for body, group in zip(vp["bodies"], groups):
        color = group_color(group)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.16)
        body.set_linewidth(0.8)

    bp = ax.boxplot(
        data, positions=positions, widths=0.34, patch_artist=True, showmeans=True,
        medianprops={"linewidth": 1.45, "color": "black"},
        boxprops={"linewidth": 1.25},
        whiskerprops={"linewidth": 1.1, "color": "black"},
        capprops={"linewidth": 1.1, "color": "black"},
        meanprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "none", "markersize": 6.0},
    )
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor((1, 1, 1, 0.78))
        patch.set_edgecolor(group_color(group))

    for i, (vals, group) in enumerate(zip(data, groups), start=1):
        if vals.size == 0:
            continue
        xj = rng.normal(i, 0.040, size=vals.size)
        ax.scatter(
            xj, vals, s=36, marker="o", facecolor=group_color(group), edgecolor="white",
            linewidth=0.45, alpha=0.38, zorder=2
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center", linespacing=1.02, fontsize=14)
    ax.set_ylabel("Runtime (s)")
    #ax.set_title("Runtime distribution", pad=10)

    for ext in formats:
        fig.savefig(out_dir / f"runtime_distribution_clinical.{ext}", dpi=dpi if ext == "png" else None)
    plt.close(fig)


def draw_memory_plot(rows: List[Dict], out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    groups = present_groups(rows)
    labels = [group_tick_label(g) for g in groups]

    # Increase horizontal separation between categories
    spacing = 1.35
    x = np.arange(len(groups), dtype=float) * spacing
    bar_width = 0.95

    cpu_peak_mean, cpu_peak_std = [], []
    cpu_delta_mean, cpu_delta_std = [], []
    gpu_mean, gpu_std = [], []

    for group in groups:
        subset = [r for r in rows if to_group(r) == group]
        peak = finite_array(r["cpu_peak_rss_mb"] for r in subset)
        delta = finite_array(r["cpu_peak_delta_mb"] for r in subset)
        gpu = finite_array(r["gpu_peak_allocated_mb"] for r in subset)
        cpu_peak_mean.append(np.mean(peak) if peak.size else np.nan)
        cpu_peak_std.append(np.std(peak) if peak.size else np.nan)
        cpu_delta_mean.append(np.mean(delta) if delta.size else np.nan)
        cpu_delta_std.append(np.std(delta) if delta.size else np.nan)
        gpu_mean.append(np.mean(gpu) if gpu.size else np.nan)
        gpu_std.append(np.std(gpu) if gpu.size else np.nan)

    gpu_has_signal = (
        np.any(np.isfinite(np.asarray(gpu_mean, dtype=float)))
        and np.nanmax(np.asarray(gpu_mean, dtype=float)) > 0
    )
    ncols = 3 if gpu_has_signal else 2

    # Slightly wider figure
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(6.2 * ncols, 2.35 * len(groups) * ncols / 2), 4.8),
        sharex=False,
    )
    axes = np.atleast_1d(axes)

    def _draw_bar(ax, means, stds, title, ylabel):
        format_ax(ax)
        colors = [group_color(g) for g in groups]
        ax.bar(
            x,
            np.asarray(means, dtype=float),
            width=bar_width,
            yerr=np.asarray(stds, dtype=float),
            capsize=3.0,
            color=colors,
            edgecolor="white",
            linewidth=0.75,
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=0,
            ha="center",
            linespacing=1.08,
            fontsize=13.5,
        )
        ax.tick_params(axis="x", pad=12)
        ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
        #ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel)

    _draw_bar(axes[0], cpu_peak_mean, cpu_peak_std, "Peak CPU RSS", "Memory (MB)")
    _draw_bar(axes[1], cpu_delta_mean, cpu_delta_std, "CPU RSS increase", "Memory (MB)")
    if gpu_has_signal:
        _draw_bar(axes[2], gpu_mean, gpu_std, "Peak GPU allocated", "Memory (MB)")

    for ext in formats:
        fig.savefig(out_dir / f"memory_summary_clinical.{ext}", dpi=dpi if ext == "png" else None)
    plt.close(fig)




def report_present_groups(rows: List[Dict]) -> None:
    groups = present_groups(rows)
    if not groups:
        print("[WARN] No method/output/warp groups found in the input CSV.")
        return
    print("[INFO] Groups present in plots:")
    for g in groups:
        print(f"  - {group_label(g)}")

def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else None
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None

    input_csv = resolve_input_csv(results_dir, csv_path)
    out_dir = resolve_output_dir(results_dir, input_csv, Path(args.out_dir).expanduser().resolve() if args.out_dir else None)
    apply_style()

    rows = load_rows(input_csv)
    if "log" in input_csv.name:
        rows = deduplicate_latest(rows)
    if not rows:
        raise RuntimeError(f"No rows found in {input_csv}")

    save_group_summary(rows, out_dir / "grouped_summary_for_plots.csv")
    report_present_groups(rows)

    scatter_runtime(
        rows, "num_samples", "N (number of samples)", "total_runtime_sec", "Runtime (s)", "Runtime vs N",
        out_dir, "runtime_vs_N_clinical", args.formats, args.dpi, args.fit, args.bootstrap, args.seed, args.legend_outside
    )
    scatter_runtime(
        rows, "num_clusters_pred", "K (predicted number of clusters)", "total_runtime_sec", "Runtime (s)", "Runtime vs K",
        out_dir, "runtime_vs_K_clinical", args.formats, args.dpi, args.fit, args.bootstrap, args.seed + 1, args.legend_outside
    )
    scatter_runtime(
        rows, "num_samples", "N (number of samples)", "cpu_peak_rss_mb", "Peak CPU RSS (MB)", "Memory vs N",
        out_dir, "memory_vs_N_clinical", args.formats, args.dpi, args.fit, args.bootstrap, args.seed + 2, args.legend_outside
    )
    scatter_runtime(
        rows, "num_clusters_pred", "K (predicted number of clusters)", "cpu_peak_rss_mb", "Peak CPU RSS (MB)", "Memory vs K",
        out_dir, "memory_vs_K_clinical", args.formats, args.dpi, args.fit, args.bootstrap, args.seed + 3, args.legend_outside
    )
    draw_distribution_plot(rows, out_dir, args.formats, args.dpi, args.seed + 4)
    draw_memory_plot(rows, out_dir, args.formats, args.dpi)

    print(f"[OK] Loaded runtime rows from: {input_csv}")
    print(f"[OK] Clinical-style plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
