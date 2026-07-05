#!/usr/bin/env python3
"""
Plot pedestrian trajectories colored by HDP-GPC cluster labels.

Layout
------
- One full-width overview panel on top with all trajectories and cluster representatives.
- One panel per cluster below, ordered in a grid with shared spatial limits and equal aspect ratio.

Direction cue
-------------
Each trajectory is drawn as small consecutive line segments whose opacity and line width
increase along the path. This creates a subtle fade-in toward the end of the trajectory.
Representatives also show start/end markers.

Optional speed cue
------------------
When the input contains a third channel interpreted as speed, the cluster representative can
be colored by speed along the path while all member trajectories remain in the cluster color.
This keeps the figure readable while still showing where clusters tend to speed up or slow down.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot trajectories colored by HDP-GPC cluster labels.")
    p.add_argument(
        "--input-npz",
        required=True,
        help="Path to pedestrian_hdp_gpc_input.npz",
    )
    p.add_argument(
        "--labels",
        required=True,
        help="Path to hdpgpc_cluster_labels.npy",
    )
    p.add_argument(
        "--metadata",
        default=None,
        help="Optional path to pedestrian_hdp_gpc_metadata.json",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: same folder as labels/input, named hdpgpc_clustered_trajectories_from_labels_speed.png",
    )
    p.add_argument(
        "--use-scaled",
        action="store_true",
        help="Plot the standardized Y array instead of Y_unscaled.",
    )
    p.add_argument(
        "--max-per-cluster",
        type=int,
        default=0,
        help="Optional cap on number of trajectories plotted per cluster. 0 means plot all.",
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns for the per-cluster panels.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when subsampling trajectories per cluster.",
    )
    p.add_argument(
        "--overview-alpha-start",
        type=float,
        default=0.03,
        help="Starting alpha for directional trajectory fading in the overview panel.",
    )
    p.add_argument(
        "--overview-alpha-end",
        type=float,
        default=0.30,
        help="Ending alpha for directional trajectory fading in the overview panel.",
    )
    p.add_argument(
        "--cluster-alpha-start",
        type=float,
        default=0.05,
        help="Starting alpha for directional trajectory fading in per-cluster panels.",
    )
    p.add_argument(
        "--cluster-alpha-end",
        type=float,
        default=0.42,
        help="Ending alpha for directional trajectory fading in per-cluster panels.",
    )
    p.add_argument(
        "--show-representative-speed",
        action="store_true",
        help="Color only the cluster representative by the speed channel (requires D>=3).",
    )
    p.add_argument(
        "--speed-cmap",
        default="viridis",
        help="Matplotlib colormap used for representative speed.",
    )
    p.add_argument(
        "--speed-vmin",
        type=float,
        default=None,
        help="Optional lower bound for the representative-speed color scale.",
    )
    p.add_argument(
        "--speed-vmax",
        type=float,
        default=None,
        help="Optional upper bound for the representative-speed color scale.",
    )
    p.add_argument(
        "--speed-robust-percentiles",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Robust percentiles used for the speed color scale when vmin/vmax are not set.",
    )
    return p.parse_args()


def extract_cluster_labels(assignments: np.ndarray, n_samples: int) -> np.ndarray:
    """Convert different assignment formats into one integer label per sample."""
    arr = np.asarray(assignments)

    if arr.ndim == 0:
        return np.full(n_samples, int(arr), dtype=int)

    if arr.ndim == 1:
        if arr.shape[0] == n_samples:
            return np.rint(arr).astype(int)
        if arr.size % n_samples != 0:
            raise ValueError(
                f"Could not reshape assignments with shape {arr.shape} into {n_samples} samples."
            )
        arr = arr.reshape(n_samples, -1)
    else:
        arr = arr.reshape(n_samples, -1)

    if arr.shape[1] == 1:
        return np.rint(arr[:, 0]).astype(int)
    return np.argmax(arr, axis=1).astype(int)


def load_center_start(metadata_path: Optional[Path]) -> bool:
    if metadata_path is None:
        return False
    if not metadata_path.exists():
        return False
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return bool(meta.get("center_start", False))


def maybe_subsample_indices(indices: np.ndarray, max_per_cluster: int, rng: np.random.Generator) -> np.ndarray:
    if max_per_cluster <= 0 or len(indices) <= max_per_cluster:
        return indices
    chosen = rng.choice(indices, size=max_per_cluster, replace=False)
    return np.sort(chosen)


def compute_global_limits(Y_plot: np.ndarray, pad_fraction: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xy = Y_plot[:, :, :2].reshape(-1, 2)
    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)

    xspan = max(float(xmax - xmin), 1e-6)
    yspan = max(float(ymax - ymin), 1e-6)
    xpad = pad_fraction * xspan
    ypad = pad_fraction * yspan

    return (float(xmin - xpad), float(xmax + xpad)), (float(ymin - ypad), float(ymax + ypad))


def apply_common_axes(ax: plt.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float], xlab: str, ylab: str) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _build_directional_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = np.column_stack([x, y]).astype(float)
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    if pts.shape[0] < 2:
        return np.empty((0, 2, 2), dtype=float)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _build_speed_segments(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.column_stack([x, y, s]).astype(float)
    good = np.isfinite(vals).all(axis=1)
    vals = vals[good]
    if vals.shape[0] < 2:
        return np.empty((0, 2, 2), dtype=float), np.empty((0,), dtype=float)
    segments = np.stack([vals[:-1, :2], vals[1:, :2]], axis=1)
    seg_speed = 0.5 * (vals[:-1, 2] + vals[1:, 2])
    return segments, seg_speed


def plot_directional_path(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color,
    alpha_start: float,
    alpha_end: float,
    lw_start: float,
    lw_end: float,
    zorder: int = 1,
) -> None:
    """Draw a trajectory using segment-wise alpha/linewidth increase to encode direction."""
    segments = _build_directional_segments(x, y)
    if segments.shape[0] == 0:
        return

    base_rgba = np.array(mcolors.to_rgba(color), dtype=float)
    nseg = segments.shape[0]
    alphas = np.linspace(alpha_start, alpha_end, nseg)
    widths = np.linspace(lw_start, lw_end, nseg)
    colors = np.tile(base_rgba, (nseg, 1))
    colors[:, 3] = alphas

    lc = LineCollection(segments, colors=colors, linewidths=widths, zorder=zorder)
    lc.set_capstyle("round")
    ax.add_collection(lc)


def plot_representative_path(
    ax: plt.Axes,
    representative: np.ndarray,
    color,
    linewidth: float = 2.8,
    marker_size: float = 18.0,
    zorder: int = 5,
) -> None:
    x = representative[:, 0]
    y = representative[:, 1]
    ax.plot(x, y, color=color, linewidth=linewidth, zorder=zorder)

    if len(x) >= 1:
        ax.scatter(
            [x[0]],
            [y[0]],
            s=marker_size,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=zorder + 1,
        )
        ax.scatter(
            [x[-1]],
            [y[-1]],
            s=marker_size,
            facecolors=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=zorder + 1,
        )


def plot_representative_speed_path(
    ax: plt.Axes,
    representative_xy: np.ndarray,
    representative_speed: np.ndarray,
    cluster_color,
    speed_norm: mcolors.Normalize,
    speed_cmap: str,
    linewidth: float = 3.0,
    marker_size: float = 18.0,
    zorder: int = 6,
) -> Optional[LineCollection]:
    x = representative_xy[:, 0]
    y = representative_xy[:, 1]

    # Thin cluster-colored underlay preserves cluster identity while the top line shows speed.
    ax.plot(x, y, color=cluster_color, linewidth=linewidth + 1.2, alpha=0.35, zorder=zorder - 1)

    segments, seg_speed = _build_speed_segments(x, y, representative_speed)
    lc: Optional[LineCollection] = None
    if segments.shape[0] > 0:
        lc = LineCollection(
            segments,
            cmap=plt.get_cmap(speed_cmap),
            norm=speed_norm,
            linewidths=linewidth,
            zorder=zorder,
        )
        lc.set_array(seg_speed)
        lc.set_capstyle("round")
        ax.add_collection(lc)

    if len(x) >= 1:
        ax.scatter(
            [x[0]],
            [y[0]],
            s=marker_size,
            facecolors="white",
            edgecolors=cluster_color,
            linewidths=1.2,
            zorder=zorder + 1,
        )
        ax.scatter(
            [x[-1]],
            [y[-1]],
            s=marker_size,
            facecolors=cluster_color,
            edgecolors="white",
            linewidths=0.5,
            zorder=zorder + 1,
        )
    return lc


def compute_speed_norm(
    Y_plot: np.ndarray,
    speed_vmin: Optional[float],
    speed_vmax: Optional[float],
    speed_robust_percentiles: Tuple[float, float],
) -> mcolors.Normalize:
    if Y_plot.shape[2] < 3:
        raise ValueError("Representative speed display requires an input with at least 3 channels: (x, y, speed).")

    speed = np.asarray(Y_plot[:, :, 2], dtype=float)
    speed = speed[np.isfinite(speed)]
    if speed.size == 0:
        raise ValueError("No finite speed values were found in channel 2.")

    if speed_vmin is None or speed_vmax is None:
        lo_pct, hi_pct = speed_robust_percentiles
        if not (0.0 <= lo_pct < hi_pct <= 100.0):
            raise ValueError("Speed robust percentiles must satisfy 0 <= LOW < HIGH <= 100.")
        lo, hi = np.percentile(speed, [lo_pct, hi_pct])
    else:
        lo, hi = np.nanmin(speed), np.nanmax(speed)

    vmin = lo if speed_vmin is None else float(speed_vmin)
    vmax = hi if speed_vmax is None else float(speed_vmax)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Speed color scale bounds must be finite.")
    if vmax <= vmin:
        span = 1.0 if vmin == 0.0 else abs(vmin) * 0.05 + 1e-6
        vmax = vmin + span

    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def save_cluster_trajectory_plot(
    Y_plot: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
    center_start: bool,
    max_per_cluster: int = 0,
    ncols: int = 3,
    seed: int = 0,
    overview_alpha_start: float = 0.03,
    overview_alpha_end: float = 0.30,
    cluster_alpha_start: float = 0.05,
    cluster_alpha_end: float = 0.42,
    show_representative_speed: bool = False,
    speed_cmap: str = "viridis",
    speed_vmin: Optional[float] = None,
    speed_vmax: Optional[float] = None,
    speed_robust_percentiles: Tuple[float, float] = (2.0, 98.0),
    speed_colorbar_label: str = "speed [m/s]",
) -> Path:
    Y_plot = np.asarray(Y_plot)
    labels = np.asarray(cluster_labels).astype(int).reshape(-1)

    if Y_plot.ndim != 3 or Y_plot.shape[2] < 2:
        raise ValueError(f"Expected trajectory array with shape (N, L, D>=2), got {Y_plot.shape}.")
    if Y_plot.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Number of trajectories and labels must match, got {Y_plot.shape[0]} trajectories and {labels.shape[0]} labels."
        )

    if show_representative_speed and Y_plot.shape[2] < 3:
        print("[WARN] --show-representative-speed requested, but the input has only 2 channels. Falling back to plain representatives.")
        show_representative_speed = False

    speed_norm = None
    if show_representative_speed:
        speed_norm = compute_speed_norm(Y_plot, speed_vmin, speed_vmax, speed_robust_percentiles)

    clusters = np.unique(labels)
    ncols = max(1, int(ncols))
    n_cluster_rows = max(1, int(math.ceil(len(clusters) / ncols)))

    fig = plt.figure(figsize=(6.0 * ncols, 4.8 + 4.8 * n_cluster_rows))
    gs = GridSpec(
        nrows=1 + n_cluster_rows,
        ncols=ncols,
        figure=fig,
        height_ratios=[1.25] + [1.0] * n_cluster_rows,
    )

    cmap = plt.get_cmap("tab20", max(len(clusters), 1))
    color_map = {c: cmap(i) for i, c in enumerate(clusters)}
    rng = np.random.default_rng(seed)

    xlab = "x displacement [m]" if center_start else "x [m]"
    ylab = "y displacement [m]" if center_start else "y [m]"
    xlim, ylim = compute_global_limits(Y_plot)

    ax_all = fig.add_subplot(gs[0, :])
    legend_handles = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        idx_plot = maybe_subsample_indices(idx, max_per_cluster, rng)
        color = color_map[c]
        for i in idx_plot:
            plot_directional_path(
                ax_all,
                Y_plot[i, :, 0],
                Y_plot[i, :, 1],
                color=color,
                alpha_start=overview_alpha_start,
                alpha_end=overview_alpha_end,
                lw_start=0.45,
                lw_end=1.15,
                zorder=1,
            )

        representative_xy = Y_plot[idx, :, :2].mean(axis=0)
        if show_representative_speed:
            representative_speed = Y_plot[idx, :, 2].mean(axis=0)
            plot_representative_speed_path(
                ax_all,
                representative_xy,
                representative_speed,
                cluster_color=color,
                speed_norm=speed_norm,
                speed_cmap=speed_cmap,
                linewidth=2.8,
                marker_size=20.0,
                zorder=6,
            )
        else:
            plot_representative_path(
                ax_all,
                representative_xy,
                color=color,
                linewidth=2.8,
                marker_size=20.0,
                zorder=6,
            )

        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.8, label=f"Cluster {int(c)} (n={len(idx)})")
        )

    title = "All trajectories and cluster representatives\n(direction fades in toward the end of each path)"
    if show_representative_speed:
        title += "\n(representative line color encodes mean speed along the path)"
    ax_all.set_title(title)
    apply_common_axes(ax_all, xlim, ylim, xlab, ylab)
    if len(clusters) <= 20:
        ax_all.legend(handles=legend_handles, loc="best", fontsize=8, ncol=min(4, max(1, len(clusters))))

    for k, c in enumerate(clusters):
        row = 1 + (k // ncols)
        col = k % ncols
        ax = fig.add_subplot(gs[row, col])

        idx = np.where(labels == c)[0]
        idx_plot = maybe_subsample_indices(idx, max_per_cluster, rng)
        color = color_map[c]
        for i in idx_plot:
            plot_directional_path(
                ax,
                Y_plot[i, :, 0],
                Y_plot[i, :, 1],
                color=color,
                alpha_start=cluster_alpha_start,
                alpha_end=cluster_alpha_end,
                lw_start=0.55,
                lw_end=1.35,
                zorder=1,
            )

        representative_xy = Y_plot[idx, :, :2].mean(axis=0)
        if show_representative_speed:
            representative_speed = Y_plot[idx, :, 2].mean(axis=0)
            plot_representative_speed_path(
                ax,
                representative_xy,
                representative_speed,
                cluster_color=color,
                speed_norm=speed_norm,
                speed_cmap=speed_cmap,
                linewidth=2.8,
                marker_size=18.0,
                zorder=5,
            )
        else:
            plot_representative_path(
                ax,
                representative_xy,
                color="black",
                linewidth=2.6,
                marker_size=18.0,
                zorder=5,
            )

        title = f"Cluster {int(c)} (n={len(idx)})"
        if max_per_cluster > 0 and len(idx) > len(idx_plot):
            title += f"\nshowing {len(idx_plot)}"
        ax.set_title(title)
        apply_common_axes(ax, xlim, ylim, xlab, ylab)

    total_slots = n_cluster_rows * ncols
    for k in range(len(clusters), total_slots):
        row = 1 + (k // ncols)
        col = k % ncols
        ax_empty = fig.add_subplot(gs[row, col])
        ax_empty.axis("off")

    if show_representative_speed and speed_norm is not None:
        sm = ScalarMappable(norm=speed_norm, cmap=plt.get_cmap(speed_cmap))
        sm.set_array([])
        cax = inset_axes(
            ax_all,
            width="2.6%",
            height="72%",
            loc="center left",
            bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
            bbox_transform=ax_all.transAxes,
            borderpad=0.0,
        )
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(speed_colorbar_label)

    fig.tight_layout(rect=[0.0, 0.0, 0.96, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(output_path)+".pdf")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    input_npz = Path(args.input_npz)
    labels_path = Path(args.labels)
    metadata_path = Path(args.metadata) if args.metadata is not None else None

    data = np.load(input_npz)
    if args.use_scaled:
        if "Y" not in data:
            raise KeyError(f"Array 'Y' not found in {input_npz}.")
        Y_plot = data["Y"]
        speed_colorbar_label = "standardized speed" if args.show_representative_speed else "speed"
        if args.show_representative_speed:
            print("[WARN] Plotting representative speed from scaled 'Y'. The colorbar will be in standardized units, not m/s.")
    else:
        if "Y_unscaled" in data:
            Y_plot = data["Y_unscaled"]
        elif "Y" in data:
            Y_plot = data["Y"]
            print("[WARN] 'Y_unscaled' not found; plotting scaled 'Y' instead.")
        else:
            raise KeyError(f"Neither 'Y_unscaled' nor 'Y' found in {input_npz}.")
        speed_colorbar_label = "speed [m/s]"

    raw_labels = np.load(labels_path)
    cluster_labels = extract_cluster_labels(raw_labels, n_samples=Y_plot.shape[0])
    center_start = load_center_start(metadata_path)

    if args.output is None:
        base_dir = labels_path.parent if labels_path.parent != Path("") else input_npz.parent
        output_path = base_dir / "hdpgpc_clustered_trajectories_from_labels_direction_speed.png"
    else:
        output_path = Path(args.output)

    saved = save_cluster_trajectory_plot(
        Y_plot=Y_plot,
        cluster_labels=cluster_labels,
        output_path=output_path,
        center_start=center_start,
        max_per_cluster=args.max_per_cluster,
        ncols=args.ncols,
        seed=args.seed,
        overview_alpha_start=args.overview_alpha_start,
        overview_alpha_end=args.overview_alpha_end,
        cluster_alpha_start=args.cluster_alpha_start,
        cluster_alpha_end=args.cluster_alpha_end,
        show_representative_speed=args.show_representative_speed,
        speed_cmap=args.speed_cmap,
        speed_vmin=args.speed_vmin,
        speed_vmax=args.speed_vmax,
        speed_robust_percentiles=tuple(args.speed_robust_percentiles),
        speed_colorbar_label=speed_colorbar_label,
    )

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Loaded {Y_plot.shape[0]} trajectories.")
    print("Cluster sizes:")
    for c, n in zip(unique_labels, counts):
        print(f"  Cluster {int(c)}: {int(n)}")
    print(f"Saved plot to: {saved}")


if __name__ == "__main__":
    main()
