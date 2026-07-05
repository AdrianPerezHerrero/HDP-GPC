#!/usr/bin/env python3
"""
Download or stream a small, curated subset of the train-station pedestrian
trajectory dataset, preprocess it into HDP-GPC-friendly tensors, and optionally
fit a user-provided HDP-GPC class.

Dataset paper:
  https://doi.org/10.1038/s41597-024-04071-9
Dataset DOI:
  https://doi.org/10.6084/m9.figshare.27058384
Code DOI:
  https://doi.org/10.5281/zenodo.13784462

Why this script exists
----------------------
The published dataset is large, and the files are arranged per camera and day.
This script lets you work with a small curated subset without manually unpacking
and browsing the full archive first.

What it does
------------
1. Reads the dataset either from:
   - the default Figshare remote ZIP URL, or
   - a local ZIP already downloaded by the user.
2. Selects a few camera/day CSV files.
3. Splits each CSV into trajectories by pedestrian ID.
4. Filters obviously low-quality tracks.
5. Resamples each trajectory to a common grid on s in [0, 1], optionally using a cosine grid that emphasizes the start and end.
6. Saves:
   - raw curated CSV files,
   - preprocessed NumPy arrays (.npz),
   - metadata (.json).
7. Optionally imports a user HDP-GPC class and calls model.fit(...).

Typical usage
-------------
# 1) Preprocess only (remote streaming from Figshare ZIP)
python hdp_gpc_pedestrian_subset.py \
    --output-dir out_pedestrian \
    --camera-prefix HB-CAM2 \
    --max-files 2 \
    --max-tracks 200

# 2) Use a local ZIP already downloaded from Figshare
python hdp_gpc_pedestrian_subset.py \
    --source /path/to/Data2.zip \
    --output-dir out_pedestrian \
    --camera-prefix PB-CAM6

# 3) Preprocess and then try to fit your own HDP-GPC class
python hdp_gpc_pedestrian_subset.py \
    --output-dir out_pedestrian \
    --hdpgpc-module your_package.models.hdpgpc \
    --hdpgpc-class HDPGPC \
    --hdpgpc-config config.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from hashlib import sha1
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

DEFAULT_REMOTE_ZIP_URL = "https://springernature.figshare.com/ndownloader/files/49279738"
CSV_COLUMNS = ["id", "t_rel_s", "date", "x_m", "y_m"]
CSV_NAME_RE = re.compile(
    r"(?P<camera>(?:HB|PB|PC)-CAM\d+)_(?P<date>\d{4}-\d{2}-\d{2})_trans\.csv$",
    flags=re.IGNORECASE,
)


@dataclass
class TrackStats:
    source_file: str
    camera: str
    acquisition_date: str
    track_id: int
    n_points: int
    duration_s: float
    max_gap_s: float
    path_length_m: float
    median_speed_mps: float
    mean_speed_mps: float
    start_x_m: float
    start_y_m: float
    end_x_m: float
    end_y_m: float


@dataclass
class StandardizationStats:
    mean: List[float]
    std: List[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curate a small subset of the train-station pedestrian dataset for HDP-GPC."
    )
    p.add_argument(
        "--source",
        type=str,
        default=DEFAULT_REMOTE_ZIP_URL,
        help=(
            "Either a local ZIP file path or the remote Figshare ZIP URL. "
            f"Default: {DEFAULT_REMOTE_ZIP_URL}"
        ),
    )
    p.add_argument("--output-dir", type=str, default="out_pedestrian_hdp_gpc")
    p.add_argument(
        "--camera-prefix",
        type=str,
        default="HB-CAM2",
        help="Camera to curate, for example HB-CAM2, HB-CAM5, PB-CAM6.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=2,
        help="How many daily CSV files to use from the chosen camera.",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=200,
        help="Maximum number of accepted trajectories to keep in the curated subset.",
    )
    p.add_argument(
        "--tracks-per-file",
        type=int,
        default=120,
        help="Upper bound on accepted trajectories per CSV file.",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Common resampling grid length for all trajectories.",
    )
    p.add_argument(
        "--date-start",
        type=str,
        default=None,
        help="Optional start date filter YYYY-MM-DD.",
    )
    p.add_argument(
        "--date-end",
        type=str,
        default=None,
        help="Optional end date filter YYYY-MM-DD.",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=8,
        help="Minimum number of distinct time points per trajectory.",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Minimum trajectory duration in seconds.",
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Maximum trajectory duration in seconds.",
    )
    p.add_argument(
        "--min-path-length",
        type=float,
        default=1.0,
        help="Minimum path length in meters.",
    )
    p.add_argument(
        "--max-path-length",
        type=float,
        default=60.0,
        help="Maximum path length in meters.",
    )
    p.add_argument(
        "--min-median-speed",
        type=float,
        default=0.05,
        help="Reject tracks whose median speed is below this value (m/s).",
    )
    p.add_argument(
        "--max-median-speed",
        type=float,
        default=3.5,
        help="Reject tracks whose median speed is above this value (m/s).",
    )
    p.add_argument(
        "--max-gap",
        type=float,
        default=5.0,
        help="Reject tracks if any adjacent time gap exceeds this value (s).",
    )
    p.add_argument(
        "--center-start",
        action="store_true",
        default=True,
        help="Subtract the starting point from each trajectory before standardization.",
    )
    p.add_argument(
        "--keep-absolute-position",
        action="store_true",
        help="Do not subtract the starting point; preserve absolute local coordinates.",
    )
    p.add_argument(
        "--include-speed-channel",
        action="store_true",
        help="Append interpolated speed as a 3rd channel, i.e. [x, y, speed].",
    )
    p.add_argument(
        "--include-heading-curvature-channels",
        action="store_true",
        help=(
            "Append sin(theta), cos(theta), and curvature kappa computed from the "
            "resampled 2D trajectory. With --include-speed-channel this yields "
            "[x, y, speed, sin_theta, cos_theta, kappa] before any optional endpoint channels."
        ),
    )
    p.add_argument(
        "--include-endpoint-channels",
        action="store_true",
        help=(
            "Append four constant channels carrying the absolute start and end "
            "positions of the trajectory: [x_ini, y_ini, x_end, y_end]. "
            "With --include-speed-channel this yields [x, y, speed, x_ini, y_ini, x_end, y_end]."
        ),
    )
    p.add_argument(
        "--endpoint-weight",
        type=float,
        default=1.0,
        help=(
            "Optional post-standardization multiplier applied only to the endpoint "
            "channels. Values > 1 make start/end position more influential during clustering."
        ),
    )
    p.add_argument(
        "--endpoint-focused-grid",
        action="store_true",
        help=(
            "Use a cosine-resampled grid that concentrates more samples near the "
            "beginning and the end of each trajectory while keeping the output "
            "dimension unchanged."
        ),
    )
    p.add_argument(
        "--preserve-original-scale",
        action="store_true",
        help=(
            "Keep the endpoint-focused cosine resampling, but preserve the "
            "original physical scale of the observations by disabling start "
            "centering and skipping dataset-level standardization. This makes "
            "the saved Y use the same units as the raw trajectories."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matching CSV members and exit.",
    )

    # Optional HDP-GPC integration.
    p.add_argument(
        "--hdpgpc-module",
        type=str,
        default=None,
        help="Import path of your HDP-GPC module, e.g. yourpkg.models.hdpgpc",
    )
    p.add_argument(
        "--hdpgpc-class",
        type=str,
        default=None,
        help="Class name inside --hdpgpc-module, e.g. HDPGPC",
    )
    p.add_argument(
        "--hdpgpc-config",
        type=str,
        default=None,
        help="Optional JSON config file passed as kwargs to the HDP-GPC constructor.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for tensor conversion when fitting the user HDP-GPC.",
    )
    return p.parse_args()


def is_remote_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def cached_download_path(source: str) -> Path:
    cache_root = Path(tempfile.gettempdir()) / "hdp_gpc_pedestrian_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(source).suffix or ".zip"
    key = sha1(source.encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{key}{suffix}"


def download_remote_zip_to_cache(source: str) -> Path:
    cache_path = cached_download_path(source)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
    print(f"[INFO] Downloading remote ZIP to local cache: {cache_path}")
    with urllib.request.urlopen(source) as resp, open(tmp_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(cache_path)
    return cache_path


@contextlib.contextmanager
def open_zip_source(source: str):
    """
    Open either a local ZIP or a remote ZIP.

    Remote reading first tries remotezip so only the needed members are fetched.
    If the host does not support HTTP range requests, it falls back to a one-time
    full download into a local cache and then opens the cached ZIP.
    """
    if is_remote_source(source):
        try:
            from remotezip import RemoteZip  # type: ignore
        except ImportError:
            RemoteZip = None

        if RemoteZip is not None:
            try:
                with RemoteZip(source) as zf:
                    yield zf
                    return
            except Exception as exc:
                msg = str(exc)
                if "RangeNotSupported" not in msg and "range requests" not in msg.lower():
                    raise
                print("[WARN] Remote host does not support HTTP range requests; falling back to full ZIP download.")

        local_zip = download_remote_zip_to_cache(source)
        with zipfile.ZipFile(local_zip) as zf:
            yield zf
    else:
        with zipfile.ZipFile(source) as zf:
            yield zf


def list_zip_members(source: str) -> List[str]:
    with open_zip_source(source) as zf:
        try:
            return list(zf.namelist())
        except AttributeError:
            return [info.filename for info in zf.infolist()]


def is_real_dataset_csv(member_name: str) -> bool:
    """
    Accept only real trajectory CSV members and reject archive metadata files
    such as AppleDouble sidecars under __MACOSX/ and names starting with ._.
    """
    p = PurePosixPath(member_name)
    name = p.name
    if "__MACOSX/" in member_name:
        return False
    if name.startswith("._"):
        return False
    if not name.lower().endswith("_trans.csv"):
        return False
    return True


def parse_member_name(member_name: str) -> Optional[Tuple[str, str]]:
    if not is_real_dataset_csv(member_name):
        return None
    base = PurePosixPath(member_name).name
    m = CSV_NAME_RE.search(base)
    if m is None:
        return None
    camera = m.group("camera").upper().replace("PC-", "PB-")
    date_str = m.group("date")
    return camera, date_str


def select_members(
    names: Sequence[str],
    camera_prefix: str,
    max_files: int,
    date_start: Optional[str],
    date_end: Optional[str],
) -> List[str]:
    camera_prefix = camera_prefix.upper().replace("PC-", "PB-")
    candidates: List[Tuple[str, str]] = []

    start_ts = pd.Timestamp(date_start) if date_start else None
    end_ts = pd.Timestamp(date_end) if date_end else None

    for name in names:
        if not is_real_dataset_csv(name):
            continue
        parsed = parse_member_name(name)
        if parsed is None:
            continue
        camera, date_str = parsed
        if camera != camera_prefix:
            continue
        day = pd.Timestamp(date_str)
        if start_ts is not None and day < start_ts:
            continue
        if end_ts is not None and day > end_ts:
            continue
        candidates.append((date_str, name))

    candidates.sort(key=lambda x: x[0])
    return [name for _, name in candidates[:max_files]]


def read_member_csv(source: str, member_name: str) -> pd.DataFrame:
    if not is_real_dataset_csv(member_name):
        raise ValueError(f"Refusing to read non-dataset CSV member: {member_name}")

    read_errors = []
    with open_zip_source(source) as zf:
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with zf.open(member_name) as fh:
                    text_fh = io.TextIOWrapper(fh, encoding=encoding, newline="")
                    df = pd.read_csv(text_fh, header=None, names=CSV_COLUMNS)
                break
            except UnicodeDecodeError as exc:
                read_errors.append(f"{encoding}: {exc}")
                continue
        else:
            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                1,
                "Could not decode CSV with tried encodings: " + "; ".join(read_errors),
            )

    df["t_rel_s"] = pd.to_numeric(df["t_rel_s"], errors="coerce")
    df["x_m"] = pd.to_numeric(df["x_m"], errors="coerce")
    df["y_m"] = pd.to_numeric(df["y_m"], errors="coerce")

    dt = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    missing = dt.isna()
    if missing.any():
        dt_fallback = pd.to_datetime(df.loc[missing, "date"], format="%d.%m.%Y %H:%M", errors="coerce")
        dt.loc[missing] = dt_fallback
    missing = dt.isna()
    if missing.any():
        dt_fallback = pd.to_datetime(df.loc[missing, "date"], format="%d.%m.%Y", errors="coerce")
        dt.loc[missing] = dt_fallback
    df["date"] = dt

    df = df.dropna(subset=["id", "t_rel_s", "x_m", "y_m"]).copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    return df


def save_raw_member(df: pd.DataFrame, member_name: str, output_dir: Path) -> None:
    raw_dir = output_dir / "raw_curated_csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / PurePosixPath(member_name).name
    df.to_csv(out_path, index=False)


def clean_single_track(track_df: pd.DataFrame) -> pd.DataFrame:
    tdf = track_df[["t_rel_s", "date", "x_m", "y_m"]].copy()
    tdf = tdf.sort_values("t_rel_s")

    # Collapse repeated times by averaging the positions.
    grouped = (
        tdf.groupby("t_rel_s", as_index=False)
        .agg({"date": "first", "x_m": "mean", "y_m": "mean"})
        .sort_values("t_rel_s")
    )
    return grouped.reset_index(drop=True)


def compute_track_stats(
    source_file: str,
    camera: str,
    acquisition_date: str,
    track_id: int,
    track_df: pd.DataFrame,
) -> TrackStats:
    t = track_df["t_rel_s"].to_numpy(dtype=float)
    x = track_df["x_m"].to_numpy(dtype=float)
    y = track_df["y_m"].to_numpy(dtype=float)

    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    step_dist = np.sqrt(dx * dx + dy * dy)
    path_length = float(step_dist.sum())
    positive = dt > 0
    speeds = step_dist[positive] / dt[positive] if np.any(positive) else np.asarray([], dtype=float)

    median_speed = float(np.median(speeds)) if speeds.size else 0.0
    mean_speed = float(np.mean(speeds)) if speeds.size else 0.0
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    max_gap = float(np.max(dt)) if len(dt) else 0.0

    return TrackStats(
        source_file=PurePosixPath(source_file).name,
        camera=camera,
        acquisition_date=acquisition_date,
        track_id=int(track_id),
        n_points=int(len(track_df)),
        duration_s=duration,
        max_gap_s=max_gap,
        path_length_m=path_length,
        median_speed_mps=median_speed,
        mean_speed_mps=mean_speed,
        start_x_m=float(x[0]),
        start_y_m=float(y[0]),
        end_x_m=float(x[-1]),
        end_y_m=float(y[-1]),
    )


def accept_track(stats: TrackStats, args: argparse.Namespace) -> bool:
    if stats.n_points < args.min_points:
        return False
    if stats.duration_s < args.min_duration or stats.duration_s > args.max_duration:
        return False
    if stats.path_length_m < args.min_path_length or stats.path_length_m > args.max_path_length:
        return False
    if stats.median_speed_mps < args.min_median_speed or stats.median_speed_mps > args.max_median_speed:
        return False
    if stats.max_gap_s > args.max_gap:
        return False
    return True


def make_resampling_grid(grid_size: int, endpoint_focused: bool) -> np.ndarray:
    """
    Build the normalized resampling grid in [0, 1].

    When endpoint_focused is True, use a cosine mapping:
        u = (1 - cos(pi * s)) / 2,  s in [0, 1]
    which allocates more points near the start and end of the trajectory.
    """
    s = np.linspace(0.0, 1.0, grid_size, dtype=float)
    if not endpoint_focused:
        return s
    return 0.5 * (1.0 - np.cos(np.pi * s))


def compute_heading_curvature_from_resampled(
    xg: np.ndarray,
    yg: np.ndarray,
    grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute heading and curvature from a resampled planar trajectory.

    Returns
    -------
    sin_theta, cos_theta, kappa : (L,), (L,), (L,)
        Heading encoded as sin/cos of the tangent angle and signed curvature
        kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2), computed with finite
        differences with respect to the common grid.
    """
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(yg, dtype=float)
    grid = np.asarray(grid, dtype=float)

    dx = np.gradient(xg, grid)
    dy = np.gradient(yg, grid)

    tangent_norm = np.sqrt(dx * dx + dy * dy)
    eps = 1e-8
    safe_norm = np.where(tangent_norm < eps, 1.0, tangent_norm)

    sin_theta = dy / safe_norm
    cos_theta = dx / safe_norm

    ddx = np.gradient(dx, grid)
    ddy = np.gradient(dy, grid)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, eps)
    kappa = (dx * ddy - dy * ddx) / denom

    stationary = tangent_norm < eps
    if np.any(stationary):
        sin_theta = sin_theta.copy()
        cos_theta = cos_theta.copy()
        sin_theta[stationary] = 0.0
        cos_theta[stationary] = 1.0

    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    return sin_theta, cos_theta, kappa


def resample_track(
    track_df: pd.DataFrame,
    grid_size: int,
    center_start: bool,
    include_speed_channel: bool,
    include_heading_curvature_channels: bool,
    include_endpoint_channels: bool,
    endpoint_focused_grid: bool,
) -> np.ndarray:
    t = track_df["t_rel_s"].to_numpy(dtype=float)
    x = track_df["x_m"].to_numpy(dtype=float)
    y = track_df["y_m"].to_numpy(dtype=float)

    if len(t) < 2 or float(t[-1] - t[0]) <= 0.0:
        raise ValueError("Trajectory must contain at least 2 strictly ordered time points.")

    u = (t - t[0]) / (t[-1] - t[0])
    grid = make_resampling_grid(grid_size=grid_size, endpoint_focused=endpoint_focused_grid)
    xg = np.interp(grid, u, x)
    yg = np.interp(grid, u, y)

    start_x = float(x[0])
    start_y = float(y[0])
    end_x = float(x[-1])
    end_y = float(y[-1])

    channels = [xg, yg]

    if include_speed_channel:
        dt = np.diff(t)
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.zeros_like(t)
        valid = dt > 0
        if np.any(valid):
            local_speed = np.sqrt(dx[valid] * dx[valid] + dy[valid] * dy[valid]) / dt[valid]
            valid_idx = np.where(valid)[0]
            speed[valid_idx + 1] = local_speed
            speed[0] = local_speed[0]
        speedg = np.interp(grid, u, speed)
        channels.append(speedg)

    if include_heading_curvature_channels:
        sin_theta, cos_theta, kappa = compute_heading_curvature_from_resampled(
            xg=xg,
            yg=yg,
            grid=grid,
        )
        channels.extend([sin_theta, cos_theta, kappa])

    if include_endpoint_channels:
        endpoint_channels = [
            np.full(grid_size, start_x, dtype=float),
            np.full(grid_size, start_y, dtype=float),
            np.full(grid_size, end_x, dtype=float),
            np.full(grid_size, end_y, dtype=float),
        ]
        channels.extend(endpoint_channels)

    arr = np.stack(channels, axis=-1)
    if center_start:
        arr[:, 0] = arr[:, 0] - arr[0, 0]
        arr[:, 1] = arr[:, 1] - arr[0, 1]
    return arr


def standardize_dataset(Y: np.ndarray) -> Tuple[np.ndarray, StandardizationStats]:
    flat = Y.reshape(-1, Y.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Yn = (Y - mean[None, None, :]) / std[None, None, :]
    return Yn, StandardizationStats(mean=mean.tolist(), std=std.tolist())


def build_channel_names(
    include_speed_channel: bool,
    include_heading_curvature_channels: bool,
    include_endpoint_channels: bool,
) -> List[str]:
    names = ["x", "y"]
    if include_speed_channel:
        names.append("speed")
    if include_heading_curvature_channels:
        names.extend(["sin_theta", "cos_theta", "kappa"])
    if include_endpoint_channels:
        names.extend(["x_ini", "y_ini", "x_end", "y_end"])
    return names


def curate_dataset(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_members = list_zip_members(args.source)
    chosen_members = select_members(
        names=all_members,
        camera_prefix=args.camera_prefix,
        max_files=args.max_files,
        date_start=args.date_start,
        date_end=args.date_end,
    )

    if not chosen_members:
        raise RuntimeError(
            f"No CSV members matched camera={args.camera_prefix!r}. "
            "Run with --dry-run to inspect matching members."
        )

    print("Matched members:")
    for name in chosen_members:
        print(f"  - {name}")

    if args.dry_run:
        return {"matched_members": chosen_members}

    accepted_tracks: List[np.ndarray] = []
    accepted_stats: List[TrackStats] = []

    for member_name in chosen_members:
        if len(accepted_tracks) >= args.max_tracks:
            break

        df = read_member_csv(args.source, member_name)
        print(
            "RAW file bounds:",
            df["x_m"].min(), df["x_m"].max(),
            df["y_m"].min(), df["y_m"].max()
        )
        save_raw_member(df, member_name, output_dir)
        parsed = parse_member_name(member_name)
        if parsed is None:
            continue
        camera, day = parsed

        accepted_this_file = 0
        for track_id, group in df.groupby("id", sort=False):
            if len(accepted_tracks) >= args.max_tracks:
                break
            if accepted_this_file >= args.tracks_per_file:
                break

            print(
                "Before clean file bounds:",
                group["x_m"].min(), group["x_m"].max(),
                group["y_m"].min(), group["y_m"].max(),
            )
            clean = clean_single_track(group)
            if len(clean) < 2:
                continue

            stats = compute_track_stats(
                source_file=member_name,
                camera=camera,
                acquisition_date=day,
                track_id=int(track_id),
                track_df=clean,
            )
            print(
                "Accepted file bounds:",
                clean["x_m"].min(), clean["x_m"].max(),
                clean["y_m"].min(), clean["y_m"].max(),
            )
            if not accept_track(stats, args):
                continue
            try:
                arr = resample_track(
                    clean,
                    grid_size=args.grid_size,
                    center_start=(False if args.preserve_original_scale else (args.center_start and not args.keep_absolute_position)),
                    include_speed_channel=args.include_speed_channel,
                    include_heading_curvature_channels=args.include_heading_curvature_channels,
                    include_endpoint_channels=args.include_endpoint_channels,
                    endpoint_focused_grid=args.endpoint_focused_grid,
                )

            except Exception:
                continue

            accepted_tracks.append(arr)
            accepted_stats.append(stats)
            accepted_this_file += 1

    if not accepted_tracks:
        raise RuntimeError(
            "No trajectories survived the curation filters. "
            "Try reducing --min-points, --min-duration, or changing the camera."
        )

    Y = np.stack(accepted_tracks, axis=0).astype(np.float32)
    channel_names = build_channel_names(
        include_speed_channel=args.include_speed_channel,
        include_heading_curvature_channels=args.include_heading_curvature_channels,
        include_endpoint_channels=args.include_endpoint_channels,
    )

    if args.preserve_original_scale:
        Y_model = Y.copy()
        std_stats = StandardizationStats(
            mean=[0.0] * Y.shape[-1],
            std=[1.0] * Y.shape[-1],
        )
    else:
        Y_model, std_stats = standardize_dataset(Y)

    if args.include_endpoint_channels and float(args.endpoint_weight) != 1.0:
        endpoint_names = {"x_ini", "y_ini", "x_end", "y_end"}
        endpoint_idx = [i for i, name in enumerate(channel_names) if name in endpoint_names]
        Y_model[..., endpoint_idx] *= float(args.endpoint_weight)

    x_grid = make_resampling_grid(args.grid_size, endpoint_focused=args.endpoint_focused_grid).astype(np.float32)[:, None]

    meta = {
        "source": args.source,
        "camera_prefix": args.camera_prefix,
        "n_tracks": int(Y_model.shape[0]),
        "grid_size": int(args.grid_size),
        "n_outputs": int(Y_model.shape[-1]),
        "channel_names": channel_names,
        "center_start": False if args.preserve_original_scale else bool(args.center_start and not args.keep_absolute_position),
        "include_speed_channel": bool(args.include_speed_channel),
        "include_heading_curvature_channels": bool(args.include_heading_curvature_channels),
        "include_endpoint_channels": bool(args.include_endpoint_channels),
        "endpoint_weight": float(args.endpoint_weight),
        "endpoint_focused_grid": bool(args.endpoint_focused_grid),
        "resampling_grid_mode": "cosine" if args.endpoint_focused_grid else "uniform",
        "preserve_original_scale": bool(args.preserve_original_scale),
        "standardized": not bool(args.preserve_original_scale),
        "chosen_members": chosen_members,
        "standardization": asdict(std_stats),
        "track_stats": [asdict(s) for s in accepted_stats],
    }

    np.savez_compressed(
        output_dir / "pedestrian_hdp_gpc_input.npz",
        x_grid=x_grid,
        Y=Y_model,
        Y_unscaled=Y,
    )
    with open(output_dir / "pedestrian_hdp_gpc_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Accepted trajectories: {Y_model.shape[0]}")
    print(f"Channel names: {channel_names}")
    print(f"Resampling grid mode: {'cosine' if args.endpoint_focused_grid else 'uniform'}")
    print(f"Preserve original scale: {bool(args.preserve_original_scale)}")
    print(f"Saved arrays to: {output_dir / 'pedestrian_hdp_gpc_input.npz'}")
    print(f"Saved metadata to: {output_dir / 'pedestrian_hdp_gpc_metadata.json'}")

    return {
        "x_grid": x_grid,
        "Y": Y_model,
        "Y_unscaled": Y,
        "metadata": meta,
        "track_stats": accepted_stats,
    }


def load_hdp_gpc_kwargs(config_path: Optional[str]) -> Dict[str, object]:
    if config_path is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("HDP-GPC config JSON must be an object/dictionary.")
    return data


def maybe_to_torch(
    x_grid: np.ndarray,
    Y: np.ndarray,
    device: str,
) -> Tuple[object, object, object]:
    if not TORCH_AVAILABLE:
        x_list = [x_grid.copy() for _ in range(Y.shape[0])]
        y_list = [Y[i].copy() for i in range(Y.shape[0])]
        return x_grid, x_list, y_list

    x_grid_t = torch.as_tensor(x_grid, dtype=torch.float32, device=device)
    x_list_t = [x_grid_t.clone() for _ in range(Y.shape[0])]
    y_list_t = [torch.as_tensor(Y[i], dtype=torch.float32, device=device) for i in range(Y.shape[0])]
    return x_grid_t, x_list_t, y_list_t


def build_and_fit_hdp_gpc(
    args: argparse.Namespace,
    x_grid: np.ndarray,
    Y: np.ndarray,
    metadata: Dict[str, object],
) -> object:
    """
    Fit the pedestrian subset with the same HDP-GPC construction used in the
    provided UCR example, adapted to multivariate pedestrian trajectories.

    Input
    -----
    x_grid : (L, 1)
        Common normalized grid used during preprocessing. The HDP-GPC fitting
        itself follows the UCR example and uses an integer basis 0..L-1.
    Y : (N, L, D)
        Batch of standardized trajectories, where D is typically 2 for [x, y],
        3 for [x, y, speed], or 6/7 when endpoint channels are appended.

    Optional config
    ---------------
    --hdpgpc-config can point to a JSON file with constructor/fit overrides,
    e.g. {
      "max_models": 30,
      "warp": false,
      "warp_updating": false,
      "method_compute_warp": "greedy",
      "mode_warp": "rough",
      "n_explore_steps": 10,
      "share_gp": true
    }

    Notes
    -----
    This function no longer uses --hdpgpc-module / --hdpgpc-class. Instead it
    directly instantiates hdpgpc.GPI_HDP using the same recipe as the uploaded
    UCR runner.
    """
    try:
        import time
        import hdpgpc.GPI_HDP as hdpgp
        from hdpgpc.get_data import compute_estimators_LDS
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import the hdpgpc package required by the UCR-style fitting "
            "function. Make sure your environment can import 'hdpgpc'."
        ) from exc

    cfg = load_hdp_gpc_kwargs(args.hdpgpc_config)

    X = np.asarray(Y, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected Y with shape (n_samples, length, n_outputs), got {X.shape}")

    n_samples, L, n_outputs = X.shape
    if n_samples < 1 or L < 2:
        raise ValueError(f"Need at least one trajectory and length >= 2, got {X.shape}")

    if TORCH_AVAILABLE:
        torch.set_default_dtype(torch.float64)

    # Same parameter bootstrapping as in run_ucr_hdpgpc.py
    std, std_dif, bound_sigma_est, bound_gamma_est = compute_estimators_LDS(X)

    sigma = float(cfg.pop("ini_sigma", std * 0.000000001))
    gamma = float(cfg.pop("ini_gamma", std_dif * 0.1))
    outputscale_ = float(cfg.pop("ini_outputscale", np.max(np.abs(X))))
    ini_lengthscale = float(cfg.pop("ini_lengthscale", 1.0))
    bound_lengthscale = tuple(cfg.pop("bound_lengthscale", (0.1, 20.0)))

    # Keep the same conservative overwrite used in the UCR example.
    default_bound_sigma = (
        max(float(std) * 1e-8, 1e-12),
        max(float(std) * 1e-5, 1e-10),
    )
    bound_sigma = tuple(cfg.pop("bound_sigma", default_bound_sigma))
    bound_gamma = tuple(cfg.pop("bound_gamma", tuple(bound_gamma_est)))

    noise_warp = float(cfg.pop("noise_warp", std * 0.1))
    default_bound_noise_warp = (
        max(noise_warp * 0.1, 1e-12),
        max(noise_warp * 0.2, 1e-10),
    )
    bound_noise_warp = tuple(cfg.pop("bound_noise_warp", default_bound_noise_warp))

    max_models = int(cfg.pop("max_models", 40))
    warp = bool(cfg.pop("warp", False))
    n_explore_steps = cfg.pop("n_explore_steps", 15)
    free_deg_MNIW = cfg.pop("free_deg_MNIV", 3)

    # Integer bases exactly as in the UCR example.
    x_basis = np.atleast_2d(np.arange(0, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(0, L, 2, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(0, L, dtype=np.float64)).T
    x_trains = np.repeat(x_train[None, :, :], n_samples, axis=0)

    model_kwargs = dict(
        x_basis_warp=x_basis_warp,
        n_outputs=n_outputs,
        kernels=cfg.pop("kernels", None),
        model_type=cfg.pop("model_type", "static"),
        ini_lengthscale=ini_lengthscale,
        bound_lengthscale=bound_lengthscale,
        ini_gamma=gamma,
        ini_sigma=sigma,
        ini_outputscale=outputscale_,
        noise_warp=noise_warp,
        bound_sigma=bound_sigma,
        bound_gamma=bound_gamma,
        bound_noise_warp=bound_noise_warp,
        warp_updating=cfg.pop("warp_updating", False),
        method_compute_warp=cfg.pop("method_compute_warp", "greedy"),
        verbose=cfg.pop("verbose", False),
        hmm_switch=cfg.pop("hmm_switch", False),
        max_models=max_models,
        mode_warp=cfg.pop("mode_warp", "rough"),
        bayesian_params=cfg.pop("bayesian_params", True),
        inducing_points=cfg.pop("inducing_points", False),
        reestimate_initial_params=cfg.pop("reestimate_initial_params", False),
        n_explore_steps=n_explore_steps,
        free_deg_MNIV=free_deg_MNIW,
        share_gp=cfg.pop("share_gp", True),
        use_snr=cfg.pop("use_snr", False),
        reduce_outputs=cfg.pop("reduce_outputs", False),
    )
    # Any remaining JSON keys are forwarded as additional constructor kwargs.
    model_kwargs.update(cfg)

    print("Fitting HDP-GPC using the UCR-style configuration...")
    print(f"  Trajectories: {n_samples}")
    print(f"  Length:       {L}")
    print(f"  Outputs:      {n_outputs}")
    print(f"  sigma:        {sigma}")
    print(f"  gamma:        {gamma}")
    print(f"  noise_warp:   {noise_warp}")
    print(f"  max_models:   {max_models}")
    print(f"  warp:         {warp}")

    sw_gp = hdpgp.GPI_HDP(x_basis, **model_kwargs)

    if TORCH_AVAILABLE and hasattr(sw_gp, "device"):
        sw_gp.device = torch.device(args.device)

    t0 = time.time()
    sw_gp.include_batch(x_trains, X, warp=warp)
    runtime_sec = float(time.time() - t0)

    resp = sw_gp.resp_assigned[-1]
    if TORCH_AVAILABLE and hasattr(resp, "detach"):
        cluster_labels = resp.detach().cpu().numpy().reshape(-1)
    else:
        cluster_labels = np.asarray(resp).reshape(-1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output_dir / "hdpgpc_cluster_labels.npy"
    np.save(labels_path, cluster_labels)

    unique_clusters, counts = np.unique(cluster_labels.astype(int), return_counts=True)
    fit_summary = {
        "n_samples": int(n_samples),
        "series_length": int(L),
        "n_outputs": int(n_outputs),
        "runtime_sec": runtime_sec,
        "camera_prefix": metadata.get("camera_prefix"),
        "chosen_members": metadata.get("chosen_members", []),
        "cluster_labels_path": str(labels_path.resolve()),
        "n_pred_clusters": int(len(unique_clusters)),
        "cluster_sizes": {str(int(c)): int(cnt) for c, cnt in zip(unique_clusters, counts)},
        "fit_recipe": "ucr_example_adaptation",
        "hyperparameters": {
            "ini_sigma": sigma,
            "ini_gamma": gamma,
            "ini_outputscale": outputscale_,
            "ini_lengthscale": ini_lengthscale,
            "bound_lengthscale": list(bound_lengthscale),
            "bound_sigma": list(bound_sigma),
            "bound_gamma": list(bound_gamma),
            "noise_warp": noise_warp,
            "bound_noise_warp": list(bound_noise_warp),
            "max_models": max_models,
            "warp": warp,
            "n_explore": n_explore_steps,
            "free_deg_MNIW": free_deg_MNIW,
        },
    }
    with open(output_dir / "hdpgpc_fit_summary.json", "w", encoding="utf-8") as f:
        json.dump(fit_summary, f, indent=2)

    print(f"Saved HDP-GPC cluster labels to: {labels_path}")
    print(f"Saved fit summary to: {output_dir / 'hdpgpc_fit_summary.json'}")
    return {
        "model": sw_gp,
        "cluster_labels": cluster_labels,
        "runtime_sec": runtime_sec,
    }


def save_quick_summary(output_dir: Path, metadata: Dict[str, object]) -> None:
    track_stats = metadata["track_stats"]
    df = pd.DataFrame(track_stats)
    summary = {
        "n_tracks": int(len(df)),
        "n_files": int(len(metadata["chosen_members"])),
        "median_duration_s": float(df["duration_s"].median()),
        "median_path_length_m": float(df["path_length_m"].median()),
        "median_speed_mps": float(df["median_speed_mps"].median()),
    }
    with open(output_dir / "curation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = curate_dataset(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        return 0

    x_grid = result["x_grid"]
    Y = result["Y"]
    metadata = result["metadata"]
    save_quick_summary(output_dir, metadata)

    build_and_fit_hdp_gpc(args, x_grid=x_grid, Y=Y, metadata=metadata)
    # except Exception as exc:  # noqa: BLE001
    #     print(f"[WARN] HDP-GPC fitting stage raised {type(exc).__name__}: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
