#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gc
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# More stable in lightweight CPU environments.
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_repo_root() -> Path:
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


def list_records(data_dir: Path) -> List[str]:
    recs: List[str] = []
    for f in sorted(data_dir.glob("*.npy")):
        if f.name.endswith("_labels.npy"):
            continue
        if "labels" in f.stem:
            continue
        rec = f.stem
        labels_f = data_dir / f"{rec}_labels.npy"
        if labels_f.exists():
            recs.append(rec)
    return recs


def znorm_per_series(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.maximum(sd, eps)
    return (X - mu) / sd


@dataclass
class RecordData:
    rec: str
    X: np.ndarray
    y: np.ndarray


def load_record(data_dir: Path, rec: str, channel: int = 0) -> RecordData:
    data = np.load(data_dir / f"{rec}.npy")
    y = np.load(data_dir / f"{rec}_labels.npy").astype(np.int64)
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (n_samples, length, n_outputs), got {data.shape}")
    if channel >= data.shape[2]:
        raise ValueError(f"Requested channel {channel} but data only has {data.shape[2]} channels")
    X = data[:, :, channel].astype(np.float32)
    X = znorm_per_series(X)
    return RecordData(rec=rec, X=X, y=y)


class ConvAutoencoder1D(nn.Module):
    def __init__(self, input_length: int, latent_dim: int = 16):
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dim = int(latent_dim)
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            h = self.encoder_cnn(torch.zeros(1, 1, self.input_length))
            self.enc_channels = h.shape[1]
            self.enc_length = h.shape[2]
            flat_dim = self.enc_channels * self.enc_length
        self.to_latent = nn.Linear(flat_dim, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, flat_dim)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, 7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        return self.to_latent(h.reshape(h.shape[0], -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).reshape(z.shape[0], self.enc_channels, self.enc_length)
        xhat = self.decoder_cnn(h)
        if xhat.shape[-1] > self.input_length:
            xhat = xhat[..., : self.input_length]
        elif xhat.shape[-1] < self.input_length:
            xhat = F.pad(xhat, (0, self.input_length - xhat.shape[-1]))
        return xhat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class DECModel(nn.Module):
    def __init__(self, encoder: ConvAutoencoder1D, n_clusters: int, alpha: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.alpha = float(alpha)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, encoder.latent_dim))

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist_sq / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.clamp(q.sum(dim=1, keepdim=True), min=1e-12)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        weight = q ** 2 / torch.clamp(q.sum(dim=0, keepdim=True), min=1e-12)
        return weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1e-12)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(X).float().unsqueeze(1)
    return DataLoader(TensorDataset(xt), batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


@torch.no_grad()
def encode_dataset(model: ConvAutoencoder1D, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    zs: List[np.ndarray] = []
    for (xb,) in make_loader(X, batch_size=batch_size, shuffle=False):
        zs.append(model.encode(xb.to(device)).cpu().numpy())
    return np.concatenate(zs, axis=0)


def train_autoencoder(
    X: np.ndarray,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    verbose: bool = True,
) -> ConvAutoencoder1D:
    model = ConvAutoencoder1D(X.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(X, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for (xb,) in loader:
            xb = xb.to(device)
            xhat, _ = model(xb)
            loss = F.mse_loss(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            count += int(xb.shape[0])
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"    [AE] epoch {epoch:03d}/{epochs} loss={total / max(count,1):.6f}")
    return model


def relabel_to_contiguous(y: np.ndarray) -> np.ndarray:
    uniq = np.unique(y)
    mapping = {u: i for i, u in enumerate(uniq)}
    return np.array([mapping[v] for v in y], dtype=np.int64)


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = relabel_to_contiguous(np.asarray(y_true))
    yp = relabel_to_contiguous(np.asarray(y_pred))
    n = max(yt.max(), yp.max()) + 1
    contingency = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(yt, yp):
        contingency[a, b] += 1
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[row_ind, col_ind].sum()) / float(len(yt))


def safe_silhouette(X: np.ndarray, y_pred: np.ndarray) -> float:
    uniq = np.unique(y_pred)
    if len(uniq) < 2 or len(uniq) >= len(y_pred):
        return float("nan")
    try:
        return float(silhouette_score(X, y_pred))
    except Exception:
        return float("nan")


def score_clustering(y_true: np.ndarray, y_pred: np.ndarray, X_embed: np.ndarray) -> Dict[str, float]:
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    return {
        "n_samples": int(len(y_true)),
        "n_true_clusters": int(len(np.unique(y_true))),
        "n_pred_clusters": int(len(np.unique(y_pred))),
        "acc": cluster_accuracy(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "ami": adjusted_mutual_info_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
        "silhouette": safe_silhouette(X_embed, y_pred),
    }


def auto_select_k(X: np.ndarray, k_min: int, k_max: int, seed: int) -> Tuple[int, float]:
    best_k = max(2, k_min)
    best_s = -np.inf
    for k in range(max(2, k_min), min(k_max, len(X) - 1) + 1):
        try:
            pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
            s = silhouette_score(X, pred)
            if s > best_s:
                best_k, best_s = int(k), float(s)
        except Exception:
            continue
    if not np.isfinite(best_s):
        best_s = float("nan")
    return best_k, best_s


def run_cae_dpgmm(latent: np.ndarray, max_components: int, dp_alpha: float, seed: int) -> np.ndarray:
    Z = StandardScaler().fit_transform(latent)
    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=dp_alpha,
        max_iter=500,
        n_init=2,
        init_params="kmeans",
        random_state=seed,
    )
    return model.fit_predict(Z).astype(np.int64)


def choose_dbscan(latent_scaled: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    best_labels: Optional[np.ndarray] = None
    best_info: Dict[str, float] = {"score": -np.inf, "eps": np.nan, "min_samples": np.nan}
    for min_samples in [3, 5, 10]:
        if len(latent_scaled) <= min_samples:
            continue
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(latent_scaled)
        dists, _ = nbrs.kneighbors(latent_scaled)
        kth = np.sort(dists[:, -1])
        eps_grid = sorted(set(float(np.quantile(kth, q)) for q in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]))
        for eps in eps_grid:
            if not np.isfinite(eps) or eps <= 0:
                continue
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(latent_scaled)
            mask = labels != -1
            uniq = np.unique(labels[mask]) if mask.any() else np.array([])
            if mask.sum() < 10 or len(uniq) < 2:
                continue
            try:
                score = silhouette_score(latent_scaled[mask], labels[mask])
            except Exception:
                continue
            if score > best_info["score"]:
                best_labels = labels.astype(np.int64)
                best_info = {
                    "score": float(score),
                    "eps": float(eps),
                    "min_samples": float(min_samples),
                    "retained_fraction": float(mask.mean()),
                    "n_clusters_no_noise": float(len(uniq)),
                }
    if best_labels is None:
        best_labels = np.zeros(len(latent_scaled), dtype=np.int64)
        best_info = {
            "score": float("nan"),
            "eps": float("nan"),
            "min_samples": float("nan"),
            "retained_fraction": 1.0,
            "n_clusters_no_noise": 1.0,
        }
    return best_labels, best_info


def run_dec(
    ae: ConvAutoencoder1D,
    X: np.ndarray,
    batch_size: int,
    dec_epochs: int,
    dec_lr: float,
    n_clusters: int,
    device: torch.device,
    seed: int,
    verbose: bool,
) -> np.ndarray:
    Z0 = encode_dataset(ae, X, batch_size=batch_size, device=device)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    km.fit(Z0)
    dec = DECModel(ae, n_clusters=n_clusters).to(device)
    with torch.no_grad():
        dec.cluster_centers.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(dec.parameters(), lr=dec_lr)
    loader = make_loader(X, batch_size=batch_size, shuffle=True)
    for epoch in range(1, dec_epochs + 1):
        dec.train()
        total = 0.0
        count = 0
        for (xb,) in loader:
            xb = xb.to(device)
            q = dec.soft_assign(dec.encoder.encode(xb))
            p = dec.target_distribution(q).detach()
            loss = F.kl_div(torch.log(torch.clamp(q, min=1e-10)), p, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            count += int(xb.shape[0])
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == dec_epochs):
            print(f"    [DEC] epoch {epoch:03d}/{dec_epochs} kl={total / max(count,1):.6f}")
    preds: List[np.ndarray] = []
    dec.eval()
    with torch.no_grad():
        for (xb,) in make_loader(X, batch_size=batch_size, shuffle=False):
            q = dec.soft_assign(dec.encoder.encode(xb.to(device)))
            preds.append(torch.argmax(q, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.int64)


def save_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels)


def rewrite_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_by_method(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = ["acc", "ari", "ami", "nmi", "homogeneity", "completeness", "v_measure", "silhouette", "fit_minutes", "n_pred_clusters"]
    methods = sorted({str(r["method"]) for r in rows})
    out: List[Dict[str, object]] = []
    for method in methods:
        subset = [r for r in rows if str(r["method"]) == method]
        row: Dict[str, object] = {"method": method, "n_records": len(subset)}
        for m in metrics:
            vals = [float(r[m]) for r in subset if m in r and np.isfinite(float(r[m]))]
            row[f"mean_{m}"] = float(np.mean(vals)) if vals else float("nan")
            row[f"std_{m}"] = float(np.std(vals)) if vals else float("nan")
        out.append(row)
    return out


def default_hdp_dir(repo_root: Path) -> Path:
    return repo_root / "results" / "cluster_labels" / "v1_UCR_new_ver"


def maybe_load_hdp_labels(hdp_dir: Path, rec: str) -> Optional[np.ndarray]:
    path = hdp_dir / f"cluster_labels_{rec}_offline.npy"
    if not path.exists():
        return None
    return np.asarray(np.load(path)).astype(np.int64)


def run_record(record: RecordData, args: argparse.Namespace, device: torch.device, out_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    print(f"  [INFO] {record.rec}: n_samples={record.X.shape[0]}, length={record.X.shape[1]}")
    t_ae = time.time()
    ae = train_autoencoder(
        X=record.X,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.ae_epochs,
        lr=args.ae_lr,
        weight_decay=args.weight_decay,
        device=device,
        verbose=not args.quiet,
    )
    ae_minutes = (time.time() - t_ae) / 60.0
    latent = encode_dataset(ae, record.X, batch_size=args.batch_size, device=device)
    latent_scaled = StandardScaler().fit_transform(latent)
    base = out_dir / "cluster_labels"

    if "cae-dpgmm" in args.methods:
        t = time.time()
        pred = run_cae_dpgmm(latent, min(args.max_components, max(2, len(record.y) - 1)), args.dp_alpha, args.seed)
        save_labels(base / "cae_dpgmm" / f"cluster_labels_{record.rec}.npy", pred)
        rows.append({"record": record.rec, "method": "CAE+DPGMM", "fit_minutes": ae_minutes + (time.time() - t) / 60.0, **score_clustering(record.y, pred, latent_scaled)})

    if "cae-dbscan" in args.methods:
        t = time.time()
        pred, info = choose_dbscan(latent_scaled)
        save_labels(base / "cae_dbscan" / f"cluster_labels_{record.rec}.npy", pred)
        rows.append({"record": record.rec, "method": "CAE+DBSCAN", "fit_minutes": ae_minutes + (time.time() - t) / 60.0, **score_clustering(record.y, pred, latent_scaled), **{f"dbscan_{k}": v for k, v in info.items()}})

    if "dec" in args.methods:
        t = time.time()
        if args.dec_k == "auto":
            dec_k, sil = auto_select_k(latent_scaled, 2, min(args.max_dec_k, max(2, len(record.y) - 1)), args.seed)
        elif args.dec_k == "oracle":
            dec_k, sil = int(len(np.unique(record.y))), float("nan")
        else:
            dec_k, sil = int(args.dec_k), float("nan")
        pred = run_dec(ae, record.X, args.batch_size, args.dec_epochs, args.dec_lr, dec_k, device, args.seed, not args.quiet)
        save_labels(base / "dec" / f"cluster_labels_{record.rec}.npy", pred)
        rows.append({"record": record.rec, "method": "DEC", "fit_minutes": ae_minutes + (time.time() - t) / 60.0, "dec_k": dec_k, "dec_auto_k_silhouette": sil, **score_clustering(record.y, pred, latent_scaled)})

    if args.include_hdp:
        hdp = maybe_load_hdp_labels(Path(args.hdp_dir), record.rec)
        if hdp is not None and len(hdp) == len(record.y):
            rows.append({"record": record.rec, "method": "HDP-GPC", "fit_minutes": float("nan"), **score_clustering(record.y, hdp, latent_scaled)})
        elif hdp is not None:
            print(f"  [WARN] HDP-GPC labels for {record.rec} have length {len(hdp)} but expected {len(record.y)}")

    del ae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep clustering baselines for MIT-BIH beat tensors")
    p.add_argument("record", nargs="?", default=None, help="Optional single record ID, e.g. 231")
    p.add_argument("--repo-root", type=str, default=None, help="Optional explicit repo root containing hdpgpc/")
    p.add_argument("--data-dir", type=str, default=None, help="Optional explicit directory with <rec>.npy and <rec>_labels.npy")
    p.add_argument("--methods", nargs="+", default=["cae-dpgmm", "cae-dbscan", "dec"], choices=["cae-dpgmm", "cae-dbscan", "dec"])
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ae-epochs", type=int, default=25)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--dec-epochs", type=int, default=20)
    p.add_argument("--dec-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-components", type=int, default=20)
    p.add_argument("--dp-alpha", type=float, default=0.5)
    p.add_argument("--dec-k", type=str, default="auto", help="auto | oracle | integer")
    p.add_argument("--max-dec-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--hdp-dir", type=str, default=None)
    p.add_argument("--include-hdp", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)
    repo_root = Path(args.repo_root) if args.repo_root is not None else find_repo_root()
    data_dir = Path(args.data_dir) if args.data_dir is not None else find_data_dir(repo_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else repo_root / "results" / "dl_clustering_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.hdp_dir is None:
        args.hdp_dir = str(default_hdp_dir(repo_root))

    recs = list_records(data_dir)
    if not recs:
        raise RuntimeError(f"No records found in {data_dir}. (Expected *.npy plus *_labels.npy)")
    if args.record is not None:
        if args.record not in recs:
            raise ValueError(f"Record {args.record} not found in {data_dir}. Example records: {recs[:10]}")
        recs = [args.record]

    print(f"[INFO] repo_root: {repo_root}")
    print(f"[INFO] data_dir:   {data_dir}")
    print(f"[INFO] out_dir:    {out_dir}")
    print(f"[INFO] device:     {device}")
    print(f"[INFO] records:    {len(recs)}")
    if args.include_hdp:
        print(f"[INFO] hdp_dir:    {args.hdp_dir}")

    all_rows: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []
    start = time.time()
    for i, rec in enumerate(recs, 1):
        print(f"\n[{i}/{len(recs)}] Processing record {rec}")
        try:
            record = load_record(data_dir, rec, channel=args.channel)
            rows = run_record(record, args, device, out_dir)
            all_rows.extend(rows)
            for row in rows:
                print(f"  [OK] {row['method']:<10} acc={float(row['acc']):.4f} ari={float(row['ari']):.4f} n_pred={int(row['n_pred_clusters'])}")
        except Exception as e:
            print(f"  [FAIL] {rec}: {repr(e)}")
            failures.append((rec, repr(e)))

    rewrite_csv(out_dir / "summary_per_record.csv", all_rows)
    rewrite_csv(out_dir / "summary_by_method.csv", summarize_by_method(all_rows))
    print(f"\n[INFO] Finished in {(time.time() - start)/60.0:.2f} minutes")
    print(f"[INFO] Per-record summary: {out_dir / 'summary_per_record.csv'}")
    print(f"[INFO] By-method summary:  {out_dir / 'summary_by_method.csv'}")
    if failures:
        print("[WARN] Failures:")
        for rec, err in failures:
            print(f"  - {rec}: {err}")


if __name__ == "__main__":
    main()
