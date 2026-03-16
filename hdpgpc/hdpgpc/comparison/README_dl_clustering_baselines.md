# Deep clustering baselines for the same MIT-BIH beat tensors used by your HDP-GPC script

This runner mirrors the data assumptions of your uploaded `run_offline_all_records.py`:

- it looks for `data/mitdb` or `data/mitbih`
- it expects one file per record: `<record>.npy`
- it expects labels in `<record>_labels.npy`
- by default it uses the first channel / first lead

## What it runs

1. **CAE + DPGMM**
2. **CAE + DBSCAN**
3. **DEC**
4. **Optional HDP-GPC scoring** using your existing `cluster_labels_<rec>_offline.npy` files

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r /path/to/requirements_dl_clustering_baselines.txt
```

## Quick start

Run one record first:

```bash
python /path/to/run_dl_clustering_baselines.py 100 --include-hdp
```

If you launch from outside the repo, pass explicit paths:

```bash
python /path/to/run_dl_clustering_baselines.py 100 \
  --repo-root /path/to/your/repo \
  --data-dir /path/to/your/repo/hdpgpc/data/mitbih \
  --include-hdp
```

Run all records:

```bash
python /path/to/run_dl_clustering_baselines.py --include-hdp
```

Fast CPU smoke test:

```bash
python /path/to/run_dl_clustering_baselines.py 100 \
  --cpu \
  --ae-epochs 10 \
  --dec-epochs 10 \
  --batch-size 64 \
  --include-hdp
```

More stable / stronger run:

```bash
python /path/to/run_dl_clustering_baselines.py \
  --ae-epochs 40 \
  --dec-epochs 30 \
  --latent-dim 32 \
  --max-components 25 \
  --include-hdp
```

## Outputs

By default results go to:

```text
<repo_root>/results/dl_clustering_baselines/
```

You will get:

- `summary_per_record.csv`
- `summary_by_method.csv`
- `cluster_labels/cae_dpgmm/cluster_labels_<rec>.npy`
- `cluster_labels/cae_dbscan/cluster_labels_<rec>.npy`
- `cluster_labels/dec/cluster_labels_<rec>.npy`

## Metrics

- `acc` = clustering accuracy after Hungarian matching
- `ari` = adjusted Rand index
- `ami` = adjusted mutual information
- `nmi` = normalized mutual information
- `homogeneity`, `completeness`, `v_measure`
- `silhouette`
- `n_pred_clusters`

## Recommended paper table

At minimum, report:

- mean ARI across records
- mean AMI across records
- mean ACC across records
- mean predicted number of clusters
- runtime per record

## Notes

- `CAE + DPGMM` and `CAE + DBSCAN` do not require a fixed K.
- `DEC` does require K, so the default here uses `--dec-k auto` based on silhouette.
- If you want an upper-bound style DEC run, use `--dec-k oracle`.
- The runner automatically looks for HDP-GPC outputs in:

```text
results/cluster_labels/v1_UCR_new_ver/cluster_labels_<rec>_offline.npy
```

If your HDP outputs are elsewhere:

```bash
python /path/to/run_dl_clustering_baselines.py --include-hdp --hdp-dir /custom/path
```
