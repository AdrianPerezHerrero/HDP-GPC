# Cluster ZIP HDP-GPC adapter

Files produced:

- `run_cluster_zip_hdpgpc.py`: main adapter script.
- `cluster_zip_inventory.csv`: inventory of the uploaded ZIP, including N-beat annotation counts.

## Run target records

Place `run_cluster_zip_hdpgpc.py` next to `analysis_one_record.py` inside your HDP-GPC repository, then run:

```bash
python run_cluster_zip_hdpgpc.py \
  --input_zip cluster.zip \
  --records n17 n17c t08 t65 t66 \
  --out_dir results/cluster_zip_hdpgpc \
  --analysis_script_dir .
```

Outputs include one `.cluster` file per processed lead, plus `.npy` beat arrays and cluster overview plots.

The output `.cluster` format is:

```text
<annotation_sample>,<cluster_label>
```

This matches the provided `t01-*.cluster` example. Use `--cluster_time_units ms` to write milliseconds instead.

## Validate extraction on t01

```bash
python run_cluster_zip_hdpgpc.py \
  --input_zip cluster.zip \
  --records t01 \
  --use_existing_clusters \
  --drop_first_beat \
  --out_dir results/t01_validation
```

With the uploaded ZIP, this reproduces the two provided `t01` `.cluster` files exactly.

## Important note about missing annotations

The uploaded `.iatr` files for `n17`, `n17c`, and `t66` contain only the WFDB EOF marker and no `N` annotations. Since the workflow extracts beats around `N` annotations, those records will be skipped unless you provide annotation files or add an R-peak detection fallback.
