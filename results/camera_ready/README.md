# Camera-ready result source of truth

Generate all files in this directory with:

```bash
python code/summarize_official_disco.py
python code/build_camera_ready_results.py
```

`headline_metrics.json` is the canonical source for numbers quoted in the
camera-ready paper. The 22-unit label-free comparison excludes STL-10 from all
methods. Official classification DiSCO is reported separately because it uses
target ground-truth labels.

The uncentered VTAB reference audit was independently rerun from the cached
feature matrices on 2026-09-04 in
`/data/Paper06/camera_ready_20260904/results/vtab_reference_uncentered`.  Its
193 cached SVDs and 169 scored non-STL-10 cells reproduce the checked-in
single-reference and leave-one-target-out regrets exactly.

The official PCA-128 DiSCO summaries are regenerated from the score CSVs and
the completed linear-probe tables.  They must not be mixed with the legacy
`disco` columns in the top-level pool tables, which came from a different,
non-official implementation.
