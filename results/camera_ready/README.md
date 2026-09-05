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

The final VTAB audit first completes 48 previously missing LP cells from cached
features, then restricts evaluation to the 24 architectures with features on
all eight tasks. The uncentered reference audit was independently rerun on the
server in
`/data/Paper06/camera_ready_20260904/results/vtab_reference_24arch_uncentered`.
Its 192 cached SVDs and 168 scored non-STL-10 cells are checked in here.

The earlier 18-architecture support produced the provisional VTAB regrets
0.053 (STL-10 reference) and 0.051 (four-reference LOO). Those values are
superseded by 0.061 (STL-10) and 0.055 (four-reference LOO) on the complete
24-architecture support after correcting the historical `n_class=0` placeholders.

The official PCA-128 DiSCO summaries are regenerated from the score CSVs and
the completed linear-probe tables.  They must not be mixed with the legacy
`disco` columns in the top-level pool tables, which came from a different,
non-official implementation.

`ssl_disco_history_summary.csv` records the historical SSL variants. The
values 0.077/0.071 in an early meta-response draft match none of the retained
score files and are treated as a transcription error; official PCA-128 values
are 0.045/0.042 (G=6/G=8).
