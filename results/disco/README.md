# Official DiSCO diagnostics

Official classification DiSCO uses target ground-truth labels. It is not a zero-label comparator.

Run from the repository root:

```sh
python code/summarize_official_disco.py
```

This regenerates official_disco_regret_summary.csv using the final supports:

| Pool | Tasks | Available candidate-task cells | G=6 regret | G=8 regret |
|---|---:|---:|---:|---:|
| MV | 5 | 595 | 0.145 | 0.145 |
| VTAB, excluding STL-10 | 7 | 168 | 0.067 | 0.040 |
| EX | 10 | 510 | 0.045 | 0.049 |
| SSL | 29 | 209 | 0.045 | 0.042 |

G is the number of consecutive near-equal spectral groups after PCA-128.
MV official scores come from the final single_preprocess_4096_pool132_scores.csv, restricted to 119 multi-variant candidates.
The two EX score CSVs are here; VTAB and SSL score CSVs are in results/.
The SSL LP table is results/paper14_crossdomain.csv; only its model, task and lp_acc fields are used here.
Other historical score columns in that retained source are not official DiSCO or verified headline SSR evidence.

The script also reports full-feature SSL diagnostics (0.057 / 0.047) separately for the appendix, never mixing them with PCA-128 scores.
For scoring user-supplied cached features, see code/run_official_disco.py --help.
