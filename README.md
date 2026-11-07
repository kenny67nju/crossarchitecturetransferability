# Width Confounding in Label-Free Cross-Architecture Model Selection: Diagnosis and Subspace Correction

Camera-ready code and analysis release for NeurIPS 2026, submission 14729.

Qin Liu, Chen Zhong, and Fengshan Zhao. State Key Laboratory of Novel Software Technology, Nanjing University. Correspondence: fs.zhao@nju.edu.cn.

This repository provides scoring and analysis code, final result tables, provenance records, verification scripts, and bilingual paper sources. It supports verification from the retained numerical results; it is **not a complete public raw-feature or end-to-end historical reproduction package**.

## Camera-ready papers and checkout

- [English camera-ready PDF](paper/Paper06_camera_ready_en.pdf): 10 content pages, 26 pages total.
- [Chinese translation PDF](paper/Paper06_camera_ready_zh.pdf): 9 content pages, 19 pages total.

Both files include the disclosures, references, appendices, and checklist. Use the English PDF for the conference submission. `main` is the public entry point; `codex/camera-ready` is the camera-ready maintenance branch. To explicitly check out that branch:

```sh
git clone --branch codex/camera-ready https://github.com/kenny67nju/crossarchitecturetransferability.git
cd crossarchitecturetransferability
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy matplotlib pypdf
```

The commands below assume this environment is active. No GPU, PyTorch, or raw feature caches are needed for the retained-result analysis. `requirements.txt` covers the broader historical extraction workflow and does not replace the PDF-specific prerequisites below. See [the file guide](docs/REPOSITORY_GUIDE.md) for current versus archived material.

## Method and scope

RankMe uses singular-value probabilities `p_i = sigma_i / sum_j sigma_j`, not squared singular values. Subspace-Corrected Ranking (SCR) is our relational score family; SSR‡ is its default member:

```text
SSR‡(H,T) = [log r_eff(H) - log SA_k(H,H_ref)] / log C(T)
```

For each candidate encoder, target and fixed-reference images are encoded by that same candidate. Alignment is the mean squared cosine of the principal angles between their top right-singular subspaces. We never compare feature subspaces from different encoders. The main protocol uses uncentered SVD, fixed k=20, and a 1e-12 lower clip before logarithms. C(T) is the target class count; for a fixed target it does not affect the ranking. Target labels are used to evaluate linear probes, not to select models with RankMe or SSR‡.

SSR‡ is invariant to a common orthogonal basis change of target and reference features. Its distinction from RankMe is dependence on target-reference geometry, not “breaking rotation invariance.” The formal width-scaling result is for RankMe under an isotropic random-feature model at fixed aspect ratio, not for arbitrary rotation-invariant scores.

## Final results and exact supports

Canonical VGG extraction arm: `single_preprocess_4096` in `results/p0_vgg_pair_20261104/p0_aggregate_v2/`. This replaces the two torchvision VGG models' target rows using 4096-dimensional hidden fully connected features and one official weights transform. It does not re-extract every candidate.

| Headline pool | Candidates | Target tasks | RankMe regret | SSR‡ regret |
|---|---:|---:|---:|---:|
| MV | 119 variants in 35 groups | 5 | 0.1440 | 0.0980 |
| VTAB | 24 architectures | 7 | 0.1100 | 0.0614 |
| EX | 51 variants in 35 groups | 10 | 0.1445 | 0.0595 |

Across 22 pool-task units (10 unique target names), mean regret is **0.133409 for RankMe and 0.068864 for SSR‡**; paired difference −0.064545, target-name-clustered 95% CI [−0.121905, −0.019750], two-sided bootstrap p=0.00194. Exact random-selection mean regret is 0.136836. Statistics and input hashes are recorded in [`paper_values.json`](results/camera_ready_p0_20261104/paper_values.json).

- STL-10 is the fixed reference and is excluded from all methods in the common target-selection aggregate.
- SSL/multimodal results cover 209 available model-task cells across 29 tasks, not a complete 8×29 rectangle. Missing historical reference manifests prevent a verified main-protocol SSR comparison, so SSL is excluded from the 22-unit headline.
- Official classification DiSCO uses target ground-truth labels. Its per-pool diagnostics are separate from the zero-label comparison; old surrogate DiSCO columns are not the official baseline.
- Auxiliary MV comparisons use 118 candidates (one missing-score variant excluded), and 115 after removing VGG. These differ from the headline 119 and the full 132-variant inventory. Random LP rounds to 0.630 on 119 candidates and 0.629 on 118 candidates.
- The five-task SSR‡→RankMe pipeline has regret 0.102 at 38 candidate evaluations, or 37 with canonical-feature reuse, excluding reference setup. It is not the best-performing zero-label pipeline tested. Evaluation counts are not measured wall-clock speedups; K counts labeled architecture evaluations, not labeled samples.

## Reproduce the retained-result analysis

Run from the repository root in a Python environment with NumPy, pandas, SciPy, and Matplotlib:

```sh
python3 code/build_p0_camera_assets.py
python3 code/figure_sigma_b_flow.py
python3 code/audit_p0_manuscript.py
python3 code/audit_dataset_citations.py
```

These commands regenerate manuscript tables/figures and verify retained numerical inputs. They do not download datasets, extract all features, or rerun all linear probes. Only final analysis inputs and the helper/source snapshots needed by these commands are published from the P0 bundle; their hashes are listed in `results/p0_vgg_pair_20261104/PUBLIC_RELEASE_MANIFEST.json`. Intermediate runs, caches, logs, and backup archives remain local. Generated manuscript derivatives go to `results/camera_ready_p0_20261104/` and `paper/`.

The statistical reference configuration is 100,000 bootstrap resamples, seed 42; target-name resampling retains every corresponding pool unit and uses a unit-weighted mean. Wilcoxon uses paired differences rounded to 10 decimals, `zero_method='wilcox'`, `method='approx'`, two-sided, no continuity correction. The recorded reference SciPy version is 1.15.3; the audit was also run with SciPy 1.17.1, NumPy 2.4.4, and pandas 3.0.2.

## Build the complete papers

Install XeLaTeX, BibTeX, and Poppler (`pdffonts`); make their executables available on PATH. Python `pypdf` is included in the lightweight setup above. On macOS these tools can be provided by MacTeX/BasicTeX and Homebrew Poppler; on Linux, by the corresponding TeX Live XeTeX/Chinese-language and Poppler packages.

The build also compiles the editable overview in `paper/fig_method_overview.tex`; the TeX installation must include `standalone`, PGF/TikZ, and `newtx` (text and math fonts). The resulting vector PDF is shared by both language versions.

The Chinese source currently selects `Arial Unicode.ttf` from `/Library/Fonts/`. For Chinese builds on other machines, install a suitably licensed font there or edit the three `setCJK...font` declarations in `paper/main_v18_zh.tex` to use an installed CJK font. A substitute font may change line breaks and page counts; check the regenerated PDF. The English build does not need this CJK font:

```sh
python3 code/build_camera_ready_pdfs.py --language en
```

With the Chinese font configured, build and audit both languages:

```sh
python3 code/build_camera_ready_pdfs.py --language both
python3 code/audit_camera_ready_pdfs.py
```

The complete deliverables are [`paper/Paper06_camera_ready_en.pdf`](paper/Paper06_camera_ready_en.pdf) and [`paper/Paper06_camera_ready_zh.pdf`](paper/Paper06_camera_ready_zh.pdf). Each contains the main text, funding and competing-interests disclosures, references, appendices, and checklist. The `combined` PDFs are byte-identical compatibility aliases. The supplement PDFs are optional appendix extracts, not required companion files. The old `main_v18.pdf` is archived under `archive/paper/history/` and is not rebuilt by this pipeline. The source names `main_v18.tex` and `main_v18_zh.tex` remain the current build inputs for compatibility.

## Release map and provenance

| Location | Role |
|---|---|
| `paper/main_v18.tex`, `paper/main_v18_zh.tex` | Current bilingual camera-ready sources |
| `paper/references.bib`, `docs/DATASET_SOURCES.md` | Dataset publications/releases and task-name mapping |
| `results/p0_vgg_pair_20261104/` | Selected final P0 score/support tables, helper/source snapshots, and public-release hash manifest; not the complete experimental bundle |
| `results/camera_ready_p0_20261104/` | Final manuscript derivatives, numeric/PDF audits, and input hashes in `paper_values.json` |
| `results/camera_ready_review_20261104/` | Stage-specific and within-family diagnostics |
| `results/camera_ready/` | Retained VTAB/EX inputs and earlier audits; consult final input hashes rather than assuming every file is current |
| `code/metrics/` | Effective-rank, subspace-alignment, SSR, and DiSCO implementations |
| `docs/CAMERA_READY_FINAL_ACCEPTANCE_20261104.md` | Acceptance checks and remaining limitations |
| `archive/` | Earlier sources and results preserved for provenance |

`archive/paper/rebuttals/` contains historical author responses; `archive/paper/history/` contains the old submission PDF and an unused historical figure. Their claims and numbers are historical, not the current results. Local-only downloads, build remnants, and backups are under ignored `archive/local/`; experimental intermediate directories are also ignored. No raw images, weights, or feature caches are included in this release.

Historical scripts such as `reproduce_all.py` and old top-level result CSVs are not the canonical camera-ready reproduction entry point. Use the final input hashes and commands above; do not mix historical scores with final LP/support tables.

## Data access, attribution, and limitations

The [dataset-source inventory](docs/DATASET_SOURCES.md) covers all 29 distinct evaluated task names, aliases, underlying sources for derived tasks, and ImageNet pretraining/reference provenance. Obtain images and pretrained weights from their original providers, subject to their terms. This repository does not redistribute those original images, model weights, or raw VGG feature caches.

Reference choice remains heuristic, final-protocol k sensitivity is not established, and performance is pool-dependent. Preprocessing is heterogeneous across retained candidates; the VGG controls do not validate every historical feature cache. Some historical environments and per-run reference records are unavailable. There is no formal guarantee from subspace alignment to downstream accuracy.

Source attribution is not a complete license audit. The release does not yet contain a separate explicit software-license file or a complete asset-by-asset license/terms inventory; the paper's Checklist retains No for these documentation gaps. No clinical use is claimed for biomedical task diagnostics.
