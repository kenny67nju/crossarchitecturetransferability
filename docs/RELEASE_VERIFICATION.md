# Final-release verification

This is a retained-result verification package, not an end-to-end reconstruction of all historical feature extraction and linear probes.

## Run from a clean checkout

Install requirements.txt, then run:

```sh
python -m unittest discover -s tests -v
python code/build_p0_camera_assets.py
python code/figure_sigma_b_flow.py
python code/summarize_official_disco.py
python code/audit_p0_manuscript.py
python code/audit_dataset_citations.py
python code/audit_float_body_references.py
```

The numerical audit checks the immutable inputs, both language versions, selection supports, and rounded-difference bootstrap/Wilcoxon statistics. Generated acceptance JSON files are local-only. No script downloads or uploads data.

With XeLaTeX, BibTeX, Poppler and the Chinese font configured as described in README:

```sh
python code/build_camera_ready_pdfs.py --language both
python code/audit_camera_ready_pdfs.py
```

Expected pagination is 10 content pages / 26 total in English and 9 / 19 in Chinese. The complete papers contain appendices and checklist. Compatibility copies and optional appendix extracts are generated locally but excluded from the release.

## Manual checks

Inspect tables and figures at publication size, conclusion placement, disclosures before references, and the checklist at the end. Automated PDF checks validate structure, fonts, links and selected numbers, not visual readability or every scientific claim.

## Evidence and limits

The public P0 manifest hashes nine retained files: seven numerical inputs and two original extraction-source snapshots. Those snapshots document preprocessing heterogeneity and are not supported extraction commands. No raw images, model weights or feature caches are distributed.

MV uses 119 candidates for the headline and 118/115 for auxiliary comparisons. VTAB excludes STL-10 on the seven-task selection support; eight-task feature diagnostics are explicitly separate. SSL uses 209 available cells across 29 tasks, and is excluded from the 22-unit headline owing to missing historical reference provenance. Official DiSCO uses labels and is reported separately.

The final selection helpers preserve the original arithmetic and tie-breaking order. The official DiSCO summary uses the final 119-candidate MV table, not a pre-repair score file. Source data remain immutable.

The cleanup does not revise paper content or evidence limitations: reference choice is heuristic, k sensitivity under the final protocol is unestablished, and full-pool preprocessing is not uniformly verified.
