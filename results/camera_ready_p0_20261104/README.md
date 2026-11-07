# Final manuscript-facing results

Canonical input arm: single_preprocess_4096 in ../p0_vgg_pair_20261104/p0_aggregate_v2/.

paper_values.json records the headline statistics, width-figure diagnostics and input hashes.
The auxiliary CSVs reproduce the 118-candidate and 115-candidate comparisons; the headline MV support remains 119 candidates.
Figure-width and pool-diagnosis CSVs correspond to the final paper.

Regenerate with code/build_p0_camera_assets.py and verify with code/audit_p0_manuscript.py from the repository root.
The generated numeric_acceptance.json and pdf_acceptance.json are local-only run reports, not distributed results.
See ../../docs/RELEASE_VERIFICATION.md for complete checks.

The two VGG variants use 4096-dimensional hidden fully connected features and one official weights transform.
Other candidates retain historical extraction; this is not a uniformly re-extracted zoo.
Missing raw caches and historical environments limit end-to-end reproduction.

Bootstrap configuration: 100,000 draws, seed 42, unit-weighted target-name clusters.
Wilcoxon: differences rounded to 10 decimals, zero_method=wilcox, method=approx, two-sided, no continuity correction.
