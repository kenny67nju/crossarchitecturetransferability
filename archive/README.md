# Historical and local-only material

Files under `archive/` are retained only for provenance and are not the current
camera-ready reproduction entry point. Use the root README and `paper/Paper06_camera_ready_en.pdf`.

- `paper/rebuttals/`: historical reviewer/AC responses. Numbers and claims may have been superseded.
- `paper/history/`: old `main_v18.pdf` and an unused historical figure; not current deliverables.
- `local/backups/`: local camera-ready backup archives and previous bibliography snapshots.
- `local/latex_build/`: old LaTeX intermediate files retained locally, no longer tracked.
- `local/reference_audits/`: downloaded third-party papers and audit working files, not redistributed.

The entire `local/` directory is ignored by Git. Moving files here does not delete them.
Older audit reports may mention their original paths; see `docs/REPOSITORY_GUIDE.md` for the mapping.

`code/legacy_uid_ssr/` contains the superseded 22-variant experiment that used
the obsolete UID-based score `r_eff / (SA_k * UID * log C)`.

The canonical paper score is
`SSR‡ = (log r_eff - log SA_k) / log C`, where `r_eff` is RankMe effective
rank with `p_i = sigma_i / sum_j sigma_j`.
