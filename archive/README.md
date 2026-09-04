# Archived experiments

Files under `archive/` are retained only for provenance and are not part of
the paper v18 reproduction path.

`code/legacy_uid_ssr/` contains the superseded 22-variant experiment that used
the obsolete UID-based score `r_eff / (SA_k * UID * log C)`.

The canonical paper score is
`SSR† = (log r_eff - log SA_k) / log C`, where `r_eff` is RankMe effective
rank with `p_i = sigma_i / sum_j sigma_j`.
