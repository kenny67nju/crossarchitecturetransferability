#!/usr/bin/env python3
"""Fail-fast consistency checks for camera-ready data, prose, and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CAMERA = RESULTS / "camera_ready"


def close(actual: float, expected: float, tolerance: float = 5e-5) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    headline = json.loads((CAMERA / "headline_metrics.json").read_text())
    common = pd.read_csv(CAMERA / "common_support_per_task.csv")
    assert len(common) == 22 and common["task"].nunique() == 10
    close(common["ssr_regret"].mean(), headline["common_support"]["ssr_mean_regret"])
    close(common["rankme_regret"].mean(), headline["common_support"]["rankme_mean_regret"])
    close(common["difference"].mean(), headline["common_support"]["mean_difference_ssr_minus_rankme"])

    for filename, rows in [
        ("pool_b_final_score_complete.csv", 660),
        ("extended_full_sota_score_complete.csv", 510),
    ]:
        table = pd.read_csv(CAMERA / filename)
        assert len(table) == rows
        assert not table[["n_class", "rankme", "reff", "sa_k", "ssr_dagger"]].isna().any().any()
        assert (table["n_class"] > 1).all()
        assert np.allclose(table["rankme"], table["reff"])
        expected = (
            np.log(table["reff"])
            - np.log(np.maximum(table["sa_k"], 1e-12))
        ) / np.log(table["n_class"])
        assert np.allclose(table["ssr_dagger"], expected)

    vtab = pd.read_csv(CAMERA / "vtab_24arch_8tasks_lp_complete.csv")
    assert len(vtab) == 192 and vtab["arch"].nunique() == 24
    assert not vtab["lp_acc"].isna().any() and (vtab["n_class"] > 1).all()
    geometry = pd.read_csv(CAMERA / "vtab_fspe_24arch_uncentered.csv")
    assert len(geometry) == 168 and not geometry.isna().any().any()

    ssl = pd.read_csv(CAMERA / "ssl_disco_history_summary.csv").set_index("variant")
    close(ssl.loc["ssl_official_pca128_g6", "mean_regret"], 0.0446761859)
    close(ssl.loc["ssl_official_pca128_g8", "mean_regret"], 0.0422780668)

    paper = (ROOT / "paper" / "main_v18.tex").read_text()
    for required in ["$0.069$", "$-0.1043$", "$0.061$", "$0.055$", "$168$"]:
        assert required in paper, f"paper missing {required}"
    for stale in ["$0.083$", "$-0.0898$", "$0.106$ over the remaining"]:
        assert stale not in paper, f"stale camera-ready value remains: {stale}"
    for figure in ["fig_sigma_b_flow.pdf", "fig_reff_d_scatter.pdf", "fig_pareto.pdf", "fig_costsplit.pdf"]:
        path = ROOT / "paper" / figure
        assert path.exists() and path.stat().st_size > 10_000

    print("Camera-ready consistency audit passed.")


if __name__ == "__main__":
    main()
