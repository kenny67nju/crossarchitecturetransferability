#!/usr/bin/env python3
"""Regenerate the camera-ready effective-rank/width diagnostic."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def pool_frames() -> dict[str, pd.DataFrame]:
    mv_scores = pd.read_csv(RESULTS / "camera_ready" / "pool_b_final_score_complete.csv")
    mv_dims = pd.read_csv(RESULTS / "truncation_ablation.csv")[["variant", "task", "feat_dim"]]
    mv = mv_scores.merge(mv_dims, on=["variant", "task"], how="inner")
    mv = mv.groupby("arch").agg(reff=("reff", "mean"), d=("feat_dim", "mean")).reset_index()

    vtab = pd.read_csv(RESULTS / "camera_ready" / "vtab_24arch_8tasks_lp_complete.csv")
    vtab = vtab.groupby("arch").agg(reff=("reff", "mean"), d=("feat_dim", "mean")).reset_index()

    ex = pd.read_csv(RESULTS / "camera_ready" / "extended_full_sota_score_complete.csv")
    ex = ex.groupby("arch").agg(reff=("reff", "mean"), d=("feat_dim", "mean")).reset_index()
    return {"MV": mv, "VTAB": vtab, "EX": ex}


def main() -> None:
    colors = {"MV": "#1f77b4", "VTAB": "#ff7f0e", "EX": "#2ca02c"}
    markers = {"MV": "o", "VTAB": "s", "EX": "^"}
    fig, ax = plt.subplots(figsize=(5.3, 3.8))
    frames = pool_frames()
    for name, frame in frames.items():
        rho = spearmanr(frame["reff"], frame["d"]).statistic
        ax.scatter(
            frame["d"], frame["reff"], s=28, alpha=0.72,
            color=colors[name], marker=markers[name], edgecolor="white",
            linewidth=0.35, label=rf"{name} ($\rho_{{\rm Sp}}={rho:+.2f}$)",
        )

    guide_x = np.array([192.0, 4096.0])
    ax.plot(guide_x, 0.22 * guide_x, "k--", linewidth=0.8, alpha=0.5,
            label="slope-1 guide")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Penultimate feature dimension $d$")
    ax.set_ylabel(r"RankMe effective rank $r_{\mathrm{eff}}$")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.3)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "fig_reff_d_scatter.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper" / "fig_reff_d_scatter.png", dpi=200, bbox_inches="tight")
    for name, frame in frames.items():
        print(name, len(frame), spearmanr(frame["reff"], frame["d"]).statistic)


if __name__ == "__main__":
    main()
