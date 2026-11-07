"""Normalize shipped result CSVs to the canonical RankMe/SSR‡ definitions.

This migration is deterministic and uses only shipped sufficient statistics.
It does not replace a full rerun from feature matrices.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from metrics.ssr import ssr_dagger


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def decode_vtab_key(arch_prefix: str, task_raw: str) -> tuple[str, str]:
    parts = str(task_raw).split("__")
    if len(parts) != 2:
        return str(arch_prefix), str(task_raw)
    suffix, task = parts
    no_underscore = f"{arch_prefix}{suffix}"
    with_underscore = f"{arch_prefix}_{suffix}"
    known = set(pd.read_csv(RESULTS / "vtab_25arch_8tasks.csv")["arch"])
    if no_underscore in known:
        return no_underscore, task
    return with_underscore, task


def migrate_pool_b() -> None:
    path = RESULTS / "pool_b_final.csv"
    df = pd.read_csv(path)
    df["ssr_dagger"] = [
        ssr_dagger(r, sa, c)
        for r, sa, c in zip(df["reff"], df["sa_k"], df["n_class"])
    ]
    df = df.drop(columns=["ssr", "scr"], errors="ignore")
    df.to_csv(path, index=False)


def migrate_vtab() -> None:
    path = RESULTS / "vtab_25arch_8tasks.csv"
    df = pd.read_csv(path)
    raw = pd.read_csv(RESULTS / "vtab_ssr_results.csv")
    keys = [decode_vtab_key(a, t) for a, t in zip(raw["arch"], raw["task"])]
    rankme_reff = {
        key: float(reff)
        for key, reff in zip(keys, raw["reff"])
        if np.isfinite(reff) and reff > 0
    }

    for idx, row in df.iterrows():
        reff = rankme_reff.get((str(row["arch"]), str(row["task"])))
        if reff is None:
            continue
        df.at[idx, "reff"] = reff
        df.at[idx, "rankme"] = np.log(reff)
        if row["n_class"] > 1 and np.isfinite(row["sa_k"]) and row["sa_k"] > 0:
            df.at[idx, "ssr_dagger"] = ssr_dagger(
                reff, float(row["sa_k"]), int(row["n_class"])
            )
        else:
            df.at[idx, "ssr_dagger"] = np.nan
    df.to_csv(path, index=False)


if __name__ == "__main__":
    migrate_pool_b()
    migrate_vtab()
    print("Migrated pool_b_final.csv and vtab_25arch_8tasks.csv")
