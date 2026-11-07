"""Selection helpers for the final fixed-support analysis.

Extracted unchanged from the final analysis functions; no repair-run I/O.
"""
import numpy as np
import pandas as pd

CAPS = [128, 256, 512, 768]

def multi(base):
    counts=base.groupby('arch').variant.nunique()
    result=base[base.arch.isin(counts[counts>=2].index)].copy()
    assert result.variant.nunique()==119
    assert (result.groupby('task').size()==119).all()
    return result

def selections(table):
    rows=[]
    metrics=['rankme','ssr_dagger','logme','hscore','uid','leep','ncti','disco_official_g6','disco_official_g8']
    canonicals=table.groupby('arch').variant.min().to_dict()
    for task,group in table.groupby('task',sort=True):
        oracle=float(group.lp_acc.max())
        rows.append(dict(task=task,method='random_exact',variant='',lp_acc=float(group.lp_acc.mean()),
                         regret=oracle-float(group.lp_acc.mean()),oracle=oracle,n_candidates=len(group),
                         evaluation_count=1,evaluation_count_reusing_canonical=1,n_scored=len(group),score_complete=True))
        for metric in metrics:
            if group[metric].isna().any() and metric in ['rankme','ssr_dagger','disco_official_g6','disco_official_g8']:
                raise ValueError(f'Missing {metric}/{task} on common candidate support')
            best=group.loc[group[metric].idxmax()]
            rows.append(dict(task=task,method=metric,variant=best.variant,lp_acc=float(best.lp_acc),
                             regret=oracle-float(best.lp_acc),oracle=oracle,n_candidates=len(group),
                             evaluation_count=len(group),evaluation_count_reusing_canonical=len(group),
                             n_scored=int(group[metric].notna().sum()),score_complete=bool(group[metric].notna().all())))
        canonical=group[group.apply(lambda row: row.variant==canonicals[row.arch],axis=1)]
        assert len(canonical)==table.arch.nunique()
        for stage1 in ['ssr_dagger','rankme','logme','hscore']:
            winner=canonical.loc[canonical[stage1].idxmax()]
            family=group[group.arch==winner.arch]
            best=family.loc[family['rankme'].idxmax()]
            rows.append(dict(task=task,method=f'two_stage_{stage1}_then_rankme',variant=best.variant,
                             lp_acc=float(best.lp_acc),regret=oracle-float(best.lp_acc),oracle=oracle,
                             n_candidates=len(group),evaluation_count=len(canonical)+len(family),
                             evaluation_count_reusing_canonical=len(canonical)+len(family)-1,
                             n_scored=int(canonical[stage1].notna().sum()),score_complete=bool(canonical[stage1].notna().all())))
    return pd.DataFrame(rows)

def selected(frame):
    sel=selections(frame)
    extra=[]
    canonicals=frame.groupby('arch').variant.min().to_dict()
    cols=['rankme_norm_d','rankme_uid','ssr_linear','ssr_inverse','ssr_half_log','ssr_double_log']+[f'reff_trunc_{cap}' for cap in CAPS]
    for task,g in frame.groupby('task'):
        oracle=float(g.lp_acc.max())
        reps=g[g.apply(lambda r:r.variant==canonicals[r.arch],axis=1)]
        for col in cols:
            if g[col].notna().any():
                winner=g.loc[g[col].idxmax()]
                extra.append(dict(task=task,method=col,variant=winner.variant,lp_acc=float(winner.lp_acc),
                                  regret=oracle-float(winner.lp_acc),oracle=oracle,n_candidates=len(g),
                                  evaluation_count=len(g),evaluation_count_reusing_canonical=len(g),
                                  n_scored=int(g[col].notna().sum()),score_complete=bool(g[col].notna().all())))
            if reps[col].notna().any():
                arch=reps.loc[reps[col].idxmax(),'arch']; family=g[g.arch==arch]
                winner=family.loc[family.rankme.idxmax()]
                extra.append(dict(task=task,method=f'two_stage_{col}_then_rankme',variant=winner.variant,
                                  lp_acc=float(winner.lp_acc),regret=oracle-float(winner.lp_acc),oracle=oracle,
                                  n_candidates=len(g),evaluation_count=len(reps)+len(family),
                                  evaluation_count_reusing_canonical=len(reps)+len(family)-1,
                                  n_scored=int(reps[col].notna().sum()),score_complete=bool(reps[col].notna().all())))
    return pd.concat([sel,pd.DataFrame(extra)],ignore_index=True)

def method_summary(sel):
    rows=[]
    for method,g in sel.groupby('method'):
        boot=np.random.default_rng(42).choice(g.regret.to_numpy(),size=(100000,len(g)),replace=True).mean(1)
        lo,hi=np.percentile(boot,[2.5,97.5])
        rows.append(dict(method=method,mean_lp=g.lp_acc.mean(),mean_regret=g.regret.mean(),
                         ci95_low=lo,ci95_high=hi,mean_evaluations=g.evaluation_count.mean(),
                         mean_evaluations_reusing_canonical=g.evaluation_count_reusing_canonical.mean(),
                         all_scores_complete=bool(g.score_complete.all()),n_tasks=len(g)))
    return pd.DataFrame(rows)
