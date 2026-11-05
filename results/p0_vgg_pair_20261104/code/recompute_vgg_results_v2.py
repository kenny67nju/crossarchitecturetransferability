#!/usr/bin/env python3
"""Recompute MV selection and 22-unit statistics with frozen non-VGG rows."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
VARIANT='vgg16.torchvision_tv_v1'
ARMS=['layer_only_4096','single_preprocess_4096']
N_BOOT=100000

def save_json(path,obj):
    with path.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def attach_official(base):
    base=base.drop(columns=['disco'],errors='ignore').copy()
    for g in [6,8]:
        scores=pd.read_csv(ROOT/'inputs'/f'mv_official_pca128_g{g}.csv')
        scores['variant']=scores.file.str.removesuffix('.npz').str.rsplit('__',n=1).str[0].str.replace('_dot_','.',regex=False)
        scores['task']=scores.file.str.removesuffix('.npz').str.rsplit('__',n=1).str[1]
        assert not scores.duplicated(['variant','task']).any()
        base=base.merge(scores[['variant','task','disco']].rename(columns={'disco':f'disco_official_g{g}'}),
                        on=['variant','task'],how='left',validate='one_to_one',sort=False)
    return base

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

def common_rows(original,selected):
    result=original.copy()
    for task in selected.task.unique():
        group=selected[selected.task==task].set_index('method')
        mask=(result.pool=='MV')&(result.task==task)
        assert mask.sum()==1
        result.loc[mask,'oracle']=group.loc['rankme','oracle']
        result.loc[mask,'random_mean']=group.loc['random_exact','lp_acc']
        result.loc[mask,'random_regret']=group.loc['random_exact','regret']
        result.loc[mask,'rankme_regret']=group.loc['rankme','regret']
        result.loc[mask,'ssr_regret']=group.loc['ssr_dagger','regret']
    result['difference']=result.ssr_regret-result.rankme_regret
    assert len(result)==22 and not result.duplicated(['pool','task']).any()
    assert 'stl10' not in result.task.str.lower().tolist()
    pd.testing.assert_frame_equal(result[result.pool!='MV'].drop(columns='difference').reset_index(drop=True),
                                  original[original.pool!='MV'].drop(columns='difference').reset_index(drop=True))
    return result

def statistics(common):
    # Explicit rounding prevents spurious ranks among mathematically tied LP differences.
    values=np.round(common.difference.to_numpy(),10)
    unit=np.random.default_rng(42).choice(values,size=(N_BOOT,len(values)),replace=True).mean(1)
    targets=np.array(sorted(common.task.unique()))
    sums=np.array([values[common.task.to_numpy()==t].sum() for t in targets])
    counts=np.array([(common.task==t).sum() for t in targets])
    draws=np.random.default_rng(42).choice(len(targets),size=(N_BOOT,len(targets)),replace=True)
    cluster=sums[draws].sum(1)/counts[draws].sum(1)
    def report(x):
        return dict(ci95=np.percentile(x,[2.5,97.5]).tolist(),
                    two_sided_sign_p=float(min(1,2*min(np.mean(x>=0),np.mean(x<=0)))))
    w=wilcoxon(values,zero_method='wilcox',correction=False,alternative='two-sided',method='approx')
    return dict(n_units=len(common),n_unique_targets=len(targets),pool_counts=common.groupby('pool').size().to_dict(),
                ssr_mean_regret=float(common.ssr_regret.mean()),rankme_mean_regret=float(common.rankme_regret.mean()),
                mean_difference=float(values.mean()),wins=int(np.sum(values<0)),ties=int(np.sum(values==0)),losses=int(np.sum(values>0)),
                unit_bootstrap=report(unit),target_cluster_bootstrap=report(cluster),
                wilcoxon_statistic=float(w.statistic),wilcoxon_p=float(w.pvalue),
                statistical_protocol=dict(n_boot=N_BOOT,seed=42,difference_round_decimals=10,
                                          wilcoxon_zero_method='wilcox',wilcoxon_method='approx',
                                          continuity_correction=False,scipy_version=scipy.__version__,
                                          cluster_estimand='unit-weighted mean; resample target names and retain all their pool units'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',default='vgg16_paired_v1'); args=ap.parse_args()
    run=ROOT/'runs'/args.run_id
    out=run/'aggregate'; out.mkdir(exist_ok=False)
    sources={p.name:digest(p) for p in (ROOT/'inputs').iterdir() if p.is_file()}
    raw=pd.read_csv(ROOT/'inputs/pool_b_final_score_complete.csv')
    assert not raw.duplicated(['variant','task']).any()
    base=attach_official(raw)
    original=pd.read_csv(ROOT/'inputs/common_support_per_task.csv')
    baseline_selection=selections(multi(base))
    baseline_common=common_rows(original,baseline_selection)
    assert np.allclose(baseline_common[['ssr_regret','rankme_regret']],original[['ssr_regret','rankme_regret']],atol=1e-12)
    baseline_selection.to_csv(out/'baseline_mv_selection.csv',index=False,mode='x')
    summaries={'baseline':statistics(baseline_common)}
    historical=pd.read_csv(run/'results/historical_flat25088/vgg_scores.csv')
    old=raw[raw.variant==VARIANT].set_index('task')
    audit=[]
    for row in historical.itertuples(index=False):
        record={'task':row.task}
        for metric in ['lp_acc','rankme','sa_k']:
            record[metric+'_historical']=float(old.loc[row.task,metric])
            record[metric+'_rerun']=float(getattr(row,metric))
            record[metric+'_difference']=float(getattr(row,metric)-old.loc[row.task,metric])
        record['lp_exact_match']=abs(record['lp_acc_difference'])<1e-10
        record['rankme_relative_error']=abs(record['rankme_difference'])/record['rankme_historical']
        audit.append(record)
    audit=pd.DataFrame(audit)
    audit.to_csv(out/'historical_reproduction_audit.csv',index=False,mode='x')
    reproduced=bool(audit.lp_exact_match.all() and (audit.rankme_relative_error<2e-4).all() and (audit.sa_k_difference.abs()<2e-4).all())
    for arm in ARMS:
        replacement=pd.read_csv(run/'results'/arm/'vgg_scores.csv')
        updated=base.copy()
        for row in replacement.to_dict('records'):
            mask=(updated.variant==VARIANT)&(updated.task==row['task'])
            assert mask.sum()==1
            for key,value in row.items():
                if key in updated.columns and key not in ['arch','variant','task']:
                    updated.loc[mask,key]=value
        pd.testing.assert_frame_equal(updated[updated.variant!=VARIANT],base[base.variant!=VARIANT])
        assert set(updated.loc[updated.variant==VARIANT,'task'])==set(historical.task)
        updated.to_csv(out/f'{arm}_pool_b_scores.csv',index=False,mode='x')
        selected=selections(multi(updated))
        selected.to_csv(out/f'{arm}_mv_selection.csv',index=False,mode='x')
        selected.groupby('method')[['lp_acc','regret','evaluation_count','evaluation_count_reusing_canonical']].mean().to_csv(out/f'{arm}_mv_summary.csv',mode='x')
        common=common_rows(original,selected)
        common.to_csv(out/f'{arm}_common_support.csv',index=False,mode='x')
        summaries[arm]=statistics(common)
    assert sources=={p.name:digest(p) for p in (ROOT/'inputs').iterdir() if p.is_file()}
    save_json(out/'summary.json',dict(historical_reproduction_passed=reproduced,
        interpretation='VGG16-only replacement sensitivity. Other encoders retain historical extraction. Not a fully standardized model zoo. Ancillary baselines retain the pre-existing missing efficientnet_b2.ra_in1k/cifar10 score; rows flagged score_complete=false are not complete-pool comparisons.',
        historical_reproduction_thresholds=dict(lp_absolute=1e-10,rankme_relative=2e-4,sa_absolute=2e-4),
        source_hashes_unchanged=True,results=summaries))
    print(json.dumps({'historical_reproduction_passed':reproduced,'statistics':summaries},indent=2),flush=True)
    print('AGGREGATION_COMPLETE',str(out),flush=True)

if __name__=='__main__': main()
