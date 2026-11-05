"""Read immutable P0 scores; derive explicitly scoped reader-facing diagnostics."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
AGG=ROOT/'results/p0_vgg_pair_20261104/p0_aggregate_v2'
OUT=ROOT/'results/camera_ready_review_20261104'

def create():
    OUT.mkdir(exist_ok=True)
    source=AGG/'single_preprocess_4096_pool132_scores.csv'
    all_rows=pd.read_csv(source)
    counts=all_rows.groupby('arch').variant.nunique()
    f=all_rows[all_rows.arch.isin(counts[counts>=2].index)].copy()
    assert f.variant.nunique()==119 and (f.groupby('task').size()==119).all()
    canonical=f.groupby('arch').variant.min().to_dict()
    rows=[]
    for method,s1,s2 in [('two_stage_ssr_dagger_then_ssr_dagger','ssr_dagger','ssr_dagger'),
                         ('two_stage_rankme_uid_then_rankme','rankme_uid','rankme')]:
        for task,g in f.groupby('task'):
            reps=g[g.variant==g.arch.map(canonical)]
            assert len(reps)==35 and reps[s1].notna().all()
            group=reps.loc[reps[s1].idxmax(),'arch']
            family=g[g.arch==group]
            assert family[s2].notna().all()
            pick=family.loc[family[s2].idxmax()]
            rows.append(dict(task=task,method=method,stage1=s1,stage2=s2,arch=group,variant=pick.variant,
                lp_acc=float(pick.lp_acc),oracle=float(g.lp_acc.max()),regret=float(g.lp_acc.max()-pick.lp_acc),
                n_candidates=len(g),n_stage1=len(reps),n_stage2=len(family),
                evaluation_count=len(reps)+len(family),evaluation_count_reusing_canonical=len(reps)+len(family)-1))
    selection=pd.DataFrame(rows);selection.to_csv(OUT/'stage_diagnostics_selections.csv',index=False)
    summaries=[]
    for method,g in selection.groupby('method'):
        boot=np.random.default_rng(42).choice(g.regret.to_numpy(),size=(100000,len(g)),replace=True).mean(1)
        lo,hi=np.percentile(boot,[2.5,97.5])
        summaries.append(dict(method=method,mean_lp=g.lp_acc.mean(),mean_regret=g.regret.mean(),ci95_low=lo,ci95_high=hi,
            mean_evaluations=g.evaluation_count.mean(),mean_evaluations_reusing_canonical=g.evaluation_count_reusing_canonical.mean()))
    summary=pd.DataFrame(summaries).set_index('method');summary.to_csv(OUT/'stage_diagnostics_summary.csv')
    cor=[]
    for task,g in f.groupby('task'):
        for score in ('rankme','ssr_dagger'):
            cor.append(dict(task=task,score=score,rho_width=spearmanr(g[score],g.feat_dim).statistic,
                            rho_lp=spearmanr(g[score],g.lp_acc).statistic))
    correlation=pd.DataFrame(cor);correlation.to_csv(OUT/'score_correlations.csv',index=False)
    tv=sorted(f.loc[f.variant.str.contains('torchvision_tv')&~f.variant.str.startswith('vgg16.'),'variant'].unique())
    pd.DataFrame({'variant':tv,'status':'retained; historical extractor applies loader transform then weights transform; individual cache provenance incomplete'}).to_csv(OUT/'retained_torchvision_variants.csv',index=False)
    assert len(tv)==23
    ties=f[(f.task=='eurosat')&np.isclose(f.lp_acc,f[f.task=='eurosat'].lp_acc.max(),atol=1e-12)]
    ties[['task','arch','variant','lp_acc']].to_csv(OUT/'eurosat_oracle_ties.csv',index=False)
    other=pd.read_csv(AGG/'layer_only_4096_common22.csv')
    primary=pd.read_csv(AGG/'single_preprocess_4096_common22.csv')
    assert other[['pool','task']].equals(primary[['pool','task']])
    assert np.allclose(other[['rankme_regret','ssr_regret']],primary[['rankme_regret','ssr_regret']],atol=1e-12)
    files=[source,AGG/'layer_only_4096_common22.csv',AGG/'single_preprocess_4096_common22.csv',
           ROOT/'results/p0_vgg_pair_20261104/inputs/historical_extract_phase_e_torchvision.py',
           ROOT/'results/p0_vgg_pair_20261104/inputs/historical_datasets_local.py']
    report=dict(support=119,n_tasks=5,uid_score='log(rankme)/(uid*log(n_class)); not raw UID',
                diagnostic_timing='post-hoc cached-score comparisons, not new encoder training or confirmatory tuning',
                stage_summary=summary.reset_index().to_dict('records'),
                correlations=correlation.groupby('score')[['rho_width','rho_lp']].mean().to_dict('index'),
                retained_torchvision_count=len(tv),preprocessing_controls_same_headline=True,
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (OUT/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return summary

if __name__=='__main__':
    print(create().to_string())
