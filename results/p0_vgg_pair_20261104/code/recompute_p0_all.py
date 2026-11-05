#!/usr/bin/env python3
"""Both-VGG replacement audit on fixed 119/132 supports, preserving all inputs."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import recompute_vgg_results_v2 as prior

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'p0_aggregate_v2'
VARIANTS=['vgg16.torchvision_tv_v1','vgg16_bn.torchvision_tv_v1']
ARMS=['layer_only_4096','single_preprocess_4096']
CAPS=[128,256,512,768]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(frame,name): frame.to_csv(OUT/name,index=False,mode='x')
def js(name,obj):
    with (OUT/name).open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)
def safe_rho(x,y):
    use=np.isfinite(x)&np.isfinite(y)
    if use.sum()<3 or len(np.unique(np.asarray(x)[use]))<2 or len(np.unique(np.asarray(y)[use]))<2: return np.nan
    return float(spearmanr(np.asarray(x)[use],np.asarray(y)[use]).statistic)

def enrich(frame):
    frame=frame.copy()
    lr=np.log(np.maximum(frame.rankme,1e-12)); lc=np.log(frame.n_class)
    sa=np.maximum(frame.sa_k,1e-12)
    frame['rankme_norm_d']=lr-np.log(frame.feat_dim)
    frame['rankme_uid']=lr/(frame.uid*lc)
    frame['ssr_linear']=(lr-sa)/lc
    frame['ssr_inverse']=(lr-np.log(np.maximum(1-frame.sa_k,1e-12)))/lc
    frame['ssr_half_log']=(lr-.5*np.log(sa))/lc
    frame['ssr_double_log']=(lr-2*np.log(sa))/lc
    return frame

def selected(frame):
    sel=prior.selections(frame)
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

def diagnostics(frame):
    rows=[]
    for task,g in frame.groupby('task'):
        for score in ['rankme','ssr_dagger','rankme_uid']+[f'reff_trunc_{cap}' for cap in CAPS]:
            rows.append(dict(task=task,score=score,n_candidates=len(g),n_scored=int(g[score].notna().sum()),
                             rho_score_width=safe_rho(g[score],g.feat_dim),rho_score_lp=safe_rho(g[score],g.lp_acc),
                             rho_width_lp=safe_rho(g.feat_dim,g.lp_acc)))
    return pd.DataFrame(rows)

def within_family(frame):
    rows=[]
    for (arch,task),g in frame.groupby(['arch','task']):
        if len(g)<2: continue
        # Allow pairs for descriptive within-family correlations, record undefined ties.
        for metric in ['rankme','ssr_dagger','rankme_uid','logme','hscore']:
            use=g[[metric,'lp_acc']].dropna()
            rho=float(spearmanr(use[metric],use.lp_acc).statistic) if len(use)>=2 and use[metric].nunique()>1 and use.lp_acc.nunique()>1 else np.nan
            rows.append(dict(arch=arch,task=task,score=metric,n_variants=len(g),n_scored=len(use),rho=rho))
    return pd.DataFrame(rows)

def budget_sweep(frame):
    # Preserves audit_v16_extras.py's RNG seed, candidate order, and shared
    # sampled families across five tasks. Also measures actual, not K+4, cost.
    families=frame.arch.unique().tolist(); tasks=sorted(frame.task.unique())
    canon={a:sorted(frame[frame.arch==a].variant.unique())[0] for a in families}
    task_map={t:frame[frame.task==t] for t in tasks}
    canonical={t:{a:task_map[t][task_map[t].variant==canon[a]].iloc[0] for a in families} for t in tasks}
    rng=np.random.default_rng(42); rows=[]; draws=[]
    for k in [1,3,5,10,15,20,25,30,35]:
        for rep in range(100):
            subset=list(rng.choice(families,size=min(k,len(families)),replace=False))
            draws.append(dict(budget=k,repeat=rep,families=subset))
            for task in tasks:
                # Historical Python max preserves candidate order on ties.
                best_arch=max(subset,key=lambda a:float(canonical[task][a].hscore))
                group=task_map[task]; g=group[group.arch==best_arch]
                winner=g.loc[g.rankme.idxmax()]
                rows.append(dict(budget=k,repeat=rep,task=task,selected_arch=best_arch,variant=winner.variant,
                                 lp_acc=float(winner.lp_acc),regret=float(group.lp_acc.max()-winner.lp_acc),
                                 historical_cost=k+4,actual_cost=k+len(g),reuse_cost=k+len(g)-1,
                                 labeled_architecture_evaluations=k))
    detail=pd.DataFrame(rows); summaries=[]
    for k,g in detail.groupby('budget'):
        repeat_lp=g.groupby('repeat').lp_acc.mean()
        summaries.append(dict(budget=k,mean_lp=g.lp_acc.mean(),sd_lp=float(repeat_lp.std(ddof=0)),
                              regret=g.regret.mean(),historical_cost=k+4,actual_cost=g.actual_cost.mean(),reuse_cost=g.reuse_cost.mean()))
    return detail,pd.DataFrame(summaries),draws

def variance(frame):
    # Evaluate only the retained map/function, not the original module's I/O.
    tree=ast.parse((ROOT/'p0_inputs/audit_P1_variance_decomp.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='variance_decomp'
           or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FAMILY_MAP' for t in n.targets)]
    ns={'np':np,'pd':pd}; exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_variance_function','exec'),ns)
    frame=frame.copy(); frame['family']=frame.arch.map(lambda a:ns['FAMILY_MAP'].get(a,'Unknown'))
    frame['rankme_log']=np.log(frame.rankme); frame['scr_star']=frame.rankme_log/np.log(frame.n_class)
    rows=[]
    for col in ['lp_acc','rankme_log','scr_star','ssr_dagger']:
        r=ns['variance_decomp'](frame,col); r.pop('per_task'); rows.append(dict(score=col,**r))
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(exist_ok=False)
    inputs={str(p.relative_to(ROOT)):sha(p) for folder in ['inputs','p0_inputs'] for p in (ROOT/folder).iterdir() if p.is_file() and not p.name.startswith('._')}
    raw=pd.read_csv(ROOT/'inputs/pool_b_final_score_complete.csv')
    trunc=pd.read_csv(ROOT/'p0_inputs/truncation_ablation.csv')
    assert not raw.duplicated(['variant','task']).any()
    assert not trunc.duplicated(['variant','task']).any()
    assert trunc.groupby('variant').feat_dim.nunique().max()==1
    dimmap=trunc.groupby('variant').feat_dim.first()
    base=prior.attach_official(raw).merge(trunc[['variant','task','feat_dim']+[f'reff_trunc_{c}' for c in CAPS]],
                                         on=['variant','task'],how='left',validate='one_to_one',sort=False)
    base.feat_dim=base.feat_dim.fillna(base.variant.map(dimmap))
    assert not base.feat_dim.isna().any()
    original=pd.read_csv(ROOT/'inputs/common_support_per_task.csv')
    all_stats={}; lineage=[]; history=[]; coverage=[]
    for arch in ['vgg16','vgg16_bn']:
        run=ROOT/'runs'/f'{arch}_p0_v2'
        hist=pd.read_csv(run/'results/historical_flat25088/vgg_scores.csv')
        for row in hist.to_dict('records'):
            old=raw[(raw.variant==row['variant'])&(raw.task==row['task'])].iloc[0]
            history.append(dict(variant=row['variant'],task=row['task'],original_lp=old.lp_acc,rerun_lp=row['lp_acc'],
                                lp_difference=row['lp_acc']-old.lp_acc,rankme_relative_error=abs(row['rankme']-old.rankme)/old.rankme,
                                sa_absolute_error=abs(row['sa_k']-old.sa_k),cache=row['target_cache'],cache_sha256=row['target_cache_sha256']))
    csv(pd.DataFrame(history),'historical_reproduction.csv')
    for arm in ['frozen_original']+ARMS:
        frame=base.copy()
        if arm!='frozen_original':
            for arch in ['vgg16','vgg16_bn']:
                run=ROOT/'runs'/f'{arch}_p0_v2'; source=run/'results'/arm/'vgg_scores.csv'
                replacement=pd.read_csv(source)
                assert len(replacement)==5
                for row in replacement.to_dict('records'):
                    mask=(frame.variant==row['variant'])&(frame.task==row['task']); assert mask.sum()==1
                    for key,val in row.items():
                        if key in frame.columns and key not in ['arch','variant','task']: frame.loc[mask,key]=val
                    frame.loc[mask,'feat_dim']=row['feature_dim']
                    lp_path=run/'results'/arm/f'lp_{row["task"]}.json'
                    lp=json.loads(lp_path.read_text())
                    assert lp['target_cache_sha256']==row['target_cache_sha256']
                    assert np.mean(np.array(lp['test_predictions'])==np.array(lp['test_labels']))==row['lp_acc']
                    lineage.append(dict(arm=arm,variant=row['variant'],task=row['task'],feature_dim=row['feature_dim'],
                                        target_cache=row['target_cache'],reference_cache=row['reference_cache'],
                                        target_sha256=row['target_cache_sha256'],reference_sha256=row['reference_cache_sha256'],
                                        scoring_script_sha256=row['scorer_sha256'],score_table=str(source),score_table_sha256=sha(source),
                                        lp_record=str(lp_path),lp_record_sha256=sha(lp_path),completed_utc=row['completed_utc']))
            pd.testing.assert_frame_equal(frame[~frame.variant.isin(VARIANTS)],base[~base.variant.isin(VARIANTS)])
            assert set(frame[frame.variant.isin(VARIANTS)].feat_dim)=={4096}
        frame=enrich(frame); csv(frame,arm+'_pool132_scores.csv')
        multi=prior.multi(frame)
        subsets={'full132':frame,'multi119':multi,'multi119_without_vgg':multi[~multi.arch.str.startswith('vgg16')],
                 'full132_without_vgg':frame[~frame.arch.str.startswith('vgg16')]}
        for label,sub in subsets.items():
            sel=selected(sub); csv(sel,f'{arm}_{label}_selection.csv')
            csv(method_summary(sel),f'{arm}_{label}_summary.csv')
            diag=diagnostics(sub); csv(diag,f'{arm}_{label}_correlations.csv')
            csv(sub.groupby(['arch','feat_dim'],as_index=False).agg(rankme=('rankme','mean'),lp_acc=('lp_acc','mean')),
                f'{arm}_{label}_width_plot_data.csv')
            for col in ['uid','logme','hscore','leep','ncti']+[f'reff_trunc_{c}' for c in CAPS]:
                missing=sub[sub[col].isna()]
                for row in missing.itertuples(): coverage.append(dict(arm=arm,support=label,metric=col,variant=row.variant,task=row.task))
            if label=='multi119':
                common=prior.common_rows(original,sel); csv(common,arm+'_common22.csv'); all_stats[arm]=prior.statistics(common)
                if arm=='frozen_original': assert np.allclose(common[['ssr_regret','rankme_regret']],original[['ssr_regret','rankme_regret']],atol=1e-12)
        csv(within_family(multi),arm+'_within_family.csv')
        csv(variance(multi),arm+'_historical_variance_diagnostic.csv')
        detail,budget,draws=budget_sweep(multi)
        csv(detail,arm+'_budget_detail.csv'); csv(budget,arm+'_budget_summary.csv'); js(arm+'_budget_draws.json',draws)
        # Explicit membership; do not conflate with undocumented 78/33-model historical diagnostics.
        transformer=multi.arch.str.contains('vit|deit|swin|beit|mae|dino',case=False,regex=True)
        for label,mask in [('cnn_explicit',~transformer),('transformer_explicit',transformer)]:
            sub=multi[mask]; csv(sub[['variant','arch','task']],arm+'_'+label+'_support.csv')
            csv(diagnostics(sub),arm+'_'+label+'_correlations.csv')
    csv(pd.DataFrame(lineage),'affected_row_lineage.csv'); csv(pd.DataFrame(coverage),'inherited_missing_scores.csv')
    assert inputs=={str(p.relative_to(ROOT)):sha(p) for folder in ['inputs','p0_inputs'] for p in (ROOT/folder).iterdir() if p.is_file() and not p.name.startswith('._')}
    js('summary.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),primary_arm='single_preprocess_4096',
                          primary_choice_reason='One official weights transform, chosen by protocol correctness rather than performance',
                          repaired_variants=VARIANTS,common22=all_stats,inputs_unchanged=True,
                          scope='Only two torchvision VGG variants replaced; all other candidate scores frozen.',
                          protocol=dict(svd_centered=False,k=20,lp_seed=0,sample_seed=42,bootstrap_seed=42,n_boot=100000),
                          warnings=['Historical exact cache hashes/timestamps remain unrecovered; old affected rows are retired, not retroactively certified.',
                                    'Auxiliary missing cell efficientnet_b2.ra_in1k/cifar10 persists; any score_complete=false outputs are partial diagnostics.',
                                    'Non-VGG width/truncation metadata inherited, not newly feature-verified.',
                                    '132/119 supports and their oracles are separate; never mix their regrets.']))
    js('source_manifest.json',dict(inputs=inputs,aggregate_script_sha256=sha(Path(__file__)),
                                  helper_sha256=sha(ROOT/'code/recompute_vgg_results_v2.py')))
    print('P0_AGGREGATE_COMPLETE',json.dumps(all_stats),flush=True)

if __name__=='__main__': main()
