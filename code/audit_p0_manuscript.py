"""Independent P0-to-manuscript numeric and bilingual checks; no data mutation."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,spearmanr
ROOT=Path(__file__).resolve().parents[1]
P0=ROOT/'results/p0_vgg_pair_20261104'
AGG=P0/'p0_aggregate_v2'
OUT=ROOT/'results/camera_ready_p0_20261104'
ARM='single_preprocess_4096'
def main():
    checks=[]
    def check(name,passed):
        assert passed,name
        checks.append(name)
    # The public release contains final analysis inputs, not intermediate runs.
    # Verify every released input; retain full-bundle checks for local-only copies.
    manifest_path=P0/'PUBLIC_RELEASE_MANIFEST.json'
    if not manifest_path.exists():
        manifest_path=P0/'P0_BUNDLE_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    for filename,v in manifest['files'].items():check('immutable:'+filename,hashlib.sha256((P0/filename).read_bytes()).hexdigest()==v['sha256'])
    values=json.loads((OUT/'paper_values.json').read_text())
    for filename,h in values['inputs'].items():check('source:'+filename,hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==h)
    common=pd.read_csv(AGG/f'{ARM}_common22.csv');frame=pd.read_csv(AGG/f'{ARM}_pool132_scores.csv')
    counts=frame.groupby('arch').variant.nunique();multi=frame[frame.arch.isin(counts[counts>=2].index)]
    check('119 fixed candidates',multi.variant.nunique()==119 and (multi.groupby('task').size()==119).all())
    check('22 fixed units',len(common)==22 and common.task.nunique()==10 and 'stl10' not in common.task.values)
    for task,g in multi.groupby('task'):
        u=common[(common.pool=='MV')&(common.task==task)].iloc[0];oracle=g.lp_acc.max()
        check(task+':oracle',abs(u.oracle-oracle)<1e-12)
        for score,col in [('rankme','rankme_regret'),('ssr_dagger','ssr_regret')]:
            check(task+':'+score,abs(oracle-g.loc[g[score].idxmax(),'lp_acc']-u[col])<1e-12)
        check(task+':random',abs(oracle-g.lp_acc.mean()-u.random_regret)<1e-12)
    d=np.round((common.ssr_regret-common.rankme_regret).to_numpy(),10)
    check('headline means',abs(d.mean()+.06454545454545454)<1e-12)
    boot=np.random.default_rng(42).choice(d,size=(100000,22),replace=True).mean(1)
    targets=sorted(common.task.unique());sums=np.array([d[common.task==t].sum() for t in targets]);counts=np.array([(common.task==t).sum() for t in targets])
    draw=np.random.default_rng(42).choice(10,size=(100000,10),replace=True);cluster=sums[draw].sum(1)/counts[draw].sum(1)
    stats=values['headline']
    check('unit CI',np.allclose(np.percentile(boot,[2.5,97.5]),stats['unit_bootstrap']['ci95'],atol=1e-12))
    check('cluster CI',np.allclose(np.percentile(cluster,[2.5,97.5]),stats['target_cluster_bootstrap']['ci95'],atol=1e-12))
    p=2*min(np.mean(cluster>=0),np.mean(cluster<=0));check('cluster p',abs(p-.00194)<1e-12)
    w=wilcoxon(d,zero_method='wilcox',method='approx',alternative='two-sided',correction=False)
    check('Wilcoxon',abs(w.pvalue-stats['wilcoxon_p'])<1e-10)
    for label,sub in [('aux118',multi[multi.variant!='efficientnet_b2.ra_in1k']),('aux115_without_vgg',multi[(multi.variant!='efficientnet_b2.ra_in1k')&~multi.arch.str.startswith('vgg16')])]:
        expected_n=118 if label=='aux118' else 115;check(label+':support',(sub.groupby('task').size()==expected_n).all())
        saved=pd.read_csv(OUT/f'{label}_summary.csv').set_index('method')
        for metric in ['rankme','ssr_dagger','logme','hscore','ncti','leep','rankme_norm_d','ssr_linear']:
            check(label+':complete:'+metric,sub[metric].notna().all())
            regrets=[];acc=[]
            for _,g in sub.groupby('task'):
                pick=g.loc[g[metric].idxmax()];acc.append(pick.lp_acc);regrets.append(g.lp_acc.max()-pick.lp_acc)
            check(label+':mean:'+metric,abs(np.mean(regrets)-saved.loc[metric,'mean_regret'])<1e-12 and abs(np.mean(acc)-saved.loc[metric,'mean_lp'])<1e-12)
        check(label+':exact random',abs(saved.loc['random_exact','mean_lp']-sub.groupby('task').lp_acc.mean().mean())<1e-12)
        for task,g in sub.groupby('task'):
            additive=(np.log(np.maximum(g.rankme,1e-12))+1-np.maximum(g.sa_k,1e-12))/np.log(g.n_class)
            check(label+task+':additive constant',np.allclose(additive-g.ssr_linear,1/np.log(g.n_class),atol=1e-12))
            check(label+task+':additive selection',additive.idxmax()==g.ssr_linear.idxmax())
    # Language-specific tables differ only in text, not their mathematical values.
    for en in (ROOT/'paper/p0_tables').glob('*_en.tex'):
        zh=en.with_name(en.name.replace('_en.tex','_zh.tex'))
        nums=lambda p:re.findall(r'\$([^$]+)\$',p.read_text())
        check('bilingual:'+en.stem,nums(en)==nums(zh))
    for filename in ['main_v18.tex','main_v18_zh.tex']:
        source=(ROOT/'paper'/filename).read_text()
        version_notes=re.findall(r'% BEGIN VERSION NOTE.*?% END VERSION NOTE',source,re.S)
        check(filename+':single correction note',len(version_notes)==1 and '-0.1043' in version_notes[0])
        source=re.sub(r'% BEGIN VERSION NOTE.*?% END VERSION NOTE','',source,flags=re.S)
        for old in ['0.319','0.173','0.1043','0.00118','0.00254','0.815','-0.538','-0.26']:
            check(filename+':no stale '+old,old not in source)
        for current in ['0.133','0.069','-0.0645','0.00194','0.00328','0.1440','-0.0460','0.388','0.527','0.519','0.058','0.101','0.305','0.512']:
            check(filename+':current '+current,current in source)
        check(filename+':VGG repair disclosure','25088' in source and 'L20' in source and '1.2.2' in source)
        for phrase in ['withdrawn','no longer','earlier claim','撤回','不再','修正后']:
            check(filename+':reader voice '+phrase,phrase not in source)
        check(filename+':checklist last',source.index(r'\section{Experimental Environment' if filename=='main_v18.tex' else r'\section{实验环境') < source.index(r'\input{checklist'))
        check(filename+':k scope',('Sensitivity to $k$ has not been established' if filename=='main_v18.tex' else '敏感性尚未验证') in source)
        check(filename+':no k plateau',all(x not in source for x in ['stable range','稳定范围','稳定区间']))
        check(filename+':additive result','0.117' in source and '1/\\log C' in source)
        check(filename+':TwoNN main citation',r'\citep{facco2017estimating}' in source.split('TwoNN')[1].split('\n')[0])
        check(filename+':strict isotropy','近各向同性条件下' not in source)
        check(filename+':regret wording','below random (' not in source)
    for lang in ['en','zh']:
        table=(ROOT/'paper/p0_tables'/f'two_stage_{lang}.tex').read_text()
        check(lang+':eight consistent strategy labels',re.findall(r'^S(\d) ',table,re.M)==[str(i) for i in range(1,9)])
        check(lang+':label-dependent strategy',('labels;' if lang=='en' else '有标签;') in table)
    a118=pd.read_csv(OUT/'aux118_summary.csv').set_index('method')
    a119=pd.read_csv(AGG/f'{ARM}_multi119_summary.csv').set_index('method')
    check('distinct random rounding',f"{a118.loc['random_exact','mean_lp']:.3f}"=='0.629' and f"{a119.loc['random_exact','mean_lp']:.3f}"=='0.630')
    check('additive regret 0.117',abs(a118.loc['ssr_linear','mean_regret']-.117)<1e-12)
    diag=ROOT/'results/camera_ready_review_20261104'
    diagnostic=json.loads((diag/'diagnostic_summary.json').read_text())
    for filename,h in diagnostic['sources'].items():
        check('diagnostic source:'+filename,hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==h)
    reported=pd.read_csv(diag/'stage_diagnostics_selections.csv')
    summ=pd.read_csv(diag/'stage_diagnostics_summary.csv').set_index('method')
    for method,stage1,stage2 in [('two_stage_rankme_uid_then_rankme','rankme_uid','rankme'),('two_stage_ssr_dagger_then_ssr_dagger','ssr_dagger','ssr_dagger')]:
        regrets=[]
        for task,g in multi.groupby('task'):
            reps=g.sort_values('variant').drop_duplicates('arch')
            check(method+task+':canonical completeness',len(reps)==35 and reps[stage1].notna().all())
            chosen_arch=reps.loc[reps[stage1].idxmax(),'arch'];group=g[g.arch==chosen_arch]
            pick=group.loc[group[stage2].idxmax()]
            recorded=reported[(reported.method==method)&(reported.task==task)].iloc[0]
            regret=g.lp_acc.max()-pick.lp_acc;regrets.append(regret)
            check(method+task+':selection',recorded.variant==pick.variant and abs(recorded.regret-regret)<1e-12)
            check(method+task+':cost',recorded.evaluation_count==35+len(group) and recorded.evaluation_count_reusing_canonical==34+len(group))
        check(method+':regret',abs(np.mean(regrets)-summ.loc[method,'mean_regret'])<1e-12)
        ci=np.percentile(np.random.default_rng(42).choice(regrets,size=(100000,5)).mean(1),[2.5,97.5])
        check(method+':CI',np.allclose(ci,summ.loc[method,['ci95_low','ci95_high']].to_numpy(dtype=float),atol=1e-12))
    cor=pd.read_csv(diag/'score_correlations.csv')
    for (task,score),g in cor.groupby(['task','score']):
        f=multi[multi.task==task]
        check(task+score+':correlation',abs(g.iloc[0].rho_width-spearmanr(f[score],f.feat_dim).statistic)<1e-12 and abs(g.iloc[0].rho_lp-spearmanr(f[score],f.lp_acc).statistic)<1e-12)
    tv=multi[multi.variant.str.contains('torchvision_tv')&~multi.variant.str.startswith('vgg16.')]
    check('23 retained torchvision variants',tv.variant.nunique()==23)
    eu=multi[multi.task=='eurosat'];ties=eu[eu.lp_acc==eu.lp_acc.max()]
    check('EuroSAT tied oracle',len(ties)==2 and abs(ties.lp_acc.iloc[0]-.955)<1e-12)
    other=pd.read_csv(AGG/'layer_only_4096_common22.csv')
    check('VGG controls same headline',np.allclose(common[['rankme_regret','ssr_regret']],other[['rankme_regret','ssr_regret']],atol=1e-12))
    answers=[]
    for filename in ['checklist.tex','checklist_zh.tex']:
        source=(ROOT/'paper'/filename).read_text()
        answers.append(re.findall(r'(?:Answer:|回答：)\s*\\answer(Yes|No|NA)\{\}',source))
    check('checklist answer parity',answers[0]==answers[1] and len(answers[0])>=15)
    report=dict(passed=True,n_checks=len(checks),checks=checks,mean_ssr=float(common.ssr_regret.mean()),mean_rankme=float(common.rankme_regret.mean()),paired_difference=float(d.mean()),cluster_p=float(p),wilcoxon_p=float(w.pvalue),
                limits=['Frozen non-VGG rows were not re-extracted.','Raw repair caches remain on server; historical provenance incomplete.','Auxiliary comparisons use explicit 118/115 supports, not 119.','k sensitivity is not established under the final evaluation protocol.'])
    (OUT/'numeric_acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
