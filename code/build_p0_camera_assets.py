"""Generate camera-ready tables/figures from immutable P0 results (no raw edits)."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_followup_diagnostics import create as followup_diagnostics

ROOT=Path(__file__).resolve().parents[1]
P0=ROOT/'results/p0_vgg_pair_20261104'
AGG=P0/'p0_aggregate_v2'
OUT=ROOT/'results/camera_ready_p0_20261104'
TEX=ROOT/'paper/p0_tables'
ARM='single_preprocess_4096'
import final_selection as calc
matplotlib.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9})

def read(name): return pd.read_csv(AGG/f'{ARM}_{name}.csv')
def table(name,lang,caption,headers,rows,label=None):
    label=label or 'tab:'+name+('_zh' if lang=='zh' else '')
    placement='!ht' if name=='two_stage' and lang=='zh' else 'htbp'
    lines=[r'\begin{table}['+placement+']',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',
           r'\setlength{\tabcolsep}{4pt}',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',
           ' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}','']
    (TEX/f'{name}_{lang}.tex').write_text('\n'.join(lines))
def num(x,d=3):return f'${x:.{d}f}$'
def fmtci(lo,hi):return f'$[{lo:.3f},{hi:.3f}]$'
def savefig(fig,name):
    fig.tight_layout();fig.savefig(ROOT/'paper'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'paper'/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)
def main():
    OUT.mkdir(exist_ok=True);TEX.mkdir(exist_ok=True)
    frame=read('pool132_scores'); multi=calc.multi(frame)
    common=read('common22');summary=read('multi119_summary').set_index('method');sel=read('multi119_selection')
    extra=followup_diagnostics()
    summary=pd.concat([summary.drop(index=extra.index,errors='ignore'),extra])
    stats=json.loads((AGG/'summary.json').read_text())['common22'][ARM]
    # Main zero-label support is not changed to accommodate a missing auxiliary score.
    aux=multi[multi.variant!='efficientnet_b2.ra_in1k']
    aux_without=aux[~aux.arch.str.startswith('vgg16')]
    auxiliary={}
    for name,f in [('aux118',aux),('aux115_without_vgg',aux_without)]:
        s=calc.selected(f);assert s.score_complete.all()
        s.to_csv(OUT/f'{name}_selections.csv',index=False)
        auxiliary[name]=calc.method_summary(s).set_index('method')
        auxiliary[name].to_csv(OUT/f'{name}_summary.csv')
    vtab=pd.read_csv(ROOT/'results/camera_ready/vtab_reference_24arch_uncentered/vtab_reference_audit_cells.csv')
    dims=pd.read_csv(ROOT/'results/camera_ready/vtab_24arch_8tasks_lp_complete.csv')[['arch','task','feat_dim']]
    vtab=vtab.merge(dims,on=['arch','task'],validate='one_to_one')
    ex=pd.read_csv(ROOT/'results/camera_ready/extended_full_sota_score_complete.csv')
    diagnosis=[]
    for name,f,rcol in [('MV',multi,'rankme'),('VTAB',vtab[vtab.task!='stl10'],'reff_rankme'),('EX',ex,'rankme')]:
        correlations=[(spearmanr(g[rcol],g.feat_dim).statistic,spearmanr(g.feat_dim,g.lp_acc).statistic) for _,g in f.groupby('task')]
        units=common[common.pool==name]
        diagnosis.append(dict(pool=name,n_arch=f.arch.nunique(),n_variants=f.variant.nunique() if 'variant' in f else f.arch.nunique(),
                              n_tasks=f.task.nunique(),rho_width=float(np.mean([x[0] for x in correlations])),
                              rho_quality=float(np.mean([x[1] for x in correlations])),rm=units.rankme_regret.mean(),random=units.random_regret.mean()))
    plot_frames={
        'VTAB':vtab.groupby('arch',as_index=False).agg(reff=('reff_rankme','mean'),d=('feat_dim','first')),
        'EX':ex.groupby('arch',as_index=False).agg(reff=('rankme','mean'),d=('feat_dim','first'))}
    figure_rhos={}
    fig,ax=plt.subplots(figsize=(5.1,2.8))
    for (name,f),color,marker in zip(plot_frames.items(),['#c76e0a','#287f57'],['s','^']):
        rho=float(spearmanr(f.reff,f.d).statistic);figure_rhos[name]=rho
        f.to_csv(OUT/f'figure_width_{name.lower()}.csv',index=False)
        ax.scatter(f.d,f.reff,label=rf'{name} ($\rho_{{Sp}}={rho:+.2f}$)',c=color,marker=marker,s=28,alpha=.8)
    x=np.array([192,4096]);ax.plot(x,.22*x,'k--',lw=.8,alpha=.5,label='slope-1 guide')
    ax.set(xscale='log',yscale='log',xlabel='Feature dimension $d$',ylabel=r'RankMe effective rank $r_{\mathrm{eff}}$')
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=8);savefig(fig,'fig_reff_d_scatter')
    methods=['random_exact','rankme','ssr_dagger','two_stage_rankme_then_rankme','two_stage_ssr_dagger_then_rankme','two_stage_ssr_dagger_then_ssr_dagger','two_stage_rankme_uid_then_rankme','two_stage_hscore_then_rankme']
    labels=['Random','Flat RankMe',r'Flat SSR$^\ddagger$',r'RankMe $\to$ RankMe',r'SSR$^\ddagger\to$ RankMe',r'SSR$^\ddagger\to$ SSR$^\ddagger$',r'$R_{\rm UID}\to$ RankMe',r'H-score $\to$ RankMe (labels)']
    colors=['#777777','#c53b35','#287db0','#8a66ad','#26835d','#c78b23','#166a76','#bf5795']
    fig,ax=plt.subplots(figsize=(5.0,3.5))
    for method,label,color,marker in zip(methods,labels,colors,['s','^','D','v','*','x','P','o']):
        r=summary.loc[method];x=r.mean_evaluations
        ax.errorbar(x,r.mean_regret,yerr=[[r.mean_regret-r.ci95_low],[r.ci95_high-r.mean_regret]],fmt=marker,c=color,label=label,capsize=3,ms=8,lw=1)
    ax.set(xlabel='Candidate evaluations per task (no reuse)',ylabel='Mean regret',xlim=(-5,128),ylim=(-.01,.36))
    ax.grid(alpha=.2);ax.legend(fontsize=7.5,loc='upper center',ncol=2);savefig(fig,'fig_pareto')
    budget=read('budget_summary');fig,ax=plt.subplots(figsize=(5.0,3.5))
    ax.plot(budget.budget,budget.regret,'o-',c='#8057a4',label='H-score then RankMe')
    ax.axhline(summary.loc['two_stage_ssr_dagger_then_rankme','mean_regret'],color='#26835d',ls='--',label=r'Zero-label SSR$^\ddagger\to$ RankMe')
    ax.axhline(summary.loc['two_stage_rankme_uid_then_rankme','mean_regret'],color='#166a76',ls=':',label=r'Zero-label $R_{\rm UID}\to$ RankMe')
    ax.set(xlabel='Stage-1 labeled architecture evaluations $K$',ylabel='Mean regret',xlim=(0,36),ylim=(0,.175))
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
    for k in [5,35]:
        r=budget[budget.budget==k].iloc[0]
        offset=(2,-15) if k==5 else (-5,10)
        ax.annotate(f'{r.regret:.3f}',(k,r.regret),xytext=offset,textcoords='offset points',fontsize=9)
    savefig(fig,'fig_costsplit')
    for lang in ['en','zh']:
        zh=lang=='zh'
        table('pool_diagnosis',lang,
              '各池的宽度与质量诊断。MV 为 119 个候选；VTAB 排除 STL-10。相关为逐任务 Spearman 的均值。' if zh else
              'Width and quality diagnostics by pool. MV uses 119 candidates; VTAB excludes STL-10. Correlations are per-task Spearman means.',
              ['池' if zh else 'Pool','架构/任务' if zh else 'Groups/tasks',r'$\rho(\mathrm{RankMe},d)$',r'$\rho(d,\mathrm{LP})$','RM 遗憾' if zh else 'RM regret','随机遗憾' if zh else 'Random regret'],
              [[r['pool'],f"${r['n_arch']}/{r['n_tasks']}$",num(r['rho_width']),num(r['rho_quality']),num(r['rm']),num(r['random'])] for r in diagnosis])
        per=[]
        names=dict(cifar10='CIFAR-10',cifar100='CIFAR-100',dtd='DTD',eurosat='EuroSAT',svhn='SVHN')
        for task,g in multi.groupby('task'):
            s=sel[sel.task==task].set_index('method');winner=g.loc[g.lp_acc.idxmax()]
            per.append([names[task],num(winner.lp_acc),num(g.lp_acc.mean()),num(s.loc['rankme','lp_acc']),num(s.loc['ssr_dagger','lp_acc']),
                        str(g.loc[g.rankme.idxmax(),'arch']).replace('_',r'\_'),str(g.loc[g.ssr_dagger.idxmax(),'arch']).replace('_',r'\_'),winner.arch.replace('_',r'\_')+(' (tie)' if task=='eurosat' else '')])
        table('main_pertask',lang,r'MV 119 候选逐任务 LP。随机列为精确期望。\SSRdag 在四个任务优于 RankMe，在 SVHN 更差。EuroSAT 另有 ViT-large 并列 oracle（LP 0.955）。' if zh else
              r'Per-task LP on 119 MV candidates. Random is the exact expectation. \SSRdag improves over RankMe on four tasks, but is worse on SVHN. EuroSAT also has a tied ViT-large oracle (LP 0.955).',
              ['任务' if zh else 'Task','Oracle','随机' if zh else 'Random','RM',r'\SSRdag','RM 选择' if zh else 'RM group',r'\SSRdag 选择' if zh else r'\SSRdag group','Oracle 组' if zh else 'Oracle group'],per)
        for name,metric in [('sota_flat','mean_regret'),('vgg_sensitivity','mean_lp')]:
            ms=['random_exact','rankme','logme','hscore','ncti','ssr_dagger','leep']
            ms.insert(2,'rankme_norm_d')
            naming={'random_exact':'随机' if zh else 'Random','rankme':'RankMe','logme':'LogME','hscore':'H-score','ncti':'NCTI','ssr_dagger':r'\SSRdag','leep':'LEEP 聚类代理' if zh else 'LEEP clustering proxy','rankme_norm_d':r'RankMe $(-\log d)$'}
            rows=[]
            for m in ms:
                a=auxiliary['aux118'].loc[m,metric];b=auxiliary['aux115_without_vgg'].loc[m,metric]
                rows.append([naming[m],num(a),num(b),num(b-a),('无标签' if zh else 'zero-label') if m in ['rankme','ssr_dagger','rankme_norm_d'] else ('参照' if zh else 'reference') if m=='random_exact' else ('有标签' if zh else 'labels')])
            caption=('辅助方法的完整共同支持：剔除一个缺项变体后为 118 个候选，再移除三个 VGG 变体后为 115 个。数值为五任务平均'+('遗憾。' if metric=='mean_regret' else ' LP。')) if zh else ('Complete auxiliary support: 118 candidates after excluding one variant with a missing score; 115 after also removing three VGG variants. Entries are five-task mean '+('regret.' if metric=='mean_regret' else 'LP.'))
            table(name,lang,caption,['方法' if zh else 'Method','含 VGG' if zh else 'With VGG','不含 VGG' if zh else 'Without VGG',r'$\Delta$','类型' if zh else 'Type'],rows)
        rows=[]
        for m,label in zip(methods,labels):
            r=summary.loc[m]
            lab={'random_exact':'S1 Random','rankme':'S2 Flat-RankMe','ssr_dagger':r'S3 Flat-\SSRdag','two_stage_rankme_then_rankme':r'S4 RankMe$\to$RankMe','two_stage_ssr_dagger_then_rankme':r'S5 \SSRdag$\to$RankMe','two_stage_ssr_dagger_then_ssr_dagger':r'S6 \SSRdag$\to$\SSRdag','two_stage_rankme_uid_then_rankme':r'S7 $R_{\mathrm{UID}}\to$RankMe','two_stage_hscore_then_rankme':r'S8 H-score$\to$RankMe (labels; $K=35$)'}[m]
            if zh: lab=lab.replace('labels;', '有标签;')
            rows.append([lab,num(r.mean_lp),num(r.mean_regret),fmtci(r.ci95_low,r.ci95_high),f'${r.mean_evaluations:g}/{r.mean_evaluations_reusing_canonical:g}$'])
        table('two_stage',lang,'MV 119 候选、五任务流程。区间为 100000 次任务 bootstrap。评估次数分别为不复用/复用规范代表；不含参考初始化。K 是使用标签的架构评估次数。' if zh else
              'Five-task MV pipelines on 119 candidates. CIs use 100,000 task bootstrap resamples. Counts are without/with canonical reuse, excluding reference setup. K counts labeled architecture evaluations.',
              ['策略' if zh else 'Strategy','平均 LP' if zh else 'Mean LP','遗憾' if zh else 'Regret',r'$95\%$ CI','次数' if zh else 'Evaluations'],rows)
    pd.DataFrame(diagnosis).to_csv(OUT/'pool_diagnosis.csv',index=False)
    # Numeric audit payload drives both language versions; input hashes prevent stale data.
    source_files=[AGG/f'{ARM}_{n}.csv' for n in ['pool132_scores','multi119_summary','multi119_selection','common22','budget_summary']]
    source_files += [ROOT/'results/camera_ready/vtab_reference_24arch_uncentered/vtab_reference_audit_cells.csv',ROOT/'results/camera_ready/vtab_24arch_8tasks_lp_complete.csv',ROOT/'results/camera_ready/extended_full_sota_score_complete.csv']
    payload=dict(headline=stats,random_mean_regret=float(common.random_regret.mean()),figure_width_rhos=figure_rhos,pool_diagnosis=diagnosis,
                 inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (OUT/'paper_values.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
