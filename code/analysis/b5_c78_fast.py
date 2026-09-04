"""
Fast B5/C7/C8 using cached SVDs — skips recomputing LP (use existing vtab_ssr_results.csv).

B5: Decode vtab_ssr_results.csv (corrupted arch/task names) → vtab_25arch_8tasks.csv
C7: SA_k with 3 reference tasks (stl10, cifar10, flowers102)
C8: SA_k at k ∈ {5, 10, 20, 50, 100}
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
from metrics.reff import effective_rank_from_singular_values

FEAT = ROOT / "results" / "vtab_features"
OUT  = Path(__file__).resolve().parent
EPS  = 1e-12

ARCHS = [
    'convnext_base','convnext_small','convnext_tiny',
    'deit_small_patch16_224','deit_tiny_patch16_224',
    'densenet121','densenet169',
    'efficientnet_b0','efficientnet_b2','efficientnet_b3',
    'mobilenetv3_large_100','mobilenetv3_small_100',
    'regnetx_032','regnety_032',
    'resnet18','resnet34','resnet50','resnet101','resnet152',
    'swin_small_patch4_window7_224','swin_tiny_patch4_window7_224',
    'vit_base_patch16_224','vit_small_patch16_224','vit_tiny_patch16_224',
    'wide_resnet50_2',
]
TASKS8 = ['cifar10','cifar100','dtd','fashion_mnist',
          'flowers102','pets','stl10','svhn']
TASKS7 = [t for t in TASKS8 if t != 'stl10']  # stl10 used as ref
REF_TASKS = ['stl10', 'cifar10', 'flowers102']
K_VALUES  = [5, 10, 20, 50, 100]


def load_Hy(arch, task):
    d = np.load(FEAT / f"{arch}__{task}.npz", allow_pickle=True)
    if 'features' in d: return d['features'].astype(np.float32), d['labels'].astype(int)
    return d['H'].astype(np.float32), d['y'].astype(int)


def reff_from_H(H):
    sv = np.linalg.svd(H, compute_uv=False)
    return effective_rank_from_singular_values(sv)


def sa_k_from_Vt(Vt1, Vt2, k):
    k1 = min(k, Vt1.shape[0], Vt2.shape[0])
    gram = Vt1[:k1] @ Vt2[:k1].T
    return float((gram**2).sum() / k1)


# ─────────────────────────────────────────────────────────────
# Step 1: Load existing lp_acc (already computed in csv)
# ─────────────────────────────────────────────────────────────
df_raw = pd.read_csv(ROOT / "results" / "vtab_ssr_results.csv")
# Decode corrupted (arch, task) → (arch_full, task_clean)
# Format: arch=prefix (e.g. "convnext"), task="suffix__actual" (e.g. "base__cifar10")
def decode_row(arch_prefix, task_raw):
    parts = task_raw.split("__")
    if len(parts) == 2:
        suffix, task_actual = parts
        # Try without underscore first: e.g. resnet + 18 → resnet18
        candidate_nounderscore = arch_prefix + suffix
        # Try with underscore: e.g. convnext + base → convnext_base
        candidate_underscore = arch_prefix + "_" + suffix
        if candidate_nounderscore in ARCHS:
            return candidate_nounderscore, task_actual
        elif candidate_underscore in ARCHS:
            return candidate_underscore, task_actual
        else:
            return candidate_underscore, task_actual  # fallback
    return arch_prefix, task_raw

decoded = [decode_row(r.arch, r.task) for _, r in df_raw.iterrows()]
df_raw['arch_full'] = [x[0] for x in decoded]
df_raw['task_clean'] = [x[1] for x in decoded]

# Restrict to known 25 archs, drop original arch/task, use decoded ones
df_lp = df_raw[df_raw['arch_full'].isin(ARCHS)].copy()
df_lp = df_lp.drop(columns=['arch','task']).rename(columns={'arch_full':'arch','task_clean':'task'})
lp_lookup = {(str(r['arch']), str(r['task'])): float(r['lp_acc']) for _, r in df_lp.iterrows()}
print(f"LP lookup entries: {len(lp_lookup)}")
print(f"  Unique archs: {df_lp['arch'].nunique()}, tasks: {df_lp['task'].nunique()}")

# ─────────────────────────────────────────────────────────────
# Step 2: Pre-compute and cache top-K right singular vectors
#         for all (arch, task) pairs
# ─────────────────────────────────────────────────────────────
print("\nCaching SVDs for all 25 × 8 = 200 (arch, task) pairs...")
MAX_K = max(K_VALUES)  # keep top-100 right singular vectors
svd_cache = {}   # (arch, task) → (Vt_topK, d, n_class)
feat_dim  = {}
reff_vals = {}

for arch in ARCHS:
    for task in TASKS8:
        try:
            H, y = load_Hy(arch, task)
        except FileNotFoundError:
            continue
        d = H.shape[1]
        nc = len(np.unique(y))
        # Truncated SVD: only compute top MAX_K right sing vecs
        # For n×d, full_matrices=False gives Vt of shape (min(n,d), d)
        # We only need the top MAX_K rows
        _, sv, Vt = np.linalg.svd(H, full_matrices=False)
        Vt_topK = Vt[:MAX_K].copy()  # (MAX_K × d)
        svd_cache[(arch, task)] = (Vt_topK, sv)
        feat_dim[(arch, task)] = d
        reff_vals[(arch, task)] = reff_from_H(H)  # use full sv for reff

        print(f"  {arch:35s} {task:15s} d={d:4d} n_sv={len(sv):4d}", end='\r')

print(f"\nDone. {len(svd_cache)} entries cached.")

# ─────────────────────────────────────────────────────────────
# B5: Build clean VTAB CSV with SA_k at k=20, ref=stl10
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("B5: Building vtab_25arch_8tasks.csv")
rows_b5 = []
for arch in ARCHS:
    for task in TASKS8:
        if (arch, task) not in svd_cache: continue
        Vt, sv = svd_cache[(arch, task)]
        d  = feat_dim[(arch, task)]
        r  = reff_vals[(arch, task)]
        nc = int(df_lp[(df_lp.arch==arch)&(df_lp.task==task)]['n_class'].values[0]) \
             if len(df_lp[(df_lp.arch==arch)&(df_lp.task==task)]) > 0 else 0
        lp = lp_lookup.get((arch, task), np.nan)
        log_c = float(np.log(nc)) if nc > 1 else 1.0

        sak = np.nan; ssr = np.nan
        if task != 'stl10' and (arch, 'stl10') in svd_cache:
            Vt_ref, _ = svd_cache[(arch, 'stl10')]
            sak = sa_k_from_Vt(Vt, Vt_ref, k=20)
            ssr = (np.log(r) - np.log(max(sak, EPS))) / log_c

        rows_b5.append({'arch':arch,'task':task,'feat_dim':d,'n_class':nc,
                        'lp_acc':lp,'reff':r,'rankme':float(np.log(r)),
                        'sa_k':sak,'ssr_dagger':ssr})

df_b5 = pd.DataFrame(rows_b5)
csv_out = ROOT / "results" / "vtab_25arch_8tasks.csv"
df_b5.to_csv(csv_out, index=False)
print(f"Saved: {csv_out}  ({len(df_b5)} rows)")

# Verify ρ(RankMe, d)
print("\n--- ρ(RankMe, feat_dim) per target task ---")
rhos_rm=[]; regs_rm=[]; rhos_ssr=[]; regs_ssr=[]
for t in TASKS7:
    sub = df_b5[df_b5.task==t].dropna(subset=['lp_acc','rankme','feat_dim'])
    if len(sub)<5: continue
    rho_rm,_=spearmanr(sub['rankme'],sub['feat_dim'])
    ora=sub['lp_acc'].max()
    rm_reg=ora-sub.loc[sub['rankme'].idxmax(),'lp_acc']
    rhos_rm.append(rho_rm); regs_rm.append(rm_reg)
    sub2=sub.dropna(subset=['ssr_dagger'])
    if len(sub2)>4:
        rho_s,_=spearmanr(sub2['ssr_dagger'],sub2['lp_acc'])
        ssr_reg=ora-sub2.loc[sub2['ssr_dagger'].idxmax(),'lp_acc']
        rhos_ssr.append(rho_s); regs_ssr.append(ssr_reg)
    print(f"  {t:15s}  ρ(RM,d)={rho_rm:+.3f}  reg_RM={rm_reg:+.3f}  reg_SSR={ssr_reg if sub2 is not None and len(sub2)>4 else float('nan'):+.3f}")
print(f"  MEAN           ρ(RM,d)={np.mean(rhos_rm):+.3f}  reg_RM={np.mean(regs_rm):+.3f}  reg_SSR={np.mean(regs_ssr) if regs_ssr else float('nan'):+.3f}")

# ─────────────────────────────────────────────────────────────
# C7: Multi-reference ablation
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("C7: Multi-reference ablation (ref ∈ {stl10, cifar10, flowers102})")
print(f"\n{'Ref task':12s}  {'ρ(SSR,LP)':>10s}  {'SSR regret':>12s}  {'n_tasks':>8s}")
print("-"*50)
for ref_task in REF_TASKS:
    rho_list=[]; reg_list=[]
    task_rows={}
    for arch in ARCHS:
        if (arch,ref_task) not in svd_cache: continue
        Vt_ref,_=svd_cache[(arch,ref_task)]
        for task in TASKS7:
            if task==ref_task: continue
            if (arch,task) not in svd_cache: continue
            Vt,_=svd_cache[(arch,task)]
            r=reff_vals[(arch,task)]
            nc_rows=df_lp[(df_lp.arch==arch)&(df_lp.task==task)]
            nc=int(nc_rows['n_class'].values[0]) if len(nc_rows)>0 else 2
            log_c=float(np.log(nc)) if nc>1 else 1.0
            lp=lp_lookup.get((arch,task),np.nan)
            sak=sa_k_from_Vt(Vt,Vt_ref,k=20)
            ssr=(np.log(r)-np.log(max(sak,EPS)))/log_c
            if task not in task_rows: task_rows[task]=[]
            task_rows[task].append({'arch':arch,'lp_acc':lp,'ssr_dagger':ssr,'reff':r})
    for task,rows in task_rows.items():
        df_t=pd.DataFrame(rows).dropna(subset=['lp_acc','ssr_dagger'])
        if len(df_t)<5: continue
        rho,_=spearmanr(df_t['ssr_dagger'],df_t['lp_acc'])
        ora=df_t['lp_acc'].max()
        reg=ora-df_t.loc[df_t['ssr_dagger'].idxmax(),'lp_acc']
        if not np.isnan(rho): rho_list.append(rho)
        reg_list.append(reg)
    print(f"  {ref_task:12s}  {np.mean(rho_list):+10.3f}  {np.mean(reg_list):+12.3f}  {len(rho_list):8d}")

# ─────────────────────────────────────────────────────────────
# C8: k ablation
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("C8: k ablation (ref=stl10, k ∈ {5,10,20,50,100})")
print(f"\n{'k':>5s}  {'ρ(SSR,LP)':>10s}  {'SSR regret':>12s}  {'mean SA_k':>10s}")
print("-"*45)
c8_rows=[]
for k in K_VALUES:
    task_rows={}
    for arch in ARCHS:
        if (arch,'stl10') not in svd_cache: continue
        Vt_ref,_=svd_cache[(arch,'stl10')]
        for task in TASKS7:
            if (arch,task) not in svd_cache: continue
            Vt,_=svd_cache[(arch,task)]
            r=reff_vals[(arch,task)]
            nc_rows=df_lp[(df_lp.arch==arch)&(df_lp.task==task)]
            nc=int(nc_rows['n_class'].values[0]) if len(nc_rows)>0 else 2
            log_c=float(np.log(nc)) if nc>1 else 1.0
            lp=lp_lookup.get((arch,task),np.nan)
            sak=sa_k_from_Vt(Vt,Vt_ref,k=k)
            ssr=(np.log(r)-np.log(max(sak,EPS)))/log_c
            if task not in task_rows: task_rows[task]=[]
            task_rows[task].append({'arch':arch,'lp_acc':lp,'ssr_dagger':ssr,
                                    'reff':r,'sa_k':sak})
    rho_l=[]; reg_l=[]; sak_l=[]
    for task,rows in task_rows.items():
        df_t=pd.DataFrame(rows).dropna(subset=['lp_acc','ssr_dagger'])
        if len(df_t)<5: continue
        rho,_=spearmanr(df_t['ssr_dagger'],df_t['lp_acc'])
        ora=df_t['lp_acc'].max()
        reg=ora-df_t.loc[df_t['ssr_dagger'].idxmax(),'lp_acc']
        if not np.isnan(rho): rho_l.append(rho); sak_l.append(df_t['sa_k'].mean())
        reg_l.append(reg)
    print(f"  k={k:3d}  {np.mean(rho_l):+10.3f}  {np.mean(reg_l):+12.3f}  {np.mean(sak_l):10.3f}")
    c8_rows.append({'k':k,'mean_rho':np.mean(rho_l),'mean_regret':np.mean(reg_l),'mean_sa_k':np.mean(sak_l)})

pd.DataFrame(c8_rows).to_csv(OUT/"c8_k_ablation.csv",index=False)
print("\nDONE")
