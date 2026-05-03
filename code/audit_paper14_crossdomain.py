"""Cross-domain validation on paper14 features (8 SSL/multimodal × ~28 tasks).
Verifies that main findings replicate in a wholly different model+task pool."""
import pandas as pd
import numpy as np
from pathlib import Path
import os, sys
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-12
rng = np.random.default_rng(42)
CACHE = Path("results/paper14_features")

def reff_(H):
    s = np.linalg.svd(H, full_matrices=False, compute_uv=False)
    p = s / max(s.sum(), 1e-12); p = p[p > 1e-12]
    return float(np.exp(-np.sum(p*np.log(p))))

def reff_truncated(H, dim_cap):
    if H.shape[1] <= dim_cap: return reff_(H)
    s = np.linalg.svd(H, full_matrices=False, compute_uv=False)
    s_t = s[:dim_cap]
    p = s_t / max(s_t.sum(), 1e-12); p = p[p > 1e-12]
    return float(np.exp(-np.sum(p*np.log(p))))

def sa_k(H_ref, H_tgt, k=20):
    _,_,Vr = np.linalg.svd(H_ref, full_matrices=False)
    _,_,Vt = np.linalg.svd(H_tgt, full_matrices=False)
    k = min(k, Vr.shape[0], Vt.shape[0])
    cos = np.linalg.svd(Vr[:k] @ Vt[:k].T, compute_uv=False)
    return float(np.mean(np.clip(cos, -1, 1)**2))

def lp_(H, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    (tr, te), = sss.split(H, y)
    sc = StandardScaler().fit(H[tr])
    clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1).fit(sc.transform(H[tr]), y[tr])
    return float(clf.score(sc.transform(H[te]), y[te]))

def disco_fast(H, y):
    classes = np.unique(y); centroids = np.array([H[y==c].mean(0) for c in classes])
    M = centroids - H.mean(0)
    s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    p = s / max(s.sum(), 1e-12); p = p[p > 1e-12]
    return float(np.exp(-np.sum(p*np.log(p))))

# Parse filenames: <task>_<model>_N1000.npz
# Tasks may include underscores (e.g. cifar100_coarse) — we know the 8 model names
MODELS = ['clip_vit_l14', 'convnext_base', 'dinov2_vit_b14', 'dino_vit_s16',
           'mocov3_resnet50', 'resnet101', 'resnet50', 'vit_b16']

files = []
for f in sorted(os.listdir(CACHE)):
    if not f.endswith('_N1000.npz'): continue
    stem = f.replace('_N1000.npz', '')
    matched_model = None
    for m in MODELS:
        if stem.endswith('_' + m):
            matched_model = m
            task = stem[:-len('_'+m)]
            files.append({'task': task, 'model': m, 'file': f})
            break
    if matched_model is None:
        # Special: imagenet_<model> as ref
        for m in MODELS:
            if stem.startswith('imagenet_') and stem == 'imagenet_'+m:
                matched_model = m
                files.append({'task': 'imagenet_REF', 'model': m, 'file': f})
                break

df_files = pd.DataFrame(files)
print(f"Parsed {len(df_files)} files. Models: {sorted(df_files.model.unique())}")
print(f"Tasks: {sorted(df_files.task.unique())}")

# Use imagenet as reference for SA_k. Compute scores per (model, task).
# Reference H: load imagenet_<model> for each model
print(f"\nLoading reference (imagenet) features per model...")
H_refs = {}
for m in MODELS:
    refs = df_files[(df_files.task == 'imagenet_REF') & (df_files.model == m)]
    if len(refs) == 0:
        print(f"  No imagenet ref for {m}; will use cifar10 as ref")
        cif10 = df_files[(df_files.task == 'cifar10') & (df_files.model == m)]
        if len(cif10) == 0: continue
        f_load = np.load(CACHE / cif10.file.iloc[0], allow_pickle=True)
        H_refs[m] = f_load['H']
    else:
        f_load = np.load(CACHE / refs.file.iloc[0], allow_pickle=True)
        H_refs[m] = f_load['H']
    print(f"  {m}: H_ref shape {H_refs[m].shape}")

# Compute scores
print(f"\nComputing scores+LP per (model, task) ...")
rows = []
for _, r in df_files.iterrows():
    if r.task == 'imagenet_REF': continue
    fpath = CACHE / r.file
    try:
        f_data = np.load(fpath, allow_pickle=True)
        H, y = f_data['H'], f_data['y']
        if len(np.unique(y)) < 2: continue
        re = reff_(H); sak = sa_k(H_refs[r.model], H)
        d_ = H.shape[1]
        # Estimate n_class from labels
        n_class = max(2, int(np.unique(y).max()) + 1)
        logc = np.log(n_class)
        rankme = np.log(max(re, EPS))
        rows.append({
            'model': r.model, 'task': r.task,
            'feat_dim': d_, 'n_class': n_class,
            'lp_acc': lp_(H, y),
            'reff': re, 'sa_k': sak,
            'rankme': rankme,
            'rankme_top256': np.log(max(reff_truncated(H, 256), EPS)),
            'rankme_norm_d': rankme - np.log(d_),
            'ssr_dagger': (rankme - np.log(max(sak, EPS))) / logc,
            'disco': disco_fast(H, y),
        })
    except Exception as e:
        print(f"  err {r.file}: {e}")

d = pd.DataFrame(rows)
d.to_csv('results/paper14_crossdomain.csv', index=False)
print(f"Saved {len(d)} rows")

# Strategy comparison
TASKS = sorted(d.task.unique())
print(f"\nPaper14 cross-domain pool: {d.model.nunique()} models × {len(TASKS)} tasks = {len(d)} cells")
print(f"Tasks: {TASKS}")
print(f"Models: {sorted(d.model.unique())}")

oracle = float(np.mean([d[d.task==t].lp_acc.max() for t in TASKS]))
rand = float(np.mean([d[d.task==t].lp_acc.mean() for t in TASKS]))
print(f"\nOracle = {oracle:.3f}    Random = {rand:.3f}    Random regret = {oracle-rand:+.3f}")

print(f"\n  {'method':32s}   mean LP   regret    Δ vs Random")
print("  " + "-"*72)
for col, label in [('rankme','Flat-RankMe'),
                    ('rankme_top256','Flat-RankMe-top256'),
                    ('rankme_norm_d','Flat-NormRM-logd'),
                    ('ssr_dagger','Flat-SSR^‡'),
                    ('disco','Flat-DiSCO')]:
    if col not in d.columns: continue
    lps = []
    for t in TASKS:
        gt = d[d.task == t]
        if gt[col].notna().sum() < 2: continue
        lps.append(gt.loc[gt[col].idxmax()].lp_acc)
    if not lps: continue
    lp_mu = float(np.mean(lps))
    print(f"  {label:32s}   {lp_mu:.3f}     {oracle-lp_mu:+.3f}    {lp_mu-rand:+.3f}")

# Per-task: which model is selected by RankMe? Is it the largest-d?
print(f"\n  Per-task RankMe pick (d=feat_dim of selected model):")
print(f"  {'task':22s}   Oracle  Random  RankMe(model/d)              SSR^‡(model)")
for t in TASKS:
    gt = d[d.task == t]
    if len(gt) < 2: continue
    o = gt.lp_acc.max(); r = gt.lp_acc.mean()
    rm_pick = gt.loc[gt['rankme'].idxmax()]
    ssr_pick = gt.loc[gt['ssr_dagger'].idxmax()]
    print(f"  {t:22s}   {o:.3f}   {r:.3f}   {rm_pick.model[:18]:18s} d={rm_pick.feat_dim:4d}   {ssr_pick.model[:18]}")

# Per-task corr(rankme, feat_dim)
from scipy.stats import spearmanr
print(f"\n  Per-task Spearman ρ(RankMe, feat_dim):")
for t in TASKS:
    gt = d[d.task == t]
    if len(gt) < 4: continue
    r, _ = spearmanr(gt['rankme'], gt['feat_dim'])
    print(f"    {t:22s}  {r:+.3f}")

# Is VGG bias replicated? In paper14 there's no vgg, but the question is:
# does Flat-RankMe systematically pick the highest-d model?
print(f"\n  Verdict on cross-domain replication:")
rm_lp = float(np.mean([d[d.task==t].loc[d[d.task==t]['rankme'].idxmax()].lp_acc for t in TASKS]))
ssr_lp = float(np.mean([d[d.task==t].loc[d[d.task==t]['ssr_dagger'].idxmax()].lp_acc for t in TASKS]))
disco_lp = float(np.mean([d[d.task==t].loc[d[d.task==t]['disco'].idxmax()].lp_acc for t in TASKS]))
print(f"    RankMe regret:  {oracle-rm_lp:+.3f}    {'WORSE' if rm_lp<rand else 'BETTER'} than Random")
print(f"    SSR^‡ regret:   {oracle-ssr_lp:+.3f}    {'WORSE' if ssr_lp<rand else 'BETTER'} than Random")
print(f"    DiSCO regret:   {oracle-disco_lp:+.3f}    {'WORSE' if disco_lp<rand else 'BETTER'} than Random")
print(f"    SSR^‡ vs DiSCO: Δ = {ssr_lp-disco_lp:+.3f}")
