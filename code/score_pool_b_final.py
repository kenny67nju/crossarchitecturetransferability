"""
Score Pool B (FINAL integrated): Phase A+B+C+D+E
=================================================

Integrates ALL extracted features:
- Phase A+B+C: phase_c_full.csv variants (existing 34 vars × 10 archs)
- Phase D: 32 single-variant cached HF archs (resnet101/152, deit, densenet, etc.)
- Phase E: torchvision additions (~26 vars across many archs)

Naming integration (N1):
- torchvision `vit_b_16` → `vit_base` arch (merges with timm vit_base 7 variants)
- torchvision `swin_s` → `swin_small` arch (merges with timm swin_small 2 variants)
- timm `vit_base_patch16_224.*` → `vit_base` arch (already in phase C)
- timm `swin_small_patch4_window7_224.*` → `swin_small` arch (already)
- timm `resnet101.a1h_in1k` + torchvision `resnet101.tv_v1/tv_v2` → arch `resnet101` (3 variants)
- ... etc.

Goal: maximize multi-variant arch count for SCR within-arch evaluation.
"""
from __future__ import annotations

import sys, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from run_final_benchmark import compute_rankme, compute_ncti, compute_disco, compute_leep
from baselines import LogME, compute_hscore
from metrics.ssr import ssr_dagger

CACHE = Path("results/vtab_features")
OUT = Path("results/pool_b_final.csv")

TASKS = ['cifar10', 'cifar100', 'dtd', 'eurosat', 'svhn']  # 5 common targets
N_CLASS = {'cifar10': 10, 'cifar100': 100, 'dtd': 47, 'eurosat': 10, 'svhn': 10}
REF = 'stl10'
K = 20


def fp(s): return s.replace('.', '_dot_')
def feat_to_full(stem):
    """Convert filename stem (with _dot_) back to original variant name."""
    return stem.replace('_dot_', '.')


# ─── Architecture canonical mapping (N1 integration) ─────────
# Maps variant_name → canonical_arch
def canonical_arch(variant_name):
    # Phase A+B+C archs (from phase_c)
    if variant_name.startswith('vit_base_patch16_224'):  return 'vit_base'
    if variant_name.startswith('vit_b_16.torchvision'):  return 'vit_base'   # merge
    if variant_name.startswith('vit_small_patch16_224'): return 'vit_small'
    if variant_name.startswith('vit_b_32.torchvision'):  return 'vit_b32'
    if variant_name.startswith('vit_base_patch32_224'):  return 'vit_b32'    # merge
    if variant_name.startswith('vit_l_16.torchvision'):  return 'vit_large'
    if variant_name.startswith('vit_large_patch16_224'): return 'vit_large'  # merge
    if variant_name.startswith('vit_large_patch14_clip'):return 'vit_large_clip'
    if variant_name.startswith('vit_base_patch32_clip'): return 'vit_b32_clip'
    if variant_name.startswith('vit_tiny_patch16_224'):  return 'vit_tiny'
    if variant_name.startswith('vit_base_patch14_dinov2'): return 'vit_dinov2'
    if variant_name.startswith('beit_base_patch16_224'): return 'beit_base'
    if variant_name.startswith('deit_base_patch16_224'): return 'deit_base'
    if variant_name.startswith('deit_small_patch16_224'):return 'deit_small'
    if variant_name.startswith('deit_tiny_patch16_224'): return 'deit_tiny'
    if variant_name.startswith('convnext_small'):        return 'convnext_small'
    if variant_name.startswith('convnext_tiny'):         return 'convnext_tiny'
    if variant_name.startswith('convnext_base'):         return 'convnext_base'
    if variant_name.startswith('swin_tiny_patch4_window7_224'): return 'swin_tiny'
    if variant_name.startswith('swin_small_patch4_window7_224'):return 'swin_small'
    if variant_name.startswith('swin_t.torchvision'):    return 'swin_tiny'   # merge
    if variant_name.startswith('swin_s.torchvision'):    return 'swin_small'  # merge
    if variant_name.startswith('swin_b.torchvision'):    return 'swin_base'
    if variant_name.startswith('resnet18'):              return 'resnet18'
    if variant_name.startswith('resnet34'):              return 'resnet34'
    if variant_name.startswith('resnet50'):              return 'resnet50'
    if variant_name.startswith('resnet101'):             return 'resnet101'
    if variant_name.startswith('resnet152'):             return 'resnet152'
    if variant_name.startswith('resnext50'):             return 'resnext50'
    if variant_name.startswith('wide_resnet50_2'):       return 'wide_resnet50_2'
    if variant_name.startswith('densenet121'):           return 'densenet121'
    if variant_name.startswith('densenet169'):           return 'densenet169'
    if variant_name.startswith('densenet201'):           return 'densenet201'
    if variant_name.startswith('efficientnet_b0'):       return 'efficientnet_b0'
    if variant_name.startswith('efficientnet_b2'):       return 'efficientnet_b2'
    if variant_name.startswith('efficientnet_b3'):       return 'efficientnet_b3'
    if variant_name.startswith('efficientnet_b4'):       return 'efficientnet_b4'
    if variant_name.startswith('tf_efficientnet'):       return 'tf_efficientnet'
    if variant_name.startswith('mobilenet_v3_large'):    return 'mobilenet_v3_large'
    if variant_name.startswith('mobilenetv3_large'):     return 'mobilenet_v3_large'  # merge
    if variant_name.startswith('mobilenet_v3_small'):    return 'mobilenet_v3_small'
    if variant_name.startswith('mobilenetv3_small'):     return 'mobilenet_v3_small'  # merge
    if variant_name.startswith('mnasnet'):               return 'mnasnet'
    if variant_name.startswith('inception'):             return 'inception_v3'
    if variant_name.startswith('regnet'):                return 'regnet'
    if variant_name.startswith('vgg16_bn'):              return 'vgg16_bn'
    if variant_name.startswith('vgg16'):                 return 'vgg16'
    if variant_name.startswith('mixer'):                 return 'mixer_b16'
    if variant_name.startswith('maxvit'):                return 'maxvit'
    return variant_name.split('.')[0]  # fallback


def load_feat(variant_name, ds):
    p = CACHE / f"{fp(variant_name)}__{ds}.npz"
    if not p.exists(): return None, None
    try:
        f = np.load(p, allow_pickle=True); k = set(f.keys())
    except Exception:
        return None, None
    if {'H', 'y'} <= k: return f['H'], f['y']
    if {'features', 'labels'} <= k: return f['features'], f['labels']
    return None, None


def discover_variants():
    """Auto-discover ALL variants from cache files matching arch_dot_recipe pattern."""
    vs = set()
    for f in os.listdir(CACHE):
        if not f.endswith('.npz') or f.startswith('.'): continue
        name = f[:-4]
        if '__' not in name: continue
        stem, _ = name.split('__', 1)
        # Reconstruct variant name
        if '_dot_' in stem:
            v = stem.replace('_dot_', '.')
        else:
            v = stem  # paper18-style canonical_id (e.g., 'beit_b')
        vs.add(v)
    return sorted(vs)


# Same metric functions as before
def effective_rank(H):
    s = np.linalg.svd(H, full_matrices=False, compute_uv=False)
    p = s / max(s.sum(), 1e-12); p = p[p>1e-12]
    return float(np.exp(-np.sum(p*np.log(p))))
def uid_twonn(H, sample=500):
    from sklearn.neighbors import NearestNeighbors
    if len(H) > sample:
        idx = np.random.default_rng(42).choice(len(H), sample, replace=False); H = H[idx]
    nn = NearestNeighbors(n_neighbors=3).fit(H); d,_ = nn.kneighbors(H)
    r1, r2 = d[:,1], d[:,2]; mu = r2/np.maximum(r1, 1e-9); mu = mu[mu > 1.001]
    if len(mu) < 5: return float('nan')
    log_mu = np.log(mu); F = (np.arange(1, len(log_mu)+1))/(len(log_mu)+1)
    return float(-np.sum(np.log(1-F))/np.sum(log_mu))
def sa_k(H_ref, H_tgt, k=K):
    _,_,Vr = np.linalg.svd(H_ref, full_matrices=False)
    _,_,Vt = np.linalg.svd(H_tgt, full_matrices=False)
    k = min(k, Vr.shape[0], Vt.shape[0])
    M = Vr[:k] @ Vt[:k].T
    cos = np.linalg.svd(M, compute_uv=False); cos = np.clip(cos, -1, 1)
    return float(np.mean(cos**2))
def linear_probe(H, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    (tr, te), = sss.split(H, y)
    sc = StandardScaler().fit(H[tr])
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf.fit(sc.transform(H[tr]), y[tr])
    return float(clf.score(sc.transform(H[te]), y[te]))


def main():
    variants = discover_variants()
    print(f"Discovered {len(variants)} variants in cache")

    # Filter: must have stl10 (REF) AND at least 1 of TASKS
    filtered = []
    for v in variants:
        H_ref, _ = load_feat(v, REF)
        if H_ref is None: continue
        has_task = any(load_feat(v, t)[0] is not None for t in TASKS)
        if not has_task: continue
        filtered.append(v)
    print(f"After filter (has REF + at least 1 task): {len(filtered)} variants")

    rows = []
    for v in filtered:
        arch = canonical_arch(v)
        H_ref, _ = load_feat(v, REF)
        for task in TASKS:
            H, y = load_feat(v, task)
            if H is None: continue
            n_c = N_CLASS[task]
            try:
                lp = linear_probe(H, y)
                reff = effective_rank(H); uid = uid_twonn(H); sak = sa_k(H_ref, H)
                rankme = compute_rankme(H); logme = LogME().fit(H, y)
                hscore = compute_hscore(H, y); leep = compute_leep(H, y)
                ncti = compute_ncti(H, y); disco = compute_disco(H, y)
            except Exception as e:
                print(f"  err {v}/{task}: {e}"); continue
            ssr = ssr_dagger(reff, sak, n_c)
            rows.append({
                'arch': arch, 'variant': v, 'task': task, 'n_class': n_c,
                'lp_acc': lp, 'reff': reff, 'uid': uid, 'sa_k': sak,
                'ssr_dagger': ssr,
                'rankme': rankme, 'logme': logme, 'hscore': hscore,
                'leep': leep, 'ncti': ncti, 'disco': disco,
            })

    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(df)} rows")
    print(f"  variants: {df.variant.nunique()}, archs: {df.arch.nunique()}")
    arch_var = df.groupby('arch').variant.nunique()
    multi = (arch_var >= 2).sum()
    print(f"  Multi-variant archs (>=2): {multi}/{len(arch_var)}")
    print("\nArch variant counts:")
    for a in arch_var.sort_values(ascending=False).index:
        print(f"  {a:25s}  {arch_var[a]} variants")

    print("\n=== Within-arch joint ρ vs LP ===")
    print(f"{'metric':12s}  joint_ρ   pos    sig_pos")
    for m in ['ssr_dagger','rankme','reff','uid','logme','hscore','leep','ncti','disco']:
        rhos=[]; sigs=[]
        for (a,t),g in df.groupby(['arch','task']):
            if g[m].nunique()<=1: continue
            r,p=spearmanr(g[m],g['lp_acc'])
            if not np.isnan(r):
                rhos.append(r); sigs.append(p<0.05)
        rhos=np.array(rhos); pos=(rhos>0).sum() if len(rhos)>0 else 0
        sig=sum(sigs) if sigs else 0
        print(f"  {m:10s}    {rhos.mean():+.3f}     {pos}/{len(rhos)}    {sig}")


if __name__ == "__main__":
    main()
