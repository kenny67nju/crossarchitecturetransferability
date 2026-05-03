"""v1.7 SOTA full extended audit — uses bsmamba env (torch 2.5.1+GPU, sklearn).
N2 — extended 51v × 9t with FULL SOTA including LogME (GPU).
Then 3-way decisive: SSR^‡ vs reff_top256 vs DiSCO across same pool.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os, sys, gc
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

EPS = 1e-12
CACHE = Path("results/vtab_features")

EXT_TASKS = ['cifar10', 'cifar100', 'dtd', 'eurosat', 'svhn',
              'fashion_mnist', 'flowers102', 'pets', 'food101', 'fgvc_aircraft']
N_CLASS = {'cifar10':10,'cifar100':100,'dtd':47,'eurosat':10,'svhn':10,
            'fashion_mnist':10,'flowers102':102,'pets':37,'food101':101,'fgvc_aircraft':100}
REF = 'stl10'

# GPU LogME — refined version
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}, torch: {torch.__version__}", flush=True)

def logme_gpu(H_np, y_np):
    """LogME via GPU with stable numerics."""
    H = torch.tensor(H_np, dtype=torch.float32, device=DEVICE)
    n, d = H.shape
    H = H - H.mean(0)
    cov = (H.T @ H) / n
    eigvals, V = torch.linalg.eigh(cov)
    s = torch.clamp(eigvals, min=1e-9)
    classes = sorted(set(y_np.tolist()))
    n_cls = len(classes)
    Y = torch.zeros((n, n_cls), device=DEVICE)
    yt = torch.tensor(y_np, device=DEVICE)
    for i, c in enumerate(classes):
        Y[yt == c, i] = 1
    Y = Y - Y.mean(0)
    Z = (H @ V).cpu().numpy()
    Y_np = Y.cpu().numpy()
    s_np = s.cpu().numpy()
    evidence = 0.0
    sum_z2 = (Z**2).sum(0)
    for k_idx in range(n_cls):
        y_k = Y_np[:, k_idx]
        z_y = (Z * y_k[:, None]).sum(0)
        alpha, beta = 1.0, 1.0
        for _ in range(40):
            t = alpha + beta * sum_z2
            m_post = beta * z_y / t
            m2 = (m_post**2).sum()
            res = y_k - Z @ m_post
            res2 = (res**2).sum()
            gamma = (beta * sum_z2 / t).sum()
            alpha_new = max(gamma / max(m2, 1e-9), 1e-9)
            beta_new = max((n - gamma) / max(res2, 1e-9), 1e-9)
            if abs(alpha_new - alpha) < 1e-3 * alpha and abs(beta_new - beta) < 1e-3 * beta: break
            alpha, beta = alpha_new, beta_new
        ev_k = 0.5 * (d * np.log(alpha) + n * np.log(beta)
                       - alpha * m2 - beta * res2 - n * np.log(2 * np.pi)
                       - np.sum(np.log(alpha + beta * sum_z2)))
        evidence += ev_k
    del H, cov, V, Y
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    return float(evidence / (n * n_cls))

# Standard helpers
def fp(s): return s.replace('.', '_dot_')
def load_feat(v, t):
    p = CACHE / f"{fp(v)}__{t}.npz"
    if not p.exists(): return None, None
    try:
        f = np.load(p, allow_pickle=True)
        if {'H', 'y'} <= set(f.keys()): return f['H'], f['y']
        if {'features', 'labels'} <= set(f.keys()): return f['features'], f['labels']
    except: return None, None
    return None, None

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

# Light SOTA fast versions
def hscore_fast(H, y):
    H = np.asarray(H, dtype=np.float32); n, d = H.shape
    if d > 2500: return float('nan')
    H = H - H.mean(0, keepdims=True)
    classes = np.unique(y); means = np.array([H[y==c].mean(0) for c in classes])
    cov = (H.T @ H) / max(n - 1, 1)
    cov_b = np.zeros_like(cov)
    for i, c in enumerate(classes):
        cov_b += (y == c).sum() * np.outer(means[i], means[i]) / n
    try: return float(np.trace(np.linalg.pinv(cov + 1e-6 * np.eye(d)) @ cov_b))
    except: return float('nan')

def leep_fast(H, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(H); Hs = sc.transform(H)
    clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1).fit(Hs, y)
    p_y_z = clf.predict_proba(Hs); classes = np.unique(y)
    p_yz_y = np.zeros((p_y_z.shape[1], len(classes)))
    for j, c in enumerate(classes): p_yz_y[:, j] = p_y_z[y == c].mean(0)
    log_p_y_x = np.log(np.maximum(p_y_z @ p_yz_y, 1e-12))
    return float(np.mean(log_p_y_x[np.arange(len(y)), [list(classes).index(yi) for yi in y]]))

def disco_fast(H, y):
    classes = np.unique(y); centroids = np.array([H[y==c].mean(0) for c in classes])
    M = centroids - H.mean(0)
    s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    p = s / max(s.sum(), 1e-12); p = p[p > 1e-12]
    return float(np.exp(-np.sum(p*np.log(p))))

def ncti_fast(H, y):
    classes = np.unique(y); centroids = np.array([H[y==c].mean(0) for c in classes])
    return float(-np.linalg.norm(centroids - centroids.mean(0), 'fro'))

# Discover
files = {}
for f in os.listdir(CACHE):
    if not f.endswith('.npz'): continue
    name = f[:-4]
    if '__' not in name: continue
    stem, t = name.split('__', 1)
    files.setdefault(stem.replace('_dot_', '.'), set()).add(t)

ext_variants = [v for v, ts in files.items() if REF in ts and
                 sum(1 for t in EXT_TASKS if t in ts) >= 7]
print(f"Variants with ≥7/{len(EXT_TASKS)} ext tasks + ref: {len(ext_variants)}", flush=True)

rows = []
for vi, v in enumerate(ext_variants, 1):
    H_ref, _ = load_feat(v, REF)
    if H_ref is None: continue
    print(f"  [{vi}/{len(ext_variants)}] {v}", flush=True)
    for t in EXT_TASKS:
        H, y = load_feat(v, t)
        if H is None: continue
        try:
            re = reff_(H); sak = sa_k(H_ref, H)
            d_ = H.shape[1]; logc = np.log(N_CLASS[t])
            rankme = np.log(max(re, EPS))
            re_top256 = reff_truncated(H, 256)
            row = {'variant': v, 'arch': v.split('.')[0], 'task': t,
                   'n_class': N_CLASS[t], 'feat_dim': d_,
                   'lp_acc': lp_(H, y),
                   'reff': re, 'reff_top256': re_top256, 'sa_k': sak,
                   'rankme': rankme,
                   'rankme_top256': np.log(max(re_top256, EPS)),
                   'rankme_norm_d': rankme - np.log(d_),
                   'ssr_dagger': (rankme - np.log(max(sak, EPS))) / logc,
                   'disco': disco_fast(H, y),
                   'hscore': hscore_fast(H, y),
                   'leep': leep_fast(H, y),
                   'ncti': ncti_fast(H, y),
                   'logme': logme_gpu(H, y)}
            rows.append(row)
            del H
            gc.collect()
        except Exception as e:
            print(f"      err {v}/{t}: {e}", flush=True)

df = pd.DataFrame(rows)
df.to_csv('results/extended_full_sota.csv', index=False)
print(f"\nSaved {len(df)} rows → results/extended_full_sota.csv", flush=True)

if len(df) == 0: sys.exit(1)
TASKS_ACTUAL = sorted(df.task.unique())
oracle = float(np.mean([df[df.task==t].lp_acc.max() for t in TASKS_ACTUAL]))
rand = float(np.mean([df[df.task==t].lp_acc.mean() for t in TASKS_ACTUAL]))
print(f"\nExtended pool replication ({len(df.variant.unique())}v × {len(TASKS_ACTUAL)}t)")
print(f"  Oracle = {oracle:.3f}    Random = {rand:.3f}")
print(f"  {'method':32s}   mean LP   regret")
for col, label in [('rankme','Flat-RankMe (LF)'),
                    ('rankme_top256','Flat-RankMe-top256 (LF)'),
                    ('rankme_norm_d','Flat-NormRM-logd (LF)'),
                    ('ssr_dagger','Flat-SSR^‡ (LF)'),
                    ('disco','Flat-DiSCO (LF)'),
                    ('logme','Flat-LogME-GPU (label)'),
                    ('hscore','Flat-H-score (label)'),
                    ('leep','Flat-LEEP (label)'),
                    ('ncti','Flat-NCTI (label)')]:
    if col not in df.columns: continue
    lps = []
    for t in TASKS_ACTUAL:
        gt = df[df.task == t]
        if gt[col].notna().sum() < 2: continue
        lps.append(gt.loc[gt[col].idxmax()].lp_acc)
    if not lps: continue
    lp_mu = float(np.mean(lps))
    print(f"  {label:32s}   {lp_mu:.3f}     {oracle-lp_mu:+.3f}")

print("\nDone.")
