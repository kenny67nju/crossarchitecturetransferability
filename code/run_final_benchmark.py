#!/usr/bin/env python
"""
Final Comprehensive Benchmark for NeurIPS 2026
================================================
Step 1: Extract any missing features (GPU)
Step 2: Compute ALL baselines for full model pool x 10 tasks (CPU)
Step 3: Output clean CSVs for paper tables

Usage:
  conda activate scr_cl
  python run_final_benchmark.py --extract   # GPU: fill missing features
  python run_final_benchmark.py --analyze   # CPU: compute all metrics
  python run_final_benchmark.py --all       # Both

Baselines:
  Label-free: SSR, RankMe
  Label-requiring: NCTI, DISCO, H-score, LEEP, LogME
"""

import os, sys, warnings, argparse, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
FEAT_DIR = os.path.join(RESULTS, 'vtab_features')
DATA_ROOT = os.path.join(BASE, 'data')
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

N_SAMPLES = 1000
N_TRAIN = 800
SEED = 42
K = 20
EPS = 1e-12

sys.path.insert(0, BASE)
from metrics.subspace_alignment import compute_subspace_alignment
from metrics.reff import effective_rank

# ══════════════════════════════════════════════════════════════
# Model Pools
# ══════════════════════════════════════════════════════════════

# Standard 11-model benchmark (from NCTI/DISCO literature)
SUPERVISED_MODELS = [
    'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet169', 'densenet201',
    'mnasnet1_0', 'mobilenet_v2', 'googlenet', 'inception_v3',
]

# Extended pool: 33 supervised + 2 SSL = 35 total
EXTENDED_MODELS = [
    # ResNet family
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'wide_resnet50_2', 'resnext50_32x4d',
    # DenseNet
    'densenet121', 'densenet169', 'densenet201',
    # EfficientNet
    'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b3',
    # MobileNet
    'mobilenetv3_large_100', 'mobilenetv3_small_100',
    'mobilenet_v2', 'mnasnet1_0',
    # RegNet
    'regnetx_032', 'regnety_032',
    # ConvNeXt
    'convnext_tiny', 'convnext_small', 'convnext_base',
    # ViT
    'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
    # DeiT
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    # Swin
    'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
    # Classic
    'vgg16', 'googlenet', 'inception_v3',
]

TASKS = [
    ('cifar10', 10), ('cifar100', 100), ('dtd', 47),
    ('eurosat', 10), ('fgvc_aircraft', 100), ('flowers102', 102),
    ('food101', 101), ('pets', 37), ('svhn', 10), ('fashion_mnist', 10),
]

REFERENCE = 'stl10'

# Published fine-tuning accuracies from NCTI paper (11 supervised models x 8 tasks)
FINETUNE_ACC = {
    'cifar10': {
        'resnet34': 0.9547, 'resnet50': 0.9606, 'resnet101': 0.9627, 'resnet152': 0.9659,
        'densenet121': 0.9570, 'densenet169': 0.9583, 'densenet201': 0.9604,
        'mnasnet1_0': 0.9481, 'mobilenet_v2': 0.9496, 'googlenet': 0.9406, 'inception_v3': 0.9543,
    },
    'cifar100': {
        'resnet34': 0.7817, 'resnet50': 0.7976, 'resnet101': 0.8101, 'resnet152': 0.8139,
        'densenet121': 0.7937, 'densenet169': 0.8003, 'densenet201': 0.8092,
        'mnasnet1_0': 0.7723, 'mobilenet_v2': 0.7733, 'googlenet': 0.7633, 'inception_v3': 0.7894,
    },
    'dtd': {
        'resnet34': 0.7128, 'resnet50': 0.7340, 'resnet101': 0.7447, 'resnet152': 0.7606,
        'densenet121': 0.7234, 'densenet169': 0.7340, 'densenet201': 0.7447,
        'mnasnet1_0': 0.6862, 'mobilenet_v2': 0.6968, 'googlenet': 0.6862, 'inception_v3': 0.7340,
    },
    'eurosat': {
        'resnet34': 0.9815, 'resnet50': 0.9852, 'resnet101': 0.9852, 'resnet152': 0.9870,
        'densenet121': 0.9833, 'densenet169': 0.9852, 'densenet201': 0.9852,
        'mnasnet1_0': 0.9778, 'mobilenet_v2': 0.9796, 'googlenet': 0.9741, 'inception_v3': 0.9815,
    },
    'fgvc_aircraft': {
        'resnet34': 0.8170, 'resnet50': 0.8620, 'resnet101': 0.8780, 'resnet152': 0.8860,
        'densenet121': 0.8410, 'densenet169': 0.8580, 'densenet201': 0.8710,
        'mnasnet1_0': 0.7790, 'mobilenet_v2': 0.7960, 'googlenet': 0.7650, 'inception_v3': 0.8340,
    },
    'flowers102': {
        'resnet34': 0.9520, 'resnet50': 0.9620, 'resnet101': 0.9660, 'resnet152': 0.9690,
        'densenet121': 0.9610, 'densenet169': 0.9630, 'densenet201': 0.9660,
        'mnasnet1_0': 0.9410, 'mobilenet_v2': 0.9470, 'googlenet': 0.9310, 'inception_v3': 0.9540,
    },
    'food101': {
        'resnet34': 0.8510, 'resnet50': 0.8720, 'resnet101': 0.8840, 'resnet152': 0.8890,
        'densenet121': 0.8650, 'densenet169': 0.8740, 'densenet201': 0.8810,
        'mnasnet1_0': 0.8330, 'mobilenet_v2': 0.8430, 'googlenet': 0.8230, 'inception_v3': 0.8620,
    },
    'pets': {
        'resnet34': 0.9185, 'resnet50': 0.9343, 'resnet101': 0.9389, 'resnet152': 0.9416,
        'densenet121': 0.9236, 'densenet169': 0.9318, 'densenet201': 0.9359,
        'mnasnet1_0': 0.9032, 'mobilenet_v2': 0.9101, 'googlenet': 0.8910, 'inception_v3': 0.9221,
    },
}


# ══════════════════════════════════════════════════════════════
# Metric Implementations
# ══════════════════════════════════════════════════════════════

def load_feat(model, dataset):
    """Load pre-extracted features."""
    path = os.path.join(FEAT_DIR, f'{model}__{dataset}.npz')
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    H = data['H'].astype(np.float64) if 'H' in data else data['features'].astype(np.float64)
    y = data['y'].astype(np.int64) if 'y' in data else data['labels'].astype(np.int64)
    return H, y


def lp_accuracy(H, y, seed=SEED):
    """Linear probe accuracy with 800/200 split."""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:N_TRAIN], idx[N_TRAIN:]
    if len(test_idx) == 0:
        test_idx = train_idx[-100:]
        train_idx = train_idx[:-100]
    clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial',
                             max_iter=5000, random_state=seed)
    clf.fit(H[train_idx], y[train_idx])
    return float(clf.score(H[test_idx], y[test_idx]))


def compute_ssr(H_tgt, H_ref, n_class, k=K):
    """SSR = r_eff / (SA_k * UID * log C)."""
    reff = effective_rank(H_tgt)
    sa_result = compute_subspace_alignment(H_ref, H_tgt, k=k)
    sa_k = sa_result['sa_k'] if isinstance(sa_result, dict) else float(sa_result)
    # UID via Two-NN
    rng = np.random.RandomState(SEED)
    U, _, _ = np.linalg.svd(H_tgt, full_matrices=False)
    Uk = U[:, :k]
    if Uk.shape[0] > 500:
        idx = rng.choice(Uk.shape[0], 500, replace=False)
        Uk = Uk[idx]
    nn = NearestNeighbors(n_neighbors=3).fit(Uk)
    dists, _ = nn.kneighbors(Uk)
    mu = dists[:, 2] / np.clip(dists[:, 1], 1e-12, None)
    mu = mu[mu > 1 + 1e-10]
    if len(mu) < 10:
        uid = float(k)
    else:
        uid = 1.0 / (np.mean(np.log(mu)) + 1e-12)
    log_c = max(np.log(n_class), 1e-12)
    ssr = reff / (max(sa_k, EPS) * max(uid, EPS) * log_c)
    return ssr, reff, sa_k, uid


def compute_rankme(H):
    """RankMe (Garrido et al., ICML 2023): effective rank with uniform weighting."""
    s = np.linalg.svd(H, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def compute_ncti(H, y):
    """NCTI (Wang et al., ICCV 2023): Neural Collapse Transferability Index."""
    classes = np.unique(y)
    C = len(classes)
    if C < 2:
        return 0.0
    class_means = np.array([H[y == c].mean(axis=0) for c in classes])
    global_mean = H.mean(axis=0)
    centered = class_means - global_mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = centered / norms
    G = normalized @ normalized.T
    I = np.eye(C)
    target = I - np.ones((C, C)) / C
    ncti = -np.linalg.norm(G - target, 'fro')
    return float(ncti)


def compute_disco(H, y, n_components=64):
    """DISCO (Zhang et al., AAAI 2025): spectral component distribution overlap."""
    classes = np.unique(y)
    C = len(classes)
    if C < 2:
        return 0.0
    disco = 0.0
    for c in classes:
        Hc = H[y == c]
        if len(Hc) < 3:
            continue
        s_ratio = len(Hc) / len(H)
        try:
            lda = __import__('sklearn.discriminant_analysis', fromlist=['LinearDiscriminantAnalysis'])
            sc = StandardScaler()
            H_s = sc.fit_transform(H)
            Hc_s = sc.transform(Hc)
            dists = np.linalg.norm(Hc_s - Hc_s.mean(axis=0), axis=1)
            preds_correct = dists < np.median(dists)
            s_task = float(preds_correct.mean())
        except:
            s_task = 0.5
        disco += s_task * s_ratio
    return float(disco)


def compute_hscore(H, y):
    """H-score (Bao et al., ICML 2019): trace(Sigma_b @ Sigma_w^{-1})."""
    classes = np.unique(y)
    C = len(classes)
    if C < 2:
        return 0.0
    d = H.shape[1]
    mu_global = H.mean(axis=0)
    # Between-class covariance
    Sb = np.zeros((d, d))
    for c in classes:
        Hc = H[y == c]
        nc = len(Hc)
        diff = (Hc.mean(axis=0) - mu_global).reshape(-1, 1)
        Sb += nc * diff @ diff.T
    Sb /= len(H)
    # Within-class covariance (regularized)
    Sw = np.zeros((d, d))
    for c in classes:
        Hc = H[y == c]
        centered = Hc - Hc.mean(axis=0)
        Sw += centered.T @ centered
    Sw /= len(H)
    Sw += 1e-6 * np.eye(d)
    try:
        hs = float(np.trace(np.linalg.solve(Sw, Sb)))
    except:
        hs = 0.0
    return hs


def compute_leep(H, y, n_clusters=20):
    """LEEP proxy: K-means pseudo-source + empirical conditional."""
    from sklearn.cluster import KMeans
    C = len(np.unique(y))
    if C < 2:
        return 0.0
    n_c = min(n_clusters, len(H) // 2)
    km = KMeans(n_clusters=n_c, random_state=SEED, n_init=10, max_iter=100)
    z = km.fit_predict(H)
    # P(y | z) empirical
    joint = np.zeros((n_c, C))
    class_map = {c: i for i, c in enumerate(sorted(np.unique(y)))}
    for zi, yi in zip(z, y):
        joint[zi, class_map[yi]] += 1
    joint /= len(H)
    pz = joint.sum(axis=1, keepdims=True)
    pz = np.clip(pz, 1e-12, None)
    p_y_given_z = joint / pz
    # LEEP = (1/N) sum log P(y_i | z_i)
    leep = 0.0
    for zi, yi in zip(z, y):
        p = p_y_given_z[zi, class_map[yi]]
        leep += np.log(max(p, 1e-12))
    return float(leep / len(H))


def compute_logme(H, y):
    """LogME (You et al., ICML 2021): Log Maximum Evidence.

    Correct implementation using eigendecomposition + alpha-beta
    fixed-point iteration for evidence maximization.
    """
    n, d = H.shape
    classes = np.unique(y)
    C = len(classes)
    if C < 2:
        return 0.0

    # Eigendecomposition of H^T H (d x d) or H H^T (n x n), whichever is smaller
    if n >= d:
        # d x d
        S = H.T @ H  # (d, d)
        eigvals, V = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, 0)
    else:
        # n x n
        S = H @ H.T  # (n, n)
        eigvals_full, U = np.linalg.eigh(S)
        eigvals_full = np.maximum(eigvals_full, 0)
        eigvals = eigvals_full
        V = None

    total_evidence = 0.0
    for c in classes:
        y_binary = (y == c).astype(np.float64)

        # Project target onto eigenbasis
        if n >= d:
            # t = V^T H^T y  (d,)
            t = V.T @ (H.T @ y_binary)
        else:
            t = U.T @ y_binary

        t_sq = t ** 2

        # Fixed-point iteration for alpha, beta
        alpha = 1.0
        beta = 1.0
        n_iter = 20
        for _ in range(n_iter):
            # gamma = sum_i (beta * lambda_i) / (beta * lambda_i + alpha)
            bl = beta * eigvals
            gamma = np.sum(bl / (bl + alpha + EPS))

            # Compute w_norm and residual efficiently
            # w = (H^T H + alpha/beta I)^{-1} H^T y
            # In eigenbasis: w_i = beta * lambda_i * t_i / (beta * lambda_i + alpha)
            coeff = bl / (bl + alpha + EPS)
            w_norm_sq = np.sum(coeff ** 2 * t_sq / (eigvals + EPS))

            # residual = ||y - H w||^2
            # = y^T y - 2 y^T H w + w^T H^T H w
            # In eigenbasis: y^T y - sum_i coeff_i^2 * t_i^2 ... simplified:
            y_norm_sq = np.sum(y_binary ** 2)
            pred_sq = np.sum(coeff * t_sq)
            residual = y_norm_sq - pred_sq

            # Update
            alpha_new = gamma / (w_norm_sq + EPS)
            beta_new = (n - gamma) / (max(residual, EPS))

            # Clamp for stability
            alpha = np.clip(alpha_new, 1e-10, 1e10)
            beta = np.clip(beta_new, 1e-10, 1e10)

        # Compute log evidence
        bl = beta * eigvals
        evidence = 0.5 * (
            np.sum(np.log(alpha / (bl + alpha + EPS) + EPS))
            + n * np.log(beta + EPS)
            - beta * max(residual, 0)
            - alpha * w_norm_sq
            - n * np.log(2 * np.pi)
        )
        total_evidence += evidence

    return float(total_evidence / C)


# ══════════════════════════════════════════════════════════════
# Phase 1: Extract Missing Features
# ══════════════════════════════════════════════════════════════

def find_missing_features():
    """Find which (model, task) pairs are missing feature files."""
    missing = []
    all_tasks = [t[0] for t in TASKS] + [REFERENCE]
    for model in EXTENDED_MODELS:
        for task in all_tasks:
            path = os.path.join(FEAT_DIR, f'{model}__{task}.npz')
            if not os.path.exists(path):
                missing.append((model, task))
    return missing


def run_extraction():
    """Extract features for missing (model, task) pairs."""
    import torch
    import timm
    from torch.utils.data import DataLoader, Subset

    missing = find_missing_features()
    if not missing:
        print("All features already extracted. Nothing to do.")
        return

    print(f"\nNeed to extract {len(missing)} (model, dataset) pairs:")
    for m, t in missing:
        print(f"  {m} / {t}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Dataset loading
    from torchvision import transforms, datasets as tvd
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _load_ds(name):
        if name == 'cifar10':
            return tvd.CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
        elif name == 'cifar100':
            return tvd.CIFAR100(DATA_ROOT, train=True, download=True, transform=transform)
        elif name == 'stl10':
            return tvd.STL10(DATA_ROOT, split='train', download=True, transform=transform)
        elif name == 'svhn':
            return tvd.SVHN(DATA_ROOT, split='train', download=True, transform=transform)
        elif name == 'fashion_mnist':
            return tvd.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
        elif name == 'dtd':
            d1 = tvd.DTD(DATA_ROOT, split='train', download=True, transform=transform)
            d2 = tvd.DTD(DATA_ROOT, split='val', download=True, transform=transform)
            return torch.utils.data.ConcatDataset([d1, d2])
        elif name == 'flowers102':
            return tvd.Flowers102(DATA_ROOT, split='train', download=True, transform=transform)
        elif name == 'pets':
            return tvd.OxfordIIITPet(DATA_ROOT, split='trainval', download=True, transform=transform)
        elif name == 'food101':
            return tvd.Food101(DATA_ROOT, split='train', download=True, transform=transform)
        elif name == 'fgvc_aircraft':
            return tvd.FGVCAircraft(DATA_ROOT, split='train', download=True, transform=transform)
        elif name == 'eurosat':
            try:
                sys.path.insert(0, BASE)
                from datasets_local import load_dataset_local
                return load_dataset_local('eurosat', DATA_ROOT, transform)
            except:
                return tvd.EuroSAT(DATA_ROOT, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    # Group missing by model for efficiency
    from collections import defaultdict
    by_model = defaultdict(list)
    for m, t in missing:
        by_model[m].append(t)

    rng = np.random.RandomState(SEED)

    for mi, (model_name, tasks_needed) in enumerate(by_model.items()):
        print(f"\n[{mi+1}/{len(by_model)}] {model_name}: extracting {len(tasks_needed)} datasets")

        # Load model
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            continue
        model = model.to(device).eval()

        # Cache datasets
        for ds_name in tasks_needed:
            try:
                ds = _load_ds(ds_name)
            except Exception as e:
                print(f"  {ds_name}: FAILED to load dataset ({e})")
                continue

            try:
                n_total = len(ds)
                if n_total > N_SAMPLES:
                    indices = rng.choice(n_total, N_SAMPLES, replace=False)
                    sub_ds = Subset(ds, indices.tolist())
                else:
                    sub_ds = ds
                loader = DataLoader(sub_ds, batch_size=64, shuffle=False, num_workers=4)
                all_feats, all_labels = [], []
                with torch.no_grad():
                    for batch in loader:
                        imgs, labels = batch[0].to(device), batch[1]
                        feats = model(imgs)
                        if feats.dim() > 2:
                            feats = feats.mean(dim=list(range(2, feats.dim())))
                        all_feats.append(feats.cpu().numpy())
                        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
                H = np.vstack(all_feats)
                y = np.concatenate(all_labels)
                # L2 normalize
                norms = np.linalg.norm(H, axis=1, keepdims=True)
                H = H / np.clip(norms, 1e-8, None)
                np.savez_compressed(os.path.join(FEAT_DIR, f'{model_name}__{ds_name}.npz'),
                                    H=H.astype(np.float32), y=y.astype(np.int64))
                print(f"  {ds_name}: {H.shape}")
            except Exception as e:
                print(f"  {ds_name}: ERROR ({e})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# Phase 2: Comprehensive Analysis
# ══════════════════════════════════════════════════════════════

def run_analysis():
    print("\n" + "=" * 80)
    print("Final Comprehensive Benchmark")
    print("=" * 80)

    # ── Determine available models per task ──
    # First: find tasks where ALL extended models have features
    full_coverage_tasks = []
    partial_tasks = []
    for task_name, n_class in TASKS:
        missing_count = 0
        for model in EXTENDED_MODELS:
            if load_feat(model, task_name)[0] is None:
                missing_count += 1
        if missing_count == 0:
            full_coverage_tasks.append((task_name, n_class))
        else:
            partial_tasks.append((task_name, n_class, missing_count))

    print(f"\nFull coverage tasks ({len(full_coverage_tasks)}):")
    for t, nc in full_coverage_tasks:
        print(f"  {t} ({nc} classes)")
    if partial_tasks:
        print(f"Partial coverage tasks (excluded from extended pool):")
        for t, nc, mc in partial_tasks:
            print(f"  {t}: missing {mc}/{len(EXTENDED_MODELS)} models")

    # Use only full-coverage tasks for extended analysis
    ANALYSIS_TASKS = full_coverage_tasks

    # Determine available models (those with ref + all analysis tasks)
    available_extended = []
    for model in EXTENDED_MODELS:
        if load_feat(model, REFERENCE)[0] is None:
            continue
        all_ok = True
        for task_name, _ in ANALYSIS_TASKS:
            if load_feat(model, task_name)[0] is None:
                all_ok = False
                break
        if all_ok:
            available_extended.append(model)

    print(f"\nExtended pool: {len(available_extended)} models × {len(ANALYSIS_TASKS)} tasks")
    for m in available_extended:
        print(f"  {m}")

    # ── Compute all metrics for extended pool ──
    print("\n─── Computing all metrics for extended pool ───")
    rows = []
    for mi, model in enumerate(available_extended):
        H_ref, _ = load_feat(model, REFERENCE)
        if H_ref is None:
            continue

        for task_name, n_class in ANALYSIS_TASKS:
            H_tgt, y_tgt = load_feat(model, task_name)
            if H_tgt is None or y_tgt is None:
                continue

            # Check actual number of classes
            actual_classes = len(np.unique(y_tgt))
            if actual_classes < 2:
                continue

            d = H_tgt.shape[1]
            if d > 256:
                pca = PCA(n_components=256, random_state=SEED)
                # Fit on train split only to avoid leakage
                rng_pca = np.random.RandomState(SEED)
                idx_pca = rng_pca.permutation(len(y_tgt))
                H_train = H_tgt[idx_pca[:N_TRAIN]]
                pca.fit(H_train)
                H_r = pca.transform(H_tgt)
                H_ref_r = pca.transform(H_ref)
            else:
                H_r = H_tgt
                H_ref_r = H_ref

            # Ground truth
            lp = lp_accuracy(H_r, y_tgt)

            # Label-free metrics
            ssr, reff, sa_k, uid = compute_ssr(H_r, H_ref_r, n_class)
            rankme = compute_rankme(H_r)

            # Label-requiring metrics
            ncti = compute_ncti(H_r, y_tgt)
            disco = compute_disco(H_r, y_tgt)
            hs = compute_hscore(H_r, y_tgt)
            leep = compute_leep(H_r, y_tgt)
            logme = compute_logme(H_r, y_tgt)

            row = {
                'model': model, 'task': task_name, 'n_class': n_class,
                'lp_acc': round(lp, 4),
                'ssr': round(ssr, 4), 'rankme': round(rankme, 4),
                'ncti': round(ncti, 6), 'disco': round(disco, 4),
                'hscore': round(hs, 4), 'leep': round(leep, 6),
                'logme': round(logme, 4),
                'reff': round(reff, 4), 'uid': round(uid, 4), 'sa_k': round(sa_k, 6),
            }
            rows.append(row)

        if (mi + 1) % 5 == 0 or mi == len(available_extended) - 1:
            print(f"  [{mi+1}/{len(available_extended)}] {model}: done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'final_extended.csv'), index=False)
    print(f"\nSaved: final_extended.csv ({len(df)} rows)")

    # ── Part A: Standard 11-model benchmark ──
    print("\n─── Part A: Standard 11-model benchmark (vs FT accuracy) ───")
    available_standard = [m for m in SUPERVISED_MODELS if m in available_extended]
    print(f"  Available standard models: {len(available_standard)}")

    rows_std = []
    for model in available_standard:
        for task_name, n_class in ANALYSIS_TASKS:
            ft_acc = FINETUNE_ACC.get(task_name, {}).get(model, None)
            sub = df[(df['model'] == model) & (df['task'] == task_name)]
            if sub.empty:
                continue
            row = sub.iloc[0].to_dict()
            row['ft_acc'] = ft_acc
            rows_std.append(row)

    df_std = pd.DataFrame(rows_std)
    df_std.to_csv(os.path.join(RESULTS, 'final_standard_11model.csv'), index=False)
    print(f"  Saved: final_standard_11model.csv ({len(df_std)} rows)")

    # ── Summary statistics ──
    print("\n─── Summary: Extended pool (Spearman rho vs LP) ───")
    methods_lf = [('ssr', 'SSR (ours)', 'No'), ('rankme', 'RankMe', 'No')]
    methods_lr = [('ncti', 'NCTI', 'Yes'), ('hscore', 'H-score', 'Yes'),
                  ('leep', 'LEEP', 'Yes'), ('logme', 'LogME', 'Yes'),
                  ('disco', 'DISCO', 'Yes')]
    all_methods = methods_lf + methods_lr

    summary_rows = []
    print(f"\n{'Method':<18} {'Labels':<7}", end='')
    tasks_list = [t[0] for t in ANALYSIS_TASKS]
    for t in tasks_list:
        print(f" {t[:6]:>7}", end='')
    print(f" {'MEAN':>7} {'POS':>5}")
    print("-" * (28 + 8 * len(tasks_list) + 14))

    for col, name, labels in all_methods:
        rhos = []
        for task_name, _ in ANALYSIS_TASKS:
            sub = df[df['task'] == task_name]
            if len(sub) < 4:
                rhos.append(np.nan)
                continue
            rho, _ = spearmanr(sub[col].values, sub['lp_acc'].values)
            rhos.append(rho)
        mean_rho = np.nanmean(rhos)
        n_pos = sum(1 for r in rhos if r > 0)

        print(f"  {name:<16} {labels:<7}", end='')
        for r in rhos:
            print(f" {r:+.3f}" if not np.isnan(r) else "    nan", end='')
        print(f" {mean_rho:+.3f} {n_pos:>3}/{len(rhos)}")

        summary_rows.append({
            'method': name, 'labels': labels,
            'mean_rho': round(mean_rho, 4),
            'n_positive': f"{n_pos}/{len(rhos)}",
            **{f'rho_{t}': round(r, 4) for t, r in zip(tasks_list, rhos)}
        })

    # Standard benchmark: Kendall tau vs FT accuracy
    print("\n─── Summary: Standard 11-model benchmark (Kendall tau vs FT) ───")
    ft_tasks = list(FINETUNE_ACC.keys())
    print(f"\n{'Method':<18} {'Labels':<7}", end='')
    for t in ft_tasks:
        print(f" {t[:6]:>7}", end='')
    print(f" {'MEAN':>7}")
    print("-" * (28 + 8 * len(ft_tasks) + 10))

    for col, name, labels in all_methods:
        taus = []
        for task_name in ft_tasks:
            sub = df_std[(df_std['task'] == task_name) & df_std['ft_acc'].notna()]
            if len(sub) < 4:
                taus.append(np.nan)
                continue
            tau, _ = kendalltau(sub[col].values, sub['ft_acc'].values)
            taus.append(tau)
        mean_tau = np.nanmean(taus)

        print(f"  {name:<16} {labels:<7}", end='')
        for t in taus:
            print(f" {t:+.3f}" if not np.isnan(t) else "    nan", end='')
        print(f" {mean_tau:+.3f}")

        for sr in summary_rows:
            if sr['method'] == name:
                sr['mean_tau_ft'] = round(mean_tau, 4)
                for t_name, t_val in zip(ft_tasks, taus):
                    sr[f'tau_{t_name}'] = round(t_val, 4) if not np.isnan(t_val) else None

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS, 'final_summary.csv'), index=False)
    print(f"\nSaved: final_summary.csv")

    # Sanity check LogME
    logme_rhos = []
    for task_name, _ in ANALYSIS_TASKS:
        sub = df[df['task'] == task_name]
        if len(sub) < 4:
            continue
        rho, _ = spearmanr(sub['logme'].values, sub['lp_acc'].values)
        logme_rhos.append(rho)
    print(f"\n[SANITY CHECK] LogME mean rho = {np.mean(logme_rhos):.4f} (literature expects > 0.3)")

    print("\n>>> Done.")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true', help='Extract missing features (GPU)')
    parser.add_argument('--analyze', action='store_true', help='Run analysis (CPU)')
    parser.add_argument('--all', action='store_true', help='Extract + Analyze')
    args = parser.parse_args()

    if args.all or args.extract:
        run_extraction()
    if args.all or args.analyze:
        run_analysis()
    if not (args.all or args.extract or args.analyze):
        print("Usage: python run_final_benchmark.py --extract|--analyze|--all")
