"""
Phase E: extract features via torchvision pretrained weights
=============================================================
Goal: For each architecture currently single-variant in our pool, add a
2nd variant using torchvision's pretrained weights (different recipe than timm.HF).

Strategy:
- For each (arch, target task), load torchvision model with DEFAULT weights
- Extract features (1000 samples) on each task
- Save as <arch_name>_dot_torchvision_v1__<task>.npz

Tasks: 6 common (cifar10/100, dtd, eurosat, svhn, stl10) + 4 extra
       (pets, flowers102, food101, fgvc_aircraft) = 10 tasks
"""
from __future__ import annotations

import os, sys, time, warnings
from pathlib import Path

# torchvision uses its own CDN, doesn't need HF
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from datasets_local import make_loader

CACHE_DIR = BASE / "results" / "vtab_features"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# (torchvision name, weights enum, output dim, identifier-for-cache-naming)
# Identifier should be DIFFERENT from existing timm.HF variants
TV_VARIANTS = [
    ('resnet101',   tvm.ResNet101_Weights.IMAGENET1K_V1, 2048, 'tv_v1'),
    ('resnet101',   tvm.ResNet101_Weights.IMAGENET1K_V2, 2048, 'tv_v2'),
    ('resnet152',   tvm.ResNet152_Weights.IMAGENET1K_V1, 2048, 'tv_v1'),
    ('resnet152',   tvm.ResNet152_Weights.IMAGENET1K_V2, 2048, 'tv_v2'),
    ('densenet121', tvm.DenseNet121_Weights.IMAGENET1K_V1, 1024, 'tv_v1'),
    ('densenet169', tvm.DenseNet169_Weights.IMAGENET1K_V1, 1664, 'tv_v1'),
    ('densenet201', tvm.DenseNet201_Weights.IMAGENET1K_V1, 1920, 'tv_v1'),
    ('efficientnet_b0', tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280, 'tv_v1'),
    ('efficientnet_b2', tvm.EfficientNet_B2_Weights.IMAGENET1K_V1, 1408, 'tv_v1'),
    ('efficientnet_b3', tvm.EfficientNet_B3_Weights.IMAGENET1K_V1, 1536, 'tv_v1'),
    ('efficientnet_b4', tvm.EfficientNet_B4_Weights.IMAGENET1K_V1, 1792, 'tv_v1'),
    ('vgg16',       tvm.VGG16_Weights.IMAGENET1K_V1,    4096, 'tv_v1'),
    ('vgg16_bn',    tvm.VGG16_BN_Weights.IMAGENET1K_V1, 4096, 'tv_v1'),
    ('inception_v3', tvm.Inception_V3_Weights.IMAGENET1K_V1, 2048, 'tv_v1'),
    ('mnasnet1_0',  tvm.MNASNet1_0_Weights.IMAGENET1K_V1, 1280, 'tv_v1'),
    ('mobilenet_v3_large', tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1, 960, 'tv_v1'),
    ('mobilenet_v3_large', tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2, 960, 'tv_v2'),
    ('mobilenet_v3_small', tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576, 'tv_v1'),
    ('swin_t',      tvm.Swin_T_Weights.IMAGENET1K_V1, 768, 'tv_v1'),
    ('swin_s',      tvm.Swin_S_Weights.IMAGENET1K_V1, 768, 'tv_v1'),
    ('swin_b',      tvm.Swin_B_Weights.IMAGENET1K_V1, 1024, 'tv_v1'),
    ('vit_b_16',    tvm.ViT_B_16_Weights.IMAGENET1K_V1, 768, 'tv_v1'),
    ('vit_b_32',    tvm.ViT_B_32_Weights.IMAGENET1K_V1, 768, 'tv_v1'),
    ('vit_l_16',    tvm.ViT_L_16_Weights.IMAGENET1K_V1, 1024, 'tv_v1'),
    ('regnet_x_3_2gf', tvm.RegNet_X_3_2GF_Weights.IMAGENET1K_V1, 1008, 'tv_v1'),
    ('regnet_y_3_2gf', tvm.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1, 1512, 'tv_v1'),
]

# Limit to 6 working tasks (pets/flowers/food/aircraft skipped due to
# datasets_local name mismatch + slow download issues)
TARGETS = ['cifar10', 'cifar100', 'dtd', 'eurosat', 'svhn', 'stl10']
N_SAMPLES = 1000
BATCH = 64


def fp(name): return name.replace(".", "_dot_")


def cache_path(arch, suffix, ds):
    """e.g., resnet101_dot_torchvision_tv_v1__cifar10.npz"""
    name = f"{arch}_dot_torchvision_{suffix}"
    return CACHE_DIR / f"{name}__{ds}.npz"


def make_feature_extractor(model, arch_name):
    """Strip classifier head; return feature module that returns (N, d) tensor."""
    m = model
    # Remove classifier — patterns vary by architecture family
    if hasattr(m, 'fc'):
        m.fc = nn.Identity()        # ResNet, RegNet
    elif hasattr(m, 'classifier'):
        m.classifier = nn.Identity()  # DenseNet, EfficientNet, MNASNet, MobileNet, VGG
    elif hasattr(m, 'heads'):
        m.heads = nn.Identity()       # ViT
    elif hasattr(m, 'head'):
        m.head = nn.Identity()        # Swin
    if hasattr(m, 'aux_logits'):
        m.aux_logits = False           # Inception
    return m


def extract_one(arch, weights, suffix, ds, device):
    out = cache_path(arch, suffix, ds)
    if out.exists():
        return 'skip'
    try:
        model = getattr(tvm, arch)(weights=weights).to(device)
    except Exception as e:
        return f'load-fail: {str(e)[:80]}'
    model = make_feature_extractor(model, arch).eval()
    transform = weights.transforms()  # use weights' built-in transform
    loader = make_loader(ds, split='train', batch_size=BATCH,
                         n_samples=N_SAMPLES, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, lab = batch[0], batch[1].numpy()
            else:
                images, lab = batch, np.zeros(len(batch), dtype=np.int64)
            # Apply weights transform (re-normalize as expected)
            images = transform(images.to(device))
            try:
                f = model(images)
            except Exception as e:
                return f'fwd-fail: {str(e)[:80]}'
            if isinstance(f, (list, tuple)):
                f = f[0]
            if f.dim() > 2:
                f = f.flatten(1)
            feats.append(f.cpu().numpy())
            labels.append(lab)
    H = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    np.savez_compressed(out, H=H, y=y)
    return f'OK shape={H.shape}'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", flush=True)

    n_total = 0; n_ok = 0; n_skip = 0; n_fail = 0
    for arch, weights, dim, suffix in TV_VARIANTS:
        print(f"\n=== {arch} ({suffix}) ===", flush=True)
        for ds in TARGETS:
            t0 = time.time()
            n_total += 1
            try:
                status = extract_one(arch, weights, suffix, ds, device)
                print(f"  {ds:14s}  {status}  ({time.time()-t0:.1f}s)", flush=True)
                if 'OK' in status: n_ok += 1
                elif 'skip' in status: n_skip += 1
                else: n_fail += 1
            except Exception as e:
                print(f"  {ds:14s}  FAIL: {str(e)[:80]}", flush=True); n_fail += 1
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nSummary: total={n_total}  OK={n_ok}  skip={n_skip}  fail={n_fail}",
          flush=True)


if __name__ == "__main__":
    main()
