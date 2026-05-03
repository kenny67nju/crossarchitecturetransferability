"""
Phase B: extract features for additional architecture variants
==============================================================

Adds 4 new architectures (each with 2-4 variants) to the SCR pool:
  - ConvNeXt-Base: 3-4 variants
  - Swin-Base: 2 variants
  - DeiT3-Base: 2 variants
  - EfficientNet-B0: 2 variants
  → ~10 new variants × 9 tasks ≈ 90 (variant, task) pairs to extract

For each (variant, dataset), extract features (1000 train samples) and save as
results/vtab_features/<variant_dot_replaced>__<dataset>.npz

Then re-run the SCR scoring to include these in the pool.

Skips already-cached pairs.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')

# Add Paper06 path so we can import datasets / probe_models
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from datasets_local import make_loader

CACHE_DIR = BASE / "results" / "vtab_features"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Use only HF-cached variants (network unavailable; HF_HUB_OFFLINE=1)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
PHASE_B_VARIANTS = [
    # ConvNeXt-Small (2 variants cached)
    'convnext_small.fb_in1k',
    'convnext_small.in12k_ft_in1k',
    # ConvNeXt-Tiny (2 variants cached)
    'convnext_tiny.fb_in1k',
    'convnext_tiny.in12k_ft_in1k',
    # Swin-Small (2 variants cached)
    'swin_small_patch4_window7_224.ms_in1k',
    'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
]

# Restrict to datasets that don't require fresh downloads (network is unstable).
# fgvc_aircraft is unsupported by datasets_local; flowers102/food101 trigger slow download.
TARGETS = ['cifar10', 'cifar100', 'dtd', 'eurosat', 'pets', 'svhn', 'stl10']

N_SAMPLES = 1000
BATCH = 64


def fp(name): return name.replace(".", "_dot_")


def cache_path(variant, ds):
    return CACHE_DIR / f"{fp(variant)}__{ds}.npz"


def extract_one(variant, ds, model, device):
    out_path = cache_path(variant, ds)
    if out_path.exists():
        return 'skip'
    loader = make_loader(ds, split='train', batch_size=BATCH,
                         n_samples=N_SAMPLES, num_workers=2)
    feats = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                lab = batch[1].numpy()
            else:
                images = batch.to(device)
                lab = np.zeros(len(images), dtype=np.int64)
            try:
                f = model(images)
            except Exception as e:
                print(f"!! forward failed {variant}/{ds}: {e}")
                return 'fail'
            if isinstance(f, (list, tuple)):
                f = f[0]
            f = f.detach().cpu().numpy()
            feats.append(f)
            labels.append(lab)
    H = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    np.savez_compressed(out_path, H=H, y=y)
    return f'OK shape={H.shape}'


def main():
    import timm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}", flush=True)

    for vname in PHASE_B_VARIANTS:
        print(f"\n=== {vname} ===", flush=True)
        try:
            t0 = time.time()
            model = timm.create_model(vname, pretrained=True, num_classes=0)
            model = model.to(device)
            print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"!! could not load {vname}: {e}", flush=True)
            continue

        for ds in TARGETS:
            t0 = time.time()
            try:
                status = extract_one(vname, ds, model, device)
                print(f"  {ds:18s}  {status}  ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"  {ds:18s}  FAIL: {e}", flush=True)

        # Free GPU
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
