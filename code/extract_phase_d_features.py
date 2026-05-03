"""
Phase D: extract features for additional 1-variant architectures (HF-cached)
============================================================================

Adds 32+ architectures (each with 1 cached variant) to the stage-1 pool.
These archs cannot do within-arch SCR (only 1 variant each) but help
demonstrate that stage-1 can scale to large architecture pools.

Targets: 6 common tasks (cifar10/100, dtd, eurosat, svhn, stl10).
Skip if cache already exists.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from datasets_local import make_loader

CACHE_DIR = BASE / "results" / "vtab_features"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PHASE_D_VARIANTS = [
    'beit_base_patch16_224.in22k_ft_in22k_in1k',
    'convnext_base.fb_in22k_ft_in1k',
    'deit_base_patch16_224.fb_in1k',
    'deit_small_patch16_224.fb_in1k',
    'deit_tiny_patch16_224.fb_in1k',
    'densenet121.ra_in1k',
    'densenet169.tv_in1k',
    'densenet201.tv_in1k',
    'efficientnet_b0.ra_in1k',
    'efficientnet_b2.ra_in1k',
    'efficientnet_b3.ra2_in1k',
    'efficientnet_b4.ra2_in1k',
    'inception_v3.tv_in1k',
    'maxvit_tiny_tf_224.in1k',
    'mixer_b16_224.goog_in21k_ft_in1k',
    'mnasnet_100.rmsp_in1k',
    'mobilenetv3_large_100.ra_in1k',
    'mobilenetv3_small_100.lamb_in1k',
    'regnetx_032.tv2_in1k',
    'regnety_032.ra_in1k',
    'regnety_040.ra3_in1k',
    'regnety_080.ra3_in1k',
    'resnet101.a1h_in1k',
    'resnet152.a1h_in1k',
    'swin_tiny_patch4_window7_224.ms_in1k',
    'tf_efficientnetv2_s.in21k_ft_in1k',
    'vgg16.tv_in1k',
    'vit_base_patch32_224.augreg_in1k',
    'vit_base_patch32_clip_224.openai',
    'vit_large_patch14_clip_224.openai',
    'vit_large_patch16_224.augreg_in21k_ft_in1k',
    'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
]
TARGETS = ['cifar10', 'cifar100', 'dtd', 'eurosat', 'svhn', 'stl10']
N_SAMPLES = 1000
BATCH = 64


def fp(name): return name.replace(".", "_dot_")
def cache_path(v, ds): return CACHE_DIR / f"{fp(v)}__{ds}.npz"


def extract_one(variant, ds, model, device):
    out = cache_path(variant, ds)
    if out.exists():
        return 'skip'
    loader = make_loader(ds, split='train', batch_size=BATCH,
                         n_samples=N_SAMPLES, num_workers=2)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device); lab = batch[1].numpy()
            else:
                images = batch.to(device); lab = np.zeros(len(images), dtype=np.int64)
            try:
                f = model(images)
            except Exception as e:
                return f'fwd-fail: {str(e)[:80]}'
            if isinstance(f, (list, tuple)): f = f[0]
            feats.append(f.detach().cpu().numpy())
            labels.append(lab)
    H = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    np.savez_compressed(out, H=H, y=y)
    return f'OK shape={H.shape}'


def main():
    import timm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", flush=True)

    n_total = 0; n_ok = 0; n_skip = 0; n_fail = 0
    for vname in PHASE_D_VARIANTS:
        print(f"\n=== {vname} ===", flush=True)
        try:
            t0 = time.time()
            model = timm.create_model(vname, pretrained=True, num_classes=0).to(device)
            print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  !! load fail: {e}", flush=True); n_fail += len(TARGETS); continue
        for ds in TARGETS:
            t0 = time.time()
            n_total += 1
            try:
                status = extract_one(vname, ds, model, device)
                print(f"  {ds:14s}  {status}  ({time.time()-t0:.1f}s)", flush=True)
                if 'OK' in status: n_ok += 1
                elif 'skip' in status: n_skip += 1
                else: n_fail += 1
            except Exception as e:
                print(f"  {ds:14s}  FAIL: {str(e)[:80]}", flush=True); n_fail += 1
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(f"\nSummary: total={n_total}  OK={n_ok}  skip={n_skip}  fail={n_fail}",
          flush=True)


if __name__ == "__main__":
    main()
