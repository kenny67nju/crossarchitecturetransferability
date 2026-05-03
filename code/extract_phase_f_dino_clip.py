"""
Phase F: extract features from cached DINO + CLIP probes
=========================================================
Uses already-cached models (no download needed):
  /root/.cache/torch/hub/dino_vitbase16_pretrain.pth
  /root/.cache/torch/hub/dino_deitsmall16_pretrain.pth
  /root/.cache/clip + open_clip (CLIP ViT-B/16, B/32 from timm HF cache too)

Naming: <model_id>__<task>.npz with model_id easily identifiable
"""
from __future__ import annotations

import os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from datasets_local import make_loader

CACHE_DIR = BASE / "results" / "vtab_features"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['cifar10', 'cifar100', 'dtd', 'eurosat', 'svhn', 'stl10']
N_SAMPLES = 1000
BATCH = 64

DINO_HUB_DIR = "/root/.cache/torch/hub/facebookresearch_dino_main"


def fp(name): return name.replace(".", "_dot_")
def cache_path(model_id, ds): return CACHE_DIR / f"{fp(model_id)}__{ds}.npz"


def extract_features(model, transform, ds, device, model_id):
    out = cache_path(model_id, ds)
    if out.exists():
        return 'skip'
    loader = make_loader(ds, split='train', batch_size=BATCH,
                         n_samples=N_SAMPLES, num_workers=2)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, lab = batch[0], batch[1].numpy()
            else:
                images, lab = batch, np.zeros(len(batch), dtype=np.int64)
            if transform is not None:
                images = transform(images)
            images = images.to(device)
            try:
                f = model(images)
            except Exception as e:
                return f'fwd-fail: {str(e)[:80]}'
            if isinstance(f, (tuple, list)):
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

    models_to_extract = []

    # ── 1. DINO ViT-B/16 from local hub cache ─────────────────────
    try:
        print("Loading DINO ViT-B/16 from torch.hub local cache...", flush=True)
        # Use torch.hub with source='local'
        if os.path.exists(DINO_HUB_DIR):
            dino_vitb = torch.hub.load(DINO_HUB_DIR, 'dino_vitb16', source='local',
                                        pretrained=True, trust_repo=True)
        else:
            dino_vitb = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',
                                        pretrained=True, trust_repo=True)
        dino_vitb = dino_vitb.to(device)
        from torchvision import transforms as T
        dino_transform = T.Compose([
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        models_to_extract.append(('dino_vitb16_facebook', dino_vitb, dino_transform))
    except Exception as e:
        print(f"  !! DINO ViT-B/16 failed: {e}", flush=True)

    # ── 2. CLIP ViT-B/16 (OpenAI) via open_clip ───────────────────
    try:
        print("Loading CLIP ViT-B/16 via open_clip (cached)...", flush=True)
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai')
        clip_model = clip_model.visual.to(device)
        # open_clip preprocess includes resize+normalize; but datasets_local
        # already returns tensors, so we apply manual normalization.
        from torchvision import transforms as T
        clip_transform = T.Compose([
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        models_to_extract.append(('clip_vitb16_openai', clip_model, clip_transform))
    except Exception as e:
        print(f"  !! CLIP ViT-B/16 failed: {e}", flush=True)

    # ── 3. CLIP ViT-B/32 (OpenAI) via open_clip ───────────────────
    try:
        print("Loading CLIP ViT-B/32 via open_clip...", flush=True)
        import open_clip
        clip_model32, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        clip_model32 = clip_model32.visual.to(device)
        from torchvision import transforms as T
        clip_transform32 = T.Compose([
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        models_to_extract.append(('clip_vitb32_openai', clip_model32, clip_transform32))
    except Exception as e:
        print(f"  !! CLIP ViT-B/32 failed: {e}", flush=True)

    print(f"\n{len(models_to_extract)} probes ready: " +
          ", ".join(m[0] for m in models_to_extract), flush=True)

    n_total = n_ok = n_skip = n_fail = 0
    for model_id, model, tf in models_to_extract:
        print(f"\n=== {model_id} ===", flush=True)
        for ds in TARGETS:
            t0 = time.time()
            n_total += 1
            try:
                status = extract_features(model, tf, ds, device, model_id)
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
