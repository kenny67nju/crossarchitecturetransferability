# Feature Extraction Guide

Features are stored as `.npz` files with keys `H` (N×d float32 features)
and `y` (N int64 labels), saved under `features/` (or the path set in
`config.yaml`).  Each file is named `{variant}__{task}.npz` where `.` in
the variant name is replaced by `_dot_` (e.g.
`resnet50_dot_tv_in1k__cifar10.npz`).

## Requirements

- CUDA GPU with ≥16 GB VRAM (tested on V100/A100)
- `pip install -r requirements.txt`  (includes torch, timm, open_clip)
- Dataset downloads (see below)

## Dataset setup

```bash
DATA_ROOT=/path/to/datasets   # set in config.yaml → data_root

# CIFAR-10/100, STL-10, SVHN, Fashion-MNIST — auto-downloaded via torchvision
# DTD  — https://www.robots.ox.ac.uk/~vgg/data/dtd/
# EuroSAT — https://github.com/phelber/EuroSAT (RGB version)
# Flowers-102 — https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
# Oxford Pets — https://www.robots.ox.ac.uk/~vgg/data/pets/
# Food-101 — https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
# FGVC-Aircraft — https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
```

## MV pool (132 timm/torchvision variants × 5 tasks + STL-10 reference)

```bash
python code/extract_phase_b_features.py   # ConvNeXt, Swin, DeiT variants
python code/extract_phase_d_features.py   # 32 single-variant HF-cached archs
python code/extract_phase_e_torchvision.py # torchvision ViT/Swin additions
```

Features saved to `features/vtab_features/` (~40 GB for full MV pool).

## VTAB pool (25 archs × 8 tasks)

```bash
python code/extract_phase_d_features.py   # same script, VTAB arch list
```

The 25 VTAB architectures are a subset of Phase D; the script skips
already-cached files, so running it twice is safe.

## SSL/multimodal pool (8 encoders × 6 tasks)

```bash
python code/extract_phase_f_dino_clip.py
```

Requires pre-downloaded model weights:
- DINO ViT-B/16 and DeiT-S/16: `torch.hub.load('facebookresearch/dino', ...)`
- CLIP ViT-B/16 and ViT-L/14: via `open_clip` or `clip` package
- DINOv2 ViT-B/14: via `torch.hub.load('facebookresearch/dinov2', ...)`

## Naming conventions

| Pool | Feature dir | N per file | Tasks |
|------|-------------|-----------|-------|
| MV pool | `features/vtab_features/` | 1000 | cifar10, cifar100, dtd, eurosat, svhn, stl10 |
| VTAB   | `features/vtab_features/` | 1000 | cifar10, cifar100, dtd, fashion_mnist, flowers102, pets, stl10, svhn |
| SSL    | `features/paper14_features/` | 1000 | task_model_N1000.npz format |

## Disk space estimate

| Pool | # files | Size |
|------|---------|------|
| MV pool (132v × 6t) | ~792 | ~40 GB |
| VTAB (25v × 8t)     | ~200 | ~10 GB |
| SSL (8e × 29t)      | ~232 |  ~8 GB |
