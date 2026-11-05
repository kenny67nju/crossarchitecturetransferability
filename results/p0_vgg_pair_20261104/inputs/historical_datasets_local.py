"""
Paper06 — Grassmann 统一框架
数据集加载模块

支持数据集（全部 torchvision 可自动下载）：
    手写域:   mnist, kmnist, fashion_mnist
    自然图像: cifar10, cifar100, stl10
    数字场景: svhn
    卫星遥感: eurosat
    细粒度:   oxford_pets, flowers102, food101
    纹理:     dtd

用法：
    loader = make_loader("cifar10", split="train", batch_size=256)
    H = probe.extract(loader, max_samples=2000)
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as dsets
import numpy as np

from config import DATA_DIR, BATCH_SIZE, SEED


# ─────────────────────────────────────────────────────────────
# 标准变换（分辨率统一到 224×224 用于大模型；32×32 用于 CNN 快速实验）
# ─────────────────────────────────────────────────────────────

def _tf_224(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def _tf_224_vit():
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _tf_32():
    """轻量化：32×32，无需大模型，用于 E1 合成实验的真实数据对。"""
    return T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ─────────────────────────────────────────────────────────────
# 单通道 → 三通道转换（MNIST 系列）
# ─────────────────────────────────────────────────────────────

class _GrayToRGB:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


def _tf_gray_224():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        _GrayToRGB(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────
# 数据集工厂
# ─────────────────────────────────────────────────────────────

def _load(name: str, split: str, transform) -> torch.utils.data.Dataset:
    """内部：根据数据集名称和 split 返回 Dataset 对象。"""
    root = str(DATA_DIR)
    train = (split == "train")

    if name == "mnist":
        return dsets.MNIST(root, train=train, transform=transform, download=True)

    elif name == "kmnist":
        return dsets.KMNIST(root, train=train, transform=transform, download=True)

    elif name == "fashion_mnist":
        return dsets.FashionMNIST(root, train=train, transform=transform, download=True)

    elif name == "cifar10":
        return dsets.CIFAR10(root, train=train, transform=transform, download=True)

    elif name == "cifar100":
        return dsets.CIFAR100(root, train=train, transform=transform, download=True)

    elif name == "stl10":
        stl_split = "train" if train else "test"
        return dsets.STL10(root, split=stl_split, transform=transform, download=True)

    elif name == "svhn":
        svhn_split = "train" if train else "test"
        return dsets.SVHN(root, split=svhn_split, transform=transform, download=True)

    elif name == "eurosat":
        # EuroSAT 没有官方 train/test split，直接用全部
        try:
            ds = dsets.EuroSAT(root, transform=transform, download=True)
        except Exception:
            ds = dsets.EuroSAT(root, transform=transform, download=False)
        return ds

    elif name == "oxford_pets":
        split_str = "trainval" if train else "test"
        return dsets.OxfordIIITPet(
            root, split=split_str, transform=transform, download=True)

    elif name == "flowers102":
        split_str = "train" if train else "test"
        return dsets.Flowers102(
            root, split=split_str, transform=transform, download=True)

    elif name == "food101":
        split_str = "train" if train else "test"
        return dsets.Food101(
            root, split=split_str, transform=transform, download=True)

    elif name == "dtd":
        split_str = "train" if train else "test"
        return dsets.DTD(root, split=split_str, transform=transform, download=True)

    else:
        raise ValueError(f"未知数据集: '{name}'")


def load_dataset(
    name: str,
    split: str = "train",
    transform_size: int = 224,
) -> torch.utils.data.Dataset:
    """
    加载数据集（完整版，供特征提取使用）。

    Args:
        name:           数据集名称（见模块 docstring）
        split:          "train" 或 "test"
        transform_size: 图像 resize 目标大小（224 用于 ViT/ResNet，32 用于 CNN 快速）

    Returns:
        torch.utils.data.Dataset
    """
    gray_datasets = {"mnist", "kmnist", "fashion_mnist"}
    if transform_size == 32:
        tf = _tf_32()
    elif name in gray_datasets:
        tf = _tf_gray_224()
    else:
        tf = _tf_224()

    return _load(name, split, tf)


def make_loader(
    name: str,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    n_samples: Optional[int] = None,
    transform_size: int = 224,
    seed: int = SEED,
    num_workers: int = 0,
) -> DataLoader:
    """
    构建 DataLoader，可选随机子采样 n_samples 条。

    Args:
        name:       数据集名称
        split:      "train" 或 "test"
        batch_size: 批大小
        n_samples:  若设定，随机子采样至此数量（保证可复现）
        transform_size: 图像大小
        seed:       随机种子（子采样时用）
        num_workers: DataLoader 工作进程数

    Returns:
        DataLoader
    """
    ds = load_dataset(name, split, transform_size)

    if n_samples is not None and n_samples < len(ds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=n_samples, replace=False)
        ds = Subset(ds, idx.tolist())

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def dataset_info(name: str) -> dict:
    """返回数据集的静态元信息（类别数、图像大小等）。"""
    INFO = {
        "mnist":         {"n_classes": 10,  "img_size": 28,  "domain": "handwritten"},
        "kmnist":        {"n_classes": 10,  "img_size": 28,  "domain": "handwritten"},
        "fashion_mnist": {"n_classes": 10,  "img_size": 28,  "domain": "handwritten"},
        "cifar10":       {"n_classes": 10,  "img_size": 32,  "domain": "natural"},
        "cifar100":      {"n_classes": 100, "img_size": 32,  "domain": "natural"},
        "stl10":         {"n_classes": 10,  "img_size": 96,  "domain": "natural"},
        "svhn":          {"n_classes": 10,  "img_size": 32,  "domain": "digit_scene"},
        "eurosat":       {"n_classes": 10,  "img_size": 64,  "domain": "satellite"},
        "oxford_pets":   {"n_classes": 37,  "img_size": 224, "domain": "finegrained"},
        "flowers102":    {"n_classes": 102, "img_size": 224, "domain": "finegrained"},
        "food101":       {"n_classes": 101, "img_size": 224, "domain": "finegrained"},
        "dtd":           {"n_classes": 47,  "img_size": 224, "domain": "texture"},
    }
    if name not in INFO:
        raise ValueError(f"未知数据集: '{name}'")
    return INFO[name]


# ─────────────────────────────────────────────────────────────
# 特征矩阵提取工具（与 ProbeModel 配合）
# ─────────────────────────────────────────────────────────────

def extract_features(
    probe,
    dataset_name: str,
    split: str = "train",
    n_samples: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用探针模型提取指定数据集的特征矩阵和标签。

    Args:
        probe:        ProbeModel 实例
        dataset_name: 数据集名称
        split:        "train" 或 "test"
        n_samples:    子采样数（None = 全部）
        batch_size:   批大小

    Returns:
        H:      (N, d) float32 特征矩阵
        labels: (N,) int64 标签数组
    """
    loader = make_loader(
        dataset_name, split=split,
        n_samples=n_samples, batch_size=batch_size,
    )

    feats, labs = [], []
    import torch
    with torch.no_grad():
        for batch in loader:
            images, targets = batch[0], batch[1]
            images = images.to(probe.device)
            f = probe._forward(images)
            feats.append(f.cpu().numpy())
            labs.append(targets.numpy() if hasattr(targets, "numpy") else np.array(targets))

    H      = np.concatenate(feats, axis=0).astype(np.float32)
    labels = np.concatenate(labs,  axis=0).astype(np.int64)
    return H, labels


# ─────────────────────────────────────────────────────────────
# 单元测试
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("datasets.py — 单元测试（仅测试 DataLoader 构建，不下载）")

    for ds_name in ["cifar10", "mnist"]:
        info = dataset_info(ds_name)
        print(f"  {ds_name}: n_classes={info['n_classes']}, domain={info['domain']}")

    print("dataset_info 测试通过 ✓")
