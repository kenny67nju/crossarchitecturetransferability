#!/usr/bin/env python3
"""Measure real feature-extraction throughput on an Ascend NPU."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_npu  # noqa: F401 - registers the npu backend
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--npu", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = datasets.CIFAR10(args.data_root, train=True, transform=transform, download=False)
    dataset = Subset(dataset, list(range(min(args.samples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=False)

    device = torch.device(f"npu:{args.npu}")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval().to(device)

    features, labels = [], []
    torch.npu.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            output = model(images.to(device, non_blocking=True))
            features.append(output.cpu().numpy())
            labels.append(target.numpy())
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    matrix = np.concatenate(features)
    print(
        {
            "device": str(device),
            "shape": matrix.shape,
            "samples": len(matrix),
            "seconds": elapsed,
            "images_per_second": len(matrix) / elapsed,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
