#!/usr/bin/env python3
"""Resumable MV/EX feature extraction on Ascend NPUs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


ALIASES = {
    "beit_b": "beit_base_patch16_224.in22k_ft_in22k_in1k",
    "clip_b32": "vit_base_patch32_clip_224.openai",
    "deit_b": "deit_base_patch16_224.fb_in1k",
    "dino_s": "vit_small_patch16_224.dino",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "mae_b": "vit_base_patch16_224.mae",
    "mixer_b16_224": "mixer_b16_224.goog_in21k_ft_in1k",
    "tf_efficientnetv2_s": "tf_efficientnetv2_s.in1k",
}

PLAIN_TORCHVISION = {
    "googlenet",
    "inception_v3",
    "mnasnet1_0",
    "mobilenet_v2",
    "resnext50_32x4d",
    "vgg16",
    "vit_b_16",
}

MANUAL_WEIGHT_DIR = Path("/data/Paper06/manual_weights")

DATASET_NAMES = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "fashion_mnist": "FashionMNIST",
    "fgvc_aircraft": "FGVCAircraft",
    "flowers102": "Flowers102",
    "food101": "Food101",
    "pets": "OxfordIIITPet",
    "svhn": "SVHN",
    "stl10": "STL10",
}


def feature_filename(variant: str, task: str) -> str:
    return f"{variant.replace('.', '_dot_')}__{task}.npz"


def make_transform(task: str, image_size: int = 224):
    operations = [
        transforms.Resize(round(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if task == "fashion_mnist":
        operations.append(transforms.Grayscale(num_output_channels=3))
    operations.append(
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )
    return transforms.Compose(operations)


class ArrowImageDataset(Dataset):
    """Adapter for an already-downloaded Hugging Face Arrow image split."""

    def __init__(self, arrow_path: Path, transform) -> None:
        from datasets import Dataset as HFDataset

        self.dataset = HFDataset.from_file(str(arrow_path))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        return self.transform(row["image"].convert("RGB")), int(row["label"])


def make_dataset(
    task: str,
    root: Path,
    hf_data_cache: Path | None = None,
    image_size: int = 224,
):
    transform = make_transform(task, image_size)
    if hf_data_cache is not None and task in {"svhn", "fashion_mnist"}:
        filename = "svhn-train.arrow" if task == "svhn" else "fashion_mnist-train.arrow"
        matches = list(hf_data_cache.rglob(filename))
        if matches:
            return ArrowImageDataset(matches[0], transform)
    if task == "cifar10":
        return datasets.CIFAR10(root, train=True, transform=transform, download=True)
    if task == "cifar100":
        return datasets.CIFAR100(root, train=True, transform=transform, download=True)
    if task == "dtd":
        return datasets.DTD(root, split="train", transform=transform, download=True)
    if task == "eurosat":
        return datasets.EuroSAT(root, transform=transform, download=True)
    if task == "fashion_mnist":
        return datasets.FashionMNIST(root, train=True, transform=transform, download=True)
    if task == "fgvc_aircraft":
        return datasets.FGVCAircraft(root, split="trainval", transform=transform, download=True)
    if task == "flowers102":
        return datasets.Flowers102(root, split="train", transform=transform, download=True)
    if task == "food101":
        return datasets.Food101(root, split="train", transform=transform, download=True)
    if task == "pets":
        return datasets.OxfordIIITPet(root, split="trainval", transform=transform, download=True)
    if task == "svhn":
        return datasets.SVHN(root, split="train", transform=transform, download=True)
    if task == "stl10":
        return datasets.STL10(root, split="train", transform=transform, download=True)
    raise ValueError(f"unsupported task: {task}")


def _torchvision_model(variant: str):
    if ".torchvision_" in variant:
        base, marker = variant.split(".torchvision_", 1)
    else:
        base, marker = variant, ""
    torchvision_names = {
        "mobilenet_v3_large": "mobilenet_v3_large",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "regnet_x_3_2gf": "regnet_x_3_2gf",
        "regnet_y_3_2gf": "regnet_y_3_2gf",
        "mnasnet1_0": "mnasnet1_0",
    }
    name = torchvision_names.get(base, base)
    weights_enum = models.get_model_weights(name)
    if "tv_v2" in marker and hasattr(weights_enum, "IMAGENET1K_V2"):
        weights = weights_enum.IMAGENET1K_V2
    elif hasattr(weights_enum, "IMAGENET1K_V1"):
        weights = weights_enum.IMAGENET1K_V1
    else:
        weights = weights_enum.DEFAULT
    model = models.get_model(name, weights=weights)
    if name in {"vgg16", "vgg16_bn"}:
        # Keep both 4096-wide hidden fully connected layers and remove only
        # the ImageNet classifier. Replacing the whole classifier would
        # incorrectly expose the 7x7x512 = 25088 convolutional flattening.
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        return model
    for head in ("fc", "classifier", "heads", "head"):
        if hasattr(model, head):
            setattr(model, head, nn.Identity())
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    return model


class OpenClipImageEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.pretrained_cfg = {"input_size": (3, 224, 224)}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images, normalize=False)


def make_model(variant: str):
    if variant == "clip_b32":
        import open_clip

        checkpoint = MANUAL_WEIGHT_DIR / "ViT-B-32.pt"
        model = open_clip.create_model("ViT-B-32", pretrained=None)
        scripted = torch.jit.load(str(checkpoint), map_location="cpu")
        incompatible = model.load_state_dict(scripted.state_dict(), strict=False)
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if key not in {"input_resolution", "context_length", "vocab_size"}
        ]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                f"CLIP checkpoint mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={unexpected}"
            )
        return OpenClipImageEncoder(model)
    if ".torchvision_" in variant or variant in PLAIN_TORCHVISION:
        return _torchvision_model(variant)
    import timm

    model_name = ALIASES.get(variant, variant)
    checkpoint = MANUAL_WEIGHT_DIR / f"{model_name}.safetensors"
    if checkpoint.exists():
        from safetensors.torch import load_file

        model = timm.create_model(model_name, pretrained=False, num_classes=0)
        incompatible = model.load_state_dict(load_file(checkpoint), strict=False)
        allowed_head_prefixes = ("head.", "fc.", "classifier.")
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith(allowed_head_prefixes)
        ]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch for {model_name}: "
                f"missing={incompatible.missing_keys}, unexpected={unexpected}"
            )
        return model
    return timm.create_model(model_name, pretrained=True, num_classes=0)


def load_manifest(path: Path) -> dict[str, list[str]]:
    variants: dict[str, list[str]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            variants.setdefault(row["variant"], []).append(row["task"])
    return {variant: sorted(set(tasks)) for variant, tasks in variants.items()}


def append_failure(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hf-data-cache", type=Path)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1000, help="0 means the full training split")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_dir / f"failures_shard{args.shard_index}.jsonl"
    device = torch.device(f"npu:{args.npu}")
    all_variants = sorted(load_manifest(args.manifest).items())
    assigned = all_variants[args.shard_index :: args.shard_count]
    print(f"device={device} variants={len(assigned)} samples={args.samples}", flush=True)

    for variant_index, (variant, tasks) in enumerate(assigned, 1):
        missing = [
            task for task in tasks if not (args.output_dir / feature_filename(variant, task)).exists()
        ]
        if not missing:
            print(f"[{variant_index}/{len(assigned)}] {variant}: all cached", flush=True)
            continue
        try:
            started = time.perf_counter()
            model = make_model(variant).eval().to(device)
            print(
                f"[{variant_index}/{len(assigned)}] {variant}: loaded in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
        except Exception as error:
            append_failure(
                failure_path,
                {"variant": variant, "stage": "load", "error": repr(error)},
            )
            print(f"[{variant_index}/{len(assigned)}] {variant}: LOAD FAILED {error}", flush=True)
            continue

        for task in missing:
            output_path = args.output_dir / feature_filename(variant, task)
            started = time.perf_counter()
            try:
                configured_size = getattr(model, "pretrained_cfg", {}).get(
                    "input_size", (3, 224, 224)
                )
                image_size = int(configured_size[-1])
                dataset = make_dataset(
                    task,
                    args.data_root,
                    args.hf_data_cache,
                    image_size=image_size,
                )
                if args.samples and args.samples < len(dataset):
                    rng = np.random.default_rng(42)
                    dataset = Subset(
                        dataset,
                        rng.choice(len(dataset), args.samples, replace=False).tolist(),
                    )
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=False,
                )
                feature_batches, label_batches = [], []
                with torch.inference_mode():
                    for images, labels in loader:
                        output = model(images.to(device))
                        if isinstance(output, (tuple, list)):
                            output = output[0]
                        if output.ndim > 2:
                            output = output.flatten(1)
                        feature_batches.append(output.float().cpu().numpy())
                        label_batches.append(np.asarray(labels))
                features = np.concatenate(feature_batches).astype(np.float32, copy=False)
                labels = np.concatenate(label_batches).astype(np.int64, copy=False)
                temporary = output_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, H=features, y=labels)
                temporary.replace(output_path)
                print(
                    f"  {task}: {features.shape} in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            except Exception as error:
                append_failure(
                    failure_path,
                    {"variant": variant, "task": task, "stage": "extract", "error": repr(error)},
                )
                print(f"  {task}: FAILED {error}", flush=True)

        del model
        gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
