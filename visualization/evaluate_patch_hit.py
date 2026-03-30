"""
Evaluate whether the highest-scoring patch falls inside the ground-truth object bbox.

Usage:
    python evaluate_patch_hit.py \
        --checkpoint ViT_190k.pth \
        --imagenet-root /path/to/imagenet
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from patch_score import DenseViT
from imagenet_dataloader import get_imagenet_val_dataloader


def bbox_to_patch_set(bbox, original_width, original_height,
                      image_size=224, patch_size=16):
    """Convert original-image bbox coordinates to a set of patch indices
    after Resize(256) + CenterCrop(224)."""
    xmin, ymin, xmax, ymax = bbox

    scale = 256.0 / min(original_width, original_height)
    rw = int(original_width * scale)
    rh = int(original_height * scale)

    crop_l = (rw - 224) // 2
    crop_t = (rh - 224) // 2

    xf_min = max(0, min(224, xmin * scale - crop_l))
    yf_min = max(0, min(224, ymin * scale - crop_t))
    xf_max = max(0, min(224, xmax * scale - crop_l))
    yf_max = max(0, min(224, ymax * scale - crop_t))

    n = image_size // patch_size
    indices = set()
    for py in range(int(yf_min // patch_size), min(int(yf_max // patch_size) + 1, n)):
        for px in range(int(xf_min // patch_size), min(int(xf_max // patch_size) + 1, n)):
            indices.add(py * n + px)
    return indices


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    hit = 0
    total = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        bboxes = batch["bbox"].cpu().numpy()
        widths = batch["image_width"].cpu().numpy()
        heights = batch["image_height"].cpu().numpy()

        output = model(images)
        scores = output["patch_scores"]  # [B, num_patches]
        top1 = scores.argmax(dim=1).cpu().numpy()  # [B]

        for i in range(len(top1)):
            patch_set = bbox_to_patch_set(
                bboxes[i].tolist(), int(widths[i]), int(heights[i])
            )
            if int(top1[i]) in patch_set:
                hit += 1
            total += 1

    return hit, total


def main():
    parser = argparse.ArgumentParser(description="LAST-ViT Patch-BBox Hit Ratio Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--imagenet-root", type=str, default=None, help="ImageNet root directory")
    parser.add_argument("--imagenet-label-dir", type=str, default=None)
    parser.add_argument("--imagenet-meta", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = DenseViT(
        image_size=224, patch_size=16,
        num_layers=12, num_heads=12,
        hidden_dim=768, mlp_dim=3072,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print("Model loaded.")

    # Build dataloader
    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_bbox=True,
        filter_multi_objects=True,
        shuffle=False,
    )
    if args.imagenet_root:
        from pathlib import Path
        root = Path(args.imagenet_root)
        dl_kwargs["val_dir"] = str(root / "val")
        if args.imagenet_label_dir:
            dl_kwargs["val_label_dir"] = args.imagenet_label_dir
        if args.imagenet_meta:
            dl_kwargs["meta_file"] = args.imagenet_meta

    dataloader = get_imagenet_val_dataloader(**dl_kwargs)

    hit, total = evaluate(model, dataloader, device)
    ratio = hit / total * 100 if total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Top-1 Patch in BBox: {hit}/{total} ({ratio:.2f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
