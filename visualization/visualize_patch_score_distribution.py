#!/usr/bin/env python
"""
Visualize Figure 2a style patch-score distributions with SAM2-refined masks.

The script:
1. Loads ImageNet validation images with single-object bbox annotations.
2. Uses cached SAM2 masks or bbox-prompted SAM2 refinement as the foreground proxy.
3. Marks a patch as foreground only when most of the patch lies inside the mask.
4. Plots foreground/background patch-score distributions for either a naive CLS ViT
   or the released LAST-ViT checkpoint.

Notes:
- All dataset and output paths must be passed explicitly from the command line.
- `--model-kind supervised_cls` may trigger a torchvision download for the
  `ViT_B_16_Weights.IMAGENET1K_V1` weights if they are not cached locally.
- SAM2 refinement may trigger a Hugging Face download for `--sam-model-id`
  when cached masks are missing and the model is not cached locally.
- `--model-kind lastvit` does not download the released checkpoint automatically;
  you must provide a local checkpoint path via `--checkpoint`, you can download it via: https://github.com/ChengShiest/LAST-ViT/releases/download/weights2/ViT_190k.pth
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms import transforms as T


@dataclass(frozen=True)
class SampleEntry:
    """Metadata for one validation image and its single-object bbox."""

    image_name: str
    image_path: Path
    bbox: tuple[float, float, float, float]
    width: int
    height: int


class LASTAggregator(nn.Module):
    """Minimal LAST-ViT token selector used for released checkpoint evaluation."""

    def __init__(
        self,
        hidden_dim: int,
        topk: int = 1,
        sigma: float | None = None,
        eps: float = 1e-6,
        score_formula: str = "repo",
    ) -> None:
        super().__init__()
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}.")
        if score_formula not in {"repo", "paper"}:
            raise ValueError(f"Unsupported score_formula={score_formula!r}.")
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.sigma = sigma if sigma is not None else hidden_dim ** 0.5
        self.eps = eps
        self.score_formula = score_formula
        self.register_buffer("_cached_kernel", torch.empty(0), persistent=False)

    def _get_kernel(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Lazily build the 1D Gaussian kernel used for low-pass filtering."""
        if self._cached_kernel.numel() != self.hidden_dim:
            positions = torch.arange(
                -self.hidden_dim // 2 + 1,
                self.hidden_dim // 2 + 1,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            kernel = torch.exp(-0.5 * (positions / self.sigma) ** 2)
            kernel = kernel / kernel.max()
            self._cached_kernel = kernel
        return self._cached_kernel.view(1, 1, self.hidden_dim).to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )

    def low_pass_filter(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Apply the frequency-domain low-pass filter from LAST-ViT."""
        original_dtype = patch_tokens.dtype
        if patch_tokens.dtype in {torch.float16, torch.bfloat16}:
            patch_tokens = patch_tokens.float()
        kernel = self._get_kernel(patch_tokens)
        spectrum = torch.fft.fft(patch_tokens, dim=-1)
        spectrum = torch.fft.fftshift(spectrum, dim=-1)
        spectrum = spectrum * kernel
        spectrum = torch.fft.ifftshift(spectrum, dim=-1)
        filtered = torch.fft.ifft(spectrum, dim=-1).real
        return filtered.to(dtype=original_dtype)

    def stability_score(
        self,
        patch_tokens: torch.Tensor,
        low_pass_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token stability scores using the repository or paper formula."""
        if self.score_formula == "repo":
            numerator = patch_tokens
            denominator = (low_pass_tokens - patch_tokens).abs().clamp_min(self.eps)
            return numerator / denominator
        return low_pass_tokens / (low_pass_tokens - patch_tokens + self.eps)

    def _vote_counts(self, selected_indices: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Count how often each patch is selected across feature channels."""
        batch_size, topk, num_channels = selected_indices.shape
        flat_indices = selected_indices.permute(0, 2, 1).reshape(batch_size, num_channels * topk)
        votes = torch.zeros(
            batch_size,
            num_patches,
            device=selected_indices.device,
            dtype=torch.long,
        )
        ones = torch.ones_like(flat_indices, dtype=votes.dtype)
        votes.scatter_add_(1, flat_indices, ones)
        return votes

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select the most stable patches and pool them into the final token."""
        _, num_patches, hidden_dim = patch_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}.")
        low_pass_tokens = self.low_pass_filter(patch_tokens)
        stability_scores = self.stability_score(patch_tokens, low_pass_tokens)
        topk = min(self.topk, num_patches)
        _, selected_indices = torch.topk(stability_scores, k=topk, dim=1, largest=True)
        selected_tokens = torch.gather(patch_tokens, 1, selected_indices)
        pooled = selected_tokens.mean(dim=1)
        vote_counts = self._vote_counts(selected_indices, num_patches)
        return {
            "cls_token": pooled,
            "vote_counts": vote_counts,
            "selected_indices": selected_indices,
        }


class DenseViT(VisionTransformer):
    """VisionTransformer wrapper that exposes patch scores for visualization."""

    def __init__(
        self,
        *args: Any,
        aggregation: str = "cls",
        topk: int = 1,
        score_formula: str = "repo",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if aggregation not in {"cls", "last"}:
            raise ValueError(f"Unsupported aggregation={aggregation!r}.")
        self.aggregation = aggregation
        self.selector = (
            LASTAggregator(hidden_dim=self.hidden_dim, topk=topk, score_formula=score_formula)
            if aggregation == "last"
            else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits, patch tokens, normalized pooled token, and patch scores."""
        x = self._process_input(x)
        batch_size = x.shape[0]
        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.encoder(x)

        encoder_cls = x[:, 0]
        patch_tokens = x[:, 1:]
        aggregation_output = None

        # The released LAST-ViT checkpoint replaces the naive CLS token with
        # a stability-based token selector. The supervised baseline keeps CLS.
        if self.selector is not None:
            aggregation_output = self.selector(patch_tokens)
            pooled = aggregation_output["cls_token"]
        else:
            pooled = encoder_cls

        logits = self.heads(pooled)
        patch_scores = F.cosine_similarity(
            patch_tokens,
            pooled.unsqueeze(1).expand_as(patch_tokens),
            dim=-1,
        )
        return {
            "logits": logits,
            "patch_tokens": patch_tokens,
            "patch_scores": patch_scores,
            "pooled_token": pooled,
            "vote_counts": None if aggregation_output is None else aggregation_output["vote_counts"],
            "selected_indices": None if aggregation_output is None else aggregation_output["selected_indices"],
        }


class ImageNetBBoxDataset(Dataset):
    """Dataset wrapper that keeps image loading simple and explicit."""

    def __init__(self, samples: list[SampleEntry], transform: T.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one image and return the tensor plus bbox metadata."""
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return {
            "image": self.transform(image),
            "image_name": sample.image_name,
            "image_path": str(sample.image_path),
            "bbox": torch.tensor(sample.bbox, dtype=torch.float32),
            "image_width": sample.width,
            "image_height": sample.height,
        }


class Sam2Refiner:
    """Thin SAM2 wrapper for bbox-prompted mask refinement."""

    def __init__(self, model_id: str, device: str, transformers_overlay: str = "") -> None:
        if transformers_overlay:
            overlay_path = Path(transformers_overlay)
            if not overlay_path.exists():
                raise FileNotFoundError(f"Transformers overlay not found: {overlay_path}")
            if str(overlay_path) not in sys.path:
                sys.path.insert(0, str(overlay_path))

        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise ImportError(
                "Sam2Model/Sam2Processor unavailable. "
                "Install a compatible transformers package or pass --transformers-overlay."
            ) from exc

        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device)
        self.device = device

    @torch.inference_mode()
    def predict(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Predict the best SAM2 mask for a single bbox prompt."""
        inputs = self.processor(
            images=image,
            input_boxes=[[[float(coord) for coord in bbox]]],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )
        mask_tensor = masks[0]
        while mask_tensor.ndim > 3:
            mask_tensor = mask_tensor[0]
        if mask_tensor.ndim == 2:
            return (mask_tensor > 0).cpu().numpy().astype(np.uint8)

        scores = outputs.iou_scores.detach().cpu().reshape(-1)
        num_candidates = mask_tensor.shape[0]
        best_idx = int(scores[:num_candidates].argmax().item()) if scores.numel() else 0
        return (mask_tensor[best_idx] > 0).cpu().numpy().astype(np.uint8)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Figure 2a style foreground/background patch-score distributions with SAM2 masks."
    )
    parser.add_argument("--model-kind", type=str, choices=["lastvit", "supervised_cls"], default="lastvit")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--imagenet-val-dir", type=str, required=True)
    parser.add_argument("--label-dir", type=str, required=True)
    parser.add_argument("--mask-cache-dir", type=str, required=True)
    parser.add_argument("--transformers-overlay", type=str, default="")
    parser.add_argument("--sam-model-id", type=str, default="facebook/sam2.1-hiera-small")
    parser.add_argument("--skip-sam2", action="store_true")
    parser.add_argument("--force-sam2", action="store_true")
    parser.add_argument("--write-mask-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.25)
    parser.add_argument("--majority-ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--output-plot", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def build_transform() -> T.Compose:
    """Build the same image transform used for patch-score evaluation."""
    return T.Compose(
        [
            T.Resize(256, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def parse_annotation(xml_path: Path, max_bbox_area_ratio: float) -> tuple[tuple[float, float, float, float], int, int] | None:
    """Parse one ImageNet bbox XML and keep only valid single-object entries."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    size_node = root.find("size")
    if size_node is None:
        return None

    width = int(size_node.findtext("width", default="0"))
    height = int(size_node.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        return None

    objects = root.findall("object")
    if len(objects) != 1:
        return None

    bbox_node = objects[0].find("bndbox")
    if bbox_node is None:
        return None

    xmin = float(bbox_node.findtext("xmin", default="0"))
    ymin = float(bbox_node.findtext("ymin", default="0"))
    xmax = float(bbox_node.findtext("xmax", default="0"))
    ymax = float(bbox_node.findtext("ymax", default="0"))
    if xmax <= xmin or ymax <= ymin:
        return None

    bbox_area = (xmax - xmin) * (ymax - ymin)
    image_area = float(width * height)
    if max_bbox_area_ratio >= 0.0 and bbox_area >= image_area * max_bbox_area_ratio:
        return None

    return (xmin, ymin, xmax, ymax), width, height


def build_image_path_map(val_dir: Path) -> dict[str, Path]:
    """Map validation image names to absolute file paths."""
    image_map: dict[str, Path] = {}
    for image_path in val_dir.rglob("*.JPEG"):
        image_map[image_path.name] = image_path
    return image_map


def discover_samples(
    val_dir: Path,
    label_dir: Path,
    max_bbox_area_ratio: float,
    limit: int,
) -> list[SampleEntry]:
    """Build the filtered ImageNet validation subset used for visualization."""
    if not val_dir.exists():
        raise FileNotFoundError(f"ImageNet val dir not found: {val_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"ImageNet bbox dir not found: {label_dir}")

    image_map = build_image_path_map(val_dir)
    samples: list[SampleEntry] = []

    for xml_path in sorted(label_dir.glob("*.xml")):
        parsed = parse_annotation(xml_path, max_bbox_area_ratio=max_bbox_area_ratio)
        if parsed is None:
            continue
        image_name = xml_path.with_suffix(".JPEG").name
        image_path = image_map.get(image_name)
        if image_path is None:
            continue
        bbox, width, height = parsed
        samples.append(
            SampleEntry(
                image_name=image_name,
                image_path=image_path,
                bbox=bbox,
                width=width,
                height=height,
            )
        )
        if limit > 0 and len(samples) >= limit:
            break

    if not samples:
        raise RuntimeError("No valid ImageNet samples found after bbox filtering.")
    return samples


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop common checkpoint prefixes so the state dict matches this script."""
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def load_checkpoint(path: str) -> Any:
    """Load a checkpoint while tolerating older pickle-style payloads."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_model(model_kind: str, checkpoint_path: str, device: str) -> tuple[DenseViT, dict[str, Any]]:
    """Build either the naive CLS baseline or the released LAST-ViT model."""
    config = {
        "image_size": 224,
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
    }

    if model_kind == "supervised_cls":
        backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model = DenseViT(**config, aggregation="cls", topk=1, score_formula="repo")
        load_result = model.load_state_dict(backbone.state_dict(), strict=True)
        model = model.to(device).eval()
        return model, {
            "model_kind": model_kind,
            "checkpoint": "torchvision_imagenet1k_v1",
            "aggregation": "cls",
            "missing_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
        }

    if not checkpoint_path:
        raise ValueError("--checkpoint is required when model-kind=lastvit.")

    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state_dict)!r}")

    model = DenseViT(**config, aggregation="last", topk=1, score_formula="repo")
    load_result = model.load_state_dict(normalize_state_dict_keys(state_dict), strict=False)
    model = model.to(device).eval()
    return model, {
        "model_kind": model_kind,
        "checkpoint": checkpoint_path,
        "aggregation": "last",
        "missing_keys": len(load_result.missing_keys),
        "unexpected_keys": len(load_result.unexpected_keys),
    }


def normalize_per_image(scores: torch.Tensor) -> torch.Tensor:
    """Normalize patch scores independently for each image to [0, 1]."""
    min_values = scores.min(dim=1, keepdim=True).values
    max_values = scores.max(dim=1, keepdim=True).values
    denom = (max_values - min_values).clamp_min(1e-8)
    return (scores - min_values) / denom


def resize_center_crop_mask(mask: np.ndarray) -> torch.Tensor:
    """Apply the same resize and crop pipeline to a binary mask."""
    mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    mask_image = TF.resize(mask_image, 256, interpolation=InterpolationMode.NEAREST)
    mask_image = TF.center_crop(mask_image, [224, 224])
    mask_array = np.array(mask_image, dtype=np.uint8)
    return torch.from_numpy(mask_array > 0)


def mask_to_patch_indices(mask: torch.Tensor, patch_size: int, majority_ratio: float) -> np.ndarray:
    """Convert a binary mask into foreground patch indices using majority overlap."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={tuple(mask.shape)}")
    height, width = mask.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Mask shape {height}x{width} is not divisible by patch size {patch_size}.")
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = mask.view(grid_h, patch_size, grid_w, patch_size)
    covered = patches.sum(dim=(1, 3)).float() / float(patch_size * patch_size)
    active = covered > majority_ratio
    indices = torch.nonzero(active, as_tuple=False)
    if indices.numel() == 0:
        return np.empty((0,), dtype=np.int64)
    return (indices[:, 0] * grid_w + indices[:, 1]).cpu().numpy().astype(np.int64)


def smooth_density(hist: np.ndarray) -> np.ndarray:
    """Apply light smoothing so the plotted curves are easier to read."""
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel = kernel / kernel.sum()
    return np.convolve(hist, kernel, mode="same")


def summarize_distribution(fg_scores: np.ndarray, bg_scores: np.ndarray, bins: int) -> dict[str, Any]:
    """Aggregate the raw patch scores into a plot-ready density summary."""
    hist_fg, edges = np.histogram(fg_scores, bins=bins, range=(0.0, 1.0), density=True)
    hist_bg, _ = np.histogram(bg_scores, bins=bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "num_foreground_scores": int(fg_scores.size),
        "num_background_scores": int(bg_scores.size),
        "foreground_mean": float(fg_scores.mean()) if fg_scores.size else None,
        "background_mean": float(bg_scores.mean()) if bg_scores.size else None,
        "foreground_q50": float(np.quantile(fg_scores, 0.5)) if fg_scores.size else None,
        "background_q50": float(np.quantile(bg_scores, 0.5)) if bg_scores.size else None,
        "foreground_q90": float(np.quantile(fg_scores, 0.9)) if fg_scores.size else None,
        "background_q90": float(np.quantile(bg_scores, 0.9)) if bg_scores.size else None,
        "hist_bin_centers": centers.tolist(),
        "foreground_density": smooth_density(hist_fg).tolist(),
        "background_density": smooth_density(hist_bg).tolist(),
    }


def make_plot(summary: dict[str, Any], out_path: Path, title: str) -> None:
    """Render the final foreground/background distribution plot."""
    centers = np.array(summary["hist_bin_centers"], dtype=np.float64)
    foreground = np.array(summary["foreground_density"], dtype=np.float64)
    background = np.array(summary["background_density"], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.3))
    ax.plot(centers, foreground, color="#1b7f5a", linewidth=2.2, label="On Object")
    ax.plot(centers, background, color="#c75b39", linewidth=2.2, label="On Background")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Normalized Patch Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def resolve_mask(
    image_path: Path,
    image_name: str,
    bbox: tuple[float, float, float, float],
    mask_cache_dir: Path | None,
    use_cache: bool,
    allow_sam2: bool,
    write_mask_cache: bool,
    refiner_holder: dict[str, Any],
    sam_model_id: str,
    device: str,
    transformers_overlay: str,
) -> tuple[np.ndarray | None, str]:
    """Load a cached mask when available, otherwise refine it with SAM2."""
    cache_path = None
    if mask_cache_dir is not None:
        cache_path = mask_cache_dir / image_name.replace(".JPEG", ".npy")
        if use_cache and cache_path.exists():
            return np.load(cache_path), "cache"

    if not allow_sam2:
        return None, "missing"

    if "refiner" not in refiner_holder:
        refiner_holder["refiner"] = Sam2Refiner(
            model_id=sam_model_id,
            device=device,
            transformers_overlay=transformers_overlay,
        )

    image = Image.open(image_path).convert("RGB")
    mask = refiner_holder["refiner"].predict(image=image, bbox=bbox)
    if cache_path is not None and write_mask_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mask.astype(np.uint8))
    return mask, "sam2"


@torch.inference_mode()
def collect_distributions(
    model: DenseViT,
    dataloader: DataLoader,
    samples_by_name: dict[str, SampleEntry],
    mask_cache_dir: Path | None,
    force_sam2: bool,
    skip_sam2: bool,
    write_mask_cache: bool,
    sam_model_id: str,
    device: str,
    majority_ratio: float,
    transformers_overlay: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect normalized foreground/background patch scores over the dataset."""
    patch_size = model.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    patch_size = int(patch_size)

    foreground_scores: list[np.ndarray] = []
    background_scores: list[np.ndarray] = []
    num_images_used = 0
    num_images_skipped = 0
    mask_sources = Counter()
    refiner_holder: dict[str, Any] = {}

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        scores = normalize_per_image(model(images)["patch_scores"]).cpu().numpy()
        image_names = batch["image_name"]

        for sample_idx, image_name in enumerate(image_names):
            sample = samples_by_name[image_name]
            # Foreground is defined by the SAM2-refined object mask instead of
            # the coarse bbox, which makes the patch split much tighter.
            mask, mask_source = resolve_mask(
                image_path=sample.image_path,
                image_name=sample.image_name,
                bbox=sample.bbox,
                mask_cache_dir=mask_cache_dir,
                use_cache=not force_sam2,
                allow_sam2=not skip_sam2,
                write_mask_cache=write_mask_cache,
                refiner_holder=refiner_holder,
                sam_model_id=sam_model_id,
                device=device,
                transformers_overlay=transformers_overlay,
            )
            mask_sources[mask_source] += 1
            if mask is None:
                num_images_skipped += 1
                continue

            # A patch counts as foreground only when most of its area is inside
            # the refined mask, matching the majority-overlap rule.
            patch_indices = mask_to_patch_indices(
                mask=resize_center_crop_mask(mask),
                patch_size=patch_size,
                majority_ratio=majority_ratio,
            )
            if patch_indices.size == 0:
                num_images_skipped += 1
                continue

            all_indices = np.arange(scores.shape[1], dtype=np.int64)
            background_mask = np.ones(scores.shape[1], dtype=bool)
            background_mask[patch_indices] = False
            background_indices = all_indices[background_mask]
            if background_indices.size == 0:
                num_images_skipped += 1
                continue

            foreground_scores.append(scores[sample_idx, patch_indices])
            background_scores.append(scores[sample_idx, background_indices])
            num_images_used += 1

        if (batch_idx + 1) % 20 == 0:
            print(
                f"processed_batches={batch_idx + 1}/{len(dataloader)} "
                f"used={num_images_used} skipped={num_images_skipped} "
                f"cache={mask_sources['cache']} sam2={mask_sources['sam2']}"
            )

    if not foreground_scores or not background_scores:
        raise RuntimeError("No valid foreground/background patch scores were collected.")

    diagnostics = {
        "num_images_used": num_images_used,
        "num_images_skipped": num_images_skipped,
        "mask_sources": dict(mask_sources),
    }
    return (
        np.concatenate(foreground_scores).astype(np.float32),
        np.concatenate(background_scores).astype(np.float32),
        diagnostics,
    )


def main() -> None:
    """Run the full visualization pipeline and save the outputs."""
    args = parse_args()

    if args.model_kind == "lastvit" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --model-kind=lastvit.")

    val_dir = Path(args.imagenet_val_dir)
    label_dir = Path(args.label_dir)
    mask_cache_dir = Path(args.mask_cache_dir)

    print("Discovering ImageNet validation samples...")
    samples = discover_samples(
        val_dir=val_dir,
        label_dir=label_dir,
        max_bbox_area_ratio=args.max_bbox_area_ratio,
        limit=args.limit,
    )
    print(f"Collected {len(samples)} valid single-object samples.")

    dataset = ImageNetBBoxDataset(samples=samples, transform=build_transform())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    print("Building model...")
    model, model_meta = build_model(
        model_kind=args.model_kind,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    title = args.title
    if not title:
        if args.model_kind == "lastvit":
            title = "LAST-ViT on ImageNet-1k"
        else:
            title = "ViT-B/16 on ImageNet-1k (Fully Supervised)"

    print("Collecting foreground/background patch-score distributions...")
    fg_scores, bg_scores, diagnostics = collect_distributions(
        model=model,
        dataloader=dataloader,
        samples_by_name={sample.image_name: sample for sample in samples},
        mask_cache_dir=mask_cache_dir,
        force_sam2=args.force_sam2,
        skip_sam2=args.skip_sam2,
        write_mask_cache=args.write_mask_cache,
        sam_model_id=args.sam_model_id,
        device=args.device,
        majority_ratio=args.majority_ratio,
        transformers_overlay=args.transformers_overlay,
    )

    summary = summarize_distribution(fg_scores=fg_scores, bg_scores=bg_scores, bins=args.bins)
    output_plot = Path(args.output_plot)
    output_json = Path(args.output_json)
    make_plot(summary=summary, out_path=output_plot, title=title)

    payload = {
        "imagenet_val_dir": str(val_dir),
        "label_dir": str(label_dir),
        "mask_cache_dir": str(mask_cache_dir),
        "foreground_proxy": "sam2_mask_majority",
        "majority_ratio": args.majority_ratio,
        "max_bbox_area_ratio": args.max_bbox_area_ratio,
        "num_discovered_samples": len(samples),
        "model": model_meta,
        "diagnostics": diagnostics,
        "summary": summary,
        "plot_path": str(output_plot),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
