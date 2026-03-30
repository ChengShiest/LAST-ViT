"""
ImageNet Validation Set DataLoader
Loads images with bounding box annotations, keeping only single-object samples.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ImageNetValDataset(Dataset):
    """
    ImageNet validation set with lazy loading and bbox annotations.

    Filters to single-object images with bbox area < 25% of image area.

    Args:
        val_dir: Directory containing class folders with validation images.
        val_label_dir: Directory containing XML annotation files.
        meta_file: Path to val.txt mapping image names to class IDs.
        transform: Image preprocessing pipeline.
        return_bbox: Whether to return bbox annotations.
        filter_multi_objects: Whether to filter out multi-object images.
        mask_dir: Optional directory for SAM2 masks.
        return_mask: Whether to return SAM2 masks (requires mask_dir).
    """

    def __init__(
        self,
        val_dir: str = "/2024233235/imagenet/val",
        val_label_dir: str = "/2024233235/imagenet/val_label/val",
        meta_file: str = "/2024233235/imagenet/meta/val.txt",
        transform: Optional[transforms.Compose] = None,
        return_bbox: bool = True,
        filter_multi_objects: bool = True,
        mask_dir: Optional[str] = None,
        return_mask: bool = False
    ):
        self.val_dir = Path(val_dir)
        self.val_label_dir = Path(val_label_dir)
        self.return_bbox = return_bbox
        self.transform = transform
        self.filter_multi_objects = filter_multi_objects
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.return_mask = return_mask and self.mask_dir is not None

        # Lightweight index: [(image_name, class_id), ...]
        self.image_list = []

        # Lazily-built image-name-to-path cache
        self._image_path_cache = None

        self._load_meta_file(meta_file)
        print(f"Loaded {len(self.image_list)} validation image indices.")

        if self.filter_multi_objects:
            self._filter_single_object_images()

    def _load_meta_file(self, meta_file: str):
        """Parse meta file, storing only image names and class IDs."""
        with open(meta_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                image_name = parts[0]
                class_id = int(parts[1])
                self.image_list.append((image_name, class_id))

    def _filter_single_object_images(self):
        """Keep only single-object images with bbox area < 25% of image area."""
        print("Filtering for single-object images with small bboxes...")
        filtered_list = []

        for image_name, class_id in self.image_list:
            xml_name = image_name.replace('.JPEG', '.xml')
            xml_path = self.val_label_dir / xml_name
            if self._is_valid_single_object(xml_path):
                filtered_list.append((image_name, class_id))

        original_count = len(self.image_list)
        self.image_list = filtered_list
        print(f"Filtered: {original_count} -> {len(self.image_list)} images "
              f"(single object + bbox < 25% image area).")

    def _is_valid_single_object(self, xml_path: Path) -> bool:
        """Check that XML has exactly one object and its bbox area < 25% of image."""
        if not xml_path.exists():
            return False

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            objects = root.findall('object')
            if len(objects) != 1:
                return False

            size_elem = root.find('size')
            if size_elem is None:
                return False

            img_width = int(size_elem.find('width').text)
            img_height = int(size_elem.find('height').text)
            img_area = img_width * img_height

            obj = objects[0]
            bndbox = obj.find('bndbox')
            if bndbox is None:
                return False

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            bbox_area = (xmax - xmin) * (ymax - ymin)
            return bbox_area < img_area * 0.25
        except Exception:
            return False

    def _count_objects_in_xml(self, xml_path: Path) -> int:
        """Quickly count the number of objects in an XML annotation."""
        if not xml_path.exists():
            return 0
        try:
            tree = ET.parse(xml_path)
            return len(tree.getroot().findall('object'))
        except Exception:
            return 0

    def _build_image_path_cache(self):
        """Lazily build a mapping from image names to file paths."""
        if self._image_path_cache is not None:
            return

        print("Building image path cache (first access)...")
        self._image_path_cache = {}

        for class_folder in self.val_dir.iterdir():
            if not class_folder.is_dir():
                continue
            for image_file in class_folder.glob("*.JPEG"):
                self._image_path_cache[image_file.name] = image_file

        print(f"Image path cache built: {len(self._image_path_cache)} images.")

    def _get_image_path(self, image_name: str) -> Optional[Path]:
        """Retrieve image path using cache."""
        if self._image_path_cache is None:
            self._build_image_path_cache()
        return self._image_path_cache.get(image_name)

    def _parse_xml_annotation(self, xml_path: Path) -> Dict:
        """Parse an XML annotation file and return bbox metadata."""
        if not xml_path.exists():
            return {
                'width': 0, 'height': 0, 'depth': 0,
                'name': '', 'bbox': [0, 0, 0, 0],
                'difficult': 0, 'truncated': 0
            }

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_elem = root.find('size')
        width = int(size_elem.find('width').text) if size_elem is not None else 0
        height = int(size_elem.find('height').text) if size_elem is not None else 0
        depth = int(size_elem.find('depth').text) if size_elem is not None else 3

        obj = root.find('object')
        if obj is not None:
            name = obj.find('name').text
            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            truncated = int(obj.find('truncated').text) if obj.find('truncated') is not None else 0
            bndbox = obj.find('bndbox')
            bbox = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
        else:
            name = ''
            bbox = [0, 0, 0, 0]
            difficult = 0
            truncated = 0

        return {
            'width': width, 'height': height, 'depth': depth,
            'name': name, 'bbox': bbox,
            'difficult': difficult, 'truncated': truncated
        }

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieve a single sample (lazy: image and XML are read here).

        Returns:
            Dictionary with keys: image, class_id, image_path, and optionally
            bbox, bbox_name, difficult, truncated, image_width, image_height,
            mask, original_mask.
        """
        image_name, class_id = self.image_list[idx]

        image_path = self._get_image_path(image_name)
        if image_path is None:
            raise FileNotFoundError(f"Image not found: {image_name}")

        original_image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = original_image.size

        if self.transform is not None:
            image = self.transform(original_image)
        else:
            image = transforms.ToTensor()(original_image)

        result = {
            'image': image,
            'class_id': class_id,
            'image_path': str(image_path)
        }

        if self.return_bbox:
            xml_name = image_name.replace('.JPEG', '.xml')
            xml_path = self.val_label_dir / xml_name
            bbox_annotation = self._parse_xml_annotation(xml_path)

            result['bbox'] = torch.tensor(bbox_annotation['bbox'], dtype=torch.float32)
            result['bbox_name'] = bbox_annotation['name']
            result['difficult'] = bbox_annotation['difficult']
            result['truncated'] = bbox_annotation['truncated']
            result['image_width'] = bbox_annotation['width']
            result['image_height'] = bbox_annotation['height']

        if self.return_mask:
            mask_name = image_name.replace('.JPEG', '.npy')
            mask_path = self.mask_dir / mask_name

            if mask_path.exists():
                original_mask = np.load(mask_path)
                mask_pil = Image.fromarray(original_mask * 255)

                if self.transform is not None:
                    for t in self.transform.transforms:
                        if isinstance(t, transforms.Resize):
                            mask_pil = transforms.functional.resize(
                                mask_pil, t.size,
                                interpolation=transforms.InterpolationMode.NEAREST
                            )
                        elif isinstance(t, transforms.CenterCrop):
                            mask_pil = transforms.functional.center_crop(mask_pil, t.size)

                mask = torch.from_numpy(np.array(mask_pil)).float() / 255.0
                mask = (mask > 0.5).float()

                result['mask'] = mask
                result['original_mask'] = torch.from_numpy(original_mask).float()
            else:
                print(f"Warning: Mask not found for {image_name}, returning zero mask")
                if hasattr(image, 'shape'):
                    mask_h, mask_w = image.shape[-2:]
                    result['mask'] = torch.zeros((mask_h, mask_w), dtype=torch.float32)
                else:
                    result['mask'] = torch.zeros((224, 224), dtype=torch.float32)

                if self.return_bbox:
                    h, w = bbox_annotation['height'], bbox_annotation['width']
                else:
                    w, h = orig_w, orig_h
                result['original_mask'] = torch.zeros((h, w), dtype=torch.float32)

        return result


def collate_fn_with_masks(batch):
    """
    Custom collate function for variable-sized original masks.

    Stacks fixed-size fields into tensors; keeps variable-size masks as lists.
    """
    result = {}

    # Stackable fields
    if 'image' in batch[0]:
        result['image'] = torch.stack([item['image'] for item in batch])
    if 'class_id' in batch[0]:
        result['class_id'] = torch.tensor([item['class_id'] for item in batch])
    if 'bbox' in batch[0]:
        result['bbox'] = torch.stack([item['bbox'] for item in batch])
    if 'difficult' in batch[0]:
        result['difficult'] = torch.tensor([item['difficult'] for item in batch])
    if 'truncated' in batch[0]:
        result['truncated'] = torch.tensor([item['truncated'] for item in batch])
    if 'image_width' in batch[0]:
        result['image_width'] = torch.tensor([item['image_width'] for item in batch])
    if 'image_height' in batch[0]:
        result['image_height'] = torch.tensor([item['image_height'] for item in batch])

    # Masks: transformed masks share the same size -> stack; original masks vary -> list
    if 'mask' in batch[0]:
        result['mask'] = torch.stack([item['mask'] for item in batch])

    # List fields
    if 'image_path' in batch[0]:
        result['image_path'] = [item['image_path'] for item in batch]
    if 'bbox_name' in batch[0]:
        result['bbox_name'] = [item['bbox_name'] for item in batch]
    if 'original_mask' in batch[0]:
        result['original_mask'] = [item['original_mask'] for item in batch]

    return result


def get_imagenet_val_dataloader(
    batch_size: int = 32,
    num_workers: int = 4,
    val_dir: str = "/2024233235/imagenet/val",
    val_label_dir: str = "/2024233235/imagenet/val_label/val",
    meta_file: str = "/2024233235/imagenet/meta/val.txt",
    transform: Optional[transforms.Compose] = None,
    return_bbox: bool = True,
    filter_multi_objects: bool = True,
    shuffle: bool = False,
    mask_dir: Optional[str] = None,
    return_mask: bool = False
) -> DataLoader:
    """
    Create an ImageNet validation DataLoader.

    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of data loading workers.
        val_dir: Validation images directory.
        val_label_dir: Annotations directory.
        meta_file: Path to val.txt.
        transform: Image preprocessing pipeline.
        return_bbox: Whether to return bbox annotations.
        filter_multi_objects: Keep only single-object images with small bboxes.
        shuffle: Whether to shuffle the dataset.
        mask_dir: Optional SAM2 mask directory.
        return_mask: Whether to return masks (requires mask_dir).

    Returns:
        A PyTorch DataLoader.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    dataset = ImageNetValDataset(
        val_dir=val_dir,
        val_label_dir=val_label_dir,
        meta_file=meta_file,
        transform=transform,
        return_bbox=return_bbox,
        filter_multi_objects=filter_multi_objects,
        mask_dir=mask_dir,
        return_mask=return_mask
    )

    collate_fn = collate_fn_with_masks if return_mask else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
