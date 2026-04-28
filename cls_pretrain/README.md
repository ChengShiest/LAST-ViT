# Classification Pretraining and Evaluation

This folder contains the ImageNet classification training and evaluation entry point for LAST-ViT. It uses Detectron2 LazyConfig for configuration and TorchVision datasets/models for the classification pipeline.

## Installation

Create a fresh Python environment, install the Python dependencies, then install Detectron2 separately:

```bash
conda create -n LAST python=3.10 pip
conda activate LAST
python -m pip install -r requirements.txt

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

If pip build isolation cannot see the already-installed PyTorch package when installing Detectron2, use:

```bash
python -m pip install --no-build-isolation -e detectron2
```

## Evaluation

Evaluate a checkpoint on an ImageNet-style validation folder with one GPU:

```bash
python cls_pretrain/lazy_train.py \
  --config-file cls_pretrain/conf.py \
  --num-gpus 1 \
  --eval-only \
  train.init_checkpoint=/path/to/ViT_190k.pth \
  dataloader.test.dataset.root=/path/to/imagenet/val
```

The validation dataloader uses `torchvision.datasets.ImageFolder`, so the dataset directory should contain one subdirectory per class. Multi-GPU evaluation is supported by increasing `--num-gpus`.

Expected results on ImageNet-1K validation with `ViT_190k.pth` after 100 epochs:

| Model | Top-1 Acc | Top-5 Acc |
|-------|-----------|-----------|
| LAST-ViT (k=1) | ~67.4% | ~87.6% |
| LAST-ViT (k=7) | ~67.6% | ~87.8% |

See Table 11 in the [paper](https://arxiv.org/pdf/2602.22394v1) for details.

## Training

Train on an ImageNet-style training folder:

```bash
python cls_pretrain/lazy_train.py \
  --config-file cls_pretrain/conf.py \
  --num-gpus 8 \
  dataloader.train.dataset.root=/path/to/imagenet/train \
  train.output_dir=/path/to/output
```
