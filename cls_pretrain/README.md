## Evaluation

We use detectron2's LazyConfig for training and evaluation. First install detectron2:

```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install -e detectron2
```

Then evaluate on ImageNet val with 1 GPU:

```bash
python cls_pretrain/lazy_train.py     --config-file cls_pretrain/conf.py     --num-gpus 1     --eval-only     train.init_checkpoint=/path/to/ViT_190k.pth     dataloader.train.dataset.root=/path/to/imagenet
```

The val dataloader uses `torchvision.datasets.ImageFolder` internally (no devkit archive needed). Multi-GPU evaluation is also supported -- simply increase `--num-gpus`.

Expected results on ImageNet-1K val with `ViT_190k.pth` (100 epochs):

| Model | Top-1 Acc | Top-5 Acc |
|-------|-----------|-----------|
| LAST-ViT (k=1) | ~67.4% | ~87.6% |
| LAST-ViT (k=7) | ~67.6% | ~87.8% |

See Table 11 in the [paper](https://arxiv.org/pdf/2602.22394v1) for detail

---

## Training

```bash
python cls_pretrain/lazy_train.py     --config-file cls_pretrain/conf.py     --num-gpus 8     dataloader.train.dataset.root=/path/to/imagenet     train.output_dir=/path/to/output
```
