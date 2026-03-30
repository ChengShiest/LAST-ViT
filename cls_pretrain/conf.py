"""
An example config file to train a ImageNet classifier with detectron2.
Model and dataloader both come from torchvision.
This shows how to use detectron2 as a general engine for any new models and tasks.

To run, use the following command:

python tools/lazyconfig_train_net.py --config-file configs/Misc/torchvision_imagenet_R_50.py \
    --num-gpus 8 dataloader.train.dataset.root=/path/to/imagenet/

"""


import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
import torchvision
from torchvision.transforms import transforms as T
from torchvision.models.resnet import ResNet, Bottleneck
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from detectron2.config import LazyCall as L
from detectron2.model_zoo import get_config
from detectron2.data.samplers import TrainingSampler, InferenceSampler
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from PIL import Image

"""
Note: Here we put reusable code (models, evaluation, data) together with configs just as a
proof-of-concept, to easily demonstrate what's needed to train a ImageNet classifier in detectron2.
Writing code in configs offers extreme flexibility but is often not a good engineering practice.
In practice, you might want to put code in your project and import them instead.
"""


def build_data_loader(dataset, batch_size, num_workers, training=True):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=(TrainingSampler if training else InferenceSampler)(len(dataset)),
        # sampler= torch.utils.data.DistributedSampler(dataset, shuffle=True) if training else InferenceSampler(len(dataset)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


class ClassificationNet(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return list(self.model.parameters())[0].device

    def forward(self, inputs):
        image, label = inputs
        pred, _ = self.model(image.to(self.device))
        if self.training:
            return F.cross_entropy(pred, label.to(self.device))
        else:
            return pred
        
class ClassificationAcc(DatasetEvaluator):
    def reset(self):
        self.corr_top1 = self.corr_top5 = self.total = 0

    def process(self, inputs, outputs):
        image, label = inputs
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)  # Get top-5 predictions
        self.corr_top1 += pred[:, 0].cpu().eq(label).cpu().sum().item()  # Top-1 accuracy
        self.corr_top5 += pred.cpu().eq(label.view(-1, 1)).cpu().sum().item()  # Top-5 accuracy
        self.total += label.size(0)

    def evaluate(self):
        all_corr_top1_total = comm.all_gather([self.corr_top1, self.total])
        all_corr_top5_total = comm.all_gather([self.corr_top5, self.total])

        corr_top1 = sum(x[0] for x in all_corr_top1_total)
        corr_top5 = sum(x[0] for x in all_corr_top5_total)
        total = sum(x[1] for x in all_corr_top1_total)

        accuracy_top1 = corr_top1 / total
        accuracy_top5 = corr_top5 / total

        return {"top1_accuracy": accuracy_top1, "top5_accuracy": accuracy_top5}


# --- End of code that could be in a project and be imported
from fvcore.transforms.transform import Transform
from detectron2.data.transforms import Augmentation, ResizeTransform


dataloader = OmegaConf.create()
dataloader.train = L(build_data_loader)(
    dataset=L(torchvision.datasets.ImageNet)(
        root="/path/to/imagenet",
        split="train",
        transform=L(T.Compose)(
            transforms=[
                L(T.RandomResizedCrop)(size=224),
                L(T.RandomHorizontalFlip)(),
                T.ToTensor(),
                L(T.Normalize)(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    ),
    batch_size=512,
    num_workers=8,
    training=True,
)


dataloader.test = L(build_data_loader)(
    dataset=L(torchvision.datasets.ImageNet)(
        root="${...train.dataset.root}",
        split="val",
        transform=L(T.Compose)(
            transforms=[
                L(T.Resize)(size=256),
                L(T.CenterCrop)(size=224),
                T.ToTensor(),
                L(T.Normalize)(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    ),
    batch_size=128,
    num_workers=8,
    training=False,
)


dataloader.evaluator = L(ClassificationAcc)()
from torchvision.models.vision_transformer import VisionTransformer, ViT_B_16_Weights

class dense_vit(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_kernel = None

    def gaussian_kernel_1d(self, kernel_size, sigma):
        kernel = torch.exp(-0.5 * (torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float() / sigma) ** 2)
        kernel = kernel / torch.max(kernel)
        return kernel

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        
        if self.cached_kernel is None:
            self.cached_kernel = self.gaussian_kernel_1d(768, 768 ** 0.5).to(x.device).unsqueeze(0).unsqueeze(0)
        x_detach = x[:, 1:]
        x = torch.fft.fft(x[:, 1:], dim=-1)
        gs_k = self.cached_kernel.to(x.device)
        x = torch.fft.fftshift(x, dim=-1)
        x = x * (gs_k)
        x = torch.fft.ifftshift(x, dim=-1)
        x = torch.fft.ifft(x, dim=-1).real
        diff =  x_detach / torch.abs(x - x_detach) 
        _, indices = torch.topk(diff, k=1, dim=1, largest=True)
        sel_p = torch.gather(x_detach, 1, indices)
        cls_token = torch.mean(sel_p, dim=1)
        return cls_token, x_detach


# VIT-B-16
model = L(ClassificationNet)(
    model=L(dense_vit)(
    image_size=224,
    patch_size=16, 
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072)
)


optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(),
    lr=0.001,
    weight_decay=0.3,
    betas=(0.9, 0.95),
)

# optimizer = L(torch.optim.SGD)(

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1, end_value=0.001
    ),
    warmup_length=1 / 100,
    warmup_factor=0.1,
)


vit_weights = ViT_B_16_Weights.IMAGENET1K_V1

train = get_config("common/train.py").train
train.max_iter = 100 * 1281167 // 512 // 8
train.output_dir = "output"
train.checkpointer['period'] = 3000
