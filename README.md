# LAST-ViT: Vision Transformers Need More Than Registers
![tea](./imgs/tea.png)


|   Scenario   |  Training Script   |     Last-ViT Weight 
| :----------------: | :----------------: |  :------:  
| Self-supervised | https://github.com/facebookresearch/dino |     [DINO](https://github.com/ChengShiest/LAST-ViT/releases/download/weights/dino0080.pth)  
| Text-supervised | https://github.com/mlfoundations/open_clip |     [CLIP](https://github.com/ChengShiest/LAST-ViT/releases/download/weights/openai_b_16.pt)  
| Label-supervised | https://github.com/ChengShiest/LAST-ViT/tree/main/cls_pretrain |     [ViT](https://github.com/ChengShiest/LAST-ViT/releases/download/weights/ViT_190k.pth)  
|||

Before
```
x = self._process_input(x)
x = torch.cat([batch_class_token, x], dim=1)
x = self.encoder(x)

cls_token = x[:, 0:1]
return cls_token
```
With Last-ViT

```
def gaussian_kernel_1d(self, kernel_size, sigma):
    kernel = torch.exp(-0.5 * (torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float() / sigma) ** 2)
    kernel = kernel / torch.max(kernel)
    return kernel
...

x = self.encoder(x)

x_detach = x[:, 1:]
x = torch.fft.fft(x[:, 1:], dim=-1)
gs_k = self.gaussian_kernel_1d
x = torch.fft.fftshift(x, dim=-1)
x = x * (gs_k)
x = torch.fft.ifftshift(x, dim=-1)
x = torch.fft.ifft(x, dim=-1).real
diff =  x_detach / torch.abs(x - x_detach) 
_, indices = torch.topk(diff, k=1, dim=1, largest=True)
sel_p = torch.gather(x_detach, 1, indices)
cls_token = torch.mean(x, dim=1)
return cls_token
```

