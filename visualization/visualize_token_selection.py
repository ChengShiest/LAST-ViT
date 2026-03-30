"""
Visualize which token positions are most frequently selected by the LAST-ViT
frequency-domain token selection mechanism across different k values.

Analyzes the forward pass of dense_vit:
- diff = x_detach / torch.abs(x - x_detach)
- _, indices = torch.topk(diff, k=k, dim=1, largest=True)
Counts how often each token position is selected across channels and samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from collections import defaultdict


class dense_vit_with_tracking(VisionTransformer):
    """dense_vit with token selection tracking for visualization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_kernel = None
        self.token_selection_counts = defaultdict(lambda: defaultdict(int))
        self.enable_tracking = False
        self.k_values = [1]
        self.sample_images = []
        self.sample_count = 0
        self.max_samples = 5

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
        x_detach = x[:, 1:]  # Strip CLS token, keep patch tokens only
        x = torch.fft.fft(x[:, 1:], dim=-1)
        gs_k = self.cached_kernel.to(x.device)
        x = torch.fft.fftshift(x, dim=-1)
        x = x * (gs_k)
        x = torch.fft.ifftshift(x, dim=-1)
        x = torch.fft.ifft(x, dim=-1).real
        diff = x_detach / torch.abs(x - x_detach) 
        
        # If tracking is enabled, record selections for different k values
        if self.enable_tracking:
            # diff shape: [batch_size, num_tokens, hidden_dim]
            # For each channel (dim=-1), select top-k tokens (dim=1)
            for k in self.k_values:
                if k <= diff.shape[1]:  # k must not exceed the number of tokens
                    # For each channel, find top-k tokens
                    # diff: [B, T, C] -> for each C, find top-k along T dimension
                    _, indices = torch.topk(diff, k=k, dim=1, largest=True)  # [B, k, C]
                    # indices: [batch_size, k, hidden_dim]
                    # Count how often each token position is selected (across all channels and batch)
                    # Note: the same token may be selected by multiple channels, counted multiple times
                    batch_size = indices.shape[0]
                    num_channels = indices.shape[2]
                    for b in range(batch_size):
                        for c in range(num_channels):
                            selected_tokens = indices[b, :, c].cpu().numpy()
                            for token_idx in selected_tokens:
                                if 0 <= token_idx < diff.shape[1]:  # Ensure valid index
                                    self.token_selection_counts[k][int(token_idx)] += 1
        
        # Original logic: select k=1
        _, indices = torch.topk(diff, k=1, dim=1, largest=True)
        sel_p = torch.gather(x_detach, 1, indices)
        cls_token = torch.mean(sel_p, dim=1)
        return cls_token, None
    
    def get_token_selection_for_image(self, image_tensor):
        """Compute token selection for a single image (for visualization)."""
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            x = self._process_input(image_tensor)
            n = x.shape[0]
            
            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.encoder(x)
            
            if self.cached_kernel is None:
                self.cached_kernel = self.gaussian_kernel_1d(768, 768 ** 0.5).to(x.device).unsqueeze(0).unsqueeze(0)
            x_detach = x[:, 1:]  # Strip CLS token, keep patch tokens only
            x = torch.fft.fft(x[:, 1:], dim=-1)
            gs_k = self.cached_kernel.to(x.device)
            x = torch.fft.fftshift(x, dim=-1)
            x = x * (gs_k)
            x = torch.fft.ifftshift(x, dim=-1)
            x = torch.fft.ifft(x, dim=-1).real
            diff = x_detach / torch.abs(x - x_detach)
            
            # For each channel, count selected tokens
            # diff shape: [1, num_tokens, hidden_dim]
            token_selection_counts = defaultdict(int)
            k = self.k_values[0]  # Use the first k value (should be 1)
            if k <= diff.shape[1]:
                _, indices = torch.topk(diff, k=k, dim=1, largest=True)  # [1, k, hidden_dim]
                # indices: [1, k, hidden_dim]
                # Note: the same token may be selected by multiple channels, counted multiple times
                num_channels = indices.shape[2]
                for c in range(num_channels):
                    selected_tokens = indices[0, :, c].cpu().numpy()
                    for token_idx in selected_tokens:
                        if 0 <= token_idx < diff.shape[1]:  # Ensure valid index
                            token_selection_counts[int(token_idx)] += 1
            
            return token_selection_counts


def load_model_and_data(imagenet_root, num_samples=1000, batch_size=32, checkpoint_path=None):
    """Load model and data."""
    # Create model
    model = dense_vit_with_tracking(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072
    )
    model.eval()
    
    # Load pretrained weights if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading pretrained weights: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle weight name prefix issues
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            # If checkpoint is a dict, try 'model' or 'state_dict' key
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
        
        # Remove 'model.' prefix from weight names
        new_state_dict = {}
        prefix_removed_count = 0
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix (6 chars)
                new_state_dict[new_key] = value
                prefix_removed_count += 1
            else:
                new_state_dict[key] = value
        
        if prefix_removed_count > 0:
            print(f"processed {prefix_removed_count} weights with 'model.' prefix")
        
        # Load weights
        try:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True)
            print("Pretrained weights loaded successfully!")
            if missing_keys:
                print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Unexpected keys: {unexpected_keys}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Attempting partial load...")
            # Try loading only matching layers
            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"Partial load succeeded, matched {len(matched_dict)}/{len(model_dict)} parameters")
    else:
        if checkpoint_path:
            print(f"Warning: pretrained weights not found: {checkpoint_path}")
        print("Using randomly initialized model weights")
    
    # Create data loader
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    try:
        dataset = ImageNet(root=imagenet_root, split='val', transform=transform)
        # Limit sample count
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return model, dataloader
    except Exception as e:
        print(f"Cannot load ImageNet dataset: {e}")
        print("Using random data instead")
        # Creating random data
        class RandomDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples, transform):
                self.num_samples = num_samples
                self.transform = transform
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Generating random images
                img = torch.randn(3, 224, 224)
                label = torch.randint(0, 1000, (1,)).item()
                return img, label
        
        dataset = RandomDataset(num_samples, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return model, dataloader


def visualize_token_selection(token_counts_dict, num_tokens=196, save_path='token_selection_visualization.png'):
    """Visualize token selection patterns for different k values."""
    k_values = sorted(token_counts_dict.keys())
    num_k = len(k_values)
    
    # Create subplots - heatmaps
    if num_k == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, (num_k + 1) // 2, figsize=(5 * ((num_k + 1) // 2), 10))
        axes = axes.flatten()
    
    # Compute token selection frequency per k value
    for idx, k in enumerate(k_values):
        counts = token_counts_dict[k]
        # Create token position frequency array
        token_freq = np.zeros(num_tokens)
        for token_idx, count in counts.items():
            if token_idx < num_tokens:
                token_freq[token_idx] = count
        
        # Normalize to [0, 1]
        if token_freq.max() > 0:
            token_freq_normalized = token_freq / token_freq.max()
        else:
            token_freq_normalized = token_freq
        
        # Reshape to 2D (14x14 for ViT-B/16)
        grid_size = int(np.sqrt(num_tokens))
        token_grid = token_freq_normalized.reshape(grid_size, grid_size)
        
        # Plot heatmap
        im = axes[idx].imshow(token_grid, cmap='hot', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        axes[idx].set_title(f'k={k} - Token Selection Frequency\n(Total: {int(token_freq.sum())})', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Patch Column', fontsize=10)
        axes[idx].set_ylabel('Patch Row', fontsize=10)
        plt.colorbar(im, ax=axes[idx], label='Normalized Selection Count')
    
    # Hide extra subplots
    for idx in range(num_k, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {save_path}")
    plt.close()
    
    # Create statistics plot: top-N most selected tokens per k
    fig, ax = plt.subplots(figsize=(14, 7))
    top_n = 30
    
    for k in k_values:
        counts = token_counts_dict[k]
        # Get top_n most selected tokens
        sorted_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if len(sorted_tokens) == 0:
            continue
        token_indices = [t[0] for t in sorted_tokens]
        token_counts = [t[1] for t in sorted_tokens]
        
        ax.plot(token_indices, token_counts, marker='o', label=f'k={k}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Token Index', fontsize=12)
    ax.set_ylabel('Selection Count (across all channels)', fontsize=12)
    ax.set_title(f'Top {top_n} Most Selected Tokens for Different k Values\n(Total selections per token across all channels)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    stats_path = save_path.replace('.png', '_stats.png')
    plt.tight_layout()
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    print(f"Statistics plot saved to: {stats_path}")
    plt.close()
    
    # Create comparison: token selection distribution across k values
    if num_k == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, num_k, figsize=(4 * num_k, 4))
    
    for idx, k in enumerate(k_values):
        counts = token_counts_dict[k]
        token_freq = np.zeros(num_tokens)
        for token_idx, count in counts.items():
            if token_idx < num_tokens:
                token_freq[token_idx] = count
        
        # Plot histogram
        axes[idx].hist(token_freq, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'k={k} - Selection Count Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Selection Count', fontsize=10)
        axes[idx].set_ylabel('Number of Tokens', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    comparison_path = save_path.replace('.png', '_distribution.png')
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Distribution comparison saved to: {comparison_path}")
    plt.close()


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize an image tensor."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def visualize_mask_on_image(image_tensor, mask, patch_size=16, image_size=224, line_width=1):
    """Overlay mask on image with red outline (external contour only)."""
    # Denormalize image
    img = denormalize_image(image_tensor.clone())
    img = img.clamp(0, 1)
    
    # Convert to numpy and rearrange: [C, H, W] -> [H, W, C]
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Create mask visualization
    grid_size = int(np.sqrt(mask.size))
    mask_2d = mask.reshape(grid_size, grid_size)
    
    # Expand mask to image size
    mask_img = np.zeros((image_size, image_size), dtype=bool)
    patch_h = image_size // grid_size
    patch_w = image_size // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            if mask_2d[i, j] > 0:
                h_start = i * patch_h
                h_end = min((i + 1) * patch_h, image_size)
                w_start = j * patch_w
                w_end = min((j + 1) * patch_w, image_size)
                mask_img[h_start:h_end, w_start:w_end] = True
    
    # Copy original image
    result = img_np.copy()
    
    # Add deep red fill in mask area
    mask_3d = mask_img[:, :, np.newaxis]
    deep_red = np.array([1.0, 0.4, 0.4])  # Deep red [R, G, B]
    fill_alpha = 0.5  # Fill alpha
    result = result * (1 - fill_alpha * mask_3d) + deep_red * fill_alpha * mask_3d
    
    # Find external contour: mask pixels adjacent to background
    # Manual dilation (3x3 structuring element)
    dilated = np.zeros_like(mask_img)
    for y in range(image_size):
        for x in range(image_size):
            if mask_img[y, x]:
                # Mark 3x3 neighborhood
                y_min = max(0, y - 1)
                y_max = min(image_size, y + 2)
                x_min = max(0, x - 1)
                x_max = min(image_size, x + 2)
                dilated[y_min:y_max, x_min:x_max] = True
    
    # Edge = dilated region - original mask
    all_edges = dilated & ~mask_img
    
    # Draw red line at edges
    edge_coords = np.where(all_edges)
    for y, x in zip(edge_coords[0], edge_coords[1]):
        # Draw red pixels (with line_width)
        y_start = max(0, y - line_width // 2)
        y_end = min(image_size, y + line_width // 2 + 1)
        x_start = max(0, x - line_width // 2)
        x_end = min(image_size, x + line_width // 2 + 1)
        result[y_start:y_end, x_start:x_end, 0] = 1.0  # red channel
        result[y_start:y_end, x_start:x_end, 1] = 0.0  # green channel
        result[y_start:y_end, x_start:x_end, 2] = 0.0  # blue channel
    
    return np.clip(result, 0, 1)


def visualize_mask_progression(model, sample_images=None, num_tokens=196, output_dir='./visualize', device='cpu'):
    """Visualize progressive mask overlay based on channel selection count thresholds.
    Subplot 1: tokens selected in > 0.5 * max channels
    Subplot 2: tokens selected in > 0.3 * max channels
    Subplot 3: tokens selected in > 0.2 * max channels
    """
    k_values = model.k_values
    patch_size = 16
    image_size = 224
    
    # If no sample images provided, create placeholder
    if sample_images is None or len(sample_images) == 0:
        sample_images = [torch.randn(3, image_size, image_size)]
    
    # Validate all images are tensors
    for img in sample_images:
        if not isinstance(img, torch.Tensor):
            raise ValueError("sample_images must contain torch.Tensor")
    
    # Move model to target device
    model = model.to(device)
    model.eval()
    
    for k in k_values:
        # Generate a 3-subplot figure per image
        for img_idx, sample_img in enumerate(sample_images):
            # Compute token selection for current image
            print(f"Processing sample {img_idx+1}/{len(sample_images)}...")
            img_tensor = sample_img.to(device)
            token_counts = model.get_token_selection_for_image(img_tensor)
            
            # Create token position frequency array
            token_freq = np.zeros(num_tokens)
            for token_idx, count in token_counts.items():
                if token_idx < num_tokens:
                    token_freq[token_idx] = count
            
            # Find max selection count for current image
            max_count = token_freq.max() if token_freq.max() > 0 else 1
            print(f"  Sample {img_idx+1} max channel selection count: {max_count:.1f}")
            
            # Define three thresholds based on per-image max count
            thresholds = [
                0.5 * max_count,  # Subplot 1: > 0.5 * Max
                0.3 * max_count,  # Subplot 2: > 0.3 * Max
                0.2 * max_count   # Subplot 3: > 0.2 * Max
            ]
            
            # Print threshold info
            print(f"  Thresholds: {thresholds[0]:.1f} (0.5*Max), {thresholds[1]:.1f} (0.3*Max), {thresholds[2]:.1f} (0.2*Max)")
            
            # Create figure with 3 subplots (left-to-right, tight layout)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)
            
            # Overlay mask on image (using CPU image)
            img_cpu = sample_img.cpu()
            
            for idx, threshold in enumerate(thresholds):
                # Create mask: 1 where channel count > threshold, 0 otherwise
                mask = np.zeros(num_tokens)
                for token_idx in range(num_tokens):
                    if token_freq[token_idx] > threshold:
                        mask[token_idx] = 1
                
                # Overlay mask on image
                img_with_mask = visualize_mask_on_image(img_cpu, mask, patch_size=patch_size, image_size=image_size)
                
                # Plot image and mask (no labels)
                axes[idx].imshow(img_with_mask)
                axes[idx].axis('off')
            
            # Saving figure
            mask_path = os.path.join(output_dir, f'mask_k{k}_sample{img_idx+1}.png')
            plt.savefig(mask_path, dpi=300, bbox_inches='tight')
            print(f"Mask visualization saved to: {mask_path}")
            plt.close()


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize ViT token selection patterns')
    parser.add_argument('--imagenet-root', type=str, default='/2024233235/imagenet',
                       help='ImageNet dataset root directory')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of samples')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='./visualize',
                       help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='/2024233235/env/last-vit/ViT_190k.pth',
                       help='Path to pretrained weights')
    
    args = parser.parse_args()
    
    # Creating output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ViT Token Selection Visualization")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Pretrained weights: {args.checkpoint}")
    print()
    
    # Load model and data
    print("Loading model and data...")
    model, dataloader = load_model_and_data(args.imagenet_root, args.num_samples, args.batch_size, args.checkpoint)
    model = model.to(args.device)
    model.enable_tracking = True
    
    print(f"Model loaded, num patch tokens: {(224 // 16) ** 2}")
    print()
    
    # Run inference, visualize each sample immediately
    print("Running inference and tracking token selections...")
    model.eval()
    num_tokens = (224 // 16) ** 2  # ViT-B/16: 14x14 = 196
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Processing batch")):
            images = images.to(args.device)
            try:
                cls_token, _ = model(images)
                
                # Visualize each image in the batch immediately
                for img_idx in range(len(images)):
                    sample_img = images[img_idx].cpu().clone()
                    sample_count += 1
                    
                    # Compute token selection and visualize immediately
                    print(f"\nProcessing and visualizing sample {sample_count}...")
                    img_tensor = sample_img.to(args.device)
                    token_counts = model.get_token_selection_for_image(img_tensor)
                    
                    # Create token position frequency array
                    token_freq = np.zeros(num_tokens)
                    for token_idx, count in token_counts.items():
                        if token_idx < num_tokens:
                            token_freq[token_idx] = count
                    
                    # Find max selection count for current image
                    max_count = token_freq.max() if token_freq.max() > 0 else 1
                    print(f"  Sample {sample_count} max channel selection count: {max_count:.1f}")
                    
                    # Define three thresholds
                    thresholds = [
                        0.5 * max_count,
                        0.3 * max_count,
                        0.2 * max_count
                    ]
                    print(f"  Thresholds: {thresholds[0]:.1f} (0.5*Max), {thresholds[1]:.1f} (0.3*Max), {thresholds[2]:.1f} (0.2*Max)")
                    
                    # Generate visualization immediately
                    k_values = model.k_values
                    patch_size = 16
                    image_size = 224
                    
                    for k in k_values:
                        # Creating figure with 3 subplots
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        fig.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)
                        
                        img_cpu = sample_img.cpu()
                        
                        for idx, threshold in enumerate(thresholds):
                            # Creating mask
                            mask = np.zeros(num_tokens)
                            for token_idx in range(num_tokens):
                                if token_freq[token_idx] > threshold:
                                    mask[token_idx] = 1
                            
                            # Overlay mask on image
                            img_with_mask = visualize_mask_on_image(img_cpu, mask, patch_size=patch_size, image_size=image_size)
                            
                            # Plotting image and mask
                            axes[idx].imshow(img_with_mask)
                            axes[idx].axis('off')
                        
                        # Saving figure
                        mask_path = os.path.join(args.output_dir, f'mask_k{k}_sample{sample_count}.png')
                        plt.savefig(mask_path, dpi=300, bbox_inches='tight')
                        print(f"  Mask visualization saved to: {mask_path}")
                        plt.close()
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}  {e}")
                continue
    
    print(f"\nProcessed and visualized {sample_count} sample images")
    
    print()
    print("Statistics complete!")
    print()
    
    # Print statistics
    print("Token Selection Statistics:")
    print("-" * 60)
    for k in sorted(model.token_selection_counts.keys()):
        counts = model.token_selection_counts[k]
        total_selections = sum(counts.values())
        unique_tokens = len(counts)
        print(f"k={k:2d}: total selections={total_selections:8d}, unique tokens={unique_tokens:3d}")
        # Show top-5 most selected tokens
        top_5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"      top 5: {top_5}")
    print()
    
    # Generate global statistics visualization
    print("Generating global statistics visualization...")
    save_path = os.path.join(args.output_dir, 'token_selection_heatmap.png')
    visualize_token_selection(model.token_selection_counts, num_tokens=num_tokens, save_path=save_path)
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

