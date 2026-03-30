"""
Dense Vision Transformer with patch-level feature extraction.
Overrides forward() to return patch tokens, classification logits, and patch scores.
"""

from torchvision.models.vision_transformer import VisionTransformer, ViT_B_16_Weights
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Tuple
from pathlib import Path


class DenseViT(VisionTransformer):
    """
    VisionTransformer with dense patch-level outputs.

    Returns patch tokens, classification logits, and patch scores
    instead of only the classification result.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning patch-level features and scores.

        Args:
            x: Input image tensor [B, C, H, W]

        Returns:
            Dictionary with cls_token, patch_tokens, logits, and patch_scores.
        """
        x = self._process_input(x)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        logits = self.heads(cls_token)

        patch_scores = self._compute_patch_scores(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            logits=logits
        )
        
        return {
            'cls_token': cls_token,
            'patch_tokens': patch_tokens,
            'logits': logits,
            'patch_scores': patch_scores,
            'num_patches': patch_tokens.shape[1],
            'hidden_dim': self.hidden_dim,
        }

    def _compute_patch_scores(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-patch relevance score via cosine similarity with the CLS token.

        Args:
            cls_token: [B, hidden_dim]
            patch_tokens: [B, num_patches, hidden_dim]
            logits: [B, num_classes]

        Returns:
            patch_scores: [B, num_patches]
        """
        B, num_patches, hidden_dim = patch_tokens.shape

        cls_token_expanded = cls_token.unsqueeze(1)
        similarity = torch.cosine_similarity(
            patch_tokens,
            cls_token_expanded.expand(-1, num_patches, -1),
            dim=-1
        )
        patch_scores = similarity
        
        return patch_scores
    
    def get_patch_grid_size(self) -> Tuple[int, int]:
        """Return the spatial grid size of patches."""
        n = self.image_size // self.patch_size
        return (n, n)

    def reshape_patch_scores_to_2d(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """Reshape 1D patch scores to a 2D grid: [B, num_patches] -> [B, H, W]."""
        B = patch_scores.shape[0]
        h, w = self.get_patch_grid_size()
        return patch_scores.reshape(B, h, w)


def load_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a pre-trained DenseViT model."""
    from torchvision.models import vit_b_16

    pretrained_vit = vit_b_16(weights=None)
    checkpoint = torch.load('/2024233235/env/last-vit/sam2/vit_b_16-c867db91.pth', map_location='cpu')
    pretrained_vit.load_state_dict(checkpoint)

    config = {
        'image_size': 224,
        'patch_size': 16,
        'num_layers': 12,
        'num_heads': 12,
        'hidden_dim': 768,
        'mlp_dim': 3072,
    }

    model = DenseViT(**config)
    print(model.load_state_dict(pretrained_vit.state_dict(), strict=False))

    model = model.to(device)
    model.eval()
    return model


def get_patch_scores(model: DenseViT, image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run inference and return the full output dictionary."""
    with torch.no_grad():
        output = model(image)
    return output


def visualize_patch_scores(
    model: DenseViT,
    image_path: str,
    save_path: str = None,
    top_k_patches: int = None
):
    """
    Visualize patch scores overlaid on the original image.

    Args:
        model: DenseViT model.
        image_path: Path to the input image.
        save_path: Output path (None to display interactively).
        top_k_patches: If set, annotate the top-k scoring patches.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from torchvision import transforms
    import numpy as np
    
    image_pil = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    transform_display = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    
    image_display = transform_display(image_pil)
    image_tensor = transform(image_pil).unsqueeze(0).to(next(model.parameters()).device)
    
    # Get patch scores
    output = get_patch_scores(model, image_tensor)
    patch_scores = output['patch_scores'][0].cpu().numpy()  # [num_patches]
    logits = output['logits'][0]
    
    # Reshape to 2D
    score_map = model.reshape_patch_scores_to_2d(output['patch_scores'])[0].cpu().numpy()  # [14, 14]
    
    # Get predicted class
    pred_class = torch.argmax(logits).item()
    pred_prob = torch.softmax(logits, dim=0)[pred_class].item()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Original image
    axes[0].imshow(image_display)
    axes[0].set_title(f'Original Image\nPred: Class {pred_class} ({pred_prob:.3f})', fontsize=12)
    axes[0].axis('off')
    
    # Subplot 2: Patch score heatmap
    im = axes[1].imshow(score_map, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Patch Scores (14x14)\nMin: {score_map.min():.3f}, Max: {score_map.max():.3f}', fontsize=12)
    axes[1].set_xlabel('Patch X')
    axes[1].set_ylabel('Patch Y')
    plt.colorbar(im, ax=axes[1])
    
    # Subplot 3: Overlay
    axes[2].imshow(image_display, alpha=0.6)
    
    # Upsample score map to image size
    from scipy.ndimage import zoom
    score_map_upsampled = zoom(score_map, (224/14, 224/14), order=1)
    im2 = axes[2].imshow(score_map_upsampled, cmap='hot', alpha=0.4, interpolation='bilinear')
    
    # If top_k specified, annotate patches
    if top_k_patches is not None:
        patch_size = 224 // 14  # 16 pixels per patch
        
        # Get top-k patches
        flat_indices = np.argsort(patch_scores)[-top_k_patches:]
        
        for idx in flat_indices:
            row = idx // 14
            col = idx % 14
            score = patch_scores[idx]
            
            # Draw rectangle
            rect = mpatches.Rectangle(
                (col * patch_size, row * patch_size),
                patch_size, patch_size,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            axes[2].add_patch(rect)
            
            # Add score label
            axes[2].text(
                col * patch_size + patch_size/2,
                row * patch_size + patch_size/2,
                f'{score:.2f}',
                color='white',
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
            )
    
    axes[2].set_title(f'Overlay (Top-{top_k_patches} patches)' if top_k_patches else 'Overlay', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'score_map': score_map,
        'pred_class': pred_class,
        'pred_prob': pred_prob,
        'patch_scores': patch_scores
    }


if __name__ == "__main__":
    # Test code
    print("="*60)
    print("DenseViT Patch Score Visualization Test")
    print("="*60)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    model = load_model(device)
    
    # Test image paths
    test_images = [
        "/2024233235/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG",
    ]
    
    print(f"\nVisualizing {len(test_images)} images...")
    
    for i, img_path in enumerate(test_images):
        if not Path(img_path).exists():
            print(f"Skip: {img_path} (not found)")
            continue
        
        print(f"\nProcessing image {i+1}/{len(test_images)}: {Path(img_path).name}")
        save_path = f"/2024233235/env/last-vit/patch_score_vis_{i}.jpg"
        
        # Visualize with top-10 patches
        result = visualize_patch_scores(
            model=model,
            image_path=img_path,
            save_path=save_path,
            top_k_patches=10
        )
        
        print(f"  Predicted class: {result['pred_class']} (confidence: {result['pred_prob']:.3f})")
        print(f"  Patch score range: [{result['score_map'].min():.3f}, {result['score_map'].max():.3f}]")
        print(f"  Mean score: {result['patch_scores'].mean():.3f}")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)