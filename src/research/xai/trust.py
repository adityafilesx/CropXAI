"""
Trust Metric Calculation
"""

import torch
import numpy as np
import logging
from src.research.xai.gradcam import GradCAM

logger = logging.getLogger(__name__)

def calculate_trust_score(model, dataloader, device, num_samples=100, noise_std=0.2):
    """
    Calculates the XAI Trust Score (SSIM between clean and noisy Grad-CAMs).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("Explanation stability NOT COMPUTED: scikit-image is required.")
        
    model.eval()
    
    # Crucial: Unfreeze the model so that gradients can reach the convolutional layers during backward pass
    for param in model.parameters():
        param.requires_grad = True
        
    # Dynamically find the appropriate layer for Grad-CAM
    target_layer = None
    
    # Standard ResNet
    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    # ConvNeXt
    elif hasattr(model, 'features') and hasattr(model.features, '7'):
        # For ConvNeXt-T, features[7] is the last ConvNeXt block stage
        target_layer = model.features[7][-1]
    # EfficientNet
    elif hasattr(model, 'features') and hasattr(model.features, '8'):
        target_layer = model.features[8]
    # Vision Transformers (ViT, Prithvi, etc.)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        # Get the last encoder block's layer norm 1 for attention map 
        target_layer = model.encoder.layers[-1].ln_1
    # Swin Transformer
    elif hasattr(model, 'features') and hasattr(model.features, '7'):
        # Swin features stage 7 is usually the last block
        target_layer = model.features[7][-1]

    if target_layer is None:
        raise ValueError(f"Explanation stability NOT COMPUTED: no supported Grad-CAM layer for {type(model).__name__}.")
        
    cam_extractor = GradCAM(model, target_layer)
    
    ssim_scores = []
    samples_processed = 0
    
    for inputs, labels in dataloader:
        if samples_processed >= num_samples:
            break
            
        inputs = inputs.to(device)
        
        # Process one image at a time
        for i in range(inputs.size(0)):
            if samples_processed >= num_samples:
                break
                
            x_clean = inputs[i].unsqueeze(0)
            
            # Generate clean CAM
            cam_clean = cam_extractor(x_clean)
            
            # Add Gaussian noise
            noise = torch.randn_like(x_clean) * noise_std
            x_noisy = x_clean + noise
            
            # Generate noisy CAM
            cam_noisy = cam_extractor(x_noisy)
            
            # Calculate SSIM
            # data_range=1.0 because cams are normalized 0-1
            score = ssim(cam_clean, cam_noisy, data_range=1.0)
            ssim_scores.append(score)
            
            samples_processed += 1
            
    if not ssim_scores:
        return 0.0
        
    trust_score = float(np.mean(ssim_scores))
    return trust_score
