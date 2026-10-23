"""
CropXAI Model Registry
======================
Dynamically loads architectures (CNNs, ViTs, VFMs) for the empirical campaign.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Retrieves the requested model architecture and configures its final layer.
    """
    logger.info(f"Loading {model_name} from registry...")
    
    # --- CNNs ---
    if model_name == "ResNet-18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    elif model_name == "ResNet-50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    elif model_name == "EfficientNet-B0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
        
    elif model_name == "ConvNeXt-T":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    # --- ViTs ---
    elif model_name == "ViT-Base":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
        
    elif model_name == "Swin-T":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
        
    elif model_name == "SegFormer-B0":
        raise ValueError("SegFormer-B0 is unavailable; substituting Swin is prohibited.")
        
    # --- VFMs (Stubs until HF weights are mounted in cluster) ---
    elif model_name in ["Prithvi-100M", "SatMAE", "DOFA", "ScaleMAE"]:
        raise ValueError(f"{model_name} is unavailable without its real checkpoint and input adapter; proxy substitution is prohibited.")
        
    else:
        raise ValueError(f"Model {model_name} not found in registry.")
