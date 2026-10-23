"""
Grad-CAM Implementation for ResNet-18
"""

import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        """
        Generates Grad-CAM heatmap for a single image tensor (1, C, H, W).
        """
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[:, class_idx].squeeze()
        score.backward(retain_graph=True)
        
        # Handle both CNN (4D: B, C, H, W) and ViT (3D: B, N, C) feature maps
        if len(self.gradients.shape) == 4:
            # Pool the gradients across spatial dimensions (CNN)
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().detach().numpy()
        elif len(self.gradients.shape) == 3:
            # ViT / Token Sequence: Shape (B, N, C)
            weights = torch.mean(self.gradients, dim=1, keepdim=True)
            weighted_activations = weights * self.activations
            cam = torch.sum(weighted_activations, dim=-1).squeeze().cpu().detach().numpy()
            
            # Reshape sequence back to 2D image grid (excluding cls token if N=197)
            seq_len = cam.shape[0] if len(cam.shape) == 1 else cam.shape[-1]
            if seq_len == 197:
                cam = cam[1:]
                seq_len = 196
                
            grid_size = int(np.sqrt(seq_len))
            if grid_size * grid_size == seq_len:
                cam = cam.reshape((grid_size, grid_size))
        else:
            cam = self.activations.squeeze().cpu().detach().numpy()
        
        # ReLU to keep only positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        # Resize to original image size (64x64)
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        cam = cam_resized.squeeze().numpy()
        
        return cam
