"""
PyTorch Evaluation Metrics
"""

import torch
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the given dataloader.
    Returns dict of metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    except ImportError:
        logger.warning("scikit-learn not installed, returning basic accuracy.")
        correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        acc = correct / len(all_labels) if all_labels else 0
        prec, rec, f1 = 0.0, 0.0, 0.0
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
