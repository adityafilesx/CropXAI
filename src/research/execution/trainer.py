"""
Real PyTorch Training Engine — Provenance-Wired
"""

import logging
import platform
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.registry import get_model
from src.data.dataset import get_dataloaders
from src.research.evaluation.real_metrics import evaluate_model
from src.research.xai.trust import calculate_trust_score
from src.research.execution.provenance_db import store_run

logger = logging.getLogger(__name__)

def _get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def _get_hardware() -> str:
    if torch.backends.mps.is_available():
        return f"Apple-Silicon MPS | {platform.processor()}"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"CPU | {platform.processor()}"

def run_real_training(dataset_id: str, model_name: str, epochs: int = 5, seed: int = 42,
                      finetune_mode: str = "frozen"):
    """
    Trains the specified model on the dataset for the specified epochs.
    All metrics are automatically persisted to the provenance database.
    """
    torch.manual_seed(seed)
    logger.info("Initializing Real PyTorch Execution Engine (seed=%d)...", seed)

    # 1. Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device.")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU — this will be slow.")

    # 2. Data Loading
    logger.info("Loading dataloaders for %s...", dataset_id)
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(dataset_id)

    # 3. Model Definition
    logger.info("Initializing %s...", model_name)
    model = get_model(model_name, num_classes=num_classes, pretrained=True)
    
    if finetune_mode == "frozen":
        for name, param in model.named_parameters():
            if "fc" not in name and "head" not in name and "classifier" not in name:
                param.requires_grad = False
                
    model = model.to(device)

    # 4. Optimizer
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=1e-3)

    # 5. Training Loop
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_metrics = evaluate_model(model, val_loader, device)
        epoch_losses.append(running_loss / len(train_loader))
        logger.info("Epoch %d/%d — Loss: %.4f — Val Acc: %.2f%%",
                    epoch + 1, epochs, epoch_losses[-1], val_metrics['accuracy'] * 100)

    # 6. Final Test Evaluation
    logger.info("Running final test evaluation...")
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info("Test Acc: %.2f%%  F1: %.4f", test_metrics['accuracy'] * 100, test_metrics['f1'])

    # 7. Trust Score
    logger.info("Calculating Trust Score (T3 Faithfulness via SSIM)...")
    trust_score = calculate_trust_score(model, test_loader, device)
    logger.info("Trust Score T3: %.4f", trust_score)

    # 8. ── PROVENANCE STORE ──────────────────────────────────────────────────
    provenance_metrics = {
        "accuracy":      test_metrics['accuracy'],
        "f1":            test_metrics['f1'],
        "precision":     test_metrics.get('precision', 0.0),
        "recall":        test_metrics.get('recall', 0.0),
        "trust_T3":      trust_score,
        "final_loss":    epoch_losses[-1] if epoch_losses else 0.0,
    }
    run_id = store_run(
        model=model_name,
        dataset=dataset_id,
        seed=seed,
        epochs=epochs,
        finetune_mode=finetune_mode,
        metrics=provenance_metrics,
        git_sha=_get_git_sha(),
        hardware=_get_hardware(),
    )
    logger.info("Results persisted to provenance DB (run_id=%s)", run_id)
    # ───────────────────────────────────────────────────────────────────────────

    return test_metrics, trust_score
