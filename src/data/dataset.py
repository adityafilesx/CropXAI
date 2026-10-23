"""
PyTorch Dataset Providers
"""

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_id: str, batch_size: int = 32):
    """
    Returns (train_loader, val_loader, test_loader, num_classes, class_names).
    """
    processed_dir = Path(f"data/processed/{dataset_id}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory {processed_dir} not found. Did you run ingestion?")
        
    # Standard ResNet/ViT augmentations/normalizations for ImageNet pretrained weights
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(processed_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(processed_dir / "val", transform=val_test_transform)
    test_dataset = datasets.ImageFolder(processed_dir / "test", transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, len(train_dataset.classes), train_dataset.classes
