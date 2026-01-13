"""
Configuration file for HRF segmentation training
PREPROCESSING VERSION (CLAHE + Z-Score)
"""

# Data settings
# Data settings
DATA_CONFIG = {
    'data_dir': '/content/drive/MyDrive/HRF-Segmentation/data',
    'image_size': None,  # None = keep original size, or specify (e.g., 512)
    'batch_size': 4,  # Reduce if out of memory
    'num_workers': 2,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'apply_augmentation': False,  # NO AUGMENTATION (Only CLAHE)
    'apply_preprocessing': True,  # CLAHE + Z-Score Normalization (ImageNet stats)
}

# Model settings
MODEL_CONFIG = {
    'name': 'UNet',
    'n_channels': 3,
    'n_classes': 1,
    'bilinear': False,
    'base_filters': 64,
}

# Loss function settings
LOSS_CONFIG = {
    'loss_name': 'dice',     # Switched to Dice Loss for direct metric optimization
    'alpha': 0.3,            # Ignored for Dice Loss
    'beta': 0.7,             # Ignored for Dice Loss
    'gamma': 4/3,            # Ignored for Dice Loss
    'smooth': 1.0,           # Smooth factor for Dice Loss
}

# Optimizer settings
OPTIMIZER_CONFIG = {
    'optimizer': 'adam',  # Options: 'adam', 'adamw', 'sgd'
    'lr': 0.001,
    'weight_decay': 0.0001,
}

# Scheduler settings
SCHEDULER_CONFIG = {
    'scheduler': 'cosine',  # Options: 'plateau', 'cosine', 'step'
    'mode': 'max',          # Ignored for cosine
    'factor': 0.5,          # Ignored for cosine
    'patience': 5,          # Ignored for cosine
}

# Training settings
TRAINING_CONFIG = {
    'num_epochs': 100,
    'early_stopping_patience': 15,
    'use_amp': True,  # Mixed precision training
    'checkpoint_dir': '/content/drive/MyDrive/HRF-Segmentation/checkpoints',
}

# Print configuration
if __name__ == '__main__':
    print("DATA CONFIG:", DATA_CONFIG)
    print("MODEL CONFIG:", MODEL_CONFIG)
    print("LOSS CONFIG:", LOSS_CONFIG)
    print("TRAINING CONFIG:", TRAINING_CONFIG)
