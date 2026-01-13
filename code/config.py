"""
Configuration file for HRF segmentation training
"""

# Data settings
DATA_CONFIG = {
    'data_dir': '/content/drive/MyDrive/HRF-Segmentation/data',
    'image_size': None,  # None = keep original size, or specify (e.g., 512)
    'batch_size': 4,  # Reduce if out of memory
    'num_workers': 2,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'apply_augmentation': False,  # NO AUGMENTATION
    'apply_preprocessing': False,
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
    'loss_name': 'focal_tversky',  # Options: 'focal_tversky', 'dice', 'dice_bce'
    'alpha': 0.7,    # Weight for false positives
    'beta': 0.3,     # Weight for false negatives
    'gamma': 4/3,    # Focal parameter
}

# Optimizer settings
OPTIMIZER_CONFIG = {
    'optimizer': 'adam',  # Options: 'adam', 'adamw', 'sgd'
    'lr': 0.001,
    'weight_decay': 0.0001,
}

# Scheduler settings
SCHEDULER_CONFIG = {
    'scheduler': 'plateau',  # Options: 'plateau', 'cosine', 'step'
    'mode': 'max',
    'factor': 0.5,
    'patience': 5,
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
