"""
Dataset class for HRF segmentation in OCT images - NO PREPROCESSING VERSION

Loads images and masks as-is, with minimal transformations (only normalization).
No resizing, rotation, contrast adjustment, or any preprocessing.

UPDATED: Uses TIFFFILE to read .ome.tiff files - proper handling for scientific data

CRITICAL FIX: Mask normalization to float32 (not boolean) for proper gradient calculation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import tifffile
from typing import Tuple, Optional, Callable, List


class HRFDataset(Dataset):
    """
    Dataset class for Hyperreflective Foci (HRF) segmentation.
    
    IMPORTANT: This version loads images AS-IS without any preprocessing,
    resizing, rotation, or contrast adjustment. Only basic normalization.
    
    UPDATED: Uses TIFFFILE for .ome.tiff support - proper scientific TIFF handling
    
    Args:
        image_dir: Directory containing OCT images
        mask_dir: Directory containing corresponding masks
        image_files: List of image filenames
        image_size: If not None, will resize. Set to None to keep original size
        apply_augmentation: If False, no augmentation applied
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_files: List[str],
        image_size: Optional[int] = None,
        apply_augmentation: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image filename
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find corresponding mask
        mask_candidates = [
            img_name.replace('.jpg', '_HRF.ome.tiff').replace('.jpeg', '_HRF.ome.tiff').replace('.png', '_HRF.ome.tiff'),
            img_name.replace('.jpg', '.tiff').replace('.jpeg', '.tiff').replace('.png', '.tiff'),
            img_name.replace('.jpg', '_mask.tiff').replace('.jpeg', '_mask.tiff').replace('.png', '_mask.tiff'),
        ]
        
        mask_path = None
        for candidate in mask_candidates:
            candidate_path = os.path.join(self.mask_dir, candidate)
            if os.path.exists(candidate_path):
                mask_path = candidate_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {img_name}")
        
        # Read mask using TIFFFILE (handles OME-XML metadata correctly)
        try:
            mask = tifffile.imread(mask_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not read mask file {mask_path}: {e}")
        
        if mask is None:
            raise FileNotFoundError(f"Mask could not be read: {mask_path}")
        
        # Ensure image is 3-channel RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            # If mask is multi-channel, take first channel
            mask = mask[:, :, 0] if mask.shape[0] > mask.shape[1] else mask[0]
        
        # ===================================================================
        # CRITICAL FIX: Proper data type conversion for mask
        # ===================================================================
        # Convert image to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # CRITICAL: Convert mask to float32 EXPLICITLY (not boolean!)
        # This ensures proper gradient calculation during training
        # mask = mask > 0  # ❌ OLD (WRONG): Creates boolean tensor
        mask = (mask > 0).astype(np.float32)  # ✅ NEW (CORRECT): Explicit float32
        # ===================================================================
        
        # Resize if specified (optional - usually None for this version)
        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # Convert to PyTorch tensors
        # Image: (H, W, C) → (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()  # (3, H, W)
        
        # Mask: (H, W) → (1, H, W)
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :]).float()  # (1, H, W)
        
        return image_tensor, mask_tensor


def create_dataloaders(
    data_dir: str,
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    batch_size: int = 4,
    num_workers: int = 2,
    image_size: Optional[int] = None,
    apply_augmentation: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root data directory
        train_files: List of training image filenames
        val_files: List of validation image filenames
        test_files: List of test image filenames
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size (if None, keeps original)
        apply_augmentation: Whether to apply augmentation (set to False for no preprocessing)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    # Create datasets - NO AUGMENTATION, NO PREPROCESSING
    train_dataset = HRFDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=train_files,
        image_size=image_size,
        apply_augmentation=False,  # Explicitly disabled
    )
    
    val_dataset = HRFDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=val_files,
        image_size=image_size,
        apply_augmentation=False,  # Explicitly disabled
    )
    
    test_dataset = HRFDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=test_files,
        image_size=image_size,
        apply_augmentation=False,  # Explicitly disabled
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation (for reproducibility)
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test one at a time for consistency
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
