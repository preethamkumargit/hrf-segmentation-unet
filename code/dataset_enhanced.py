"""
Dataset class for HRF segmentation in OCT images - ENHANCED VERSION

Features:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement
2. Albumentations for robust data augmentation
3. Maintains original image resolution (938x625)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import tifffile
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HRFDatasetEnhanced(Dataset):
    """
    Enhanced Dataset class for HRF segmentation.
    
    Includes:
    - CLAHE preprocessing (always applied to enhance HRF visibility)
    - Data Augmentation (optional, using Albumentations)
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_files: List[str],
        image_size: Optional[int] = None,  # Kept for compatibility, but usually None
        apply_augmentation: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        
        # Define Augmentation Pipeline
        if self.apply_augmentation:
            self.transform = A.Compose([
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5), # REMOVED: Causes shape mismatch (H,W swap) in batches for non-square images
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                
                # Intensity Augmentations (Subtle, to avoid washing out HRF)
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            ])
        else:
            self.transform = None

    def apply_clahe(self, image):
        """
        Apply CLAHE to the L-channel of LAB image to enhance contrast
        without shifting colors significantly.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final

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
        
        # Read mask using TIFFFILE
        try:
            mask = tifffile.imread(mask_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not read mask file {mask_path}: {e}")
        
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] if mask.shape[0] > mask.shape[1] else mask[0]
            
        # FIX: Convert to float32 BEFORE augmentation to avoid cv2.flip error with uint32
        # Also binarize here to ensure we are augmenting a binary mask
        mask = (mask > 0).astype(np.float32)
            
        # ===================================================================
        # PREPROCESSING: CLAHE
        # ===================================================================
        image = self.apply_clahe(image)
        
        # ===================================================================
        # AUGMENTATION (Albumentations)
        # ===================================================================
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # ===================================================================
        # NORMALIZATION & TENSOR CONVERSION
        # ===================================================================
        # Convert image to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Mask is already float32 from above
        # mask = (mask > 0).astype(np.float32)
        
        # Resize if specified (optional)
        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # Convert to PyTorch tensors
        # Image: (H, W, C) → (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Mask: (H, W) → (1, H, W)
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :]).float()
        
        return image_tensor, mask_tensor


def create_dataloaders(
    data_dir: str,
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    batch_size: int = 4,
    num_workers: int = 2,
    image_size: Optional[int] = None,
    apply_augmentation: bool = True, # Default to True now
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders with ENHANCED dataset.
    """
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    # Train dataset: With Augmentation + CLAHE
    train_dataset = HRFDatasetEnhanced(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=train_files,
        image_size=image_size,
        apply_augmentation=apply_augmentation, 
    )
    
    # Val/Test dataset: No Augmentation, but YES CLAHE (for consistency)
    val_dataset = HRFDatasetEnhanced(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=val_files,
        image_size=image_size,
        apply_augmentation=False,
    )
    
    test_dataset = HRFDatasetEnhanced(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=test_files,
        image_size=image_size,
        apply_augmentation=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
