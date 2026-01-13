"""
Training module for HRF segmentation
Handles model training, validation, and checkpoint management
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, Optional
import numpy as np

from metrics import evaluate_batch, MetricsTracker


class Trainer:
    """
    Trainer class for model training and validation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        use_amp: bool = True,
        log_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        self.log_wandb = log_wandb
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.scaler = GradScaler() if use_amp else None
        self.best_val_dice = 0.0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics_tracker = MetricsTracker()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            batch_metrics = evaluate_batch(outputs, masks)
            metrics_tracker.update(batch_metrics)
            epoch_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item(), 'dice': batch_metrics['dice']})
        
        avg_metrics = metrics_tracker.get_average()
        avg_metrics['loss'] = epoch_loss / len(self.train_loader)
        
        return avg_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        epoch_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                batch_metrics = evaluate_batch(outputs, masks)
                metrics_tracker.update(batch_metrics)
                epoch_loss += loss.item()
                
                pbar.set_postfix({'loss': loss.item(), 'dice': batch_metrics['dice']})
        
        avg_metrics = metrics_tracker.get_average()
        avg_metrics['loss'] = epoch_loss / len(self.val_loader)
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filename: str = 'checkpoint.pth'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path,weights_only=False, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 15,
        resume_from: Optional[str] = None,
    ):
        start_epoch = 0
        patience_counter = 0
        
        if resume_from is not None and os.path.exists(resume_from):
            start_epoch, metrics = self.load_checkpoint(resume_from)
            if metrics is not None and 'dice' in metrics:
                self.best_val_dice = metrics['dice']
                print(f"Restored best validation Dice: {self.best_val_dice:.4f}")
            start_epoch += 1
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        
        # Learning rate warmup
        base_lr = self.optimizer.param_groups[0]['lr']
        warmup_epochs = 3
        
        for epoch in range(start_epoch, num_epochs):
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"Warmup LR: {warmup_lr:.6f}")
            
            # Training and validation
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # Scheduler step (after warmup)
            if epoch >= warmup_epochs and self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"LR: {current_lr:.6f}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")
            
            # Save best model
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.save_checkpoint(epoch, val_metrics, 'best_model.pth')
                patience_counter = 0
                print(f"✓ Best model saved! Dice: {self.best_val_dice:.4f}")
            else:
                patience_counter += 1
            
            # Save latest
            self.save_checkpoint(epoch, val_metrics, 'latest_model.pth')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {patience_counter} epochs without improvement")
                break
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print("="*50)
