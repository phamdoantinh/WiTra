import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import time
import json
from datetime import datetime


class Trainer:
    """Trainer class for WiFi indoor positioning with multi-task learning"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        logger = None,
        norm_params: Optional[Tuple] = None,
    ):
        """
        Args:
            model: Model instance (already created from model/witra.py)
            device: Device to use ('cuda' or 'cpu')
            logger: Logger instance for logging
            norm_params: (min_x, max_x, min_y, max_y, max_range) for denormalizing coordinates
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.logger = logger
        self.norm_params = norm_params
        
        # Training components (to be set in setup_training)
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_coord_loss': [],
            'train_building_loss': [],
            'train_floor_loss': [],
            'train_diff_loss': [],
            'train_sim_loss': [],
            'train_sp_diff_loss': [],
            
            'val_loss': [],
            'val_coord_loss': [],
            'val_building_loss': [],
            'val_floor_loss': [],
            'val_diff_loss': [],
            'val_sim_loss': [],
            'val_sp_diff_loss': [],
            'val_coord_error': [],
            'val_building_acc': [],
            'val_floor_acc': [],
            
            'test_coord_error': [],
            'test_building_acc': [],
            'test_floor_acc': [],
        }
        
        if self.logger:
            self.logger.info(f"Trainer initialized on device: {self.device}")
            if self.device.type == 'cuda':
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"Trainer initialized on device: {self.device}")
            if self.device.type == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def setup_training(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        coord_loss_weight: float = 1.0,
        building_loss_weight: float = 1.0,
        floor_loss_weight: float = 1.0,
        w_diff: float = 0.01,
        w_sim: float = 0.0,
        w_sp_diff: float = 0.01,
    ):
        """Setup optimizer, scheduler and loss function"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Store loss weights
        self.coord_loss_weight = coord_loss_weight
        self.building_loss_weight = building_loss_weight
        self.floor_loss_weight = floor_loss_weight
        self.w_diff = w_diff
        self.w_sim = w_sim
        self.w_sp_diff = w_sp_diff
        
        msg = (f"Training setup complete:\n"
               f"   Learning rate: {learning_rate}\n"
               f"   Loss weights: coord={coord_loss_weight}, building={building_loss_weight}, floor={floor_loss_weight}\n"
               f"   Auxiliary weights: diff={w_diff}, sim={w_sim}, sp_diff={w_sp_diff}")
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def setup_scheduler(
        self,
        scheduler_type: str = 'cosine',
        num_epochs: int = 100,
        warmup_epochs: int = 5,
    ):
        """Setup learning rate scheduler"""
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        msg = f"Scheduler: {scheduler_type}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def compute_loss(
        self,
        coord_pred: torch.Tensor,
        coord_true: torch.Tensor,
        building_pred: torch.Tensor,
        building_true: torch.Tensor,
        floor_pred: torch.Tensor,
        floor_true: torch.Tensor,
        aux_losses: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss"""
        # Coordinate loss (MSE or Huber)
        coord_loss = F.mse_loss(coord_pred, coord_true)
        
        # Building classification loss
        building_loss = F.cross_entropy(building_pred, building_true)
        
        # Floor classification loss
        floor_loss = F.cross_entropy(floor_pred, floor_true)
        
        # Combined loss
        total_loss = (
            self.coord_loss_weight * coord_loss +
            self.building_loss_weight * building_loss +
            self.floor_loss_weight * floor_loss
        )
        
        diff_val, sim_val, sp_diff_val = 0.0, 0.0, 0.0
        if aux_losses is not None:
            l_diff, l_sim, l_sp_diff = aux_losses
            diff_val = l_diff.item()
            sim_val = l_sim.item()
            sp_diff_val = l_sp_diff.item()
            
            w_diff = getattr(self, 'w_diff', 0.01)
            w_sim = getattr(self, 'w_sim', 0.01)
            w_sp_diff = getattr(self, 'w_sp_diff', 0.01)
            
            total_loss = total_loss + w_diff * l_diff + w_sim * l_sim + w_sp_diff * l_sp_diff
        
        loss_dict = {
            'total': total_loss.item(),
            'coord': coord_loss.item(),
            'building': building_loss.item(),
            'floor': floor_loss.item(),
            'diff': diff_val,
            'sim': sim_val,
            'sp_diff': sp_diff_val,
        }
        
        return total_loss, loss_dict
    
    def train_epoch(
        self,
        train_loader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_coord_loss = 0
        total_building_loss = 0
        total_floor_loss = 0
        total_diff_loss = 0
        total_sim_loss = 0
        total_sp_diff_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', colour='green')
        for batch_idx, batch in enumerate(pbar):
            # Get data from batch
            features = batch['rssi features'].to(self.device)
            coords = batch['coords'].to(self.device)
            building = batch['building'].to(self.device)
            floor = batch['floor'].to(self.device)
            
            # Forward pass
            coord_pred, floor_pred, building_pred, aux_losses = self.model(features, return_ortho_loss=True)
            
            # Check for NaN in predictions
            if torch.isnan(coord_pred).any():
                if self.logger:
                    self.logger.warning(f"WARNING: NaN detected in coord predictions at epoch {epoch}, batch {batch_idx}!")
                else:
                    print(f"WARNING: NaN detected in coord predictions at epoch {epoch}, batch {batch_idx}!")
            
            # Compute loss (match parameter order: coord, building, floor)
            loss, loss_dict = self.compute_loss(
                coord_pred, coords,
                building_pred, building,
                floor_pred, floor,
                aux_losses=aux_losses
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total']
            total_coord_loss += loss_dict['coord']
            total_building_loss += loss_dict['building']
            total_floor_loss += loss_dict['floor']
            total_diff_loss += loss_dict['diff']
            total_sim_loss += loss_dict['sim']
            total_sp_diff_loss += loss_dict['sp_diff']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'coord': f"{loss_dict['coord']:.4f}",
                'diff': f"{loss_dict['diff']:.4f}",
                'sim': f"{loss_dict['sim']:.4f}",
            })
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'coord_loss': total_coord_loss / num_batches,
            'building_loss': total_building_loss / num_batches,
            'floor_loss': total_floor_loss / num_batches,
            'diff_loss': total_diff_loss / num_batches,
            'sim_loss': total_sim_loss / num_batches,
            'sp_diff_loss': total_sp_diff_loss / num_batches,
        }
        
        return metrics
    
    def denorm_coords(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize coordinates from [0,1] scale back to original coordinate system"""
        if self.norm_params is None:
            return coords_norm
        
        max_range = self.norm_params
        coords_denorm = coords_norm.clone()
        # Khôi phục: nhân max_range để về hệ tọa độ đã transform (gốc 0,0)
        coords_denorm[:, 0] = coords_denorm[:, 0] * max_range  # LONGITUDE
        coords_denorm[:, 1] = coords_denorm[:, 1] * max_range  # LATITUDE
        return coords_denorm
    
    def norm_coords(self, coords_orig: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates from transformed scale to [0,1] scale"""
        if self.norm_params is None:
            return coords_orig
        
        max_range = self.norm_params
        coords_norm = coords_orig.clone()
        # Chuẩn hóa: chia max_range
        coords_norm[:, 0] = coords_norm[:, 0] / max_range  # LONGITUDE
        coords_norm[:, 1] = coords_norm[:, 1] / max_range  # LATITUDE
        return coords_norm
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        split: str = 'val',
    ) -> Dict[str, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        
        # Check for NaN in model weights
        has_nan_weights = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                has_nan_weights = True
                if self.logger:
                    self.logger.warning(f"WARNING: NaN in model weights: {name}")
                else:
                    print(f"WARNING: NaN in model weights: {name}")
        
        if has_nan_weights:
            msg = f"ERROR: Model weights contain NaN! Cannot evaluate {split} set."
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
        
        total_loss = 0.0
        total_coord_loss = 0.0
        total_building_loss = 0.0
        total_floor_loss = 0.0
        total_diff_loss = 0.0
        total_sim_loss = 0.0
        total_sp_diff_loss = 0.0
        num_batches = 0
        
        all_coord_errors = []
        all_building_correct = []
        all_floor_correct = []
        
        batch_count = 0
        for batch in tqdm(data_loader, desc=f'Evaluating {split}', colour='blue'):
            # Get data
            features = batch['rssi features'].to(self.device)
            coords = batch['coords'].to(self.device)
            building = batch['building'].to(self.device)
            floor = batch['floor'].to(self.device)
            
            # Forward pass
            if split != 'test':
                coord_pred, floor_pred, building_pred, aux_losses = self.model(features, return_ortho_loss=True)
            else:
                coord_pred, floor_pred, building_pred = self.model(features)
                aux_losses = None
            
            # Check if predictions contain NaN during iteration
            if torch.isnan(coord_pred).any():
                # Find which samples have NaN
                nan_mask = torch.isnan(coord_pred).any(dim=1)
                num_nan = nan_mask.sum().item()
                if self.logger:
                    self.logger.warning(f"WARNING: {split} batch {batch_count}: {num_nan}/{len(coord_pred)} samples have NaN predictions!")
                else:
                    print(f"WARNING: {split} batch {batch_count}: {num_nan}/{len(coord_pred)} samples have NaN predictions!")
                
                # Skip NaN samples for metric calculation
                valid_mask = ~nan_mask
                if valid_mask.any():
                    coord_pred = coord_pred[valid_mask]
                    coords = coords[valid_mask]
                    building = building[valid_mask]
                    floor = floor[valid_mask]
                    building_pred = building_pred[valid_mask]
                    floor_pred = floor_pred[valid_mask]
                else:
                    continue  # Skip this batch entirely
            batch_count += 1
            
            # Compute loss (only for train/val, not test)
            if split != 'test':
                loss, loss_dict = self.compute_loss(
                    coord_pred, coords,
                    building_pred, building,
                    floor_pred, floor,
                    aux_losses=aux_losses
                )
                
                total_loss += loss_dict['total']
                total_coord_loss += loss_dict['coord']
                total_building_loss += loss_dict['building']
                total_floor_loss += loss_dict['floor']
                total_diff_loss += loss_dict['diff']
                total_sim_loss += loss_dict['sim']
                total_sp_diff_loss += loss_dict['sp_diff']
                num_batches += 1
            
            # Calculate metrics: denormalize predictions and ground-truth coordinates back to meters
            coord_pred_denorm = self.denorm_coords(coord_pred)  # [0,1] → [0, max_range]
            coords_denorm = self.denorm_coords(coords)          # [0,1] → [0, max_range]
            
            # Coordinate error (Euclidean distance in transformed coordinate space)
            coord_error = torch.norm(coord_pred_denorm - coords_denorm, dim=1)
            all_coord_errors.extend(coord_error.cpu().numpy())
            
            # Building accuracy
            building_pred_class = torch.argmax(building_pred, dim=1)
            building_correct = (building_pred_class == building).float()
            all_building_correct.extend(building_correct.cpu().numpy())
            
            # Floor accuracy
            floor_pred_class = torch.argmax(floor_pred, dim=1)
            floor_correct = (floor_pred_class == floor).float()
            all_floor_correct.extend(floor_correct.cpu().numpy())
        
        # Compute metrics
        metrics = {
            'coord_error_mean': np.mean(all_coord_errors),
            'coord_error_median': np.median(all_coord_errors),
            'building_acc': np.mean(all_building_correct) * 100,
            'floor_acc': np.mean(all_floor_correct) * 100,
        }
        
        # Add loss metrics only for train/val (not test)
        if split != 'test':
            metrics['loss'] = total_loss / num_batches
            metrics['coord_loss'] = total_coord_loss / num_batches
            metrics['building_loss'] = total_building_loss / num_batches
            metrics['floor_loss'] = total_floor_loss / num_batches
            metrics['diff_loss'] = total_diff_loss / num_batches
            metrics['sim_loss'] = total_sim_loss / num_batches
            metrics['sp_diff_loss'] = total_sp_diff_loss / num_batches
        
        return metrics
    
    def fit(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        num_epochs: int = 100,
        save_dir: str = './checkpoints',
        save_every: int = 10,
        test_every: int = 10,
    ):
        """Train the model"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_error = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader, split='val')
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_coord_loss'].append(train_metrics['coord_loss'])
            self.history['train_building_loss'].append(train_metrics['building_loss'])
            self.history['train_floor_loss'].append(train_metrics['floor_loss'])
            self.history['train_diff_loss'].append(train_metrics['diff_loss'])
            self.history['train_sim_loss'].append(train_metrics['sim_loss'])
            self.history['train_sp_diff_loss'].append(train_metrics['sp_diff_loss'])
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_coord_loss'].append(val_metrics['coord_loss'])
            self.history['val_building_loss'].append(val_metrics['building_loss'])
            self.history['val_floor_loss'].append(val_metrics['floor_loss'])
            self.history['val_diff_loss'].append(val_metrics['diff_loss'])
            self.history['val_sim_loss'].append(val_metrics['sim_loss'])
            self.history['val_sp_diff_loss'].append(val_metrics['sp_diff_loss'])
            self.history['val_coord_error'].append(val_metrics['coord_error_mean'])
            self.history['val_building_acc'].append(val_metrics['building_acc'])
            self.history['val_floor_acc'].append(val_metrics['floor_acc'])
            
            # Log results
            log_msg = (
                f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)\n"
                f"  Train: loss={train_metrics['loss']:.4f} (coord={train_metrics['coord_loss']:.4f}, bld={train_metrics['building_loss']:.4f}, flr={train_metrics['floor_loss']:.4f}, diff={train_metrics['diff_loss']:.4f}, sim={train_metrics['sim_loss']:.4f}, sp={train_metrics['sp_diff_loss']:.4f})\n"
                f"  Valid: loss={val_metrics['loss']:.4f} (coord={val_metrics['coord_loss']:.4f}, bld={val_metrics['building_loss']:.4f}, flr={val_metrics['floor_loss']:.4f}, diff={val_metrics['diff_loss']:.4f}, sim={val_metrics['sim_loss']:.4f}, sp={val_metrics['sp_diff_loss']:.4f})\n"
                f"  Metrics: error={val_metrics['coord_error_mean']:.2f}m (med={val_metrics['coord_error_median']:.2f}m) | bld_acc={val_metrics['building_acc']:.2f}% | flr_acc={val_metrics['floor_acc']:.2f}%"
            )
            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)
            
            # Save best model
            if val_metrics['coord_error_mean'] < best_val_error:
                best_val_error = val_metrics['coord_error_mean']
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_error': best_val_error,
                    'max_range': self.norm_params,  # Lưu max_range để test sau này
                }, best_model_path)
                msg = f"  New best model saved! Error: {best_val_error:.2f}m"
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
            
            # Save checkpoint
            if epoch % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_error': val_metrics['coord_error_mean'],
                    'max_range': self.norm_params,  # Lưu max_range
                }, checkpoint_path)
                msg = f"  Checkpoint saved: epoch_{epoch}.pth"
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
            
            # Test every N epochs
            if test_loader is not None and epoch % test_every == 0:
                msg = f"\nTesting at epoch {epoch}..."
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
                
                test_metrics = self.evaluate(test_loader, split='test')
                
                test_msg = (
                    f"  Error: {test_metrics['coord_error_mean']:.2f}m (median: {test_metrics['coord_error_median']:.2f}m) | "
                    f"Building: {test_metrics['building_acc']:.2f}% | "
                    f"Floor: {test_metrics['floor_acc']:.2f}%"
                )
                if self.logger:
                    self.logger.info(test_msg)
                else:
                    print(test_msg)
        
        msg = f"\nTraining complete! Best val error: {best_val_error:.2f}m"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        
        return self.history
    
    def test(
        self,
        test_loader,
    ) -> Dict[str, float]:
        """Test the model"""
        test_metrics = self.evaluate(test_loader, split='test')
        
        # Update history
        self.history['test_coord_error'].append(test_metrics['coord_error_mean'])
        self.history['test_building_acc'].append(test_metrics['building_acc'])
        self.history['test_floor_acc'].append(test_metrics['floor_acc'])
        
        # Print results
        msg = (
            f"\nTest Results:\n"
            f"  Error: {test_metrics['coord_error_mean']:.2f}m (median: {test_metrics['coord_error_median']:.2f}m)\n"
            f"  Building Acc: {test_metrics['building_acc']:.2f}%\n"
            f"  Floor Acc: {test_metrics['floor_acc']:.2f}%"
        )
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        
        return test_metrics
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int = None,
        metrics: Dict = None,
    ):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'epoch': epoch,
            'metrics': metrics,
            'max_range': self.norm_params,  # Lưu max_range
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
    ):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load max_range if available
        if 'max_range' in checkpoint:
            self.norm_params = checkpoint['max_range']
            print(f"   Max range loaded: {self.norm_params:.2f}")
        elif 'norm_params' in checkpoint:
            # Backward compatibility with old checkpoints
            old_params = checkpoint['norm_params']
            if isinstance(old_params, tuple):
                self.norm_params = old_params[4]  # Extract max_range from old format
            else:
                self.norm_params = old_params
            print(f"   Max range loaded (old format): {self.norm_params:.2f}")
        
        if load_optimizer and checkpoint['optimizer_state_dict'] is not None:
            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from {path}")
        if checkpoint.get('epoch'):
            print(f"   Epoch: {checkpoint['epoch']}")
        if checkpoint.get('metrics'):
            print(f"   Metrics: {checkpoint['metrics']}")
    
    def save_history(
        self,
        path: str,
    ):
        """Save training history to JSON"""
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        history_serializable = convert_to_serializable(self.history)
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"History saved to {path}")
