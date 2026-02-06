"""
Training loop для обучения модели сегментации
Поддерживает:
- CUDA и CPU
- Mixed Precision Training (FP16)
- TensorBoard логирование
- Checkpointing
- Early Stopping
- Learning Rate Scheduling
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional

from metrics import SegmentationMetrics, format_metrics


class Trainer:
    """
    Класс для обучения модели сегментации
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Args:
            model: Модель для обучения
            criterion: Loss функция
            optimizer: Оптимизатор
            device: Устройство (cuda/cpu)
            train_loader: DataLoader для train
            val_loader: DataLoader для validation
            config: Словарь с конфигурацией
            scheduler: Learning rate scheduler (опционально)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.scheduler = scheduler
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # TensorBoard
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            log_dir = Path(config.get('log_dir', './results/logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Checkpoints
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        
        # Метрики
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()
        
        # История
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'lr': []
        }
        
        print(f"\n{'='*80}")
        print("TRAINER ИНИЦИАЛИЗИРОВАН")
        print(f"{'='*80}")
        print(f"Устройство: {device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"TensorBoard: {self.use_tensorboard}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"{'='*80}\n")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Одна эпоха обучения
        
        Args:
            epoch: Номер эпохи
        
        Returns:
            dict: Метрики за эпоху
        """
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.train_loader)
        accumulation_steps = 2  # Накапливаем градиенты 2 батча
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Forward pass с mixed precision
            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / accumulation_steps  # Нормализуем loss
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Метрики (loss уже нормализован, умножаем обратно для отображения)
            running_loss += loss.item() * accumulation_steps
            self.train_metrics.update(outputs.detach(), masks)
            
            # Обновляем progress bar
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'iou': f'{self.train_metrics.get_iou():.4f}'
            })
        
        # Средние метрики за эпоху
        avg_loss = running_loss / num_batches
        metrics = self.train_metrics.get_all_metrics()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Валидация
        
        Args:
            epoch: Номер эпохи
        
        Returns:
            dict: Метрики за эпоху
        """
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]  ")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Метрики
                running_loss += loss.item()
                self.val_metrics.update(outputs, masks)
                
                # Обновляем progress bar
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'iou': f'{self.val_metrics.get_iou():.4f}'
                })
        
        # Средние метрики за эпоху
        avg_loss = running_loss / num_batches
        metrics = self.val_metrics.get_all_metrics()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        Полный цикл обучения
        
        Args:
            num_epochs: Количество эпох
            start_epoch: Начальная эпоха (для resume)
        """
        print(f"\n{'='*80}")
        print(f"НАЧАЛО ОБУЧЕНИЯ: {num_epochs} эпох")
        print(f"{'='*80}\n")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch + 1)
            
            # Validation
            val_metrics = self.validate_epoch(epoch + 1)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['iou'])
                else:
                    self.scheduler.step()
            
            # Текущий learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Время эпохи
            epoch_time = time.time() - epoch_start_time
            
            # Сохраняем историю
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['lr'].append(current_lr)
            
            # TensorBoard logging
            if self.use_tensorboard:
                self.log_metrics(epoch + 1, train_metrics, val_metrics, current_lr)
            
            # Печатаем результаты эпохи
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
            print(f"{'-'*80}")
            print(format_metrics(train_metrics, prefix='Train'))
            print(format_metrics(val_metrics, prefix='Val  '))
            print(f"{'='*80}\n")
            
            # Сохранение лучшей модели
            if val_metrics['iou'] > self.best_metric:
                print(f"✓ Новая лучшая модель! IoU: {self.best_metric:.4f} → {val_metrics['iou']:.4f}")
                self.best_metric = val_metrics['iou']
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch + 1, is_best=True)
            else:
                self.epochs_without_improvement += 1
                print(f"✗ IoU не улучшился ({self.epochs_without_improvement}/{self.early_stopping_patience})")
            
            # Периодическое сохранение
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                break
            
            print()
        
        # Закрываем TensorBoard writer
        if self.use_tensorboard:
            self.writer.close()
        
        print(f"\n{'='*80}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*80}")
        print(f"Лучший IoU: {self.best_metric:.4f}")
        print(f"{'='*80}\n")
    
    def log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float
    ):
        """Логирование в TensorBoard"""
        # Loss
        self.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        # IoU
        self.writer.add_scalars('IoU', {
            'train': train_metrics['iou'],
            'val': val_metrics['iou']
        }, epoch)
        
        # F1
        self.writer.add_scalars('F1', {
            'train': train_metrics['f1'],
            'val': val_metrics['f1']
        }, epoch)
        
        # Accuracy
        self.writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)
        
        # Precision & Recall
        self.writer.add_scalars('Precision', {
            'train': train_metrics['precision'],
            'val': val_metrics['precision']
        }, epoch)
        
        self.writer.add_scalars('Recall', {
            'train': train_metrics['recall'],
            'val': val_metrics['recall']
        }, epoch)
        
        # Learning rate
        self.writer.add_scalar('Learning_Rate', lr, epoch)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Сохранение checkpoint
        
        Args:
            epoch: Номер эпохи
            is_best: Является ли лучшей моделью
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Сохраняем
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  → Сохранён checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  → Сохранён checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Загрузка checkpoint
        
        Args:
            checkpoint_path: Путь к checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✓ Загружен checkpoint: {checkpoint_path}")
        print(f"  Эпоха: {checkpoint['epoch']}")
        print(f"  Best IoU: {self.best_metric:.4f}")
        
        return checkpoint['epoch']


if __name__ == '__main__':
    # Пример использования
    print("Этот модуль предназначен для импорта")
    print("Используйте train_script.py для запуска обучения")
