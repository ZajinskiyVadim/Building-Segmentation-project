"""
Главный скрипт для обучения модели сегментации
Запуск: python train_script.py --config configs/config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Добавляем src в path
sys.path.append(str(Path(__file__).parent / 'src'))

from dataset import create_dataloaders
from model import create_model
from losses import get_loss_function
from train import Trainer


def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    print("\n" + "="*80)
    print("ИНИЦИАЛИЗАЦИЯ ОБУЧЕНИЯ")
    print("="*80)
    print(f"Config: {args.config}")
    print("="*80 + "\n")
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Доступно памяти: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # DataLoaders
    print("Создание DataLoader'ов...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config['data']['root'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        patch_size=config['dataset']['patch_size'],
        stride=config['dataset']['train_stride'],
        cache_images=config['dataset'].get('cache_images', False),
    )
    print()
    
    # Модель
    print("Создание модели...")
    model = create_model(config['model'], device)
    print()
    
    # Loss функция
    print("Создание loss функции...")
    criterion = get_loss_function(config['training']['loss'])
    print()
    
    # Optimizer
    print("Создание оптимизатора...")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    print(f"Optimizer: Adam")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")
    print()
    
    # Scheduler
    scheduler_config = config['training']['scheduler']
    scheduler_name = scheduler_config.get('name', 'ReduceLROnPlateau')
    
    print("Создание scheduler...")
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Максимизируем IoU
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            #verbose=True
        )
        print(f"Scheduler: ReduceLROnPlateau")
        print(f"  Factor: {scheduler_config.get('factor', 0.5)}")
        print(f"  Patience: {scheduler_config.get('patience', 5)}")
    
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-6
        )
        print(f"Scheduler: CosineAnnealingLR")
        print(f"  T_max: {config['training']['epochs']}")
    
    else:
        scheduler = None
        print("Scheduler: None")
    print()
    
    # Trainer
    trainer_config = {
        'use_amp': True,  # Mixed precision для ускорения
        'use_tensorboard': config['logging']['use_tensorboard'],
        'log_dir': config['logging']['save_dir'] + '/tensorboard',
        'checkpoint_dir': config['checkpoints']['save_dir'],
        'early_stopping_patience': config['training']['early_stopping']['patience'],
    }
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        scheduler=scheduler,
    )
    
    # Resume from checkpoint (опционально)
    start_epoch = 0
    if args.resume:
        print(f"Загрузка checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print()
    
    # Обучение
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)
    
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("="*80)
    print(f"Лучшая модель сохранена: {trainer_config['checkpoint_dir']}/best_model.pth")
    print(f"TensorBoard логи: {trainer_config['log_dir']}")
    print("="*80 + "\n")
    
    print("Для просмотра результатов в TensorBoard:")
    print(f"  tensorboard --logdir {trainer_config['log_dir']}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение модели сегментации')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Путь к конфигурационному файлу'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Путь к checkpoint для продолжения обучения'
    )
    
    args = parser.parse_args()
    
    main(args)
