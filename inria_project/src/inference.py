"""
Inference модуль для предсказаний на изображениях
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def predict_patch(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    use_amp: bool = True
) -> torch.Tensor:
    """
    Предсказание для одного патча
    
    Args:
        model: Обученная модель
        image: Входное изображение [1, 3, H, W]
        device: Устройство (cuda/cpu)
        use_amp: Использовать mixed precision
    
    Returns:
        prediction: Предсказание [1, 1, H, W] (логиты)
    """
    model.eval()
    
    image = image.to(device)
    
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                prediction = model(image)
        else:
            prediction = model(image)
    
    return prediction


def predict_large_image(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 512,
    use_amp: bool = True,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Предсказание для большого изображения с разбивкой на патчи
    
    Args:
        model: Обученная модель
        image: RGB изображение [H, W, 3], значения [0, 255]
        device: Устройство
        patch_size: Размер патча
        stride: Шаг для патчей
        use_amp: Mixed precision
        threshold: Порог бинаризации
    
    Returns:
        pred_mask: Бинарная маска [H, W], значения 0/1
        pred_probs: Вероятности [H, W], значения [0, 1]
    """
    model.eval()
    
    H, W, C = image.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    
    # Подготовка выходных массивов
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.int32)
    
    # ImageNet нормализация
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Генерация координат патчей
    patch_coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch_coords.append((y, x))
    
    # Добавляем граничные патчи если нужно
    if (H - patch_size) % stride != 0:
        y = H - patch_size
        for x in range(0, W - patch_size + 1, stride):
            patch_coords.append((y, x))
    
    if (W - patch_size) % stride != 0:
        x = W - patch_size
        for y in range(0, H - patch_size + 1, stride):
            patch_coords.append((y, x))
    
    # Угловой патч
    if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
        patch_coords.append((H - patch_size, W - patch_size))
    
    # Inference для каждого патча
    with torch.no_grad():
        for y, x in tqdm(patch_coords, desc="Processing patches"):
            # Вырезаем патч
            patch = image[y:y+patch_size, x:x+patch_size, :]
            
            # Нормализация
            patch_norm = (patch / 255.0 - mean) / std
            
            # [H, W, C] -> [C, H, W] -> [1, C, H, W]
            patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).float()
            patch_tensor = patch_tensor.to(device)
            
            # Предсказание
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(patch_tensor)
            else:
                pred = model(patch_tensor)
            
            # Sigmoid для получения вероятностей
            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            # Накапливаем предсказания
            pred_sum[y:y+patch_size, x:x+patch_size] += pred_prob
            pred_count[y:y+patch_size, x:x+patch_size] += 1
    
    # Усредняем перекрывающиеся области
    pred_probs = pred_sum / np.maximum(pred_count, 1)
    
    # Бинаризация
    pred_mask = (pred_probs > threshold).astype(np.uint8)
    
    return pred_mask, pred_probs


def predict_batch(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Предсказание для батча изображений
    
    Args:
        model: Модель
        images: Батч изображений [B, 3, H, W]
        device: Устройство
        use_amp: Mixed precision
        threshold: Порог
    
    Returns:
        masks: Бинарные маски [B, H, W]
        probs: Вероятности [B, H, W]
    """
    model.eval()
    
    images = images.to(device)
    
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
        else:
            outputs = model(images)
    
    # Sigmoid для вероятностей
    probs = torch.sigmoid(outputs).squeeze(1)  # [B, H, W]
    
    # Бинаризация
    masks = (probs > threshold).float()
    
    return masks, probs


def load_model_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device,
    strict: bool = True
) -> nn.Module:
    """
    Загрузка модели из checkpoint
    
    Args:
        checkpoint_path: Путь к checkpoint
        model: Модель (архитектура)
        device: Устройство
        strict: Строгая загрузка весов
    
    Returns:
        model: Модель с загруженными весами
    """
    print(f"Загрузка checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Загружаем веса модели
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    model = model.to(device)
    model.eval()
    
    # Выводим информацию
    print(f"✓ Checkpoint загружен:")
    print(f"  Эпоха: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best metric: {checkpoint.get('best_metric', 'N/A'):.4f}")
    
    return model


def predict_from_file(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 512,
    use_amp: bool = True,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Предсказание из файла изображения
    
    Args:
        model: Модель
        image_path: Путь к изображению
        device: Устройство
        patch_size: Размер патча
        stride: Шаг
        use_amp: Mixed precision
        threshold: Порог
    
    Returns:
        pred_mask: Бинарная маска
        pred_probs: Вероятности
    """
    # Загружаем изображение
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Предсказание
    pred_mask, pred_probs = predict_large_image(
        model=model,
        image=image,
        device=device,
        patch_size=patch_size,
        stride=stride,
        use_amp=use_amp,
        threshold=threshold
    )
    
    return pred_mask, pred_probs


if __name__ == '__main__':
    print("Этот модуль предназначен для импорта")
    print("Используйте evaluate.py для тестирования модели")
