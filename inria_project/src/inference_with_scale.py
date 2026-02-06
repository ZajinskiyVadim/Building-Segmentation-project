"""
Обёртка для inference с автоматическим определением масштаба
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union, Dict
from pathlib import Path
from PIL import Image

# Импорты из существующих модулей
from inference import predict_large_image as _predict_large_image
from area_calculator_advanced import AreaCalculatorAdvanced


def predict_and_calculate_area(
    model: torch.nn.Module,
    image_source: Union[str, Path, np.ndarray],
    device: torch.device,
    patch_size: int = 512,
    stride: int = 512,
    use_amp: bool = True,
    threshold: float = 0.5,
    pixel_size_m: Optional[float] = None,
    auto_extract_scale: bool = True
) -> Dict:
    """
    Полный pipeline: Сегментация + Расчёт площади с автоматическим масштабом
    
    Args:
        model: Обученная модель
        image_source: Путь к изображению (str/Path) или numpy array
        device: Устройство (cuda/cpu)
        patch_size: Размер патча для inference
        stride: Шаг патчей
        use_amp: Mixed precision
        threshold: Порог бинаризации
        pixel_size_m: Явно указанный масштаб (если None - авто)
        auto_extract_scale: Автоматически извлекать масштаб из GeoTIFF
    
    Returns:
        dict с результатами:
            - pred_mask: Бинарная маска
            - pred_probs: Вероятности
            - areas: Словарь с площадями
            - pixel_size_m: Использованный масштаб
    """
    
    # 1. Загрузка изображения
    if isinstance(image_source, (str, Path)):
        image_path = Path(image_source)
        image = np.array(Image.open(image_path).convert('RGB'))
        has_path = True
    else:
        image = image_source
        image_path = None
        has_path = False
    
    # 2. Сегментация (независимо от масштаба!)
    pred_mask, pred_probs = _predict_large_image(
        model=model,
        image=image,
        device=device,
        patch_size=patch_size,
        stride=stride,
        use_amp=use_amp,
        threshold=threshold
    )
    
    # 3. Определение масштаба
    if pixel_size_m is not None:
        # Явно указан - используем его
        calculator = AreaCalculatorAdvanced(pixel_size_m=pixel_size_m)
        scale_source = f"manual ({pixel_size_m}m)"
        
    elif auto_extract_scale and has_path and image_path.suffix.lower() in ['.tif', '.tiff']:
        # Попытка извлечь из GeoTIFF
        try:
            calculator = AreaCalculatorAdvanced.from_geotiff(image_path)
            scale_source = f"geotiff ({calculator.pixel_size_m}m)"
        except Exception as e:
            print(f"⚠️  Не удалось извлечь масштаб из GeoTIFF: {e}")
            print(f"    Использую значение по умолчанию: 0.3 м/пиксель")
            calculator = AreaCalculatorAdvanced(pixel_size_m=0.3)
            scale_source = "default (0.3m)"
    else:
        # Значение по умолчанию (INRIA)
        calculator = AreaCalculatorAdvanced(pixel_size_m=0.3)
        scale_source = "default (0.3m)"
    
    # 4. Расчёт площади
    areas = calculator.calculate_area(pred_mask)
    
    # 5. Результат
    return {
        'pred_mask': pred_mask,
        'pred_probs': pred_probs,
        'areas': areas,
        'pixel_size_m': calculator.pixel_size_m,
        'scale_source': scale_source,
        'image_shape': image.shape
    }


def batch_predict_and_calculate(
    model: torch.nn.Module,
    image_paths: list,
    device: torch.device,
    **kwargs
) -> list:
    """
    Batch обработка с автоматическим масштабом для каждого изображения
    
    Args:
        model: Модель
        image_paths: Список путей к изображениям
        device: Устройство
        **kwargs: Аргументы для predict_and_calculate_area
    
    Returns:
        Список результатов для каждого изображения
    """
    results = []
    
    for img_path in image_paths:
        print(f"\n📷 Обработка: {Path(img_path).name}")
        
        result = predict_and_calculate_area(
            model=model,
            image_source=img_path,
            device=device,
            **kwargs
        )
        
        areas = result['areas']
        print(f"   Масштаб: {result['pixel_size_m']} м/пиксель ({result['scale_source']})")
        print(f"   Площадь: {areas['area_ha']:.2f} га ({areas['area_m2']:.0f} м²)")
        print(f"   Покрытие: {areas['coverage_percent']:.2f}%")
        
        results.append(result)
    
    return results


# Пример использования
if __name__ == '__main__':
    import sys
    sys.path.append('src')
    
    from model import create_model
    from inference import load_model_from_checkpoint
    
    print("="*80)
    print("Тест автоматического определения масштаба")
    print("="*80)
    
    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Устройство: {device}")
    
    # Конфигурация модели (пример)
    model_config = {
        'architecture': 'unet',
        'encoder': 'resnet50',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1
    }
    
    model = create_model(model_config, device)
    
    # Загрузка checkpoint
    checkpoint_path = "models/checkpoints/best_model.pth"
    if Path(checkpoint_path).exists():
        model = load_model_from_checkpoint(checkpoint_path, model, device)
        print(f"✅ Модель загружена: {checkpoint_path}")
    else:
        print(f"⚠️  Checkpoint не найден: {checkpoint_path}")
        print("   Используем неинициализированную модель (только для теста)")
    
    # Тест 1: С автоматическим извлечением масштаба
    print("\n" + "="*80)
    print("Тест 1: Автоматическое извлечение масштаба из GeoTIFF")
    print("="*80)
    
    test_image = "data/AerialImageDataset/train/images/austin1.tif"
    
    if Path(test_image).exists():
        result = predict_and_calculate_area(
            model=model,
            image_source=test_image,
            device=device,
            patch_size=512,
            stride=512,
            threshold=0.5,
            auto_extract_scale=True  # Автоматически!
        )
        
        print(f"\n✅ Результаты:")
        print(f"   Масштаб: {result['pixel_size_m']} м/пиксель")
        print(f"   Источник: {result['scale_source']}")
        print(f"   Площадь: {result['areas']['area_ha']:.4f} га")
        print(f"   Покрытие: {result['areas']['coverage_percent']:.2f}%")
    else:
        print(f"⚠️  Тестовое изображение не найдено: {test_image}")
    
    # Тест 2: С явным указанием масштаба
    print("\n" + "="*80)
    print("Тест 2: Явное указание масштаба")
    print("="*80)
    
    # Создаём тестовое изображение
    test_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    
    result = predict_and_calculate_area(
        model=model,
        image_source=test_array,
        device=device,
        pixel_size_m=0.5,  # Явно указываем!
        auto_extract_scale=False
    )
    
    print(f"\n✅ Результаты:")
    print(f"   Масштаб: {result['pixel_size_m']} м/пиксель")
    print(f"   Источник: {result['scale_source']}")
    
    # Тест 3: Batch обработка
    print("\n" + "="*80)
    print("Тест 3: Batch обработка")
    print("="*80)
    
    test_images = [
        "data/AerialImageDataset/train/images/austin1.tif",
        "data/AerialImageDataset/train/images/chicago1.tif",
        "data/AerialImageDataset/train/images/vienna1.tif"
    ]
    
    # Фильтруем существующие
    existing_images = [img for img in test_images if Path(img).exists()]
    
    if existing_images:
        results = batch_predict_and_calculate(
            model=model,
            image_paths=existing_images,
            device=device,
            patch_size=512,
            stride=512,
            auto_extract_scale=True
        )
        
        print(f"\n✅ Обработано {len(results)} изображений")
    else:
        print("⚠️  Тестовые изображения не найдены")
    
    print("\n" + "="*80)
    print("✓ Все тесты завершены!")
    print("="*80)
