"""
Улучшенный модуль для расчёта площади с автоматическим извлечением масштаба из GeoTIFF
"""

import numpy as np
from typing import Dict, Optional, List, Union, Tuple
from pathlib import Path


class AreaCalculatorAdvanced:
    """
    Улучшенный класс для расчёта площади застройки
    с автоматическим извлечением масштаба из GeoTIFF метаданных
    """
    
    def __init__(self, pixel_size_m: Optional[float] = None):
        """
        Args:
            pixel_size_m: Размер пикселя в метрах
                         Если None, будет извлечён из GeoTIFF метаданных
        """
        self.pixel_size_m = pixel_size_m if pixel_size_m is not None else 0.3
        self.pixel_area_m2 = self.pixel_size_m ** 2
    
    @staticmethod
    def extract_pixel_size_from_geotiff(image_path: Union[str, Path]) -> Tuple[float, float]:
        """
        Извлечение размера пикселя из GeoTIFF метаданных
        
        Args:
            image_path: Путь к GeoTIFF файлу
        
        Returns:
            (pixel_width_m, pixel_height_m): Размер пикселя в метрах
        
        Raises:
            ImportError: Если rasterio не установлен
            FileNotFoundError: Если файл не найден
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "Для автоматического извлечения масштаба требуется rasterio.\n"
                "Установите: pip install rasterio"
            )
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        
        with rasterio.open(image_path) as src:
            # Affine transform содержит информацию о масштабе
            transform = src.transform
            
            # Размер пикселя в единицах CRS (обычно метры)
            pixel_width = abs(transform.a)   # Ширина пикселя
            pixel_height = abs(transform.e)  # Высота пикселя
            
            # CRS (система координат)
            crs = src.crs
            
            print(f"📐 Метаданные GeoTIFF:")
            print(f"   CRS: {crs}")
            print(f"   Pixel size: {pixel_width:.4f} × {pixel_height:.4f} м")
            print(f"   Transform: {transform}")
        
        return pixel_width, pixel_height
    
    @classmethod
    def from_geotiff(cls, image_path: Union[str, Path]) -> 'AreaCalculatorAdvanced':
        """
        Создать AreaCalculator с автоматическим извлечением масштаба из GeoTIFF
        
        Args:
            image_path: Путь к GeoTIFF файлу
        
        Returns:
            AreaCalculatorAdvanced с корректным масштабом
        
        Example:
            >>> calculator = AreaCalculatorAdvanced.from_geotiff('austin1.tif')
            >>> areas = calculator.calculate_area(mask)
        """
        pixel_width, pixel_height = cls.extract_pixel_size_from_geotiff(image_path)
        
        # Используем среднее значение (обычно они равны)
        pixel_size_m = (pixel_width + pixel_height) / 2
        
        if abs(pixel_width - pixel_height) > 0.01:
            print(f"⚠️  Предупреждение: пиксели не квадратные! "
                  f"({pixel_width:.4f} × {pixel_height:.4f})")
            print(f"    Используется среднее значение: {pixel_size_m:.4f} м")
        
        return cls(pixel_size_m=pixel_size_m)
    
    def calculate_area(
        self,
        mask: np.ndarray,
        return_dict: bool = True
    ) -> Union[Dict[str, float], float]:
        """
        Расчёт площади застройки
        
        Args:
            mask: Бинарная маска [H, W], значения 0/1
            return_dict: Возвращать словарь с разными единицами
        
        Returns:
            dict с площадями в разных единицах или float (area_m2)
        """
        # Количество пикселей зданий
        building_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        
        # Площадь в м²
        area_m2 = building_pixels * self.pixel_area_m2
        
        # Площадь в га (1 га = 10,000 м²)
        area_ha = area_m2 / 10000
        
        # Площадь в км² (для больших территорий)
        area_km2 = area_m2 / 1000000
        
        # Процент покрытия
        coverage_percent = (building_pixels / total_pixels) * 100
        
        if return_dict:
            return {
                'building_pixels': int(building_pixels),
                'total_pixels': int(total_pixels),
                'pixel_size_m': float(self.pixel_size_m),
                'pixel_area_m2': float(self.pixel_area_m2),
                'area_m2': float(area_m2),
                'area_ha': float(area_ha),
                'area_km2': float(area_km2),
                'coverage_percent': float(coverage_percent)
            }
        else:
            return area_m2
    
    def calculate_per_class_area(
        self,
        pred_mask: np.ndarray,
        gt_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Расчёт площади с разбивкой по классам предсказания
        
        Args:
            pred_mask: Предсказанная маска [H, W]
            gt_mask: Ground truth маска [H, W] (опционально)
        
        Returns:
            dict с площадями TP, FP, TN, FN
        """
        result = {
            'predicted': self.calculate_area(pred_mask)
        }
        
        if gt_mask is not None:
            # True Positives, False Positives, True Negatives, False Negatives
            tp_mask = (pred_mask == 1) & (gt_mask == 1)
            fp_mask = (pred_mask == 1) & (gt_mask == 0)
            tn_mask = (pred_mask == 0) & (gt_mask == 0)
            fn_mask = (pred_mask == 0) & (gt_mask == 1)
            
            result['ground_truth'] = self.calculate_area(gt_mask)
            result['true_positive'] = self.calculate_area(tp_mask)
            result['false_positive'] = self.calculate_area(fp_mask)
            result['true_negative'] = self.calculate_area(tn_mask)
            result['false_negative'] = self.calculate_area(fn_mask)
            
            # Accuracy площади
            pred_area = result['predicted']['area_m2']
            gt_area = result['ground_truth']['area_m2']
            
            if gt_area > 0:
                area_error_m2 = abs(pred_area - gt_area)
                area_error_percent = (area_error_m2 / gt_area) * 100
                area_accuracy = 100 - area_error_percent
                
                result['area_metrics'] = {
                    'error_m2': float(area_error_m2),
                    'error_percent': float(area_error_percent),
                    'accuracy_percent': float(area_accuracy)
                }
        
        return result
    
    def format_area(self, area_m2: float, unit: str = 'auto') -> str:
        """
        Форматирование площади для вывода
        
        Args:
            area_m2: Площадь в м²
            unit: Единица измерения ('auto', 'm2', 'ha', 'km2')
        
        Returns:
            Отформатированная строка
        """
        if unit == 'auto':
            if area_m2 < 10000:  # < 1 га
                return f"{area_m2:.2f} м²"
            elif area_m2 < 1000000:  # < 1 км²
                return f"{area_m2 / 10000:.2f} га"
            else:
                return f"{area_m2 / 1000000:.2f} км²"
        elif unit == 'm2':
            return f"{area_m2:.2f} м²"
        elif unit == 'ha':
            return f"{area_m2 / 10000:.2f} га"
        elif unit == 'km2':
            return f"{area_m2 / 1000000:.2f} км²"
        else:
            raise ValueError(f"Unknown unit: {unit}")
    
    def print_summary(self, areas: Dict[str, Dict[str, float]]):
        """
        Вывод красивой сводки по площадям
        
        Args:
            areas: Словарь с площадями (из calculate_per_class_area)
        """
        print("\n" + "="*80)
        print("РАСЧЁТ ПЛОЩАДИ ЗАСТРОЙКИ")
        print("="*80)
        
        # Масштаб
        pred = areas['predicted']
        print(f"\nМАССТАБ:")
        print(f"  Размер пикселя: {pred['pixel_size_m']:.4f} м")
        print(f"  Площадь пикселя: {pred['pixel_area_m2']:.6f} м²")
        
        # Predicted
        print(f"\nПРЕДСКАЗАНО:")
        print(f"  Площадь застройки: {self.format_area(pred['area_m2'])}")
        print(f"  Количество пикселей: {pred['building_pixels']:,}")
        print(f"  Покрытие: {pred['coverage_percent']:.2f}%")
        
        # Ground Truth
        if 'ground_truth' in areas:
            gt = areas['ground_truth']
            print(f"\nGROUND TRUTH:")
            print(f"  Площадь застройки: {self.format_area(gt['area_m2'])}")
            print(f"  Количество пикселей: {gt['building_pixels']:,}")
            print(f"  Покрытие: {gt['coverage_percent']:.2f}%")
            
            # Разница
            diff_m2 = pred['area_m2'] - gt['area_m2']
            diff_percent = (diff_m2 / gt['area_m2']) * 100 if gt['area_m2'] > 0 else 0
            
            print(f"\nРАЗНИЦА:")
            print(f"  Абсолютная: {self.format_area(abs(diff_m2))}")
            print(f"  Относительная: {diff_percent:+.2f}%")
            
            # Метрики точности площади
            if 'area_metrics' in areas:
                metrics = areas['area_metrics']
                print(f"\nТОЧНОСТЬ ПЛОЩАДИ:")
                print(f"  Ошибка: {self.format_area(metrics['error_m2'])} "
                      f"({metrics['error_percent']:.2f}%)")
                print(f"  Точность: {metrics['accuracy_percent']:.2f}%")
            
            # True Positives, False Positives, etc.
            if 'true_positive' in areas:
                tp = areas['true_positive']
                fp = areas['false_positive']
                fn = areas['false_negative']
                
                print(f"\nДЕТАЛИЗАЦИЯ:")
                print(f"  True Positive:  {self.format_area(tp['area_m2'])} "
                      f"({tp['coverage_percent']:.2f}%)")
                print(f"  False Positive: {self.format_area(fp['area_m2'])} "
                      f"({fp['coverage_percent']:.2f}%)")
                print(f"  False Negative: {self.format_area(fn['area_m2'])} "
                      f"({fn['coverage_percent']:.2f}%)")
        
        print("="*80 + "\n")


# Пример использования
if __name__ == '__main__':
    print("="*80)
    print("Тест AreaCalculatorAdvanced")
    print("="*80)
    
    # Пример 1: С явным указанием масштаба (как сейчас)
    print("\n1️⃣  СПОСОБ 1: Явное указание масштаба")
    print("-" * 80)
    
    calculator_manual = AreaCalculatorAdvanced(pixel_size_m=0.3)
    
    # Тестовая маска
    mask = np.zeros((5000, 5000), dtype=np.uint8)
    mask[1000:2000, 1000:2000] = 1  # Квадрат 1000x1000 пикселей
    
    areas = calculator_manual.calculate_area(mask)
    
    print(f"Pixel size: {areas['pixel_size_m']} м")
    print(f"Building pixels: {areas['building_pixels']:,}")
    print(f"Area: {areas['area_m2']:.2f} м² = {areas['area_ha']:.4f} га")
    
    # Пример 2: Автоматическое извлечение из GeoTIFF
    print("\n2️⃣  СПОСОБ 2: Автоматическое извлечение из GeoTIFF")
    print("-" * 80)
    
    # Путь к реальному GeoTIFF файлу
    geotiff_path = "data/AerialImageDataset/train/images/austin1.tif"
    
    if Path(geotiff_path).exists():
        try:
            # Создание calculator с автоматическим масштабом
            calculator_auto = AreaCalculatorAdvanced.from_geotiff(geotiff_path)
            
            areas_auto = calculator_auto.calculate_area(mask)
            
            print(f"\n✅ Масштаб извлечён автоматически!")
            print(f"Pixel size: {areas_auto['pixel_size_m']} м")
            print(f"Area: {areas_auto['area_m2']:.2f} м²")
            
        except ImportError as e:
            print(f"\n⚠️  {e}")
            print("Для автоматического извлечения установите: pip install rasterio")
    else:
        print(f"\n⚠️  Файл не найден: {geotiff_path}")
        print("Пропускаем тест автоматического извлечения")
    
    # Пример 3: Расчёт с GT маской
    print("\n3️⃣  СПОСОБ 3: Сравнение с Ground Truth")
    print("-" * 80)
    
    gt_mask = np.zeros((5000, 5000), dtype=np.uint8)
    gt_mask[1000:1900, 1000:1900] = 1  # Чуть меньший квадрат
    
    calculator = AreaCalculatorAdvanced(pixel_size_m=0.3)
    areas_detailed = calculator.calculate_per_class_area(mask, gt_mask)
    
    calculator.print_summary(areas_detailed)
    
    print("✓ Все тесты пройдены успешно!")
