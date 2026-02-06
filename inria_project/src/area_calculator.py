"""
Модуль для расчёта площади застройки по сегментационной маске
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Optional, List, Optional, List, Optional, List


class AreaCalculator:
    """
    Класс для расчёта площади застройки
    """
    
    def __init__(self, pixel_size_m: float = 0.3):
        """
        Args:
            pixel_size_m: Размер пикселя в метрах (по умолчанию 0.3м для INRIA)
        """
        self.pixel_size_m = pixel_size_m
        self.pixel_area_m2 = pixel_size_m ** 2  # м² на пиксель
    
    def calculate_area(
        self,
        mask: np.ndarray,
        return_dict: bool = True
    ) -> Dict[str, float]:
        """
        Расчёт площади застройки
        
        Args:
            mask: Бинарная маска [H, W], значения 0/1
            return_dict: Возвращать словарь с разными единицами
        
        Returns:
            dict с площадями в разных единицах
        """
        # Количество пикселей зданий
        building_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        
        # Площадь в м²
        area_m2 = building_pixels * self.pixel_area_m2
        
        # Площадь в га (1 га = 10,000 м²)
        area_ha = area_m2 / 10000
        
        # Процент покрытия
        coverage_percent = (building_pixels / total_pixels) * 100
        
        if return_dict:
            return {
                'building_pixels': int(building_pixels),
                'total_pixels': int(total_pixels),
                'area_m2': float(area_m2),
                'area_ha': float(area_ha),
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
        
        # Predicted
        pred = areas['predicted']
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
            
            # True Positives, False Positives, etc.
            if 'true_positive' in areas:
                tp = areas['true_positive']
                fp = areas['false_positive']
                fn = areas['false_negative']
                
                print(f"\nДЕТАЛИЗАЦИЯ:")
                print(f"  True Positive:  {self.format_area(tp['area_m2'])} ({tp['coverage_percent']:.2f}%)")
                print(f"  False Positive: {self.format_area(fp['area_m2'])} ({fp['coverage_percent']:.2f}%)")
                print(f"  False Negative: {self.format_area(fn['area_m2'])} ({fn['coverage_percent']:.2f}%)")
        
        print("="*80 + "\n")


def calculate_batch_areas(
    pred_masks: np.ndarray,
    gt_masks: Optional[np.ndarray] = None,
    pixel_size_m: float = 0.3
) -> List[Dict[str, Dict[str, float]]]:
    """
    Расчёт площадей для батча масок
    
    Args:
        pred_masks: Предсказанные маски [B, H, W]
        gt_masks: GT маски [B, H, W] (опционально)
        pixel_size_m: Размер пикселя в метрах
    
    Returns:
        Список словарей с площадями для каждого изображения
    """
    calculator = AreaCalculator(pixel_size_m=pixel_size_m)
    
    batch_size = pred_masks.shape[0]
    results = []
    
    for i in range(batch_size):
        pred_mask = pred_masks[i]
        gt_mask = gt_masks[i] if gt_masks is not None else None
        
        areas = calculator.calculate_per_class_area(pred_mask, gt_mask)
        results.append(areas)
    
    return results


if __name__ == '__main__':
    # Пример использования
    print("="*80)
    print("Тест AreaCalculator")
    print("="*80)
    
    # Создаём тестовую маску
    mask = np.zeros((5000, 5000), dtype=np.uint8)
    mask[1000:2000, 1000:2000] = 1  # Квадрат 1000x1000 пикселей
    
    # Создаём GT маску (чуть отличается)
    gt_mask = np.zeros((5000, 5000), dtype=np.uint8)
    gt_mask[1000:1900, 1000:1900] = 1  # Квадрат 900x900 пикселей
    
    # Расчёт
    calculator = AreaCalculator(pixel_size_m=0.3)
    areas = calculator.calculate_per_class_area(mask, gt_mask)
    
    # Вывод
    calculator.print_summary(areas)
    
    print("✓ AreaCalculator работает корректно!")
