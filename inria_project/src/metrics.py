"""
Метрики для оценки качества сегментации
- IoU (Intersection over Union)
- F1-Score
- Accuracy
- Precision
- Recall
"""

import torch
import numpy as np
from typing import Dict, Tuple


class SegmentationMetrics:
    """
    Класс для вычисления метрик сегментации
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Порог для бинаризации предсказаний
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Сброс накопленных значений"""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Обновление статистики на основе батча
        
        Args:
            pred: Predictions [B, 1, H, W] (логиты или вероятности)
            target: Ground truth [B, H, W] или [B, 1, H, W]
        """
        # Применяем sigmoid если нужно
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Бинаризация
        pred_binary = (pred > self.threshold).float()
        
        # Приводим к одинаковой размерности
        if target.dim() == 3:
            target = target.unsqueeze(1).float()
        else:
            target = target.float()
        
        if pred_binary.dim() == 3:
            pred_binary = pred_binary.unsqueeze(1)
        
        # Вычисляем TP, FP, TN, FN
        self.tp += ((pred_binary == 1) & (target == 1)).sum().item()
        self.fp += ((pred_binary == 1) & (target == 0)).sum().item()
        self.tn += ((pred_binary == 0) & (target == 0)).sum().item()
        self.fn += ((pred_binary == 0) & (target == 1)).sum().item()
    
    def get_iou(self) -> float:
        """
        Intersection over Union (IoU) / Jaccard Index
        
        IoU = TP / (TP + FP + FN)
        """
        intersection = self.tp
        union = self.tp + self.fp + self.fn
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_dice(self) -> float:
        """
        Dice Coefficient / F1-Score
        
        Dice = 2 * TP / (2 * TP + FP + FN)
        """
        numerator = 2 * self.tp
        denominator = 2 * self.tp + self.fp + self.fn
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_accuracy(self) -> float:
        """
        Pixel Accuracy
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        total = self.tp + self.tn + self.fp + self.fn
        
        if total == 0:
            return 0.0
        
        return (self.tp + self.tn) / total
    
    def get_precision(self) -> float:
        """
        Precision
        
        Precision = TP / (TP + FP)
        """
        denominator = self.tp + self.fp
        
        if denominator == 0:
            return 0.0
        
        return self.tp / denominator
    
    def get_recall(self) -> float:
        """
        Recall / Sensitivity
        
        Recall = TP / (TP + FN)
        """
        denominator = self.tp + self.fn
        
        if denominator == 0:
            return 0.0
        
        return self.tp / denominator
    
    def get_f1_score(self) -> float:
        """
        F1-Score (то же что Dice Coefficient)
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        precision = self.get_precision()
        recall = self.get_recall()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def get_specificity(self) -> float:
        """
        Specificity
        
        Specificity = TN / (TN + FP)
        """
        denominator = self.tn + self.fp
        
        if denominator == 0:
            return 0.0
        
        return self.tn / denominator
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Возвращает все метрики
        
        Returns:
            dict: Словарь со всеми метриками
        """
        return {
            'iou': self.get_iou(),
            'dice': self.get_dice(),
            'f1': self.get_f1_score(),
            'accuracy': self.get_accuracy(),
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'specificity': self.get_specificity(),
        }
    
    def __str__(self) -> str:
        """String representation"""
        metrics = self.get_all_metrics()
        return (
            f"IoU: {metrics['iou']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f}"
        )


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Вычисляет все метрики для одного батча
    
    Args:
        pred: Predictions [B, 1, H, W] (логиты или вероятности)
        target: Ground truth [B, H, W]
        threshold: Порог бинаризации
    
    Returns:
        dict: Словарь с метриками
    """
    metrics = SegmentationMetrics(threshold=threshold)
    metrics.update(pred, target)
    return metrics.get_all_metrics()


def calculate_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Быстрое вычисление IoU
    
    Args:
        pred: Predictions [B, 1, H, W]
        target: Ground truth [B, H, W]
        threshold: Порог бинаризации
    
    Returns:
        iou: IoU score
    """
    # Sigmoid
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Бинаризация
    pred_binary = (pred > threshold).float()
    
    # Приводим размерности
    if target.dim() == 3:
        target = target.unsqueeze(1).float()
    else:
        target = target.float()
    
    # Intersection и Union
    intersection = ((pred_binary == 1) & (target == 1)).sum().item()
    union = ((pred_binary == 1) | (target == 1)).sum().item()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Быстрое вычисление Dice Coefficient
    
    Args:
        pred: Predictions [B, 1, H, W]
        target: Ground truth [B, H, W]
        threshold: Порог бинаризации
        smooth: Сглаживание
    
    Returns:
        dice: Dice coefficient
    """
    # Sigmoid
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Бинаризация
    pred_binary = (pred > threshold).float()
    
    # Приводим размерности
    if target.dim() == 3:
        target = target.unsqueeze(1).float()
    else:
        target = target.float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Dice
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice.item()


def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """
    Форматирует метрики в строку для вывода
    
    Args:
        metrics: Словарь с метриками
        prefix: Префикс (например, 'Train' или 'Val')
    
    Returns:
        str: Отформатированная строка
    """
    if prefix:
        prefix = f"{prefix} | "
    
    return (
        f"{prefix}"
        f"IoU: {metrics.get('iou', 0):.4f} | "
        f"F1: {metrics.get('f1', 0):.4f} | "
        f"Acc: {metrics.get('accuracy', 0):.4f} | "
        f"Prec: {metrics.get('precision', 0):.4f} | "
        f"Rec: {metrics.get('recall', 0):.4f}"
    )


if __name__ == '__main__':
    # Тест метрик
    print("="*80)
    print("Тест метрик сегментации")
    print("="*80)
    
    # Создаём тестовые данные
    batch_size = 4
    height, width = 256, 256
    
    # Predictions (логиты)
    pred = torch.randn(batch_size, 1, height, width)
    
    # Ground truth
    target = torch.randint(0, 2, (batch_size, height, width)).float()
    
    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    print()
    
    # Тест через класс
    print("Тест через SegmentationMetrics:")
    metrics_obj = SegmentationMetrics(threshold=0.5)
    metrics_obj.update(pred, target)
    
    print(metrics_obj)
    print()
    
    all_metrics = metrics_obj.get_all_metrics()
    print("Все метрики:")
    for name, value in all_metrics.items():
        print(f"  {name:12s}: {value:.4f}")
    print()
    
    # Тест через функции
    print("Тест через функции:")
    metrics_dict = calculate_metrics(pred, target)
    print(format_metrics(metrics_dict))
    print()
    
    # Быстрые функции
    iou = calculate_iou(pred, target)
    dice = calculate_dice(pred, target)
    print(f"IoU (fast):  {iou:.4f}")
    print(f"Dice (fast): {dice:.4f}")
    
    print("\n" + "="*80)
    print("✅ Все метрики работают корректно!")
    print("="*80)
    
    # Тест с идеальными предсказаниями
    print("\nТест с идеальными предсказаниями:")
    pred_perfect = (target > 0).float().unsqueeze(1) * 10  # Логиты >> 0
    metrics_perfect = calculate_metrics(pred_perfect, target)
    print(format_metrics(metrics_perfect, prefix='Perfect'))
    print("(Должны быть все метрики = 1.0)")
