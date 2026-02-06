"""
Модуль для визуализации результатов сегментации
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple, List
from pathlib import Path


def visualize_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    title: str = "Prediction",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Визуализация предсказания
    
    Args:
        image: RGB изображение [H, W, 3]
        pred_mask: Предсказанная маска [H, W]
        gt_mask: Ground truth маска [H, W] (опционально)
        title: Заголовок
        save_path: Путь для сохранения (опционально)
        figsize: Размер фигуры
    """
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Предсказание
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title(f'Prediction')
    axes[1].axis('off')
    
    # Ground truth (если есть)
    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    plt.show()


def visualize_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    title: str = "Overlay",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Визуализация с наложением маски на изображение
    
    Args:
        image: RGB изображение [H, W, 3]
        pred_mask: Предсказанная маска [H, W]
        gt_mask: GT маска [H, W] (опционально)
        alpha: Прозрачность наложения
        title: Заголовок
        save_path: Путь для сохранения
        figsize: Размер фигуры
    """
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Оригинал
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Предсказание с наложением
    axes[1].imshow(image)
    # Создаём красную маску для зданий
    overlay_pred = np.zeros_like(image)
    overlay_pred[pred_mask > 0] = [255, 0, 0]  # Красный для зданий
    axes[1].imshow(overlay_pred, alpha=alpha)
    axes[1].set_title('Prediction Overlay')
    axes[1].axis('off')
    
    # GT с наложением (если есть)
    if gt_mask is not None:
        axes[2].imshow(image)
        overlay_gt = np.zeros_like(image)
        overlay_gt[gt_mask > 0] = [0, 255, 0]  # Зелёный для GT
        axes[2].imshow(overlay_gt, alpha=alpha)
        axes[2].set_title('Ground Truth Overlay')
        axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_comparison(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    title: str = "Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Детальное сравнение предсказания с GT
    
    Показывает: TP (зелёный), FP (красный), FN (синий)
    
    Args:
        image: RGB изображение [H, W, 3]
        pred_mask: Предсказание [H, W]
        gt_mask: Ground truth [H, W]
        title: Заголовок
        save_path: Путь для сохранения
        figsize: Размер фигуры
    """
    # Вычисляем TP, FP, FN
    tp = (pred_mask == 1) & (gt_mask == 1)  # True Positive - зелёный
    fp = (pred_mask == 1) & (gt_mask == 0)  # False Positive - красный
    fn = (pred_mask == 0) & (gt_mask == 1)  # False Negative - синий
    
    # Создаём RGB маску ошибок
    error_mask = np.zeros_like(image)
    error_mask[tp] = [0, 255, 0]    # Зелёный - правильно
    error_mask[fp] = [255, 0, 0]    # Красный - ложное срабатывание
    error_mask[fn] = [0, 0, 255]    # Синий - пропуск
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Ряд 1: Исходные данные
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_mask, cmap='gray')
    axes[0, 1].set_title('Prediction')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt_mask, cmap='gray')
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    
    # Ряд 2: Анализ ошибок
    axes[1, 0].imshow(image)
    axes[1, 0].imshow(error_mask, alpha=0.5)
    axes[1, 0].set_title('Error Analysis')
    axes[1, 0].axis('off')
    
    # Легенда
    legend_elements = [
        mpatches.Patch(color='green', label='True Positive'),
        mpatches.Patch(color='red', label='False Positive'),
        mpatches.Patch(color='blue', label='False Negative')
    ]
    axes[1, 0].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Только TP
    tp_viz = np.zeros_like(image)
    tp_viz[tp] = [0, 255, 0]
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(tp_viz, alpha=0.7)
    axes[1, 1].set_title(f'True Positive ({np.sum(tp):,} px)')
    axes[1, 1].axis('off')
    
    # FP и FN
    error_viz = np.zeros_like(image)
    error_viz[fp] = [255, 0, 0]
    error_viz[fn] = [0, 0, 255]
    axes[1, 2].imshow(image)
    axes[1, 2].imshow(error_viz, alpha=0.7)
    axes[1, 2].set_title(f'Errors (FP: {np.sum(fp):,}, FN: {np.sum(fn):,})')
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(
    metrics: dict,
    title: str = "Metrics",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Визуализация метрик в виде bar chart
    
    Args:
        metrics: Словарь с метриками
        title: Заголовок
        save_path: Путь для сохранения
        figsize: Размер фигуры
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color='steelblue', alpha=0.8)
    
    # Добавляем значения над барами
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_result_grid(
    images: List[np.ndarray],
    predictions: List[np.ndarray],
    gt_masks: Optional[List[np.ndarray]] = None,
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 15)
):
    """
    Создаёт сетку результатов для нескольких изображений
    
    Args:
        images: Список изображений
        predictions: Список предсказаний
        gt_masks: Список GT масок (опционально)
        titles: Заголовки для каждого изображения
        save_path: Путь для сохранения
        figsize: Размер фигуры
    """
    n_images = len(images)
    n_cols = 4 if gt_masks else 3  # Image, Pred, GT, Overlay
    
    fig, axes = plt.subplots(n_images, n_cols, figsize=figsize)
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_images):
        # Original
        axes[i, 0].imshow(images[i])
        if titles and i < len(titles):
            axes[i, 0].set_title(titles[i])
        else:
            axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        # Prediction
        axes[i, 1].imshow(predictions[i], cmap='gray')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        
        # GT
        if gt_masks:
            axes[i, 2].imshow(gt_masks[i], cmap='gray')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
            
            # Overlay comparison
            overlay = np.zeros_like(images[i])
            tp = (predictions[i] == 1) & (gt_masks[i] == 1)
            fp = (predictions[i] == 1) & (gt_masks[i] == 0)
            fn = (predictions[i] == 0) & (gt_masks[i] == 1)
            overlay[tp] = [0, 255, 0]
            overlay[fp] = [255, 0, 0]
            overlay[fn] = [0, 0, 255]
            
            axes[i, 3].imshow(images[i])
            axes[i, 3].imshow(overlay, alpha=0.5)
            axes[i, 3].set_title('Error Analysis')
            axes[i, 3].axis('off')
        else:
            # Just overlay prediction
            axes[i, 2].imshow(images[i])
            overlay_pred = np.zeros_like(images[i])
            overlay_pred[predictions[i] > 0] = [255, 0, 0]
            axes[i, 2].imshow(overlay_pred, alpha=0.5)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    print("Этот модуль предназначен для импорта")
    print("Примеры использования см. в notebooks/02_evaluation.ipynb")
