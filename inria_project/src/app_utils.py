"""
Вспомогательные функции для Streamlit приложения
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend для Streamlit


def load_example_images():
    """
    Загрузка примеров изображений из папки examples/
    
    Returns:
        list: Список словарей с информацией о примерах
    """
    examples_dir = Path('examples')
    
    if not examples_dir.exists():
        return []
    
    examples = []
    
    # Поддерживаемые форматы
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    for ext in extensions:
        for img_path in examples_dir.glob(f'*{ext}'):
            examples.append({
                'name': img_path.stem,
                'path': str(img_path),
                'extension': ext
            })
    
    return sorted(examples, key=lambda x: x['name'])


def apply_colormap_to_mask(mask: np.ndarray, colormap: str = 'Red') -> np.ndarray:
    """
    Применение цветовой схемы к бинарной маске
    
    Args:
        mask: Бинарная маска (H, W)
        colormap: Название цветовой схемы
    
    Returns:
        np.ndarray: Цветная маска (H, W, 3)
    """
    # Нормализация маски
    mask_normalized = (mask * 255).astype(np.uint8)
    
    # Выбор цветовой схемы
    if colormap == 'Red':
        # Красная маска
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[..., 0] = mask_normalized  # R
    
    elif colormap == 'Blue':
        # Синяя маска
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[..., 2] = mask_normalized  # B
    
    elif colormap == 'Green':
        # Зелёная маска
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[..., 1] = mask_normalized  # G
    
    elif colormap == 'Jet':
        # Jet colormap
        colored_mask = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
        colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    elif colormap == 'Viridis':
        # Viridis colormap
        colored_mask = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_VIRIDIS)
        colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    else:
        # По умолчанию красная
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[..., 0] = mask_normalized
    
    return colored_mask


def create_overlay_image(
    image: np.ndarray,
    mask_colored: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Создание overlay изображения (маска поверх оригинала)
    
    Args:
        image: Оригинальное изображение (H, W, 3)
        mask_colored: Цветная маска (H, W, 3)
        alpha: Прозрачность маски (0.0 - 1.0)
    
    Returns:
        np.ndarray: Overlay изображение (H, W, 3)
    """
    # Убедимся что размеры совпадают
    if image.shape[:2] != mask_colored.shape[:2]:
        mask_colored = cv2.resize(
            mask_colored,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Наложение
    overlay = image.copy().astype(np.float32)
    mask_float = mask_colored.astype(np.float32)
    
    # Только там где маска не нулевая
    mask_binary = (mask_colored.sum(axis=-1) > 0)[..., None]
    
    overlay = np.where(
        mask_binary,
        overlay * (1 - alpha) + mask_float * alpha,
        overlay
    ).astype(np.uint8)
    
    return overlay


def create_side_by_side_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    probs: np.ndarray
) -> np.ndarray:
    """
    Создание изображения с тремя панелями: Original | Mask | Overlay
    
    Args:
        image: Оригинальное изображение
        mask: Бинарная маска
        probs: Карта вероятностей
    
    Returns:
        np.ndarray: Сравнительное изображение
    """
    # Размеры
    h, w = image.shape[:2]
    
    # Создаём маску в цвете
    mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    mask_colored[mask > 0] = [255, 0, 0]  # Красный
    
    # Создаём overlay
    overlay = create_overlay_image(image, mask_colored, alpha=0.5)
    
    # Объединяем горизонтально
    comparison = np.hstack([image, mask_colored, overlay])
    
    return comparison


def plot_area_statistics(probabilities: np.ndarray) -> plt.Figure:
    """
    График статистики площади (гистограмма вероятностей)
    
    Args:
        probabilities: Плоский массив вероятностей
    
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Гистограмма
    ax.hist(probabilities, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Порог
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    
    # Оформление
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def format_area(area_m2: float) -> str:
    """
    Форматирование площади с автоматическим выбором единиц
    
    Args:
        area_m2: Площадь в м²
    
    Returns:
        str: Отформатированная строка
    """
    if area_m2 < 10000:
        return f"{area_m2:.2f} м²"
    elif area_m2 < 1000000:
        return f"{area_m2 / 10000:.2f} га"
    else:
        return f"{area_m2 / 1000000:.2f} км²"


def create_summary_card(areas: dict) -> str:
    """
    Создание HTML карточки с результатами
    
    Args:
        areas: Словарь с данными о площади
    
    Returns:
        str: HTML разметка
    """
    html = f"""
    <div class="metric-card">
        <h3>📊 Результаты сегментации</h3>
        <table style="width:100%">
            <tr>
                <td><b>Площадь застройки:</b></td>
                <td>{areas['area_ha']:.2f} га ({areas['area_m2']:,.0f} м²)</td>
            </tr>
            <tr>
                <td><b>Пикселей зданий:</b></td>
                <td>{areas['building_pixels']:,}</td>
            </tr>
            <tr>
                <td><b>Покрытие:</b></td>
                <td>{areas['coverage_percent']:.2f}%</td>
            </tr>
        </table>
    </div>
    """
    return html


def resize_image_for_display(image: np.ndarray, max_size: int = 1000) -> np.ndarray:
    """
    Изменение размера изображения для отображения
    
    Args:
        image: Изображение
        max_size: Максимальный размер (ширина или высота)
    
    Returns:
        np.ndarray: Изображение с изменённым размером
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    # Вычисляем масштаб
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Изменяем размер
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized


def create_heatmap_overlay(
    image: np.ndarray,
    probs: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Создание heatmap overlay (карта вероятностей поверх изображения)
    
    Args:
        image: Оригинальное изображение
        probs: Карта вероятностей (0-1)
        alpha: Прозрачность heatmap
    
    Returns:
        np.ndarray: Heatmap overlay
    """
    # Нормализация вероятностей
    probs_normalized = (probs * 255).astype(np.uint8)
    
    # Применение colormap
    heatmap = cv2.applyColorMap(probs_normalized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize если нужно
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(
            heatmap,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def calculate_building_statistics(mask: np.ndarray, pixel_size_m: float = 0.3) -> dict:
    """
    Расчёт детальной статистики о зданиях
    
    Args:
        mask: Бинарная маска
        pixel_size_m: Размер пикселя в метрах
    
    Returns:
        dict: Статистика
    """
    # Связные компоненты (отдельные здания)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )
    
    # Убираем фон (label 0)
    num_buildings = num_labels - 1
    
    # Площади отдельных зданий
    building_areas = []
    for i in range(1, num_labels):
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        area_m2 = area_pixels * (pixel_size_m ** 2)
        building_areas.append(area_m2)
    
    # Сортируем по размеру
    building_areas = sorted(building_areas, reverse=True)
    
    return {
        'num_buildings': num_buildings,
        'building_areas_m2': building_areas,
        'largest_building_m2': building_areas[0] if building_areas else 0,
        'smallest_building_m2': building_areas[-1] if building_areas else 0,
        'average_building_m2': np.mean(building_areas) if building_areas else 0
    }


def export_results_to_geojson(
    mask: np.ndarray,
    transform: tuple = None,
    crs: str = 'EPSG:4326'
) -> dict:
    """
    Экспорт результатов в GeoJSON (для будущего функционала)
    
    Args:
        mask: Бинарная маска
        transform: Affine transform (если известен)
        crs: Coordinate reference system
    
    Returns:
        dict: GeoJSON структура
    """
    # TODO: Реализация векторизации маски в полигоны
    # Требует rasterio и shapely
    
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    return geojson
