"""
Скрипт для оценки модели на test set
Запуск: python evaluate.py --checkpoint models/checkpoints/best_model.pth
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json

sys.path.append(str(Path(__file__).parent / 'src'))

from dataset import create_dataloaders
from model import create_model
from metrics import SegmentationMetrics, format_metrics
from inference import load_model_from_checkpoint, predict_batch
from area_calculator import AreaCalculator, calculate_batch_areas
from visualize import (
    visualize_prediction,
    visualize_comparison,
    plot_metrics_comparison,
    create_result_grid
)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: Path,
    num_visualize: int = 5,
    use_amp: bool = True
):
    """
    Оценка модели на тестовом наборе данных
    
    Args:
        model: Обученная модель
        dataloader: DataLoader для test set
        device: Устройство
        save_dir: Директория для сохранения результатов
        num_visualize: Количество изображений для визуализации
        use_amp: Использовать mixed precision
    """
    model.eval()
    
    # Создаём директории
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Метрики
    metrics_tracker = SegmentationMetrics(threshold=0.5)
    area_calculator = AreaCalculator(pixel_size_m=0.3)
    
    # Для визуализации
    viz_images = []
    viz_preds = []
    viz_gts = []
    viz_titles = []
    
    # Список всех результатов
    all_results = []
    
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*80 + "\n")
    
    # Inference на всём test set
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Предсказание
            pred_masks, pred_probs = predict_batch(
                model=model,
                images=images,
                device=device,
                use_amp=use_amp,
                threshold=0.5
            )
            
            # Обновляем метрики
            metrics_tracker.update(pred_probs.unsqueeze(1), masks)
            
            # Расчёт площадей
            pred_masks_np = pred_masks.cpu().numpy()
            gt_masks_np = masks.cpu().numpy()
            
            batch_areas = calculate_batch_areas(
                pred_masks=pred_masks_np,
                gt_masks=gt_masks_np,
                pixel_size_m=0.3
            )
            
            all_results.extend(batch_areas)
            
            # Сохраняем для визуализации
            if batch_idx < num_visualize:
                # Берём первое изображение из батча
                img = images[0].cpu().numpy().transpose(1, 2, 0)
                # Денормализация
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img * std + mean) * 255
                img = np.clip(img, 0, 255).astype(np.uint8)
                
                viz_images.append(img)
                viz_preds.append(pred_masks_np[0])
                viz_gts.append(gt_masks_np[0])
                viz_titles.append(f"Test Image {batch_idx + 1}")
    
    # Финальные метрики
    final_metrics = metrics_tracker.get_all_metrics()
    
    # Вывод метрик
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*80)
    print(format_metrics(final_metrics, prefix='Test'))
    print("="*80 + "\n")
    
    # Средняя площадь застройки
    avg_pred_area = np.mean([r['predicted']['area_ha'] for r in all_results])
    avg_gt_area = np.mean([r['ground_truth']['area_ha'] for r in all_results])
    
    print(f"ПЛОЩАДЬ ЗАСТРОЙКИ (среднее по тестовой выборке):")
    print(f"  Predicted: {avg_pred_area:.2f} га")
    print(f"  Ground Truth: {avg_gt_area:.2f} га")
    print(f"  Разница: {avg_pred_area - avg_gt_area:+.2f} га ({((avg_pred_area - avg_gt_area) / avg_gt_area * 100):+.2f}%)")
    print()
    
    # Сохраняем метрики в JSON
    results_json = {
        'metrics': final_metrics,
        'area_stats': {
            'avg_predicted_ha': float(avg_pred_area),
            'avg_ground_truth_ha': float(avg_gt_area),
            'avg_difference_ha': float(avg_pred_area - avg_gt_area),
            'avg_difference_percent': float((avg_pred_area - avg_gt_area) / avg_gt_area * 100) if avg_gt_area > 0 else 0
        },
        'num_test_images': len(all_results)
    }
    
    results_path = save_dir / 'test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"✓ Результаты сохранены: {results_path}")
    
    # Визуализация метрик
    plot_metrics_comparison(
        metrics=final_metrics,
        title="Test Set Metrics",
        save_path=vis_dir / 'metrics_chart.png'
    )
    
    # Визуализация примеров
    create_result_grid(
        images=viz_images,
        predictions=viz_preds,
        gt_masks=viz_gts,
        titles=viz_titles,
        save_path=vis_dir / 'test_results_grid.png'
    )
    
    # Детальные сравнения для первых N изображений
    for i in range(min(3, len(viz_images))):
        visualize_comparison(
            image=viz_images[i],
            pred_mask=viz_preds[i],
            gt_mask=viz_gts[i],
            title=viz_titles[i],
            save_path=vis_dir / f'comparison_{i+1}.png'
        )
    
    print(f"\n✓ Визуализации сохранены: {vis_dir}")
    print(f"✓ Всего протестировано изображений: {len(all_results)}")
    
    return final_metrics, all_results


def main(args):
    # Загружаем конфигурацию
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*80)
    print("ИНИЦИАЛИЗАЦИЯ ТЕСТИРОВАНИЯ")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*80 + "\n")
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # DataLoader (только test)
    print("Создание test DataLoader...")
    _, _, test_loader = create_dataloaders(
        data_root=config['data']['root'],
        batch_size=args.batch_size,
        num_workers=0,  # Для stability
        patch_size=config['dataset']['patch_size'],
        stride=config['dataset']['test_stride'],
    )
    print()
    
    # Модель
    print("Создание модели...")
    model = create_model(config['model'], device)
    print()
    
    # Загрузка checkpoint
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device
    )
    print()
    
    # Директория для результатов
    save_dir = Path(args.output_dir)
    
    # Оценка
    metrics, results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=save_dir,
        num_visualize=args.num_visualize,
        use_amp=True
    )
    
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*80)
    print(f"Результаты сохранены в: {save_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Оценка модели на test set')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/checkpoints/best_model.pth',
        help='Путь к checkpoint модели'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Путь к конфигурационному файлу'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Директория для сохранения результатов'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size для тестирования'
    )
    
    parser.add_argument(
        '--num-visualize',
        type=int,
        default=5,
        help='Количество изображений для визуализации'
    )
    
    args = parser.parse_args()
    
    main(args)
