"""
Скрипт для оценки модели на данных с Ground Truth
Использует TRAIN set INRIA, т.к. TEST set не имеет публичных GT масок

Запуск: python evaluate_with_gt.py --checkpoint models/checkpoints/best_model.pth --num-test 36
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json
from PIL import Image

sys.path.append(str(Path(__file__).parent / 'src'))

from model import create_model
from metrics import SegmentationMetrics, format_metrics
from inference import load_model_from_checkpoint, predict_large_image
from area_calculator import AreaCalculator
from visualize import visualize_comparison, plot_metrics_comparison, create_result_grid


def evaluate_on_images_with_gt(
    model: torch.nn.Module,
    image_paths: list,
    gt_paths: list,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 512,
    save_dir: Path = None,
    num_visualize: int = 5
):
    """Оценка модели на изображениях с Ground Truth"""
    model.eval()
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = save_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    metrics_tracker = SegmentationMetrics(threshold=0.5)
    area_calculator = AreaCalculator(pixel_size_m=0.3)
    
    viz_images = []
    viz_preds = []
    viz_gts = []
    viz_titles = []
    all_results = []
    
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ (с Ground Truth)")
    print("="*80 + "\n")
    print(f"Количество изображений: {len(image_paths)}\n")
    
    for idx, (img_path, gt_path) in enumerate(tqdm(zip(image_paths, gt_paths), total=len(image_paths), desc="Evaluation")):
        image = np.array(Image.open(img_path).convert('RGB'))
        gt_mask = (np.array(Image.open(gt_path)) > 0).astype(np.uint8)
        
        pred_mask, pred_probs = predict_large_image(
            model=model,
            image=image,
            device=device,
            patch_size=patch_size,
            stride=stride,
            use_amp=True,
            threshold=0.5
        )
        
        pred_tensor = torch.from_numpy(pred_probs).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0)
        
        metrics_tracker.update(pred_tensor, gt_tensor)
        
        areas = area_calculator.calculate_per_class_area(pred_mask=pred_mask, gt_mask=gt_mask)
        all_results.append(areas)
        
        if idx < num_visualize:
            viz_images.append(image)
            viz_preds.append(pred_mask)
            viz_gts.append(gt_mask)
            viz_titles.append(Path(img_path).stem)
    
    final_metrics = metrics_tracker.get_all_metrics()
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*80)
    print(format_metrics(final_metrics, prefix='Test'))
    print("="*80 + "\n")
    
    avg_pred_area = np.mean([r['predicted']['area_ha'] for r in all_results])
    avg_gt_area = np.mean([r['ground_truth']['area_ha'] for r in all_results])
    
    print(f"ПЛОЩАДЬ ЗАСТРОЙКИ (среднее):")
    print(f"  Predicted: {avg_pred_area:.2f} га")
    print(f"  Ground Truth: {avg_gt_area:.2f} га")
    if avg_gt_area > 0:
        diff_percent = ((avg_pred_area - avg_gt_area) / avg_gt_area) * 100
        print(f"  Разница: {avg_pred_area - avg_gt_area:+.2f} га ({diff_percent:+.2f}%)")
    print()
    
    if save_dir:
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
        
        plot_metrics_comparison(metrics=final_metrics, title="Test Set Metrics", save_path=vis_dir / 'metrics_chart.png')
        create_result_grid(images=viz_images, predictions=viz_preds, gt_masks=viz_gts, titles=viz_titles, save_path=vis_dir / 'test_results_grid.png')
        
        for i in range(min(3, len(viz_images))):
            visualize_comparison(image=viz_images[i], pred_mask=viz_preds[i], gt_mask=viz_gts[i], title=viz_titles[i], save_path=vis_dir / f'comparison_{i+1}.png')
        
        print(f"✓ Визуализации сохранены: {vis_dir}")
    
    print(f"✓ Всего протестировано изображений: {len(all_results)}")
    
    return final_metrics, all_results


def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*80)
    print("ИНИЦИАЛИЗАЦИЯ ТЕСТИРОВАНИЯ (с Ground Truth)")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    data_root = Path(config['data']['root'])
    train_images_dir = data_root / 'train' / 'images'
    train_gt_dir = data_root / 'train' / 'gt'
    
    all_images = sorted(list(train_images_dir.glob('*.tif')))
    
    # Берём последние N изображений для тестирования
    test_image_paths = all_images[-args.num_test:]
    test_gt_paths = [train_gt_dir / img.name for img in test_image_paths]
    
    print(f"Тестовая выборка:")
    print(f"  Всего train изображений: {len(all_images)}")
    print(f"  Используем для теста: {len(test_image_paths)} (последние {args.num_test})")
    print()
    
    print("Создание модели...")
    model = create_model(config['model'], device)
    print()
    
    model = load_model_from_checkpoint(checkpoint_path=args.checkpoint, model=model, device=device)
    print()
    
    save_dir = Path(args.output_dir)
    
    metrics, results = evaluate_on_images_with_gt(
        model=model,
        image_paths=test_image_paths,
        gt_paths=test_gt_paths,
        device=device,
        patch_size=config['dataset']['patch_size'],
        stride=config['dataset']['test_stride'],
        save_dir=save_dir,
        num_visualize=args.num_visualize
    )
    
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*80)
    print(f"Результаты сохранены в: {save_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Оценка модели на данных с GT')
    
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth', help='Путь к checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Путь к config')
    parser.add_argument('--output-dir', type=str, default='results/evaluation_with_gt', help='Директория результатов')
    parser.add_argument('--num-visualize', type=int, default=5, help='Количество для визуализации')
    parser.add_argument('--num-test', type=int, default=36, help='Количество тестовых изображений')
    
    args = parser.parse_args()
    main(args)
