#!/usr/bin/env python3
"""
Скрипт для загрузки датасета INRIA Aerial Image Labeling Dataset
Источник: https://project.inria.fr/aerialimagelabeling/
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    """
    Загружает файл с отображением прогресса
    
    Args:
        url: URL файла для загрузки
        destination: Путь для сохранения файла
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def extract_archive(archive_path, extract_to):
    """
    Извлекает ZIP архив
    
    Args:
        archive_path: Путь к архиву
        extract_to: Директория для извлечения
    """
    print(f"Извлечение {archive_path.name}...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Архив извлечен в {extract_to}")


def download_inria_dataset(data_dir='./data'):
    """
    Загружает и извлекает датасет INRIA
    
    Args:
        data_dir: Директория для сохранения данных
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # URLs для загрузки датасета INRIA
    # Примечание: актуальные ссылки могут измениться, проверьте на официальном сайте
    urls = {
        'train_images': 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001',
        'train_images_2': 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002',
        'train_images_3': 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003',
        'train_images_4': 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004',
        'train_images_5': 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005',
    }
    
    print("="*80)
    print("ЗАГРУЗКА ДАТАСЕТА INRIA AERIAL IMAGE LABELING")
    print("="*80)
    print("\nПРИМЕЧАНИЕ:")
    print("Датасет INRIA доступен на официальном сайте:")
    print("https://project.inria.fr/aerialimagelabeling/")
    print("\nДля автоматической загрузки вам нужно:")
    print("1. Скачать файлы вручную с официального сайта")
    print("2. Или использовать команду wget/curl с актуальными ссылками")
    print("\nФайлы датасета:")
    print("- NEW2-AerialImageDataset.zip (train + test images)")
    print("Размер: ~11 GB")
    print("="*80)
    
    # Структура после распаковки
    print("\nОжидаемая структура директорий после распаковки:")
    print("""
    data/
    └── AerialImageDataset/
        ├── train/
        │   ├── images/
        │   │   ├── austin1.tif
        │   │   ├── austin2.tif
        │   │   └── ...
        │   └── gt/
        │       ├── austin1.tif
        │       ├── austin2.tif
        │       └── ...
        └── test/
            └── images/
                ├── bellingham1.tif
                └── ...
    """)
    
    print("\nИНСТРУКЦИЯ ПО РУЧНОЙ ЗАГРУЗКЕ:")
    print("1. Перейдите на https://project.inria.fr/aerialimagelabeling/files/")
    print("2. Скачайте 'NEW2-AerialImageDataset.zip'")
    print("3. Распакуйте в директорию './data/'")
    print("4. Убедитесь, что структура соответствует указанной выше")
    
    # Альтернативный способ - через wget (если доступен)
    print("\n" + "="*80)
    print("АЛЬТЕРНАТИВА: Загрузка через wget")
    print("="*80)
    print("\nВыполните в терминале:")
    print("cd data/")
    print("wget https://files.inria.fr/aerialimagelabeling/NEW2-AerialImageDataset.zip")
    print("unzip NEW2-AerialImageDataset.zip")
    print("="*80)
    

def verify_dataset(data_dir='./data/AerialImageDataset'):
    """
    Проверяет наличие и корректность датасета
    
    Args:
        data_dir: Путь к директории датасета
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Датасет не найден в {data_path}")
        print("Пожалуйста, скачайте датасет вручную")
        return False
    
    # Проверка структуры
    train_images = data_path / 'train' / 'images'
    train_gt = data_path / 'train' / 'gt'
    test_images = data_path / 'test' / 'images'
    
    required_dirs = [train_images, train_gt, test_images]
    
    print("\n" + "="*80)
    print("ПРОВЕРКА ДАТАСЕТА")
    print("="*80)
    
    all_good = True
    for dir_path in required_dirs:
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path.relative_to(data_path.parent)}")
        
        if exists and dir_path.name == 'images':
            num_files = len(list(dir_path.glob('*.tif')))
            print(f"  └─ Найдено изображений: {num_files}")
            
            if 'train' in str(dir_path):
                expected = 180
            else:
                expected = 180
                
            if num_files != expected:
                print(f"  ⚠️  Ожидалось {expected} изображений")
                all_good = False
        
        if exists and dir_path.name == 'gt':
            num_files = len(list(dir_path.glob('*.tif')))
            print(f"  └─ Найдено масок: {num_files}")
            
            if num_files != 180:
                print(f"  ⚠️  Ожидалось 180 масок")
                all_good = False
    
    print("="*80)
    
    if all_good:
        print("✓ Датасет проверен успешно!")
        return True
    else:
        print("⚠️  Обнаружены проблемы с датасетом")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Загрузка датасета INRIA')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Директория для сохранения данных')
    parser.add_argument('--verify', action='store_true',
                        help='Только проверить наличие датасета')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(os.path.join(args.data_dir, 'AerialImageDataset'))
    else:
        download_inria_dataset(args.data_dir)
        print("\nПосле загрузки запустите с флагом --verify для проверки:")
        print("python download_dataset.py --verify")
