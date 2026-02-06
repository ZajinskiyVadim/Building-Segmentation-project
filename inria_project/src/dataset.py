"""
Кастомный Dataset и DataLoader для датасета INRIA Aerial Image Labeling
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class INRIADataset(Dataset):
    """
    Кастомный Dataset для INRIA Aerial Image Labeling Dataset
    
    Структура датасета:
    AerialImageDataset/
    ├── train/
    │   ├── images/
    │   │   ├── austin1.tif
    │   │   └── ...
    │   └── gt/
    │       ├── austin1.tif
    │       └── ...
    └── test/
        └── images/
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        cities: Optional[List[str]] = None,
        patch_size: int = 512,
        stride: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        cache_images: bool = True,
    ):
        """
        Args:
            data_root: Путь к корневой директории датасета (AerialImageDataset/)
            split: 'train', 'val' или 'test'
            cities: Список городов для загрузки. Если None, загружаются все города
            patch_size: Размер патчей для нарезки изображений
            stride: Шаг для нарезки патчей. Если None, равен patch_size (без перекрытия)
            transform: Аугментации (albumentations)
            normalize: Применять нормализацию ImageNet
            cache_images: Кэшировать изображения в RAM для ускорения
        """
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.transform = transform
        self.normalize = normalize
        self.cache_images = cache_images

        # Кэш для изображений и масок
        self._image_cache = {}
        self._mask_cache = {}
        
        # Маппинг городов для train/test
        self.train_cities = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
        self.test_cities = ['bellingham', 'bloomington', 'innsbruck', 'san', 'tyrol-e']
        
        # Определяем директории
        if split in ['train', 'val']:
            self.images_dir = self.data_root / 'train' / 'images'
            self.masks_dir = self.data_root / 'train' / 'gt'
        else:
            self.images_dir = self.data_root / 'test' / 'images'
            self.masks_dir = None  # Тестовые маски не публикуются
        
        # Получаем список файлов
        self.image_files = self._get_image_files(cities)
        
        # Создаем индекс патчей
        self.patches_index = self._create_patches_index()
        
        print(f"Инициализирован {split} датасет:")
        print(f"  - Количество изображений: {len(self.image_files)}")
        print(f"  - Количество патчей: {len(self.patches_index)}")
        print(f"  - Размер патча: {patch_size}x{patch_size}")
        print(f"  - Stride: {self.stride}")
    
    def _get_image_files(self, cities: Optional[List[str]] = None) -> List[Path]:
        """Получает список файлов изображений"""
        all_files = sorted(self.images_dir.glob('*.tif'))
        
        if cities is None:
            return all_files
        
        # Фильтруем по городам
        filtered_files = []
        for file in all_files:
            for city in cities:
                if file.stem.lower().startswith(city.lower()):
                    filtered_files.append(file)
                    break
        
        return filtered_files
    
    def _create_patches_index(self) -> List[Tuple[int, int, int]]:
        """
        Создает индекс всех патчей
        Returns:
            List of (image_idx, row_start, col_start)
        """
        patches = []
        
        for img_idx in range(len(self.image_files)):
            # Для создания индекса открываем только первое изображение
            # чтобы узнать размер (все изображения 5000x5000)
            image_size = 5000  # Известный размер для INRIA
            
            # Вычисляем количество патчей
            n_rows = (image_size - self.patch_size) // self.stride + 1
            n_cols = (image_size - self.patch_size) // self.stride + 1
            
            for i in range(n_rows):
                for j in range(n_cols):
                    row_start = i * self.stride
                    col_start = j * self.stride
                    patches.append((img_idx, row_start, col_start))
        
        return patches
    
    def __len__(self) -> int:
        return len(self.patches_index)
    
    def _load_image(self, img_idx: int) -> np.ndarray:
        """Загружает изображение с кэшированием"""
        if self.cache_images and img_idx in self._image_cache:
            return self._image_cache[img_idx]

        image_path = self.image_files[img_idx]
        image = np.array(Image.open(image_path))  # Shape: (5000, 5000, 3)

        if self.cache_images:
            self._image_cache[img_idx] = image

        return image

    def _load_mask(self, img_idx: int) -> Optional[np.ndarray]:
        """Загружает маску с кэшированием"""
        if self.masks_dir is None:
            return None

        if self.cache_images and img_idx in self._mask_cache:
            return self._mask_cache[img_idx]

        image_path = self.image_files[img_idx]
        mask_path = self.masks_dir / image_path.name
        mask = np.array(Image.open(mask_path))  # Shape: (5000, 5000)

        if self.cache_images:
            self._mask_cache[img_idx] = mask

        return mask

    def __getitem__(self, idx: int) -> dict:
        """
        Возвращает патч изображения и маски

        Returns:
            dict with keys:
                - 'image': Tensor [C, H, W]
                - 'mask': Tensor [H, W] (только для train/val)
                - 'filename': str
                - 'patch_coords': (row_start, col_start)
        """
        img_idx, row_start, col_start = self.patches_index[idx]

        # Загружаем изображение (с кэшированием)
        image_path = self.image_files[img_idx]
        image = self._load_image(img_idx)

        # Вырезаем патч
        row_end = row_start + self.patch_size
        col_end = col_start + self.patch_size
        image_patch = image[row_start:row_end, col_start:col_end]

        # Загружаем маску (если есть)
        mask = self._load_mask(img_idx)
        if mask is not None:
            mask_patch = mask[row_start:row_end, col_start:col_end]
            # Бинаризация маски: 0 - фон, 1 - здание
            mask_patch = (mask_patch > 0).astype(np.uint8)
        else:
            mask_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        
        # Применяем аугментации
        if self.transform is not None:
            transformed = self.transform(image=image_patch, mask=mask_patch)
            image_patch = transformed['image']
            mask_patch = transformed['mask']
        else:
            # Конвертируем в тензор вручную
            image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).float() / 255.0
            mask_patch = torch.from_numpy(mask_patch).long()
            
            # Нормализация ImageNet
            if self.normalize:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_patch = (image_patch - mean) / std
        
        return {
            'image': image_patch,
            'mask': mask_patch,
            'filename': image_path.stem,
            'patch_coords': (row_start, col_start),
        }
    
    def get_full_image(self, image_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает полное изображение и маску (без нарезки на патчи)
        
        Args:
            image_idx: Индекс изображения
            
        Returns:
            image: np.ndarray [H, W, 3]
            mask: np.ndarray [H, W] (or None for test)
        """
        image_path = self.image_files[image_idx]
        image = np.array(Image.open(image_path))
        
        if self.masks_dir is not None:
            mask_path = self.masks_dir / image_path.name
            mask = np.array(Image.open(mask_path))
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = None
        
        return image, mask


def get_train_transforms(patch_size: int = 512) -> A.Compose:
    """
    Аугментации для обучающей выборки
    
    Args:
        patch_size: Размер патча
    """
    return A.Compose([
        # Геометрические преобразования
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),  # scale_limit=0.1 означает 90%-110%
            translate_percent=(-0.1, 0.1),  # shift_limit=0.1
            rotate=(-15, 15),  # rotate_limit=15
            shear=(-5, 5),
            p=0.5
        ),
        
        # Цветовые преобразования
        A.OneOf([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
        ], p=0.5),
        
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),  # Нормализованный диапазон [0, 1]
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        
        # Нормализация и конвертация в тензор
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(patch_size: int = 512) -> A.Compose:
    """
    Трансформации для валидации/теста (только нормализация)
    
    Args:
        patch_size: Размер патча
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def create_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    patch_size: int = 512,
    stride: Optional[int] = None,
    val_split: float = 0.15,
    cache_images: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создает DataLoader'ы для train, validation и test
    
    Args:
        data_root: Путь к AerialImageDataset/
        batch_size: Размер батча
        num_workers: Количество workers для загрузки
        patch_size: Размер патчей
        stride: Шаг для нарезки (если None, равен patch_size)
        val_split: Доля валидационной выборки (по количеству изображений)
        cache_images: Кэшировать изображения в RAM для ускорения

    Returns:
        train_loader, val_loader, test_loader
    """
    # Города для train/val
    train_cities = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    
    # Разделение на train и val (по первым изображениям каждого города)
    # Для простоты: первые 5 изображений каждого города идут в val
    val_indices_per_city = 5  # ~25 изображений в val (14%)
    
    # Создаем датасеты
    train_transform = get_train_transforms(patch_size)
    val_transform = get_val_transforms(patch_size)
    
    # Train dataset (исключая первые 5 из каждого города)
    train_dataset = INRIADataset(
        data_root=data_root,
        split='train',
        cities=train_cities,
        patch_size=patch_size,
        stride=stride,
        transform=train_transform,
        cache_images=cache_images,
    )

    # Val dataset (только первые 5 из каждого города)
    val_dataset = INRIADataset(
        data_root=data_root,
        split='val',
        cities=train_cities,
        patch_size=patch_size,
        stride=patch_size,  # Без перекрытия для валидации
        transform=val_transform,
        cache_images=cache_images,
    )

    # Test dataset
    test_cities = ['bellingham', 'bloomington', 'innsbruck', 'san', 'tyrol-e']
    test_dataset = INRIADataset(
        data_root=data_root,
        split='test',
        cities=test_cities,
        patch_size=patch_size,
        stride=patch_size,  # Без перекрытия для теста
        transform=val_transform,
        cache_images=cache_images,
    )
    
    # Создаем DataLoader'ы
    # persistent_workers=True для ускорения (не пересоздавать workers)
    use_persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    print("\n" + "="*80)
    print("DATALOADER'Ы СОЗДАНЫ")
    print("="*80)
    print(f"Train: {len(train_dataset)} патчей, {len(train_loader)} батчей")
    print(f"Val:   {len(val_dataset)} патчей, {len(val_loader)} батчей")
    print(f"Test:  {len(test_dataset)} патчей, {len(test_loader)} батчей")
    print(f"Batch size: {batch_size}")
    print("="*80)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Пример использования
    data_root = './data/AerialImageDataset'
    
    # Создаем DataLoader'ы
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=4,
        patch_size=512,
        stride=256,  # Перекрытие 50% для train
    )
    
    # Проверяем один батч
    print("\nПроверка батча:")
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    print(f"Mask values: {batch['mask'].unique()}")
    print(f"Filenames: {batch['filename'][:2]}")
