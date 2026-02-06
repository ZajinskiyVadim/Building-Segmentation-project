"""
Архитектуры моделей для сегментации зданий
Поддерживаемые модели:
- U-Net (с различными encoders)
- U-Net++ 
- DeepLabV3+
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class SegmentationModel(nn.Module):
    """
    Обёртка для моделей сегментации из segmentation_models_pytorch
    Упрощает создание и использование различных архитектур
    """
    
    def __init__(
        self,
        architecture: str = 'unet',
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        """
        Args:
            architecture: Архитектура модели ('unet', 'unetplusplus', 'deeplabv3plus')
            encoder_name: Название энкодера ('resnet50', 'efficientnet-b4', etc.)
            encoder_weights: Веса энкодера ('imagenet', None)
            in_channels: Количество входных каналов (3 для RGB)
            classes: Количество классов (1 для бинарной сегментации)
            activation: Функция активации на выходе (None для логитов)
        """
        super().__init__()
        
        self.architecture = architecture
        self.encoder_name = encoder_name
        
        # Создаём модель в зависимости от архитектуры
        if architecture == 'unet':
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'unetplusplus':
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        print(f"Создана модель {architecture} с энкодером {encoder_name}")
        if encoder_weights:
            print(f"Загружены pretrained веса: {encoder_weights}")
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def get_num_parameters(self):
        """Возвращает количество параметров модели"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def freeze_encoder(self):
        """Замораживает веса энкодера"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print("Энкодер заморожен")
    
    def unfreeze_encoder(self):
        """Размораживает веса энкодера"""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        print("Энкодер разморожен")


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Создаёт модель на основе конфигурации
    
    Args:
        config: Словарь с параметрами модели
        device: Устройство (cuda/cpu)
    
    Returns:
        model: Модель на нужном устройстве
    """
    model = SegmentationModel(
        architecture=config.get('name', 'unet'),
        encoder_name=config.get('encoder_name', 'resnet50'),
        encoder_weights=config.get('encoder_weights', 'imagenet'),
        in_channels=config.get('in_channels', 3),
        classes=config.get('classes', 1),
        activation=config.get('activation', None),
    )
    
    model = model.to(device)
    
    # Выводим информацию о модели
    params = model.get_num_parameters()
    print(f"\nПараметры модели:")
    print(f"  Всего: {params['total']:,}")
    print(f"  Обучаемых: {params['trainable']:,}")
    print(f"  Замороженных: {params['non_trainable']:,}")
    print(f"  Устройство: {device}")
    
    return model


# =============================================================================
# Кастомная U-Net (если не хотим использовать segmentation_models_pytorch)
# =============================================================================

class DoubleConv(nn.Module):
    """Двойная свёртка: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling: Upsample -> Conv -> Concat -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        
        # Concat along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Выходная свёртка 1x1"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Кастомная реализация U-Net
    Используйте если не хотите зависимости от segmentation_models_pytorch
    """
    
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512], bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        
        # Decoder
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # Output
        self.outc = OutConv(features[0], num_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # Тест модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Тест SegmentationModel (с smp)
    print("="*80)
    print("Тест SegmentationModel (U-Net + ResNet50)")
    print("="*80)
    
    model_config = {
        'name': 'unet',
        'encoder_name': 'resnet50',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1,
    }
    
    model = create_model(model_config, device)
    
    # Тестовый вход
    x = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        y = model(x)
    
    print(f"\nВход: {x.shape}")
    print(f"Выход: {y.shape}")
    print(f"Min/Max выхода: {y.min().item():.4f} / {y.max().item():.4f}")
    
    # Тест кастомной U-Net
    print("\n" + "="*80)
    print("Тест Custom UNet")
    print("="*80)
    
    custom_unet = UNet(in_channels=3, num_classes=1).to(device)
    
    with torch.no_grad():
        y_custom = custom_unet(x)
    
    print(f"\nВход: {x.shape}")
    print(f"Выход: {y_custom.shape}")
    print(f"Min/Max выхода: {y_custom.min().item():.4f} / {y_custom.max().item():.4f}")
    
    # Количество параметров
    total_params = sum(p.numel() for p in custom_unet.parameters())
    print(f"\nПараметров в Custom U-Net: {total_params:,}")
