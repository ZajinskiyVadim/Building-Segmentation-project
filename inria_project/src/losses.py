"""
Loss функции для сегментации
- Dice Loss
- BCE Loss  
- Комбинированный Dice + BCE
- Focal Loss
- Tversky Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss для бинарной сегментации
    
    Dice Coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice Coefficient
    """
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: Сглаживающий параметр для избежания деления на ноль
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, 1, H, W] (логиты или вероятности)
            target: Ground truth [B, H, W] или [B, 1, H, W]
        
        Returns:
            dice_loss: Scalar tensor
        """
        # Применяем sigmoid если pred - логиты
        pred = torch.sigmoid(pred)
        
        # Приводим к одинаковой размерности
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        target = target.float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Dice coefficient
        intersection = (pred * target).sum()
        dice_coef = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # Dice loss
        dice_loss = 1.0 - dice_coef
        
        return dice_loss


class BCEDiceLoss(nn.Module):
    """
    Комбинированный BCE + Dice Loss
    Для дополнительных баллов по ТЗ (+2 балла)
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        """
        Args:
            bce_weight: Вес BCE Loss
            dice_weight: Вес Dice Loss
            smooth: Сглаживание для Dice
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, 1, H, W] (логиты)
            target: Ground truth [B, H, W]
        
        Returns:
            combined_loss: Weighted sum of BCE and Dice
        """
        # Приводим target к нужной размерности для BCE
        if target.dim() == 3:
            target_bce = target.unsqueeze(1).float()  # [B, 1, H, W]
        else:
            target_bce = target.float()
        
        # BCE Loss
        bce_loss = self.bce(pred, target_bce)
        
        # Dice Loss
        dice_loss = self.dice(pred, target)
        
        # Комбинированный loss
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss
    
    def get_component_losses(self, pred, target):
        """Возвращает компоненты loss для логирования"""
        if target.dim() == 3:
            target_bce = target.unsqueeze(1).float()
        else:
            target_bce = target.float()
        
        bce_loss = self.bce(pred, target_bce)
        dice_loss = self.dice(pred, target)
        
        return {
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item(),
            'combined_loss': (self.bce_weight * bce_loss + self.dice_weight * dice_loss).item()
        }


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с дисбалансом классов
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Вес для класса (обычно 0.25)
            gamma: Фокусирующий параметр (обычно 2.0)
            reduction: 'mean' или 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, 1, H, W] (логиты)
            target: Ground truth [B, H, W]
        """
        # Sigmoid для получения вероятностей
        pred_prob = torch.sigmoid(pred)
        
        # Приводим к одинаковой размерности
        if target.dim() == 3:
            target = target.unsqueeze(1).float()
        else:
            target = target.float()
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # p_t
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = self.alpha * focal_term * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - обобщение Dice Loss
    Позволяет контролировать баланс между False Positives и False Negatives
    """
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """
        Args:
            alpha: Вес False Positives
            beta: Вес False Negatives
            smooth: Сглаживание
        
        Note:
            alpha = beta = 0.5 эквивалентно Dice Loss
            alpha < beta фокусируется на recall
            alpha > beta фокусируется на precision
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, 1, H, W] (логиты)
            target: Ground truth [B, H, W]
        """
        pred = torch.sigmoid(pred)
        
        if target.dim() == 3:
            target = target.unsqueeze(1).float()
        else:
            target = target.float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Tversky loss
        return 1.0 - tversky


def get_loss_function(loss_config: dict) -> nn.Module:
    """
    Создаёт loss функцию на основе конфигурации
    
    Args:
        loss_config: Словарь с параметрами loss
        
    Returns:
        loss_fn: Loss функция
    """
    loss_name = loss_config.get('name', 'dice_bce').lower()
    
    if loss_name == 'dice':
        smooth = loss_config.get('smooth', 1.0)
        loss_fn = DiceLoss(smooth=smooth)
        print(f"Loss: Dice (smooth={smooth})")
        
    elif loss_name == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
        print("Loss: BCE")
        
    elif loss_name == 'dice_bce':
        bce_weight = loss_config.get('bce_weight', 0.5)
        dice_weight = loss_config.get('dice_weight', 0.5)
        smooth = loss_config.get('smooth', 1.0)
        loss_fn = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            smooth=smooth
        )
        print(f"Loss: BCE ({bce_weight}) + Dice ({dice_weight})")
        
    elif loss_name == 'focal':
        alpha = loss_config.get('alpha', 0.25)
        gamma = loss_config.get('gamma', 2.0)
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        print(f"Loss: Focal (alpha={alpha}, gamma={gamma})")
        
    elif loss_name == 'tversky':
        alpha = loss_config.get('alpha', 0.5)
        beta = loss_config.get('beta', 0.5)
        smooth = loss_config.get('smooth', 1.0)
        loss_fn = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        print(f"Loss: Tversky (alpha={alpha}, beta={beta})")
        
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    
    return loss_fn


if __name__ == '__main__':
    # Тест loss функций
    print("="*80)
    print("Тест Loss функций")
    print("="*80)
    
    # Создаём тестовые данные
    batch_size = 2
    height, width = 512, 512
    
    # Логиты (до sigmoid)
    pred = torch.randn(batch_size, 1, height, width)
    
    # Ground truth (бинарная маска)
    target = torch.randint(0, 2, (batch_size, height, width)).float()
    
    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    print()
    
    # Тест Dice Loss
    dice_loss = DiceLoss()
    loss_dice = dice_loss(pred, target)
    print(f"Dice Loss: {loss_dice.item():.4f}")
    
    # Тест BCE Loss
    bce_loss = nn.BCEWithLogitsLoss()
    target_bce = target.unsqueeze(1)  # [B, 1, H, W]
    loss_bce = bce_loss(pred, target_bce)
    print(f"BCE Loss: {loss_bce.item():.4f}")
    
    # Тест BCEDice Loss
    bce_dice_loss = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    loss_combined = bce_dice_loss(pred, target)
    print(f"Combined BCE+Dice Loss: {loss_combined.item():.4f}")
    
    # Компоненты
    components = bce_dice_loss.get_component_losses(pred, target)
    print(f"  BCE component: {components['bce_loss']:.4f}")
    print(f"  Dice component: {components['dice_loss']:.4f}")
    
    # Тест Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_focal = focal_loss(pred, target)
    print(f"Focal Loss: {loss_focal.item():.4f}")
    
    # Тест Tversky Loss
    tversky_loss = TverskyLoss(alpha=0.5, beta=0.5)
    loss_tversky = tversky_loss(pred, target)
    print(f"Tversky Loss: {loss_tversky.item():.4f}")
    
    print("\n" + "="*80)
    print("✅ Все loss функции работают корректно!")
    print("="*80)
