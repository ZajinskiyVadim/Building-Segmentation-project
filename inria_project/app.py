"""
🏙️ Building Segmentation from Satellite Images
Streamlit приложение для сегментации зданий на спутниковых снимках

Автор: Vadim
Дата: Январь 2026
Модель: U-Net + ResNet50 (IoU: 0.8022)
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import yaml
from pathlib import Path
import sys

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent / 'src'))

from model import create_model
from inference import load_model_from_checkpoint, predict_large_image
from area_calculator_advanced import AreaCalculatorAdvanced
from app_utils import (
    load_example_images,
    create_overlay_image,
    create_side_by_side_comparison,
    plot_area_statistics,
    apply_colormap_to_mask
)


# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ПРИЛОЖЕНИЯ
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Building Segmentation",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_cached(config_path: str, checkpoint_path: str):
    """Загрузка модели с кешированием"""
    
    # Загрузка конфига
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание модели
    model = create_model(config['model'], device)
    
    # Загрузка весов
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device
    )
    
    return model, device, config


# ═══════════════════════════════════════════════════════════════
# SIDEBAR - НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    """Рендер боковой панели с настройками"""
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/city.png", width=80)
        st.title("⚙️ Настройки")
        
        st.markdown("---")
        
        # Модель
        st.subheader("🤖 Модель")
        config_path = st.text_input(
            "Путь к конфигу",
            value="configs/config.yaml",
            help="Путь к файлу конфигурации YAML"
        )
        
        checkpoint_path = st.text_input(
            "Путь к checkpoint",
            value="models/checkpoints/best_model.pth",
            help="Путь к обученной модели"
        )
        
        st.markdown("---")
        
        # Параметры inference
        st.subheader("🔧 Параметры")
        
        threshold = st.slider(
            "Порог (threshold)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Порог для бинаризации вероятностей"
        )
        
        patch_size = st.selectbox(
            "Размер патча",
            options=[256, 512, 1024],
            index=1,
            help="Размер патча для inference"
        )
        
        stride = st.selectbox(
            "Stride (шаг)",
            options=[128, 256, 512, 1024],
            index=2,
            help="Шаг скользящего окна (меньше = больше перекрытие)"
        )
        
        st.markdown("---")
        
        # Визуализация
        st.subheader("🎨 Визуализация")
        
        colormap = st.selectbox(
            "Цветовая схема",
            options=['Red', 'Blue', 'Green', 'Jet', 'Viridis'],
            index=0,
            help="Цветовая схема для маски зданий"
        )
        
        overlay_alpha = st.slider(
            "Прозрачность overlay",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Прозрачность наложения маски на изображение"
        )
        
        st.markdown("---")
        
        # Информация о модели
        st.subheader("📊 О модели")
        st.info("""
        **Архитектура:** U-Net + ResNet50  
        **Датасет:** INRIA Aerial Images  
        **Test IoU:** 0.8022  
        **Точность площади:** 98.63%
        """)
        
        st.markdown("---")
        
        # GitHub
        st.markdown("""
        <div style='text-align: center'>
            <a href='https://github.com/yourusername/inria-building-segmentation' target='_blank'>
                <img src='https://img.icons8.com/fluency/48/000000/github.png' width='32'/>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    return {
        'config_path': config_path,
        'checkpoint_path': checkpoint_path,
        'threshold': threshold,
        'patch_size': patch_size,
        'stride': stride,
        'colormap': colormap,
        'overlay_alpha': overlay_alpha
    }


# ═══════════════════════════════════════════════════════════════
# ГЛАВНАЯ СТРАНИЦА
# ═══════════════════════════════════════════════════════════════

def main():
    """Главная функция приложения"""
    
    # Header
    st.markdown("<h1 class='main-header'>🏙️ Building Segmentation from Satellite Images</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Автоматическая сегментация зданий и расчёт площади застройки</p>", unsafe_allow_html=True)
    
    # Sidebar
    settings = render_sidebar()
    
    # Проверка наличия файлов
    if not Path(settings['config_path']).exists():
        st.error(f"❌ Конфиг не найден: {settings['config_path']}")
        st.stop()
    
    if not Path(settings['checkpoint_path']).exists():
        st.error(f"❌ Checkpoint не найден: {settings['checkpoint_path']}")
        st.stop()
    
    # Загрузка модели
    with st.spinner("🔄 Загрузка модели..."):
        try:
            model, device, config = load_model_cached(
                settings['config_path'],
                settings['checkpoint_path']
            )
            st.success(f"✅ Модель загружена! Устройство: {device}")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Загрузить изображение", "🖼️ Примеры", "ℹ️ О проекте"])
    
    # ───────────────────────────────────────────────────────────
    # TAB 1: ЗАГРУЗКА ИЗОБРАЖЕНИЯ
    # ───────────────────────────────────────────────────────────
    
    with tab1:
        st.header("📤 Загрузите спутниковый снимок")
        
        uploaded_file = st.file_uploader(
            "Выберите изображение (PNG, JPG, JPEG, TIF)",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="Поддерживаются форматы: PNG, JPG, JPEG, TIF, TIFF"
        )
        
        if uploaded_file is not None:
            process_image(uploaded_file, model, device, settings)
    
    # ───────────────────────────────────────────────────────────
    # TAB 2: ПРИМЕРЫ
    # ───────────────────────────────────────────────────────────
    
    with tab2:
        st.header("🖼️ Примеры изображений")
        
        # Загрузка примеров
        examples = load_example_images()
        
        if not examples:
            st.warning("⚠️ Примеры изображений не найдены в папке `examples/`")
        else:
            # Выбор примера
            example_names = [ex['name'] for ex in examples]
            selected_example = st.selectbox(
                "Выберите пример",
                options=example_names,
                help="Выберите пример изображения для демонстрации"
            )
            
            # Найти выбранный пример
            example = next(ex for ex in examples if ex['name'] == selected_example)
            
            # Кнопка запуска
            if st.button("🚀 Запустить сегментацию", type="primary", use_container_width=True):
                process_image(example['path'], model, device, settings)
    
    # ───────────────────────────────────────────────────────────
    # TAB 3: О ПРОЕКТЕ
    # ───────────────────────────────────────────────────────────
    
    with tab3:
        st.header("ℹ️ О проекте")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Цель проекта")
            st.write("""
            Автоматическая сегментация зданий на спутниковых снимках 
            с использованием глубокого обучения. Приложение позволяет:
            
            - 🏗️ Определить границы зданий
            - 📏 Рассчитать площадь застройки
            - 📊 Визуализировать результаты
            - 💾 Скачать результаты
            """)
            
            st.subheader("🧠 Архитектура модели")
            st.write("""
            **U-Net** с **ResNet50** encoder:
            - Pretrained на ImageNet
            - 32.5M параметров
            - Mixed Precision Training (FP16)
            - Комбинированная loss (Dice + BCE)
            """)
        
        with col2:
            st.subheader("📊 Результаты")
            
            # Метрики
            metrics_data = {
                "IoU": 0.8022,
                "F1-Score": 0.8903,
                "Accuracy": 0.9363,
                "Precision": 0.8843,
                "Recall": 0.8964
            }
            
            for metric, value in metrics_data.items():
                st.metric(
                    label=metric,
                    value=f"{value:.4f}",
                    delta=f"+{((value - 0.60) / 0.60 * 100):.1f}% vs baseline" if metric == "IoU" else None
                )
            
            st.subheader("🗂️ Датасет")
            st.write("""
            **INRIA Aerial Image Labeling Dataset**
            - 360 изображений (180 train, 180 test)
            - 10 городов (США, Австрия)
            - Разрешение: 0.3 м/пиксель
            - Площадь покрытия: ~810 км²
            """)
        
        st.markdown("---")
        
        st.subheader("📚 Технологии")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Фреймворки:**")
            st.write("- PyTorch 2.11.0")
            st.write("- Streamlit")
            st.write("- Albumentations")
        
        with col2:
            st.write("**Модели:**")
            st.write("- U-Net")
            st.write("- ResNet50")
            st.write("- Segmentation Models")
        
        with col3:
            st.write("**Инструменты:**")
            st.write("- TensorBoard")
            st.write("- OpenCV")
            st.write("- NumPy, Matplotlib")


# ═══════════════════════════════════════════════════════════════
# ОБРАБОТКА ИЗОБРАЖЕНИЯ
# ═══════════════════════════════════════════════════════════════

def process_image(image_source, model, device, settings):
    """
    Обработка изображения: inference + визуализация
    
    Args:
        image_source: Путь к файлу или uploaded file
        model: Модель
        device: Устройство (cuda/cpu)
        settings: Настройки из sidebar
    """
    
    # Загрузка изображения
    try:
        if isinstance(image_source, (str, Path)):
            image = np.array(Image.open(image_source).convert('RGB'))
        else:
            image = np.array(Image.open(image_source).convert('RGB'))
    except Exception as e:
        st.error(f"❌ Ошибка загрузки изображения: {e}")
        return
    
    st.markdown("---")
    
    # Информация об изображении
    col1, col2, col3 = st.columns(3)
    col1.metric("🖼️ Ширина", f"{image.shape[1]} px")
    col2.metric("🖼️ Высота", f"{image.shape[0]} px")
    col3.metric("💾 Размер", f"{image.nbytes / 1024 / 1024:.2f} MB")
    
    # Предпросмотр
    st.subheader("📷 Оригинальное изображение")
    st.image(image, use_container_width=True, caption="Спутниковый снимок")
    
    # Кнопка запуска
    if st.button("🚀 Запустить сегментацию", type="primary", use_container_width=True, key="process_btn"):
        
        # Inference
        with st.spinner("🤖 Выполняется сегментация..."):
            try:
                pred_mask, pred_probs = predict_large_image(
                    model=model,
                    image=image,
                    device=device,
                    patch_size=settings['patch_size'],
                    stride=settings['stride'],
                    use_amp=True,
                    threshold=settings['threshold']
                )
                
                st.success("✅ Сегментация завершена!")
                
            except Exception as e:
                st.error(f"❌ Ошибка inference: {e}")
                return
        
        # Расчёт площади с автоматическим масштабом
        try:
            # Попытка извлечь масштаб из GeoTIFF (если это файл)
            if isinstance(image_source, (str, Path)):
                image_path = Path(image_source) if isinstance(image_source, str) else image_source
                if image_path.suffix.lower() in ['.tif', '.tiff']:
                    try:
                        calculator = AreaCalculatorAdvanced.from_geotiff(image_path)
                        st.info(f"📐 Масштаб извлечён из GeoTIFF: {calculator.pixel_size_m} м/пиксель")
                    except Exception as e:
                        st.warning(f"⚠️ Не удалось извлечь масштаб из GeoTIFF. Используется значение по умолчанию: 0.3 м/пиксель")
                        calculator = AreaCalculatorAdvanced(pixel_size_m=0.3)
                else:
                    calculator = AreaCalculatorAdvanced(pixel_size_m=0.3)
            else:
                calculator = AreaCalculatorAdvanced(pixel_size_m=0.3)
        except Exception as e:
            st.warning(f"⚠️ Ошибка определения масштаба: {e}. Используется 0.3 м/пиксель")
            calculator = AreaCalculatorAdvanced(pixel_size_m=0.3)
        
        areas = calculator.calculate_area(pred_mask)
        
        # ═══════════════════════════════════════════════════════
        # РЕЗУЛЬТАТЫ
        # ═══════════════════════════════════════════════════════
        
        st.markdown("---")
        st.header("📊 Результаты сегментации")
        
        # Метрики площади
        col1, col2, col3, col4 = st.columns(4)
        
        # Информация о масштабе
        st.markdown(f"""
        <div style='background-color: #e3f2fd; padding: 0.5rem; border-radius: 0.3rem; margin-bottom: 1rem;'>
            📐 <b>Масштаб:</b> {areas['pixel_size_m']} м/пиксель 
            ({areas['pixel_area_m2']:.4f} м² на пиксель)
        </div>
        """, unsafe_allow_html=True)
        
        with col1:
            st.metric(
                "🏗️ Площадь застройки",
                f"{areas['area_ha']:.2f} га",
                help="Площадь в гектарах (1 га = 10,000 м²)"
            )
        
        with col2:
            st.metric(
                "📏 В квадратных метрах",
                f"{areas['area_m2']:,.0f} м²",
                help="Площадь в квадратных метрах"
            )
        
        with col3:
            st.metric(
                "📍 Пикселей зданий",
                f"{areas['building_pixels']:,}",
                help="Количество пикселей, классифицированных как здания"
            )
        
        with col4:
            st.metric(
                "📈 Покрытие",
                f"{areas['coverage_percent']:.2f}%",
                help="Процент площади занятой зданиями"
            )
        
        # ═══════════════════════════════════════════════════════
        # ВИЗУАЛИЗАЦИЯ
        # ═══════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("🎨 Визуализация результатов")
        
        # Создание визуализаций
        mask_colored = apply_colormap_to_mask(pred_mask, colormap=settings['colormap'])
        overlay_image = create_overlay_image(image, mask_colored, alpha=settings['overlay_alpha'])
        comparison_image = create_side_by_side_comparison(image, pred_mask, pred_probs)
        
        # Tabs для разных визуализаций
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "🔴 Маска зданий",
            "🌈 Overlay",
            "📊 Probability Map",
            "📐 Сравнение"
        ])
        
        with viz_tab1:
            st.image(mask_colored, use_container_width=True, caption="Маска сегментированных зданий")
        
        with viz_tab2:
            st.image(overlay_image, use_container_width=True, caption="Overlay маски на оригинальное изображение")
        
        with viz_tab3:
            st.image(pred_probs, use_container_width=True, caption="Карта вероятностей", clamp=True)
            
            # Гистограмма вероятностей
            fig = plot_area_statistics(pred_probs.flatten())
            st.pyplot(fig)
        
        with viz_tab4:
            st.image(comparison_image, use_container_width=True, caption="Сравнение: Оригинал | Маска | Overlay")
        
        # ═══════════════════════════════════════════════════════
        # СКАЧИВАНИЕ
        # ═══════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("💾 Скачать результаты")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Маска
            mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
            mask_bytes = io.BytesIO()
            mask_pil.save(mask_bytes, format='PNG')
            
            st.download_button(
                label="📥 Скачать маску",
                data=mask_bytes.getvalue(),
                file_name="building_mask.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col2:
            # Overlay
            overlay_pil = Image.fromarray(overlay_image)
            overlay_bytes = io.BytesIO()
            overlay_pil.save(overlay_bytes, format='PNG')
            
            st.download_button(
                label="📥 Скачать overlay",
                data=overlay_bytes.getvalue(),
                file_name="building_overlay.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col3:
            # Результаты (JSON)
            import json
            results_json = json.dumps({
                'area_m2': float(areas['area_m2']),
                'area_ha': float(areas['area_ha']),
                'building_pixels': int(areas['building_pixels']),
                'coverage_percent': float(areas['coverage_percent']),
                'image_size': image.shape[:2],
                'threshold': settings['threshold']
            }, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Скачать данные (JSON)",
                data=results_json,
                file_name="building_data.json",
                mime="application/json",
                use_container_width=True
            )


# ═══════════════════════════════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import io
    main()
