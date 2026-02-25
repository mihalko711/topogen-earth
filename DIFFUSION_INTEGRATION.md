# Интеграция Conditional Diffusion Model с SOM-VAE

## Обзор

Conditional Diffusion Model для улучшения качества реконструкции SOM-VAE через моделирование residual `r = x - x_hat`.

## Архитектура

### Forward Process
```
r_t = sqrt(alpha_bar[t]) * r + sqrt(1 - alpha_bar[t]) * eps
```

### Conditional Generation
- **Input**: (r_t, t, condition=x_hat)
- **Output**: predicted noise eps_pred
- **Final**: x = x_hat + r_0

## Файлы

### Основной модуль
- `src/diffusion_vae.py` - полная реализация conditional diffusion model

### Компоненты
1. **DiffusionScheduler** - управление шумом и временными шагами
2. **ConditionalUNet** - U-Net с time embedding и conditional input
3. **ConditionalDiffusionVAE** - основной класс для обучения и генерации

## Интеграция с SOM-VAE

### Шаг 1: Подготовка SOM-VAE
```python
# Убедитесь, что у вас есть обученная SOM-VAE модель
som_vae_model = EuroSAT_GlobalSOM_Deep(...)
# Загрузите веса или обучите модель
```

### Шаг 2: Создание Diffusion Model
```python
from src.diffusion_vae import create_diffusion_vae, train_diffusion_vae

# Создание conditional diffusion model
diffusion_vae = create_diffusion_vae(som_vae_model, device='cuda')
```

### Шаг 3: Обучение Diffusion Model
```python
# Полное обучение
diffusion_vae, history = train_diffusion_vae(
    som_vae_model=som_vae_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=50,
    batch_size=32,
    lr=1e-4,
    device='cuda'
)
```

### Шаг 4: Генерация улучшенных изображений
```python
from src.diffusion_vae import generate_enhanced_images, visualize_enhancement

# Генерация улучшенных изображений
original, som_recon, enhanced = generate_enhanced_images(
    som_vae_model=som_vae_model,
    diffusion_vae=diffusion_vae,
    dataset=val_dataset,
    num_samples=8,
    device='cuda'
)

# Визуализация результатов
visualize_enhancement(original, som_recon, enhanced)
```

## Добавление в som_vae.ipynb

### Импорты
Добавьте в начало ноутбука:
```python
# Diffusion model imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('')), '../src'))
from diffusion_vae import create_diffusion_vae, train_diffusion_vae
from diffusion_vae import generate_enhanced_images, visualize_enhancement
```

### Новые ячейки для обучения

#### Ячейка 1: Создание Diffusion Model
```python
# Создание conditional diffusion model
diffusion_vae = create_diffusion_vae(model, device='cuda', timesteps=1000)
print("Diffusion model created successfully!")
```

#### Ячейка 2: Обучение Diffusion Model
```python
# Обучение conditional diffusion model
diffusion_vae, diffusion_history = train_diffusion_vae(
    som_vae_model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=30,
    batch_size=16,  # Меньший размер для экономии памяти
    lr=1e-4,
    device='cuda'
)
```

#### Ячейка 3: Визуализация обучения Diffusion
```python
# Визуализация истории обучения diffusion model
plt.figure(figsize=(10, 4))
plt.plot(diffusion_history['train_loss'], label='Train Loss')
plt.plot(diffusion_history['val_loss'], label='Val Loss')
plt.title('Diffusion Model Training')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
```

#### Ячейка 4: Генерация улучшенных изображений
```python
# Генерация улучшенных изображений
original, som_recon, enhanced = generate_enhanced_images(
    som_vae_model=model,
    diffusion_vae=diffusion_vae,
    dataset=val_dataset,
    num_samples=6,
    device='cuda'
)

# Визуализация результатов
visualize_enhancement(original, som_recon, enhanced)
```

#### Ячейка 5: Количественная оценка
```python
# Оценка качества реконструкции
def calculate_metrics(original, reconstructed, enhanced):
    """Расчет MSE и PSNR"""
    mse_som = F.mse_loss(reconstructed, original).item()
    mse_enhanced = F.mse_loss(enhanced, original).item()
    
    psnr_som = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse_som)))
    psnr_enhanced = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse_enhanced)))
    
    print(f"SOM-VAE MSE: {mse_som:.4f}, PSNR: {psnr_som:.2f} dB")
    print(f"Enhanced MSE: {mse_enhanced:.4f}, PSNR: {psnr_enhanced:.2f} dB")
    print(f"Improvement: {((mse_som - mse_enhanced) / mse_som * 100):.1f}%")

# Расчет метрик
original_tensor = torch.stack(original)
som_tensor = torch.stack(som_recon)
enhanced_tensor = torch.stack(enhanced)

calculate_metrics(original_tensor, som_tensor, enhanced_tensor)
```

## Гиперпараметры

### Diffusion Model
- `timesteps`: 1000 (количество шагов диффузии)
- `time_dim`: 256 (размерность time embedding)
- `beta_schedule`: 'cosine' (тип расписания)

### Training
- `epochs`: 30-50
- `batch_size`: 16-32 (зависит от GPU памяти)
- `lr`: 1e-4 (скорость обучения)
- `optimizer`: AdamW

## Преимущества подхода

1. **Условная генерация**: Использует SOM-VAE реконструкцию как условие
2. **Residual обучение**: Моделирует только разницу, что эффективнее
3. **Совместная оптимизация**: SOM-VAE + Diffusion работают вместе
4. **Гибкость**: Можно использовать с любой обученной SOM-VAE

## Вычислительные требования

### GPU Memory
- SOM-VAE: ~2-4 GB
- Diffusion U-Net: ~1-2 GB
- Обучение: ~6-8 GB (batch_size=16)

### Training Time
- SOM-VAE: ~30-60 минут (15 эпох)
- Diffusion: ~45-90 минут (30 эпох)
- Total: ~2-3 часа на RTX 3080

## Возможные проблемы

1. **Memory Error**: Уменьшите batch_size
2. **Slow Training**: Уменьшите timesteps или epochs
3. **Poor Quality**: Увеличьте epochs или измените lr
4. **Mode Collapse**: Проверьте данные и гиперпараметры

## Дальнейшие улучшения

1. **Multi-scale U-Net**: Добавьте skip connections
2. **Attention**: Добавьте self-attention слои
3. **Better Scheduling**: Попробуйте другие beta schedules
4. **Progressive Training**: Пошаговое увеличение timesteps

---

*Последнее обновление: 2026-02-01*
