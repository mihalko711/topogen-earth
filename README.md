# TopoGen Earth

**Generative topological atlas of satellite imagery**

TopoGen Earth — проект по генерации и классификации спутниковых изображений с использованием самоорганизующихся карт (SOM) и диффузионных моделей.

---

## 📌 Активные ноутбуки

### 1. **SOM-VAE Model** — `notebooks/som_vae_renewed.ipynb`

Основная реализация модели **SOM-VAE** (Self-Organizing Map + Variational Autoencoder) для классификации спутниковых изображений EuroSAT.

**Что внутри:**
- Полный пайплайн обучения SOM-VAE с нуля
- Архитектура **VAE с bottleneck** (классический автоэнкодер)
- Самоорганизующаяся карта для кластеризации латентных представлений
- Визуализация результатов классификации
- Реконструкция изображений из латентного пространства

**Ключевые компоненты:**
```python
# Encoder: Image → Latent (bottleneck)
# SOM: Clustering in latent space
# Decoder: Latent → Image reconstruction
```

**Датасет:** EuroSAT (21 000 изображений, 10 классов)

---

### 2. **DDPM Model** — `notebooks/new_DDPM.ipynb`

Реализация **Denoising Diffusion Probabilistic Model** для генерации спутниковых изображений.

**Что внутри:**
- Полная реализация DDPM с **UNet** архитектурой
- Различные расписания бетта (cosine, linear, quadratic, sigmoid)
- Обучение модели на EuroSAT датасете
- Генерация новых изображений из шума
- Conditional generation (опционально)

**Ключевые особенности:**
- 1000 шагов диффузии
- Time embeddings для условной генерации
- Attention механизмы в UNet
- Progressive sampling для генерации

---

## 🗂 Структура проекта

```
topogen-earth/
├── notebooks/
│   ├── som_vae_renewed.ipynb    # 🔴 Активный: SOM-VAE (VAE + SOM)
│   ├── new_DDPM.ipynb           # 🔴 Активный: DDPM (UNet-based)
│   ├── som_vae.ipynb            # Архивная версия SOM-VAE
│   ├── diffusion_implementation.ipynb  # Архивная версия DDPM
│   └── ...
│
├── src/
│   ├── som_vae/
│   │   ├── model.py             # SOM-VAE архитектура (bottleneck VAE)
│   │   ├── training.py          # Функции обучения
│   │   └── visualization.py     # Визуализация результатов
│   │
│   ├── diffusion_model/
│   │   ├── model.py             # DDPM UNet архитектура
│   │   ├── training.py          # Обучение диффузии
│   │   ├── inference.py         # Генерация изображений
│   │   └── README.md            # Документация по модулю
│   │
│   ├── datasets/
│   │   └── dataset.py           # EuroSAT датасет классы
│   │
│   ├── resnet_eurosat/          # ResNet для классификации
│   └── diffusion_vae.py         # Conditional Diffusion VAE
│
├── finetune_script.py           # Скрипт дообучения моделей
├── train_from_scratch_script.py # Обучение с нуля
│
├── configs/                     # Hydra конфигурации
├── checkpoints/                 # Сохранённые модели (.gitignored)
├── outputs/                     # Результаты обучения (.gitignored)
│
└── docs/
    ├── SOM_VAE_EuroSAT.md       # Документация SOM-VAE
    ├── DIFFUSION_INTEGRATION.md # Интеграция диффузии
    └── FINETUNE_README.md       # Гайд по дообучению
```

---

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install torch torchvision torchaudio
pip install matplotlib numpy pandas polars
pip install tqdm scikit-learn
pip install kagglehub
pip install hydra-core omegaconf
```

### Запуск SOM-VAE

1. Откройте `notebooks/som_vae_renewed.ipynb`
2. Датасет загрузится автоматически через Kaggle Hub
3. Запустите все ячейки для обучения модели

### Запуск DDPM

1. Откройте `notebooks/new_DDPM.ipynb`
2. Настройте гиперпараметры (timesteps, beta schedule)
3. Обучите модель и генерируйте изображения

### Дообучение моделей

```bash
python finetune_script.py \
  --config-path=./configs \
  --config-name=finetune_config \
  paths.model_path=./checkpoints/pretrained.pt \
  paths.data_path=./data \
  training.epochs=20
```

---

## 📊 Датасеты

### EuroSAT
- **21 000** спутниковых изображений
- **10** классов земного покрова
- Размер: 64×64 RGB
- Источник: [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

**Классы:**
1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

---

## 🧪 Архитектуры моделей

### SOM-VAE (VQ-VAE style)
- **Encoder:** Conv layers → **Bottleneck** → латентное пространство (64-256 dim)
- **SOM:** Самоорганизующаяся карта (grid_size × grid_size) для кластеризации
- **Decoder:** Bottleneck → Conv layers → реконструкция
- **Loss:** Reconstruction + SOM topology + Commitment loss

### DDPM (UNet-based)
- **Backbone:** **UNet** с attention блоками
- **Timesteps:** 1000 шагов диффузии
- **Beta Schedule:** Cosine (по умолчанию)
- **Conditioning:** Time embeddings + class labels (опционально)

---

## 📈 Результаты

### SOM-VAE
- Классификация изображений через ближайших соседей в SOM
- Визуализация карты признаков
- Реконструкция изображений из латентного пространства

### DDPM
- Генерация реалистичных спутниковых изображений
- Conditional generation по классам
- Интерполяция в латентном пространстве

---

## 🛠 Конфигурация

Проект использует **Hydra** для управления конфигурациями:

```yaml
# configs/finetune_config.yaml
model:
  grid_size: 10
  latent_dim: 64

training:
  epochs: 20
  batch_size: 32
  learning_rate: 5e-6
  alpha: 0.5  # commitment loss
  beta: 0.3   # SOM loss
  gamma: 0.2  # classification loss
```

---

## 📝 Дополнительная документация

- [SOM_VAE_EuroSAT.md](./SOM_VAE_EuroSAT.md) — Подробная документация по SOM-VAE
- [DIFFUSION_INTEGRATION.md](./DIFFUSION_INTEGRATION.md) — Интеграция диффузионной модели
- [FINETUNE_README.md](./FINETUNE_README.md) — Гайд по дообучению моделей
- [src/diffusion_model/README.md](./src/diffusion_model/README.md) — API диффузионной модели

---

## 🔒 Безопасность

✅ Проект проверен на отсутствие:
- API ключей и секретов
- Персональных данных
- Чувствительной информации

Все чувствительные файлы добавлены в `.gitignore`:
- `checkpoints/` — веса моделей
- `outputs/` — логи и результаты
- `*.env` — переменные окружения

---

## 👥 Авторы

Проект разработан для создания генеративного топологического атласа спутниковых изображений.

---

*Последнее обновление: 2026-02-25*
