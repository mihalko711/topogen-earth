# Fine-tuning SOM-VAE Model

Этот проект содержит скрипт для дообучения (fine-tuning) модели SOM-VAE с использованием Hydra для управления конфигурациями.

## Структура проекта

```
topogen-earth/
├── finetune_script.py          # Основной скрипт дообучения
├── configs/                    # Конфигурационные файлы Hydra
│   ├── config.yaml             # Главный конфиг
│   └── conf/                   # Подконфиги
│       ├── default.yaml        # Конфиг по умолчанию
│       ├── model/              # Конфиги модели
│       │   └── som_vae.yaml
│       └── training/           # Конфиги тренировки
│           └── default.yaml
├── checkpoints/                # Чекпоинты моделей
├── notebooks/                  # Jupyter ноутбуки
├── src/                        # Исходный код проекта
└── ...
```

## Запуск дообучения

Для запуска дообучения используйте следующую команду:

```bash
python finetune_script.py \
  --config-path=../configs \
  --config-name=config \
  paths.model_path=./checkpoints/pretrained_model.pt \
  paths.train_data_path=./data/train \
  paths.val_data_path=./data/val \
  paths.output_dir=./checkpoints \
  paths.output_name=finetuned_model \
  training.epochs=20 \
  training.learning_rate=5e-6
```

## Конфигурации

Hydra позволяет легко управлять параметрами модели и обучения:

- `configs/conf/model/som_vae.yaml` - параметры архитектуры модели
- `configs/conf/training/default.yaml` - гиперпараметры обучения
- `configs/conf/default.yaml` - конфигурация по умолчанию

Вы можете создавать собственные конфигурации и использовать их через параметр `--config-name`.

## Параметры

### Пути:
- `paths.model_path` - путь к предварительно обученной модели
- `paths.train_data_path` - путь к обучающим данным
- `paths.val_data_path` - путь к валидационным данным
- `paths.output_dir` - директория для сохранения результатов
- `paths.output_name` - имя файла для сохранения модели

### Параметры обучения:
- `training.epochs` - количество эпох дообучения
- `training.batch_size` - размер батча
- `training.learning_rate` - скорость обучения
- `training.alpha` - вес commitment loss
- `training.beta` - вес SOM loss
- `training.gamma` - вес classification loss
- `training.device` - устройство для обучения ('cuda' или 'cpu')