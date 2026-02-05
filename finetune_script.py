#!/usr/bin/env python3
"""
Скрипт для дообучения (fine-tuning) модели SOM-VAE
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from datetime import datetime
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.som_vae.model import EuroSAT_GlobalSOM_Deep
from src.som_vae.training import (
    som_vae_loss,
    train_som_vae_pretrained,
    load_model,
    save_model
)
from src.datasets.dataset import EuroSATDataset, get_train_transform, get_val_transform


@hydra.main(version_base=None, config_path="configs", config_name="finetune_config")
def main(cfg: DictConfig):
    print("Конфигурация:", cfg)
    
    # Определение устройства
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device != "cpu" else 'cpu')
    print(f"Используется устройство: {device}")

    # Загрузка предварительно обученной модели
    if cfg.paths.model_path is None:
        print("Ошибка: Не указан путь к предварительно обученной модели (model_path)")
        return

    print(f"Загрузка модели из {cfg.paths.model_path}...")
    model, metadata = load_model(cfg.paths.model_path, device=device, params={"grid_size" : cfg.model.grid_size, "latent_dim" : cfg.model.latent_dim})
    
    # Обновляем скорость обучения для дообучения
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    print(f"Модель загружена")
    
    # Подготовка данных
    print(f"Загрузка обучающих/валидационных данных из {cfg.paths.data_path}...")

    # Используем трансформации для обучения и валидации
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # Загружаем датасеты
    train_dataset = EuroSATDataset(csv_path=os.path.join(cfg.paths.data_path, "train.csv"), image_root=cfg.paths.data_path, transform=train_transform, return_label=True)
    val_dataset = EuroSATDataset(csv_path=os.path.join(cfg.paths.data_path, "validation.csv"), image_root=cfg.paths.data_path, transform=val_transform, return_label=True)
    
    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")
    
    # Дообучение модели
    print("Начинаем дообучение...")
    
    history = train_som_vae_pretrained(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        device=device,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        alpha=cfg.training.alpha,
        beta=cfg.training.beta,
        gamma=cfg.training.gamma,
        info_interval=cfg.training.info_interval,
        img_save_name=os.path.join(cfg.paths.img_path, cfg.model.name)
    )
    
    # Сохранение дообученной модели
    print(f"Сохранение дообученной модели в {cfg.paths.output_dir}...")

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # # Обновляем параметры обучения в чекпоинте
    # metadata['training']['learning_rate'] = cfg.training.learning_rate
    # metadata['training']['alpha'] = cfg.training.alpha
    # metadata['training']['beta'] = cfg.training.beta
    # metadata['training']['gamma'] = cfg.training.gamma

    # Сохраняем модель с новым именем
    save_model(model, path=os.path.join(cfg.paths.output_dir, cfg.paths.output_name), metadata=metadata)
    
    print(f"Дообученная модель успешно сохранена!")
    
    # Сохраняем историю обучения
    import json
    history_path = os.path.join(cfg.paths.output_dir, f"{cfg.paths.output_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"История обучения сохранена: {history_path}")


if __name__ == "__main__":
    main()