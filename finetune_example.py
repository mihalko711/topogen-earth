#!/usr/bin/env python3
"""
Пример запуска скрипта дообучения
"""

# Пример команды для запуска:
# python finetune_script.py \
#   --config-path=../configs \
#   --config-name=config \
#   paths.model_path=./checkpoints/pretrained_model.pt \
#   paths.train_data_path=./data/train \
#   paths.val_data_path=./data/val \
#   paths.output_dir=./checkpoints \
#   paths.output_name=finetuned_model \
#   training.epochs=20 \
#   training.learning_rate=5e-6 \
#   training.alpha=0.8 \
#   training.beta=1.2 \
#   training.gamma=1.0

print("Примеры запуска скрипта дообучения SOM-VAE:")
print()
print("# Запуск с параметрами по умолчанию:")
print("python finetune_script.py \\")
print("  --config-path=../configs \\")
print("  --config-name=config \\")
print("  paths.model_path=./checkpoints/pretrained_model.pt \\")
print("  paths.train_data_path=./data/train \\")
print("  paths.val_data_path=./data/val")
print()
print("# Запуск с измененными гиперпараметрами:")
print("python finetune_script.py \\")
print("  --config-path=../configs \\")
print("  --config-name=config \\")
print("  paths.model_path=./checkpoints/pretrained_model.pt \\")
print("  paths.train_data_path=./data/train \\")
print("  paths.val_data_path=./data/val \\")
print("  training.epochs=20 \\")
print("  training.learning_rate=5e-6 \\")
print("  training.alpha=0.8 \\")
print("  training.beta=1.2 \\")
print("  training.gamma=1.0")
print()
print("Примечание: Убедитесь, что пути к данным и модели существуют.")