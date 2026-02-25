import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import json
from datetime import datetime
import os

# Import the model class needed for loading
from .model import EuroSAT_GlobalSOM_Deep
from .visualization import plot_som_reconstruction_map


def som_vae_loss(x, x_hat_e, x_hat_q, z_e, z_q, indices, logits, targets, som_layer, 
                 alpha=1.0, beta=1.0, gamma=1.0):
    """
    Комбинированная функция потерь для SOM-VAE
    
    Args:
        x: исходные изображения
        x_hat_e: реконструкция из z_e (до квантования)
        x_hat_q: реконструкция из z_q (после квантования)
        z_e: латентные представления до квантования
        z_q: латентные представления после квантования
        indices: индексы победителей на SOM сетке
        logits: предсказания классификатора
        targets: истинные метки классов
        som_layer: SOM слой для расчета соседей
        alpha: вес commitment loss
        beta: вес SOM loss
        gamma: вес classification loss
    
    Returns:
        total_loss, l_reconstruction, l_commitment, l_som, l_cls
    """
    # 1. Reconstruction Loss (MSE между оригиналом и обеими реконструкциями)
    l_rec_e = F.mse_loss(x_hat_e, x)
    l_rec_q = F.mse_loss(x_hat_q, x)
    l_reconstruction = l_rec_e + l_rec_q
    
    # 2. Commitment Loss (регуляризация квантования)
    l_commitment = F.mse_loss(z_e, z_q)
    
    # 3. SOM Loss (регуляризация соседей)
    neighbor_indices = som_layer.get_neighbors(indices)
    neighbors = som_layer.embeddings[neighbor_indices]
    z_e_target = z_e.detach().view(z_e.shape[0], -1).unsqueeze(1)
    l_som = torch.mean((neighbors - z_e_target)**2)
    
    # 4. Classification Loss (кросс-энтропия)
    l_cls = F.cross_entropy(logits, targets)
    
    # Итоговый лосс с учетом всех компонентов
    total_loss = l_reconstruction + alpha * l_commitment + beta * l_som + gamma * l_cls
    
    return total_loss, l_reconstruction, l_commitment, l_som, l_cls


def restart_som_with_data(model, train_loader, device):
    """
    Инициализация весов SOM реальными данными из первого батча
    """
    model.eval()
    with torch.no_grad():
        # Берем один батч
        images, _, _ = next(iter(train_loader))
        z_e = model.encoder(images.to(device)) # [B, C, H, W]
        z_e_flat = z_e.view(z_e.size(0), -1)   # [B, Dim]
        
        num_embeddings = model.som.embeddings.size(0)
        
        # Если батч меньше, чем число узлов, просто повторяем его
        indices = torch.arange(num_embeddings) % z_e_flat.size(0)
        initial_weights = z_e_flat[indices]
        
        # Прямое копирование в веса
        model.som.embeddings.data.copy_(initial_weights)
    print(f"Карта SOM инициализирована {num_embeddings} векторами из данных.")


def train_som_vae(model, train_dataset, val_dataset, optimizer, device, 
                  epochs=50, batch_size=128, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Обучение SOM-VAE модели
    """
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    
    # История обучения для визуализации
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'nmi': []
    }
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = {'total': 0, 'rec': 0, 'comm': 0, 'som': 0, 'cls': 0}
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for data, labels, _ in pbar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Прямой проход
            x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(data)
            
            # Расчет лосса
            loss, l_rec, l_comm, l_som, l_cls = som_vae_loss(
                data, x_hat_e, x_hat_q, z_e, z_q, indices, logits, labels, model.som,
                alpha=alpha, beta=beta, gamma=gamma
            )
            
            loss.backward()
            optimizer.step()
            
            # Считаем точность
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Накопление лоссов
            train_losses['total'] += loss.item()
            train_losses['rec'] += l_rec.item()
            train_losses['comm'] += l_comm.item()
            train_losses['som'] += l_som.item()
            train_losses['cls'] += l_cls.item()
            
            pbar.set_postfix({
                'L': f"{loss.item():.3f}",
                'Rec': f"{l_rec.item():.3f}",
                'Comm': f"{l_comm.item():.3f}",
                'som': f"{l_som.item():.3f}",
                'Cls': f"{l_cls.item():.3f}",
                'Acc': f"{100 * train_correct / train_total:.1f}%"
            })

        # --- ФАЗА ВАЛИДАЦИИ ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_total_loss = 0
        all_indices = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels, _ in tqdm(val_loader, desc='Validation', leave=False):
                data, labels = data.to(device), labels.to(device)
                x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(data)
                
                # Расчет лосса для валидации
                loss, _, _, _, _ = som_vae_loss(
                    data, x_hat_e, x_hat_q, z_e, z_q, indices, logits, labels, model.som,
                    alpha=alpha, beta=beta, gamma=gamma
                )
                
                # Точность на валидации
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_total_loss += loss.item()

                all_indices.append(indices.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        # Расчет метрик
        flat_indices = np.concatenate(all_indices).ravel()
        flat_labels = np.concatenate(all_labels).ravel()
        current_nmi = normalized_mutual_info_score(flat_labels, flat_indices)

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Сохранение истории
        history['train_loss'].append(train_losses['total'] / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_total_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['nmi'].append(current_nmi)
        
        tqdm.write(f"Summary Epoch {epoch}:")
        tqdm.write(f"Train Loss: {history['train_loss'][-1]:.4f}| Val Loss: {history['val_loss'][-1]:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | NMI: {current_nmi:.4f}")

    return history


def train_som_vae_pretrained(model, train_dataset, val_dataset, optimizer, device, 
                  epochs=10, batch_size=128, info_interval=5, alpha=1.0, beta=1.0, gamma=1.0, 
                  img_save_name='pic.png', scheduler=None):
    """
    Обучение SOM-VAE модели с поддержкой learning rate scheduler'а
    
    Параметры:
        scheduler: опционально - экземпляр torch.optim.lr_scheduler.*
                   Поддерживаются все типы, включая ReduceLROnPlateau (авто-детект)
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from sklearn.metrics import normalized_mutual_info_score

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    
    # История обучения + трекинг LR
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'nmi': [],
        'lr': []  # <-- новый ключ для отслеживания LR
    }
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = {'total': 0, 'rec': 0, 'comm': 0, 'som': 0, 'cls': 0}
        train_correct = 0
        train_total = 0
        
        # Получаем текущий LR для отображения
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train] LR={current_lr:.2e}")
        
        for data, labels, _ in pbar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Прямой проход
            x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(data)
            
            # Расчет лосса
            loss, l_rec, l_comm, l_som, l_cls = som_vae_loss(
                data, x_hat_e, x_hat_q, z_e, z_q, indices, logits, labels, model.som,
                alpha=alpha, beta=beta, gamma=gamma
            )
            
            loss.backward()
            optimizer.step()
            
            # Считаем точность
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Накопление лоссов
            train_losses['total'] += loss.item()
            train_losses['rec'] += l_rec.item()
            train_losses['comm'] += l_comm.item()
            train_losses['som'] += l_som.item()
            train_losses['cls'] += l_cls.item()
            
            pbar.set_postfix({
                'L': f"{loss.item():.3f}",
                'Rec': f"{l_rec.item():.3f}",
                'Comm': f"{l_comm.item():.3f}",
                'som': f"{l_som.item():.3f}",
                'Cls': f"{l_cls.item():.3f}",
                'Acc': f"{100 * train_correct / train_total:.1f}%"
            })

        # --- ФАЗА ВАЛИДАЦИИ ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_total_loss = 0
        all_indices = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels, _ in tqdm(val_loader, desc='Validation', leave=False):
                data, labels = data.to(device), labels.to(device)
                x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(data)
                
                # Расчет лосса для валидации
                loss, _, _, _, _ = som_vae_loss(
                    data, x_hat_e, x_hat_q, z_e, z_q, indices, logits, labels, model.som,
                    alpha=alpha, beta=beta, gamma=gamma
                )
                
                # Точность на валидации
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_total_loss += loss.item()

                all_indices.append(indices.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        # Расчет метрик
        flat_indices = np.concatenate(all_indices).ravel()
        flat_labels = np.concatenate(all_labels).ravel()
        current_nmi = normalized_mutual_info_score(flat_labels, flat_indices)
        val_loss_avg = val_total_loss / len(val_loader)

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Сохранение истории
        history['train_loss'].append(train_losses['total'] / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        history['nmi'].append(current_nmi)
        history['lr'].append(current_lr)  # <-- сохраняем текущий LR до шага скедулера
        
        # Шаг скедулера после валидации
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss_avg)  # для плато — передаём метрику
            else:
                scheduler.step()  # для остальных — просто шаг
        
        # Вывод саммари с LR
        new_lr = optimizer.param_groups[0]['lr']
        tqdm.write(f"Summary Epoch {epoch}:")
        tqdm.write(f"LR: {current_lr:.2e} → {new_lr:.2e} | "
                   f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {val_loss_avg:.4f} | "
                   f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | NMI: {current_nmi:.4f}")
        
        if epoch % info_interval == 0:
            plot_som_reconstruction_map(model, save_path=f"{img_save_name}_ep{epoch}.png")

    return history


def save_model(model, path, metadata=None):
    """Сохраняет модель и опциональные метаданные"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }, path)


def load_model(path, device='cpu', params={}):
    """Загружает модель на указанный девайс"""
    
    checkpoint = torch.load(path, map_location=device)
    model = EuroSAT_GlobalSOM_Deep(**params)  # передай свои параметры при необходимости
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['metadata']