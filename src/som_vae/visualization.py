import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random


def visualize_som_sample(model, dataset, class_num=None, device="cuda"):
    model.eval()

    # --- 1. Случайный пример ---
    if class_num is not None:
        stop_fl = False
        try_cnt = 0
        while not stop_fl:
            try_cnt += 1
            idx = random.randint(0, len(dataset) - 1)
            if dataset[idx][1] == class_num:
                stop_fl = True
            elif try_cnt > 1000:
                raise Exception('too much attempts')
    else:
        idx = random.randint(0, len(dataset) - 1)

    sample = dataset[idx]
    if isinstance(sample, tuple):
        x = sample[0]
        label = sample[1] if len(sample) > 1 else None
    else:
        x = sample
        label = None

    x = x.unsqueeze(0).to(device)

    # --- 2. Прогон через модель ---
    with torch.no_grad():
        x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(x)

    # --- 3. Активации SOM ---
    activations = model.som.calc_activations(z_e)  # [1, K]
    grid_h, grid_w = model.som.grid_h, model.som.grid_w
    act_map = activations[0].view(grid_h, grid_w).detach().cpu()

    winner = indices[0].item()
    win_r = winner // grid_w
    win_c = winner % grid_w

    # --- 4. Победивший embedding ---
    emb = model.som.embeddings[winner]  # [C*H*W]
    emb = emb.view(1, model.som.c, model.som.h, model.som.w)

    with torch.no_grad():
        winner_recon = model.decode(emb.to(device)).cpu()

    # --- 5. Денормализация ---
    def denorm(img):
        img = img.squeeze().permute(1, 2, 0)
        img = img * 0.5 + 0.5
        return img.clamp(0, 1)

    x = denorm(x.cpu())
    x_hat_e = denorm(x_hat_e.cpu())
    x_hat_q = denorm(x_hat_q.cpu())
    winner_recon = denorm(winner_recon)

    # --- 6. Визуализация ---
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    axs[0][0].imshow(x)
    axs[0][0].axis('off')
    axs[0][0].set_title(r"Исходная картинка")

    axs[0][1].imshow(x_hat_e)
    axs[0][1].axis('off')
    axs[0][1].set_title(r"Реконструкция сразу после кодировщика($\hat{x_e}$)")

    axs[0][2].imshow(x_hat_q)
    axs[0][2].axis('off')
    axs[0][2].set_title(r"Реконструкция узла-победителя($\hat{x_q}$)"+ f", индекс{indices[0]}")

    act_map = axs[0][3].imshow(act_map, cmap='magma')
    fig.colorbar(act_map, ax=axs[0][3])
    axs[0][3].set_title(r"Карта активации узлов")

    neighbors = model.som.get_neighbors(indices)[0]

    for num, neighbor in enumerate(neighbors):
        emb = model.som.embeddings[neighbor]
        emb = emb.view(1, model.som.c, model.som.h, model.som.w)
        
        with torch.no_grad():
            neighbor_recon = model.decode(emb.to(device)).cpu()

        x_hat_neighbor = denorm(neighbor_recon)

        axs[1][num].imshow(x_hat_neighbor)
        axs[1][num].axis('off')
        axs[1][num].set_title(f"Реконструкция  соседа по индексу {neighbor}")

    fig.suptitle(f"Отчет по реконструкции для класса {dataset[idx][-1]}")
    fig.tight_layout()
    plt.show()


def plot_som_map(model, device='cuda'):
    som_map = model.som.embeddings.view(-1, model.som.c, model.som.h, model.som.w)
    with torch.no_grad():
        som_cls = model.classifier(som_map.to(device)).cpu()

    fig, ax = plt.subplots(3, 4, figsize=(20, 15))

    for cls_num in range(10):
        som_cls_act = som_cls[:,cls_num]
        temperature = ax[cls_num // 4][cls_num % 4].imshow(som_cls_act.reshape(model.som.grid_h, model.som.grid_w))
        fig.colorbar(temperature, ax=ax[cls_num // 4][cls_num % 4])
        ax[cls_num // 4][cls_num % 4].set_title(f"Карта логитов класса {cls_num} для som-map")

    ax[2][2].imshow(som_cls.argmax(axis=1).reshape(model.som.grid_h, model.som.grid_w), cmap='rainbow')
    ax[2][3].imshow(som_cls.argmin(axis=1).reshape(model.som.grid_h, model.som.grid_w), cmap='rainbow')
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def plot_som_reconstruction_map(model, figsize=(20, 20), save_path=None):
    """
    Визуализирует карту реконструкций: каждый узел SOM → декодированное изображение.
    Результат: сетка [grid_h x grid_w] изображений.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Берём все веса SOM и приводим к форме [N, C, H, W]
    embeddings = model.som.embeddings  # [N, C*H*W]
    z_q = embeddings.view(-1, model.som.c, model.som.h, model.som.w).to(device)
    
    # 2. Декодируем все шаблоны сразу
    reconstructions = model.decode(z_q)  # [N, 3, 64, 64]
    
    # 3. Нормализуем из [-1, 1] (Tanh) → [0, 1] для визуализации
    reconstructions = (reconstructions + 1) / 2.0
    
    # 4. Собираем мозаику
    grid_h, grid_w = model.som.grid_h, model.som.grid_w
    img_h, img_w = reconstructions.shape[2], reconstructions.shape[3]
    
    mosaic = torch.zeros(3, grid_h * img_h, grid_w * img_w)
    for idx in range(grid_h * grid_w):
        r = idx // grid_w
        c = idx % grid_w
        mosaic[:, r*img_h:(r+1)*img_h, c*img_w:(c+1)*img_w] = reconstructions[idx]
    
    # 5. Отображаем
    plt.figure(figsize=figsize)
    plt.imshow(mosaic.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f'SOM Reconstruction Map ({grid_h}×{grid_w} nodes)')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()


def analyze_latent_similarity(z_e, z_q, show=True):
    """
    z_e, z_q: [1, C, H, W]
    """
    z_e = z_e.detach().cpu()
    z_q = z_q.detach().cpu()

    # --- flatten ---
    ze_flat = z_e.view(1, -1)
    zq_flat = z_q.view(1, -1)

    # --- Метрики ---
    mse = F.mse_loss(z_q, z_e).item()
    l2 = torch.norm(ze_flat - zq_flat, dim=1).item()
    cos = F.cosine_similarity(ze_flat, zq_flat).item()

    # --- Spatial error map ---
    spatial_err = ((z_e - z_q) ** 2).mean(dim=1)[0]  # [H, W]

    # --- Channel-wise error ---
    channel_err = ((z_e - z_q) ** 2).mean(dim=(2, 3))[0]  # [C]

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        # spatial
        im0 = axs[0].imshow(spatial_err, cmap="inferno")
        axs[0].set_title("Spatial MSE (8×8)")
        plt.colorbar(im0, ax=axs[0])

        # channels
        axs[1].plot(channel_err.numpy())
        axs[1].set_title("Channel-wise MSE")
        axs[1].set_xlabel("Channel")
        axs[1].set_ylabel("Error")

        # summary text
        axs[2].axis("off")
        axs[2].text(0.05, 0.7, f"MSE: {mse:.6f}", fontsize=14)
        axs[2].text(0.05, 0.5, f"L2 norm: {l2:.4f}", fontsize=14)
        axs[2].text(0.05, 0.3, f"Cos sim: {cos:.4f}", fontsize=14)

        plt.tight_layout()
        plt.show()

    return {
        "mse": mse,
        "l2": l2,
        "cosine": cos,
        "spatial_err": spatial_err,
        "channel_err": channel_err
    }


def analyze_random(model, dataset, device="cuda"):
    model.eval()

    # --- 1. Случайный пример ---
    idx = random.randint(0, len(dataset) - 1)

    sample = dataset[idx]
    if isinstance(sample, tuple):
        x = sample[0]
        label = sample[1] if len(sample) > 1 else None
    else:
        x = sample
        label = None

    x = x.unsqueeze(0).to(device)

    # --- 2. Прогон через модель ---
    with torch.no_grad():
        x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(x)

    return analyze_latent_similarity(z_e, z_q)


def visualize_enhancement(original, som_recon, enhanced):
    num_samples = len(original)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        imgs = [original[i], som_recon[i], enhanced[i]]
        titles = ['Original', 'SOM-VAE', 'Enhanced']
        for j in range(3):
            # Денормализация из [-1, 1] в [0, 1] для корректного отображения
            img = imgs[j].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 0.5 + 0.5, 0, 1)
            
            axes[i, j].imshow(img)
            axes[i, j].set_title(titles[j])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()