import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSpatialSOMLayer(nn.Module):
    def __init__(self, grid_h, grid_w, channels, latent_h, latent_w, alpha_grad=1.0):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.c, self.h, self.w = channels, latent_h, latent_w
        self.template_dim = channels * latent_h * latent_w
        self.embeddings = nn.Parameter(torch.randn(grid_h * grid_w, self.template_dim))
        self.alpha_grad = alpha_grad  # вес градиентного члена

    def _compute_distances(self, z_e_flat, embeddings_flat):
        """
        z_e_flat: [B, D]
        embeddings_flat: [K, D]
        Возвращает: [B, K] — квадратичные расстояния (без корня для скорости)
        """
        # Оптимизированная реализация через матричные операции
        # ||a - b||² = ||a||² + ||b||² - 2⟨a, b⟩
        z_norm = (z_e_flat ** 2).sum(dim=1, keepdim=True)      # [B, 1]
        e_norm = (embeddings_flat ** 2).sum(dim=1, keepdim=True)  # [K, 1]
        dot = z_e_flat @ embeddings_flat.T                     # [B, K]
        
        distances = z_norm + e_norm.T - 2 * dot                # [B, K]
        return torch.clamp(distances, min=0.0)  # стабилизация против отрицательных значений из-за численной ошибки

    def forward(self, z_e):
        batch_size = z_e.size(0)
        z_e_flat = z_e.view(batch_size, -1)  # [B, D]
    
        # Считаем расстояния с учётом градиентов
        distances = self._compute_distances(z_e_flat, self.embeddings)  # [B, K]
    
        indices = torch.argmin(distances, dim=1)  # [B]
    
        z_q_flat = self.embeddings[indices]  # [B, D]
        z_q = z_q_flat.view(batch_size, self.c, self.h, self.w)
    
        return z_q, indices

    def get_neighbors(self, indices):
        # Логика поиска соседей по 2D сетке
        row = indices // self.grid_w
        col = indices % self.grid_w
        
        def get_idx(r, c):
            r = torch.clamp(r, 0, self.grid_h - 1)
            c = torch.clamp(c, 0, self.grid_w - 1)
            return r * self.grid_w + c

        return torch.stack([
            get_idx(row-1, col), get_idx(row+1, col), 
            get_idx(row, col-1), get_idx(row, col+1)
        ], dim=1)

    def calc_activations(self, z_e):
        batch_size = z_e.size(0)
        z_e_flat = z_e.view(batch_size, -1)
        
        distances = self._compute_distances(z_e_flat, self.embeddings)  # [B, K]
        
        # Инвертируем, чтобы близость → большое число
        activations = 1.0 / (distances + 1e-8)
        
        return activations


class EuroSAT_GlobalSOM_Deep(nn.Module):
    def __init__(self, in_channels=3, grid_size=(16, 16), latent_dim=(32, 4, 4), num_classes=10):
        super().__init__()
        c_l, h_l, w_l = latent_dim

        # ---------------------
        # 1. ЭНКОДЕР (64x64 -> 8x8)
        # ---------------------
        self.encoder = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),  # ✅ Исправлено: было 128 → теперь 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, latent_dim[0], 4, stride=2, padding=1),  # 8 -> 4 ✅ Исправлен комментарий
            nn.BatchNorm2d(latent_dim[0]),
            nn.LeakyReLU(0.2),
        )

        # ---------------------
        # 2. SOM СЛОЙ
        # ---------------------
        self.som = GlobalSpatialSOMLayer(grid_size[0], grid_size[1], c_l, h_l, w_l)

        # ---------------------
        # 3. ДЕКОДЕР (8x8 -> 64x64)
        # ---------------------
        self.decoder = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(latent_dim[0], 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 8 -> 16
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        
            # 16 -> 32
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        
            # 32 -> 64
            nn.ConvTranspose2d(128, in_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )


        # ---------------------
        # 4. Классификатор
        # ---------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_l * h_l * w_l, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.som(z_e)
        logits = self.classifier(z_q)
        x_hat_e = self.decoder(z_e)
        x_hat_q = self.decoder(z_q)
        return x_hat_e, x_hat_q, z_e, z_q, indices, logits

    def decode(self, z):
        return self.decoder(z)