class TopoResNet(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=64, pretrained=True):
        super().__init__()
        

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        
        num_features = self.backbone.fc.in_features
        
        # замена родной головы на пустышку
        self.backbone.fc = nn.Identity()
        

        # создание своего промежуточного слоя нужного размера
        self.projection = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # финальный слой для классификации
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, return_embedding=False):
        # прогон через resnet (без старой головы)
        features = self.backbone(x)
        
        embedding = self.projection(features)
        
        # для получения просто эмбеддинга
        if return_embedding:
            return embedding
        
        logits = self.classifier(embedding)
        return logits, embedding

    def save_weights(self, path: str):
        """
        Сохраняет веса модели в файл.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_classes": self.classifier.out_features,
                "embedding_dim": self.projection[-1].out_features,
            },
            path
        )

    def load_weights(self, path: str, map_location="cpu"):
        """
        Загружает веса модели из файла.
        """
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["model_state_dict"])



def base_train_loop(model, criterion, optimizer, train_loader, test_loader, epochs):
    epoch_loop = tqdm(range(epochs), desc="Итерации по эпохам", leave=False)
    history = {
        "train_loss": [],
        "test_loss": [],
        'train_acc': [],
        "test_acc": []
    }
    for epoch in epoch_loop:
        model.train()
        train_total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Итерация по батчам(трейн)", leave=False):
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                raise ValueError("Для Fine-tuning нужны метки классов!")
    
            optimizer.zero_grad()
            
            logits, _ = model(imgs)
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item()
            
            # Точность (Accuracy)
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        test_total_loss = 0
        test_correct = 0
        test_total = 0
        for batch in tqdm(test_loader, desc="Итерация по батчам(тест)", leave=False):
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                raise ValueError("Для Fine-tuning нужны метки классов!")
    
            with torch.no_grad():
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
            
            test_total_loss += loss.item()
            

            _, predicted = torch.max(logits.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
        
        
            
        epoch_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        epoch_loop.set_postfix(
            train_loss=train_total_loss/len(train_loader),
            train_acc=train_correct/train_total,
            test_loss=test_total_loss/len(test_loader),
            test_acc=test_correct/test_total
        )

        history['train_loss'].append(train_total_loss/len(train_loader))
        history['train_acc'].append(train_correct/train_total)
        history['test_loss'].append(test_total_loss/len(test_loader))
        history['test_acc'].append(test_correct/test_total)

    return model, history