"""
02_treino_classificador.py
Treinamento de uma Rede Neural Convolucional (CNN) completa usando PyTorch.
Classifica imagens do dataset Fashion-MNIST (10 classes de roupas/calçados).
Executa rápido na CPU (cerca de 30-40 segundos) e salva os pesos treinados.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Definição da Arquitetura da Rede Neural (CNN)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Bloco Convolucional 1: Entrada 1 canal (cinza) -> 16 canais
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        )
        # Bloco Convolucional 2: 16 canais -> 32 canais
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )
        # Classificador (Fully Connected / Linear)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x

def main():
    print("=" * 65)
    print(" TREINAMENTO DE UMA CNN EM VISÃO COMPUTACIONAL (Fashion-MNIST)")
    print("=" * 65)

    # Definir dispositivo: GPU (se configurada) ou CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo em uso: {device}")

    # 2. Pré-processamento e Carga de Dados (Transformações e DataLoaders)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Média 0.5, Desvio 0.5 -> [-1, 1]
    ])

    print("[*] Baixando/carregando dataset Fashion-MNIST...")
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"[*] Total amostras de Treino: {len(train_dataset)} | Teste: {len(test_dataset)}")

    # 3. Instanciar Modelo, Critério de Perda e Otimizador
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # 4. Loop de Treinamento (3 Épocas para demonstração rápida)
    epochs = 3
    print(f"\n[*] Iniciando treinamento por {epochs} épocas...\n")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        print(f" -> Época [{epoch}/{epochs}] - Perda: {epoch_loss:.4f} | Acurácia Treino: {epoch_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n[*] Treinamento concluído em {elapsed:.1f} segundos.")

    # 5. Avaliação no Conjunto de Teste
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = (test_correct / test_total) * 100
    print(f"[*] Acurácia Final no Conjunto de Teste: {test_acc:.2f}%\n")

    # 6. Salvar os Pesos do Modelo
    model_path = "modelo_cnn_fashion.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Pesos do modelo salvos com sucesso em: {model_path}")
    print("=" * 65)

if __name__ == '__main__':
    main()
