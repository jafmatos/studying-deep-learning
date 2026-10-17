# 📙 Guia Técnico: PyTorch para Visão Computacional (Os 20% Fundamentais)
> **Projeto Âncora:** Diagnóstico de Patologias em Folhas e Culturas Agrícolas (AgroVision)  
> **Objetivo:** Dominar a arquitetura convolucional, a organização de software modular, o fluxo de autograd e o treinamento científico em PyTorch com rigor acadêmico.

---

## 1. Engenharia de Software: Estrutura Profissional de um Projeto em PyTorch

Projetos amadores de Deep Learning costumam sofrer da *"Síndrome do Jupyter Notebook"*: misturam carregamento de arquivos, definição de redes neurais, loop de treinamento e interface gráfica em um único arquivo gigantesco ou copiam e colam classes de rede de um script para o outro.

Em projetos profissionais e de pesquisa séria, aplicamos o princípio de **Separação de Responsabilidades (SoC - Separation of Concerns)** e o princípio **DRY (*Don't Repeat Yourself*)** com uma arquitetura modular limpa:

```text
agrovision/
│
├── config.py                 # 1. Configurações Globais e Hiperparâmetros
│                             #    (LR, Batch Size, Épocas, Caminhos, Seed, Device)
│
├── src/                      # 2. Pacote Central (Lógica Reutilizável)
│   ├── __init__.py
│   │
│   ├── models/               # Camada de Modelos (Arquiteturas Puras)
│   │   ├── __init__.py
│   │   ├── simple_cnn.py     # Definição ÚNICA da SimpleCNN (nn.Module)
│   │   └── resnet_custom.py  # Adaptações de Transfer Learning
│   │
│   ├── data/                 # Camada de Dados (I/O e Pré-Processamento)
│   │   ├── __init__.py
│   │   ├── dataset.py        # Classes CustomDataset e Loaders
│   │   └── transforms.py     # Pipelines de Data Augmentation
│   │
│   └── engine/               # Camada de Execução (O Motor de Treino e Predição)
│       ├── __init__.py
│       ├── trainer.py        # Funções train_one_epoch() e evaluate()
│       └── predictor.py      # Função limpa de inferência para produção
│
├── train.py                  # 3. Ponto de Entrada de Treino (Orquestrador CLI)
├── app.py                    # 4. Ponto de Entrada da Interface (Streamlit)
│                             #    (Apenas importa de src/ e desenha a tela)
└── requirements.txt
```

### As 4 Regras de Ouro dessa Arquitetura:
1. **`models/` não sabe de onde vêm os dados:** A classe `nn.Module` só recebe um Tensor `(B, C, H, W)` e devolve Logits. Ela não abre arquivos, não lê CSV e não sabe o que é Streamlit.
2. **`data/` não sabe qual modelo vai rodar:** Ele só se preocupa em ler imagens do disco, aplicar transformações (`transforms`) e entregar lotes organizados via `DataLoader`.
3. **`engine/` é o maestro:** Ele importa o modelo de `models/`, os dados de `data/`, executa o loop `forward -> loss -> backward -> step` e salva o arquivo de pesos `.pth`.
4. **`app.py` nunca redeclara redes neurais:** Ele apenas importa a classe de `src.models` e a função de predição de `src.engine.predictor`. Se você alterar a rede amanhã, o `app.py` continua funcionando sem precisar mexer em uma única linha!

---

## 2. O Motor Matemático: Tensores, Autograd e Dispositivos

O PyTorch se diferencia de bibliotecas de álgebra linear clássica por duas propriedades:
1. **Aceleração em Hardware:** Tensores podem ser movidos instantaneamente entre CPU e GPU/VRAM.
2. **Autograd (Diferenciação Automática):** Constrói dinamicamente um Grafo Computacional Direcionado Acíclico (DAG) para calcular derivadas parciais ($\nabla L$) via regra da cadeia com uma única chamada: `loss.backward()`.

```python
import torch

# 1. Alocação dinâmica de dispositivo (Device Agnostic Code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando no dispositivo: {device}")

# 2. Tensores de Entrada para Visão:
# O PyTorch EXIGE 4 dimensões no formato: (Batch, Canais, Altura, Largura)
lote_folhas = torch.randn(16, 3, 224, 224, device=device) # Lote de 16 imagens RGB

# 3. Gerenciamento de Memória (Crucial para não dar 'CUDA Out of Memory')
# Na inferência ou validação, SEMPRE desative o grafo de derivadas:
with torch.no_grad(): # Economiza até 70% de memória RAM/VRAM!
    predicoes = lote_folhas * 2.0
```

---

## 3. Aritmética e Intuição de Convoluções 2D (`nn.Conv2d`)

Em Machine Learning tradicional com dados tabulares, cada neurônio possui um peso para cada coluna. Se aplicássemos uma camada densa (`Linear`) diretamente em uma imagem $224 \times 224 \times 3$, seriam necessários mais de **150.000 pesos por neurônio**, explodindo o número de parâmetros e causando *overfitting* imediato.

A Convolução resolve isso com **Compartilhamento de Pesos** e **Invariância Espacial**: pequenos filtros matemáticos (*kernels*, geralmente $3 \times 3$) deslizam pela imagem procurando características locais (bordas, texturas de fungos, veios de folhas).

```
Kernel 3x3 deslizando sobre a matriz de pixels:
[ 1  0 -1 ]
[ 2  0 -2 ]  *  Sub-região 3x3 da folha  ==>  Soma ponderada  ==>  1 pixel no mapa de saída
[ 1  0 -1 ]
```

### A Fórmula Fundamental da Resolução de Saída:
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
Onde:
* $W$ = Dimensão de entrada (ex: $224$).
* $K$ = Tamanho do Kernel (ex: $3$).
* $P$ = *Padding* (pixels de borda adicionados com valor zero).
* $S$ = *Stride* (passo com que o filtro desliza).

> **A Regra de Ouro do Padding:** Para manter o tamanho da imagem idêntico após uma convolução com Kernel $3 \times 3$, use sempre **`padding=1`** e **`stride=1`**:
> $$O = \frac{224 - 3 + 2(1)}{1} + 1 = 224$$

---

## 4. O Arsenal de Camadas Convolucionais

```python
import torch.nn as nn

# 1. nn.Conv2d: O Extrator de Padrões
# in_channels=3 (RGB), out_channels=32 (aprenderá 32 filtros diferentes)
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

# 2. nn.BatchNorm2d: Acelerador e Estabilizador
# Normaliza as ativações de cada canal para média 0 e desvio 1 durante o treino.
# Evita que gradientes explodam ou sumam e permite taxas de aprendizado maiores.
bn = nn.BatchNorm2d(num_features=32)

# 3. nn.ReLU: A Não-Linearidade
# f(x) = max(0, x). Zera valores negativos e ativa apenas padrões relevantes.
relu = nn.ReLU(inplace=True)

# 4. nn.MaxPool2d: Redução Espacial
# Mantém o valor mais intenso em janelas 2x2. Corta Altura e Largura pela metade (224 -> 112).
# Reduz custo computacional e confere tolerância a pequenas mudanças de posição.
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 5. nn.AdaptiveAvgPool2d: O Segredo da Flexibilidade
# Transforma qualquer mapa de ativação em um tamanho fixo (ex: 1x1 ou 7x7)
# É o que permite redes modernas aceitarem imagens de tamanhos diferentes!
adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
```

---

## 5. Arquitetura Modular e Rastreamento de Dimensões (*Shape Tracking*)

Como saber exatamente qual número colocar no `in_features` da primeira camada `Linear` (`nn.Linear(?, 64)`)? 

Basta acompanhar a evolução dimensional do tensor camada por camada:

```
Entrada:                    (Batch, 1,  28, 28)
Conv1 (1 -> 16, pad=1):     (Batch, 16, 28, 28)
MaxPool1 (stride=2):        (Batch, 16, 14, 14)  <- Metade do tamanho
Conv2 (16 -> 32, pad=1):    (Batch, 32, 14, 14)
MaxPool2 (stride=2):        (Batch, 32,  7,  7)  <- Metade do tamanho
Flatten():                  (Batch, 32 * 7 * 7)  = (Batch, 1568)
Linear (1568 -> 64):        (Batch, 64)
```

Por isso, na linha de declaração da camada densa, escrevemos:  
`nn.Linear(32 * 7 * 7, 64)`. Esse cálculo não é um número mágico: é a multiplicação exata de **$\text{Canais} \times \text{Altura} \times \text{Largura}$** que saíram do último bloco convolucional!

```python
class CNNClassificadoraDeFolhas(nn.Module):
    def __init__(self, num_classes=3): # Ex: [0=Saudável, 1=Ferrugem, 2=Mancha-Bacteriana]
        super(CNNClassificadoraDeFolhas, self).__init__()
        
        # 1. Bloco Extrator Convolucional
        self.features = nn.Sequential(
            # Bloco 1: 3 canais -> 32 canais (224x224 -> 112x112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Bloco 2: 32 canais -> 64 canais (112x112 -> 56x56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloco 3: 64 canais -> 128 canais (56x56 -> 28x28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Reduz qualquer mapa espacial para 1x1 (Global Average Pooling)
        # Transforma (Batch, 128, 28, 28) em (Batch, 128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2. Bloco Classificador Final
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4), # Desliga 40% dos neurônios aleatoriamente no treino
            nn.Linear(64, num_classes) # Saída bruta (Logits)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits
```

---

## 6. Dataset e DataLoader com Otimização de Performance

Em projetos reais, os dados não cabem todos na memória RAM de uma vez. O `Dataset` ensina o PyTorch a ler **uma foto por vez do disco**, e o `DataLoader` cria os lotes em segundo plano usando múltiplos núcleos de CPU (*multiprocessing*).

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class FolhasCSVDataset(Dataset):
    """Lê um arquivo CSV contendo: caminho_foto,rotulo"""
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caminho = self.df.iloc[idx, 0]
        rotulo = int(self.df.iloc[idx, 1])
        
        # Carregamento sob demanda (Lazy Loading)
        img = Image.open(caminho).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, rotulo

# Configuração de Alta Performance do DataLoader:
# num_workers: usa núcleos paralelos da CPU para carregar o próximo lote enquanto a GPU treina
# pin_memory: aloca memória em páginas fixas para transferência ultrarrápida para a GPU
loader_treino = DataLoader(
    dataset_treino, 
    batch_size=32, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=(device.type == "cuda")
)
```

---

## 7. O Loop Científico de Treino, Validação e Diagnóstico de Curvas

Este é o padrão rigoroso de treino com cálculo de métricas e prevenção de overfitting (*Early Stopping*):

```python
import torch.optim as optim

modelo = CNNClassificadoraDeFolhas(num_classes=3).to(device)

# Tratamento de Classes Desbalanceadas (Típico em Agronomia):
# Se temos 800 folhas saudáveis, 150 com míldio e apenas 50 com ferrugem:
pesos_classes = torch.tensor([1.0, 5.3, 16.0]).to(device) # Inverso da frequência
criterion = nn.CrossEntropyLoss(weight=pesos_classes)

# Otimizador com decaimento de peso (Weight Decay = regularização L2)
optimizer = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)

# Redutor de Taxa de Aprendizado: Se a perda de validação estagnar por 2 épocas, corta o LR pela metade
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

def executar_treinamento(modelo, loader_treino, loader_val, epocas=10):
    melhor_loss_val = float('inf')

    for epoca in range(1, epocas + 1):
        # 1. MODO DE TREINO
        modelo.train() # Ativa Dropout e atualiza estatísticas do BatchNorm
        perda_acumulada_treino = 0.0
        acertos_treino, total_treino = 0, 0

        for imagens, labels in loader_treino:
            imagens, labels = imagens.to(device), labels.to(device)

            optimizer.zero_grad()            # 1. Zera resquícios de gradientes anteriores
            outputs = modelo(imagens)        # 2. Forward pass (Logits)
            loss = criterion(outputs, labels) # 3. Erro da época
            loss.backward()                  # 4. Cálculo das derivadas (Backprop)
            optimizer.step()                 # 5. Atualização dos pesos

            perda_acumulada_treino += loss.item() * imagens.size(0)
            _, pred = torch.max(outputs, 1)
            acertos_treino += (pred == labels).sum().item()
            total_treino += labels.size(0)

        loss_treino = perda_acumulada_treino / total_treino
        acc_treino = (acertos_treino / total_treino) * 100

        # 2. MODO DE VALIDAÇÃO (Sem cálculo de gradiente)
        modelo.eval() # Desativa Dropout e congela BatchNorm
        perda_acumulada_val = 0.0
        acertos_val, total_val = 0, 0

        with torch.no_grad():
            for imagens, labels in loader_val:
                imagens, labels = imagens.to(device), labels.to(device)
                outputs = modelo(imagens)
                loss = criterion(outputs, labels)

                perda_acumulada_val += loss.item() * imagens.size(0)
                _, pred = torch.max(outputs, 1)
                acertos_val += (pred == labels).sum().item()
                total_val += labels.size(0)

        loss_val = perda_acumulada_val / total_val
        acc_val = (acertos_val / total_val) * 100

        # Atualiza a taxa de aprendizado conforme a perda de validação
        scheduler.step(loss_val)

        print(f"Época [{epoca}/{epocas}] | "
              f"Treino Loss: {loss_treino:.4f} Acc: {acc_treino:.1f}% | "
              f"Val Loss: {loss_val:.4f} Acc: {acc_val:.1f}%")

        # Early Stopping / Salvamento do Melhor Modelo
        if loss_val < melhor_loss_val:
            melhor_loss_val = loss_val
            torch.save(modelo.state_dict(), "melhor_modelo_folhas.pth")
            print("  --> [SALVO] Novo melhor modelo alcançado!")
```

### Como Diagnosticar as Curvas na Apresentação Técnica:
* **Overfitting (Sobreajuste):** `Treino Loss` continua caindo ($0.30 \to 0.15 \to 0.05$), mas `Val Loss` começa a subir ($0.40 \to 0.55 \to 0.70$). A rede decorou as folhas de treino. *Solução: aumentar Dropout, aplicar mais Data Augmentation ou usar pesos pré-treinados.*
* **Underfitting (Subajuste):** Tanto `Treino Loss` quanto `Val Loss` permanecem altas. O modelo não tem capacidade de aprender. *Solução: aumentar número de camadas, treinar por mais épocas ou aumentar a taxa de aprendizado.*

---

## 📌 Checklist do Pesquisador em Visão Computacional (PyTorch 80/20)

| Problema | Solução PyTorch | Explicação Técnica |
| :--- | :--- | :--- |
| **Código duplicado e acoplado** | Arquitetura modular (`src/models`, `src/data`) | Isola modelo, dados e treino seguindo o princípio DRY |
| **Dimensões quebrando no Linear** | Rastrear shapes: $(\text{Canais} \times H \times W)$ | Determina o número exato de entradas do `in_features` |
| **Gradientes acumulando errado** | `optimizer.zero_grad()` | PyTorch por padrão soma gradientes a cada backward |
| **VRAM estourando na avaliação** | `with torch.no_grad():` | Desliga o grafo de derivadas na validação |
| **Dropout atuando no teste** | `modelo.eval()` | Garante predições determinísticas na inferência |
| **Imagens de tamanhos variados** | `nn.AdaptiveAvgPool2d((1,1))` | Padroniza a saída convolucional antes do classificador |
| **Classes agrícolas raras** | `nn.CrossEntropyLoss(weight=...)` | Dá penalidade maior para erros na classe minoritária |
| **Perda estagnando no platô** | `scheduler.step(loss_val)` | Reduz a taxa de aprendizado para ajuste fino dos pesos |
