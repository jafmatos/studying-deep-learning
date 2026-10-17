# 📗 Guia Técnico: TorchVision para Visão Computacional (Os 20% Fundamentais)
> **Projeto Âncora:** Diagnóstico de Patologias em Folhas e Culturas Agrícolas (AgroVision)  
> **Objetivo:** Dominar Data Augmentation estratégico para dados de campo, Transfer Learning com **ResNet** e **MobileNet**, interpretabilidade com Grad-CAM e a transição para detecção com **YOLO**.

---

## 1. Organização do Dataset e `datasets.ImageFolder`

Em projetos reais de pesquisa com culturas agrícolas, as fotos coletadas no campo são organizadas em pastas com os nomes das condições patológicas:

```text
dataset_folhas/
├── treino/ (70% das fotos)
│   ├── saudavel/
│   ├── ferrugem/
│   └── mancha_bacteriana/
├── validacao/ (15% das fotos - para ajuste de hiperparâmetros)
│   ├── saudavel/
│   ├── ferrugem/
│   └── mancha_bacteriana/
└── teste/ (15% das fotos - nunca vistas pelo modelo)
    ├── saudavel/
    ├── ferrugem/
    └── mancha_bacteriana/
```

O `ImageFolder` mapeia essa hierarquia de diretórios diretamente para números inteiros (labels) em ordem alfabética:

```python
from torchvision import datasets

dataset_treino = datasets.ImageFolder(root="dataset_folhas/treino", transform=transformacoes_treino)

print("Nomes das Classes:", dataset_treino.classes)
# ['ferrugem', 'mancha_bacteriana', 'saudavel']

print("Dicionário de Classes para IDs:", dataset_treino.class_to_idx)
# {'ferrugem': 0, 'mancha_bacteriana': 1, 'saudavel': 2}
```

---

## 2. Data Augmentation Estratégico para Imagens de Campo

Em agronomia e biologia, raramente temos 100.000 imagens rotuladas. Geralmente temos algumas centenas. Se treinarmos uma rede profunda com poucas imagens, ela decorará as fotos em vez de aprender as patologias (**Overfitting brutal**).

O **Data Augmentation** simula as variações reais da lavoura a cada época de treinamento:

```python
from torchvision import transforms

# 1. Pipeline de TREINO: Aleatório e Estocástico (cada época a imagem parece diferente)
transformacoes_treino = transforms.Compose([
    # A) Recorte com Zoom Aleatório: força a rede a reconhecer a doença tanto de perto quanto de longe
    transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
    
    # B) Espelhamento Horizontal: folhas não têm lado fixo na natureza
    transforms.RandomHorizontalFlip(p=0.5),
    
    # C) Rotação Suave: simula a câmera do agrônomo inclinada na lavoura
    transforms.RandomRotation(degrees=20),
    
    # D) Variações Fotométricas: simula sol do meio-dia, sombra de nuvem e câmeras com sensores diferentes
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
    
    # E) Conversão para Tensor [0.0, 1.0] no formato (C, H, W)
    transforms.ToTensor(),
    
    # F) Normalização com os padrões do ImageNet (Média e Desvio Padrão)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 2. Pipeline de VALIDAÇÃO E TESTE: Estritamente Determinístico!
# NUNCA aplique rotações ou crops aleatórios no conjunto de validação/teste.
# A avaliação precisa ser reprodutível e fiel à imagem original.
transformacoes_teste = transforms.Compose([
    transforms.Resize(256),             # Redimensiona mantendo proporção
    transforms.CenterCrop(224),         # Recorta os 224x224 pixels centrais
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

---

## 3. Transfer Learning em Profundidade: O Segredo da Pesquisa Científica

Treinar uma rede convolucional do zero exige milhões de imagens e semanas de processamento. No **Transfer Learning**, aproveitamos redes treinadas na base de dados **ImageNet** (mais de 1,2 milhão de imagens distribuídas em 1.000 categorias).

### A Hierarquia Visual das Camadas:
1. **Camadas Iniciais (Conv1, Conv2):** Aprendem conceitos primitivos universais (bordas retas, diagonais, gradientes de luz, cores básicas).
2. **Camadas Intermediárias (Conv3, Conv4):** Aprendem texturas complexas (rugosidade, estrias foliares, pontilhados de ferrugem).
3. **Camadas Finais (Conv5, FC):** Aprendem partes inteiras de objetos e tomam a decisão final.

---

### Estratégia A: Feature Extraction (Congelamento Total)
Congelamos **todas as camadas convolucionais** e treinamos apenas a nova camada linear final:

```python
from torchvision import models
import torch.nn as nn

# 1. Carregar a ResNet18 com os pesos mais recentes
modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Congelar os parâmetros (desativa o cálculo de gradientes para as convoluções)
for param in modelo.parameters():
    param.requires_grad = False

# 3. Substituir a cabeça final (fc = Fully Connected layer)
# A ResNet18 possui 512 neurônios na penúltima camada
num_features = modelo.fc.in_features # 512
modelo.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 3) # 3 classes de patologia da folha
)

# Vantagem: Treina em minutos mesmo na CPU! Impossível estragar os pesos prévios.
```

---

### Estratégia B: Fine-Tuning Progressivo (O Salto para >95% de Acurácia)
Após treinar a cabeça final por algumas épocas, descongelamos o **último bloco convolucional** da ResNet (`layer4`) para que ela refine seus extratores de textura para as lesões agrícolas específicas:

```python
# 1. Descongelar o último bloco da ResNet (layer4)
for param in modelo.layer4.parameters():
    param.requires_grad = True

# 2. O PULO DO GATO: Otimizador com Taxas de Aprendizado Diferenciadas!
# Usamos uma taxa minúscula na base convolucional (para não esquecer o ImageNet)
# e uma taxa maior na camada de classificação:
optimizer = torch.optim.Adam([
    {'params': modelo.layer4.parameters(), 'lr': 1e-5}, # Taxa muito baixa (Ajuste fino)
    {'params': modelo.fc.parameters(),     'lr': 1e-3}  # Taxa normal
])
```

---

## 4. Comparativo de Arquiteturas para a Seleção de IC

O orientador certamente perguntará sobre a escolha entre **ResNet** e **MobileNet**:

| Característica | ResNet18 / ResNet50 | MobileNetV3 (Small/Large) |
| :--- | :--- | :--- |
| **Principal Inovação** | Conexões Residuais (*Skip Connections*) | Convoluções Separáveis (*Depthwise Separable*) |
| **Tamanho em Disco** | ~45 MB (ResNet18) a ~100 MB (ResNet50) | ~10 MB (MobileNetV3-Small) |
| **Número de Parâmetros**| ~11.7 Milhões | ~2.5 Milhões (4x menos parâmetros!) |
| **Cenário de Aplicação** | Servidor, laboratório, alta precisão de diagnóstico | Aplicativo de celular do agricultor, drone ou Raspberry Pi |
| **Como adaptar a cabeça**| `modelo.fc = nn.Linear(512, num_classes)` | `modelo.classifier[3] = nn.Linear(1024, num_classes)` |

### Código de Adaptação da MobileNetV3:
```python
modelo_mobile = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# Congelar
for param in modelo_mobile.parameters():
    param.requires_grad = False

# Na MobileNetV3, o classificador é um nn.Sequential. A última camada é o índice 3:
in_features = modelo_mobile.classifier[3].in_features # 1024
modelo_mobile.classifier[3] = nn.Linear(in_features, 3)
```

---

## 5. Interpretabilidade Científica: Grad-CAM (Onde a Rede Olha?)

Em avaliações técnicas e bancas de pesquisa, uma pergunta crucial é:  
> *"Como você sabe que a sua rede aprendeu a identificar o fungo da folha e não a terra no fundo ou a mão de quem segurava a planta?"*

A técnica padrão da ciência para responder isso é o **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Ele calcula a derivada das predições em relação aos mapas de ativação do último bloco convolucional (`layer4`) para gerar um **mapa de calor** sobre a imagem:

```
[ Imagem Original ]   +   [ Grad-CAM Heatmap ]   ==>   [ Diagnóstico Visual Auditável ]
 Foto da folha             Áreas vermelhas onde         Confirma que a atenção
 com mancha amarela        o modelo focou a atenção     está 100% sobre a lesão foliar!
```

* Se o mapa de calor acender sobre a folha $\to$ **Modelo confiável cientificamente.**
* Se o mapa acender sobre a madeira da mesa ou o céu $\to$ **Modelo viciado (*Clever Hans effect*), precisa de mais Data Augmentation.**

---

## 6. Pipeline de Produção para Streamlit

Como integrar o modelo treinado com TorchVision na interface do Streamlit para inferência em milissegundos:

```python
import torch
from PIL import Image

def inferir_imagem_folha(caminho_ou_upload, modelo, classes, device):
    """Executa a inferência de ponta a ponta e retorna classes e probabilidades."""
    modelo.eval()
    
    # 1. Carrega imagem PIL
    img = Image.open(caminho_ou_upload).convert('RGB')
    
    # 2. Aplica as transformações determinísticas de teste
    tensor_img = transformacoes_teste(img).unsqueeze(0).to(device) # Shape: (1, 3, 224, 224)
    
    # 3. Inferência com Softmax para probabilidades percentuais
    with torch.no_grad():
        logits = modelo(tensor_img)
        probabilidades = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    idx_campeao = probabilidades.argmax()
    resultado = {
        "diagnostico": classes[idx_campeao],
        "confianca": float(probabilidades[idx_campeao] * 100),
        "todas_probabilidades": {classes[i]: float(probabilidades[i] * 100) for i in range(len(classes))}
    }
    return resultado
```

---

## 7. A Transição para o YOLO (Classificação vs. Detecção)

Quando o orientador pedir para quantificar o dano da planta, a tarefa muda de nível:

* **Classificação (ResNet / MobileNet via TorchVision):**
  * Pergunta: *"Qual é a patologia predominante nesta folha?"*
  * Saída: `Ferrugem (96.4% de probabilidade)`.
* **Detecção de Objetos (YOLO):**
  * Pergunta: *"Onde e quantas pústulas de ferrugem existem nesta folha?"*
  * Saída: `5 Caixas Delimitadoras [x, y, w, h]` com as coordenadas de cada lesão individual.
  * Aplicação: Permite calcular a **Severidade Foliar** ($\text{Área das Lesões} / \text{Área Total da Folha} \times 100$).

---

## 📌 Checklist do Pesquisador em Visão Computacional (TorchVision 80/20)

| Problema | Solução TorchVision | Justificativa Técnica |
| :--- | :--- | :--- |
| **Poucas fotos de campo** | `RandomResizedCrop` + `ColorJitter` | Evita overfitting forçando invariância de escala e iluminação |
| **Avaliação injusta/ruidosa** | Pipeline determinístico (`Resize` + `CenterCrop`) | Garante métricas reprodutíveis no conjunto de teste |
| **Treinar do zero é inviável** | `models.resnet18(weights="DEFAULT")` | Herda extratores de bordas e formas pré-treinados no ImageNet |
| **Modelo muito pesado para app** | `models.mobilenet_v3_small` | Convoluções separáveis reduzem o peso para ~10 MB |
| **Ajuste para novo dataset** | Congelar base convolucional e trocar `model.fc` | Transfere aprendizado sem destruir os pesos originais |
