# 📗 Guia Técnico: TorchVision para Visão Computacional

> **Objetivo:** Dominar a biblioteca oficial **TorchVision** para carregamento prático de imagens (`ImageFolder`), pipelines de aumento de dados (**Data Augmentation**), reutilização de redes consagradas (**Transfer Learning** com ResNet e MobileNet) e interpretabilidade visual (**Grad-CAM**).
> **Pré-requisitos assimilados:** Fundamentos de tensores e derivadas ([GUIA 00](GUIA_00_FUNDAMENTOS_TEORICOS.md)), intuição de convoluções ([CS231n](CS231N_CONVOLUCOES_RESUMO.md)) e o ciclo de treino em PyTorch ([GUIA 02](GUIA_02_PYTORCH_CV.md)).

---

# 1. Carregamento Automático com `datasets.ImageFolder`

### O que este módulo faz?

Em vez de escrever loops manuais para ler pastas e mapear arquivos com código customizado, o TorchVision possui o `datasets.ImageFolder`. Ele inspeciona automaticamente a estrutura de pastas do disco e cria um `Dataset` pronto para consumo.

### Como as pastas devem ser organizadas no disco:

O `ImageFolder` espera que as fotos estejam divididas em subpastas cujos nomes representam os rótulos das classes:

```text
dataset_folhas/
├── treino/
│   ├── ferrugem/             <-- todas as fotos de ferrugem
│   ├── mancha_bacteriana/    <-- todas as fotos de mancha bacteriana
│   └── saudavel/             <-- todas as fotos de folhas saudáveis
└── validacao/
    ├── ferrugem/
    ├── mancha_bacteriana/
    └── saudavel/
```

### Código e Explicação Detalhada:

```python
from torchvision import datasets

# Cria o dataset lendo a hierarquia de pastas
dataset_treino = datasets.ImageFolder(
    root="dataset_folhas/treino", 
    transform=transformacoes_treino
)

# 1. Lista com os nomes das classes ordenados alfabeticamente:
print("Classes encontradas:", dataset_treino.classes)
# Saída: ['ferrugem', 'mancha_bacteriana', 'saudavel']

# 2. Mapeamento de texto para IDs inteiros que a rede entende:
print("Mapeamento de IDs:", dataset_treino.class_to_idx)
# Saída: {'ferrugem': 0, 'mancha_bacteriana': 1, 'saudavel': 2}
```

#### O que cada parâmetro determina:

* `root="dataset_folhas/treino"`: O caminho do diretório raiz onde estão as subpastas das classes.
* `transform=transformacoes_treino`: O pipeline de transformações (redimensionamento, conversão para tensor e normalização) que será aplicado individualmente a cada foto quando o lote for montado.

---

# 2. Data Augmentation (Aumento de Dados) Estratégico

### 1. O que é e por que é necessário?

Em problemas reais, raramente temos centenas de milhares de fotos rotuladas. Se treinarmos uma rede profunda com poucas imagens estáticas, ela decorará as fotos em vez de aprender os traços visuais da doença (**Overfitting**).

O **Data Augmentation** resolve isso alterando a imagem dinamicamente a cada época na memória: a foto sofre pequenos cortes, rotações e variações de cor aleatórias. A rede nunca vê a mesmíssima imagem duas vezes seguidas, forçando-a a aprender características invariantes.

---

### 2. O Pipeline de Treino (Aleatório & Estocástico)

```python
from torchvision import transforms

transformacoes_treino = transforms.Compose([
    # A) Recorte com zoom aleatório e redimensionamento
    transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
  
    # B) Espelhamento horizontal aleatório
    transforms.RandomHorizontalFlip(p=0.5),
  
    # C) Rotação angular suave
    transforms.RandomRotation(degrees=20),
  
    # D) Variações fotométricas de luz e cor
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
  
    # E) Conversão de PIL para Tensor PyTorch
    transforms.ToTensor(),
  
    # F) Normalização pelos parâmetros do ImageNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

#### O que cada método e seus parâmetros determinam:

* **`transforms.Compose([...])`**:
  * *O que faz:* Encadeia uma lista de operações de transformação em sequência. A imagem entra na primeira função e o resultado vai para a seguinte.
* **`transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0))`**:
  * *O que faz:* Recorta uma área aleatória da foto e a redimensiona para um tamanho fixo. Isso simula a câmera tirando fotos tanto de perto quanto de longe (invariância de escala).
  * `size=224`: A resolução final da imagem quadrada ($224 \times 224$ pixels).
  * `scale=(0.7, 1.0)`: O intervalo de área recortada. O valor `(0.7, 1.0)` significa que o recorte conterá entre 70% e 100% da área original da imagem.
* **`transforms.RandomHorizontalFlip(p=0.5)`**:
  * *O que faz:* Inverte a imagem horizontalmente como em um espelho. Em imagens naturais (como folhas), a patologia pode estar no lado esquerdo ou direito indiferentemente.
  * `p=0.5`: Probabilidade de 50% de o espelhamento ocorrer a cada chamada.
* **`transforms.RandomRotation(degrees=20)`**:
  * *O que faz:* Gira a imagem em torno do seu centro para simular fotos tiradas em ângulos inclinados.
  * `degrees=20`: Sorteia um ângulo entre $-20^\circ$ e $+20^\circ$.
* **`transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2)`**:
  * *O que faz:* Altera aleatoriamente os valores de cor e iluminação da imagem para simular dias nublados, sol do meio-dia ou câmeras com sensores diferentes.
  * `brightness=0.25`: Altera o brilho em até $\pm 25\%$.
  * `contrast=0.25`: Altera o contraste em até $\pm 25\%$.
  * `saturation=0.2`: Altera a vivacidade das cores em até $\pm 20\%$.
* **`transforms.ToTensor()`**:
  * *O que faz:* Uma das funções mais importantes do TorchVision. Ela executa duas conversões obrigatórias:
    1. Muda a ordem dos eixos de PIL $(H, W, C)$ para PyTorch $(C, H, W)$.
    2. Converte os pixels que eram inteiros no intervalo $[0, 255]$ para números em ponto flutuante `float32` no intervalo normalizado de $[0.0, 1.0]$.
* **`transforms.Normalize(mean=[...], std=[...])`**:
  * *O que faz:* Aplica a padronização matemática canal por canal:
    $$
    \text{pixel}_{\text{normalizado}} = \frac{\text{pixel} - \text{média}}{\text{desvio\_padrão}}
    $$
  * *Por que esses valores específicos?*
    `mean=[0.485, 0.456, 0.406]` e `std=[0.229, 0.224, 0.225]` são as médias e desvios exatos dos canais R, G e B do dataset **ImageNet** (1,2 milhão de imagens). Redes pré-treinadas esperam obrigatoriamente que as imagens de entrada estejam nessa mesma escala estatística para que seus filtros funcionem corretamente.

---

### 3. O Pipeline de Teste e Validação (Estritamente Determinístico)

> **Regra de Ouro:** NUNCA aplique rotações, espelhamentos ou cortes aleatórios nos dados de validação ou teste! A validação deve ser reprodutível e avaliar a foto original sem distorções estocásticas.

```python
transformacoes_teste = transforms.Compose([
    # 1. Redimensiona o menor lado da foto para 256 pixels mantendo a proporção original
    transforms.Resize(256),
  
    # 2. Recorta os 224x224 pixels centrais exatos
    transforms.CenterCrop(224),
  
    # 3. Converte para Tensor [0.0, 1.0] no formato (C, H, W)
    transforms.ToTensor(),
  
    # 4. Mesma normalização do treino
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

#### Por que `Resize(256)` seguido de `CenterCrop(224)`?

Se redimensionarmos diretamente uma foto retangular para $224 \times 224$, a imagem é espremida (*distorção de aspecto*).
O padrão clássico redimensiona a imagem proporcionalmente até 256 pixels e faz um recorte limpo do miolo de $224 \times 224$, preservando a geometria natural dos objetos.

---

# 3. Transfer Learning (Aprendizado por Transferência)

Treinar uma rede convolucional profunda do zero exige milhões de imagens rotuladas e semanas de processamento. No **Transfer Learning**, reaproveitamos uma rede já treinada no ImageNet.

### Como a rede "enxerga" em camadas:

* **Camadas Iniciais:** Detectam bordas, linhas e gradientes de luz (universais para qualquer imagem).
* **Camadas Intermediárias:** Detectam texturas, pontilhados e rugosidades.
* **Camadas Finais:** Detectam formas específicas dos objetos originais do ImageNet.

Substituímos apenas a última camada para adaptar a rede ao nosso problema de classificação.

---

### Abordagem A: Feature Extraction (Congelamento Total)

Congelamos **todas as camadas convolucionais** (mantendo os pesos originais do ImageNet travados) e treinamos apenas uma nova cabeça linear:

```python
from torchvision import models
import torch.nn as nn

# 1. Carrega a ResNet18 com os melhores pesos pré-treinados
modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. CONGELAMENTO: Desativa o cálculo de gradientes em todas as camadas
for param in modelo.parameters():
    param.requires_grad = False

# 3. SUBSTITUIÇÃO DA CAMADA FINAL (fc = Fully Connected layer)
# Consultamos quantos valores a penúltima camada da ResNet18 entrega (512)
num_features = modelo.fc.in_features 

# Substituímos a camada original de 1000 classes por uma nova com 3 classes
modelo.fc = nn.Sequential(
    nn.Linear(in_features=num_features, out_features=128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=128, out_features=3) # 3 classes do nosso problema
)
```

#### O que cada método e propriedade faz:

* `models.ResNet18_Weights.DEFAULT`: Instrui o PyTorch a baixar automaticamente a versão mais recente e precisa dos pesos treinados no ImageNet.
* `param.requires_grad = False`:
  * *O que faz:* Define que o PyTorch **não deve calcular derivadas** para aquele peso durante o `loss.backward()`.
  * *Resultado prático:* Os pesos da base convolucional ficam imunes a alterações e a velocidade de treinamento aumenta drasticamente, exigindo pouquíssima VRAM.
* `modelo.fc.in_features`:
  * *O que faz:* Lê dinamicamente o número de conexões que chegam na camada linear final da ResNet (na ResNet18 são 512). Isso evita colocar números "chumbados" no código.

---

### Abordagem B: Fine-Tuning Progressivo (Ajuste Fino de Alta Precisão)

Após treinar a camada linear final por algumas épocas, podemos destravar o **último bloco convolucional** da ResNet (`layer4`) para que ele ajuste suas texturas específicas para o nosso problema:

```python
# 1. Destrava apenas o último bloco convolucional (layer4)
for param in modelo.layer4.parameters():
    param.requires_grad = True

# 2. OTIMIZADOR COM TAXAS DE APRENDIZADO DIFERENCIADAS (Learning Rate Splitting)
optimizer = torch.optim.Adam([
    # Taxa minúscula no bloco convolucional (para não destruir os filtros prévios):
    {'params': modelo.layer4.parameters(), 'lr': 1e-5},
  
    # Taxa padrão na cabeça de classificação:
    {'params': modelo.fc.parameters(),     'lr': 1e-3}
])
```

#### Como funciona passar uma lista de dicionários para o otimizador?

O PyTorch permite definir grupos de parâmetros independentes. Em vez de aplicar a mesma taxa de aprendizado para a rede toda, usamos:

* `1e-5` ($0.00001$) na base convolucional: faz ajustes microscópicos nos filtros já treinados.
* `1e-3` ($0.001$) na camada linear: treina com velocidade normal os pesos recém-criados.

---

# 4. Escolhendo a Arquitetura: ResNet vs. MobileNet

| Característica                  | ResNet-18                                   | MobileNetV3-Small                                   |
| :------------------------------- | :------------------------------------------ | :-------------------------------------------------- |
| **Inovação Técnica**    | Conexões Residuais (*Skip Connections*)  | Convoluções Separáveis (*Depthwise Separable*) |
| **Tamanho do Arquivo**     | ~45 MB                                      | ~10 MB                                              |
| **Número de Parâmetros** | ~11.7 milhões                              | ~2.5 milhões (mais de 4x mais leve!)               |
| **Melhor Caso de Uso**     | Treinamento robusto em desktop / servidores | Dispositivos de borda, celulares e Raspberry Pi     |

### Como adaptar a cabeça da MobileNetV3:

Na ResNet, a camada final se chama `modelo.fc`. Na MobileNetV3, o classificador final é um `nn.Sequential` chamado `classifier`, e a camada de saída é o elemento de índice 3:

```python
modelo_mobile = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# 1. Congelar toda a rede
for param in modelo_mobile.parameters():
    param.requires_grad = False

# 2. Descobrir o número de entradas da última camada (índice 3 do classificador)
in_features = modelo_mobile.classifier[3].in_features # 1024

# 3. Substituir apenas a camada final
modelo_mobile.classifier[3] = nn.Linear(in_features=in_features, out_features=3)
```

---

# 5. Interpretabilidade Visual: Grad-CAM (Onde a Rede Olha?)

### Por que isto é fundamental?

Redes neurais são frequentemente criticadas como "caixas-pretas". Se você treina um modelo para identificar doenças em folhas, como ter certeza de que a rede aprendeu o formato do fungo, e não a cor do solo no fundo ou a mão de quem segurava a planta?

A técnica padrão para auditar redes convolucionais é o **Grad-CAM (Gradient-weighted Class Activation Mapping)**:

```
[ Imagem de Entrada ]  ──►  [ Rede Convolucional ]  ──►  [ Gradientes do Último Bloco ]
                                                                      │
                                                                      ▼
                                                            [ Mapa de Calor (Heatmap) ]
```

* **Como interpretar o mapa de calor:**
  * Áreas em **vermelho/amarelo**: Regiões com maior peso na tomada de decisão da rede. Se focarem na lesão $\to$ o modelo aprendeu o padrão correto.
  * Áreas em **azul**: Regiões ignoradas pelo modelo.

---

# 6. Pipeline Completo de Predição para Aplicações Web

Abaixo está a função pronta para receber o upload de uma foto em uma interface (como Streamlit) e devolver o diagnóstico percentual:

```python
import torch
from PIL import Image

def inferir_imagem(caminho_ou_arquivo, modelo, classes, transform, device):
    """
    Executa a inferência de ponta a ponta em uma imagem avulsa.
    Retorna o nome da classe prevista e a confiança percentual.
    """
    # 1. Coloca a rede em modo determinístico
    modelo.eval()
  
    # 2. Abre a imagem e garante 3 canais RGB
    imagem = Image.open(caminho_ou_arquivo).convert("RGB")
  
    # 3. Aplica o pipeline determinístico de teste e adiciona a dimensão de lote com unsqueeze(0)
    # Shape resultante: (1, 3, 224, 224)
    tensor_batch = transform(imagem).unsqueeze(0).to(device)
  
    # 4. Desativa gravação de derivadas
    with torch.no_grad():
        logits = modelo(tensor_batch)
      
        # Converte logits em probabilidades percentuais de 0.0 a 1.0
        # .squeeze(0): remove a dimensão de lote -> vetor 1D
        # .cpu(): move o tensor da memória da GPU para a RAM
        # .numpy(): converte o tensor em array do NumPy
        probabilidades = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
      
    # Obtém o índice da classe com a maior probabilidade
    idx_vencedor = probabilidades.argmax()
  
    return {
        "classe_prevista": classes[idx_vencedor],
        "confianca": float(probabilidades[idx_vencedor] * 100),
        "todas_probabilidades": {
            classes[i]: float(probabilidades[i] * 100) 
            for i in range(len(classes))
        }
    }
```

#### O que os métodos de conversão no passo 4 fazem:

* **`torch.softmax(logits, dim=1)`**: Converte pontuações arbitrárias em probabilidades que somam $1.0$ ($100\%$) ao longo das classes (`dim=1`).
* **`.squeeze(0)`**: Remove a dimensão unitária de lote que criamos com `unsqueeze(0)`, transformando a matriz $(1, 3)$ em um vetor unidimensional de 3 números.
* **`.cpu()`**: **Obrigatório antes do NumPy.** A biblioteca NumPy opera exclusivamente na memória RAM da CPU; se tentarmos converter um tensor que reside na VRAM da GPU direto para NumPy, o PyTorch levantará uma exceção. O método `.cpu()` realiza essa transferência de hardware.
* **`.numpy()`**: Transforma o tensor numérico em uma estrutura nativa do NumPy.
* **`probabilidades.argmax()`**: Devolve o índice inteiro do maior elemento do array.

---

# GUIA DE BOLSO: TorchVision 80/20

| Situação                                  | Método TorchVision                                      | O que ele resolve                                                     |
| :------------------------------------------ | :------------------------------------------------------- | :-------------------------------------------------------------------- |
| **Pastas organizadas por classe**     | `datasets.ImageFolder(root, transform)`                | Lê diretórios e gera IDs inteiros automaticamente.                  |
| **Poucas imagens no dataset**         | `RandomResizedCrop` + `ColorJitter`                  | Evita overfitting gerando variações de escala e luz a cada época.  |
| **Avaliação justa e estável**      | `Resize(256)` + `CenterCrop(224)`                    | Avalia o centro da foto sem distorcer proporções espaciais.         |
| **Treinar rede do zero é inviável** | `models.resnet18(weights="DEFAULT")`                   | Herda extratores treinados em 1,2 milhão de imagens do ImageNet.     |
| **Congelar camadas pré-treinadas**   | `for p in model.parameters(): p.requires_grad = False` | Impede o cálculo de derivadas, preservando os pesos da base.         |
| **Trocar saída para novo problema**  | `modelo.fc = nn.Linear(512, num_classes)`              | Adapta a camada de decisão final para o número de classes desejado. |
| **Modelo leve para celular/borda**    | `models.mobilenet_v3_small`                            | Convoluções separáveis reduzem o peso da rede para ~10 MB.         |
