# 📙 Guia Prático: PyTorch para Visão Computacional

> **Para quem é este guia:** Para quem está tendo o primeiro contato com o PyTorch. Nenhum método ou parâmetro é assumido como óbvio: para cada chamada de função, você verá **o que o método faz** e **o que cada parâmetro passado determina**.
> **Pré-requisitos assimilados:** Fundamentos de tensores e derivadas ([GUIA 00](GUIA_00_FUNDAMENTOS_TEORICOS.md)), intuição convolucional ([CS231n](CS231N_CONVOLUCOES_RESUMO.md)) e manipulação com PIL/NumPy ([GUIA 01](GUIA_01_NUMPY_CV.md)).

---

# PARTE 1: O "Script Plano" Passo a Passo

Antes de criar arquiteturas avançadas, precisamos ver as engrenagens do PyTorch rodando em um fluxo contínuo.

Vamos construir esse fluxo em **5 etapas sequenciais**. Em cada uma, responderemos:

1. *O que esta etapa faz?*
2. *Por que ela é necessária?*
3. *O que o código faz e o que cada parâmetro significa?*

---

## Etapa 1: Onde o código vai rodar? (Hardware & Dispositivo)

### 1. O que esta etapa faz?

Identifica se o computador tem uma placa de vídeo (GPU) com suporte a aceleração gráfica (como AMD ROCm ou NVIDIA CUDA) ou se usará o processador central (CPU).

### 2. Por que ela é necessária?

O computador possui duas memórias físicas separadas:

* **Memória RAM:** Acesso pela CPU.
* **Memória VRAM:** Acesso pela GPU.

O PyTorch **não adivinha** onde você quer executar as contas. Se você tentar multiplicar um modelo alocado na GPU por dados que estão na memória da CPU, o programa trava com o erro:
`RuntimeError: Tensor on cuda:0 not on cpu`.
Portanto, o primeiro passo de qualquer projeto é definir a variável que representa o destino de memória.

### 3. Código e Explicação Detalhada:

```python
import torch

# torch.device: cria um objeto que representa o hardware onde os tensores residirão
# torch.cuda.is_available(): devolve True se uma GPU compatível (NVIDIA ou AMD ROCm) foi detectada
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositivo selecionado: {device}")
```

#### Parâmetros do `torch.device(...)`:

* `"cuda"`: Identificador padrão do PyTorch para placas gráficas dedicadas (mesmo no caso de placas AMD, a API compatível ROCm responde pelo nome `"cuda"`).
* `"cpu"`: A memória convencional do sistema, utilizada caso não haja placa de vídeo configurada.

---

## Etapa 2: Com o que vamos alimentar a rede? (Dados Sintéticos)

### 1. O que esta etapa faz?

Cria matrizes numéricas simuladas (tensores) que imitam um lote de imagens reais e suas respectivas respostas corretas (rótulos).

### 2. Por que ela é necessária?

Se tentarmos ler arquivos `.jpg` do disco rígido logo na largada, teremos de lidar com arquivos corrompidos, caminhos errados de pastas e resoluções diferentes.
Com dados simulados (*dummy tensors*), nós **isolamos o problema**: garantimos que as dimensões da rede e o treinamento funcionam antes de plugar o disco rígido.

Em Visão Computacional, tensores no PyTorch sempre seguem uma ordem estrita de 4 eixos:

$$
\mathbf{(Batch,\ Canais,\ Altura,\ Largura)} \iff (B, C, H, W)
$$

### 3. Código e Explicação Detalhada:

```python
# 1. IMAGENS SIMULADAS
x = torch.randn(16, 3, 224, 224, device=device)

# 2. RÓTULOS (GABARITO) SIMULADOS
y = torch.randint(low=0, high=3, size=(16,), device=device)

print(f"Formato da entrada (x): {x.shape}") # torch.Size([16, 3, 224, 224])
print(f"Formato dos rótulos (y): {y.shape}") # torch.Size([16])
```

#### O que o método `torch.randn(...)` faz?

Gera números aleatórios seguindo a **Distribuição Normal padrão** (Gaussiana, com média $\mu = 0$ e desvio padrão $\sigma = 1$). Esses valores contêm números positivos e negativos próximos de zero, simulando com perfeição uma imagem que já foi normalizada.

#### O que os parâmetros de `torch.randn(16, 3, 224, 224, device=device)` determinam?

* `16`: Tamanho do lote (**Batch Size**). Representa 16 fotos simultâneas.
* `3`: Número de **Canais** de cor ($C$). O valor 3 representa imagens coloridas em RGB (Vermelho, Verde, Azul).
* `224`: **Altura** da foto em pixels ($H$).
* `224`: **Largura** da foto em pixels ($W$).
* `device=device`: Informa onde alocar esses números imediatamente (na VRAM da GPU).

#### O que o método `torch.randint(...)` faz?

Sorteia números inteiros dentro de um intervalo delimitado, representando o ID numérico da classe correta de cada foto (ex: 0 = folha saudável, 1 = ferrugem, 2 = mancha).

#### O que os parâmetros de `torch.randint(low=0, high=3, size=(16,), device=device)` determinam?

* `low=0`: Menor número sorteável (inclusivo).
* `high=3`: Limite superior (exclusivo). Portanto, ele só sorteará os números inteiros `0`, `1` ou `2` (3 classes no total).
* `size=(16,)`: A forma da matriz resultante. Uma tupla de um único elemento `(16,)` cria um vetor unidimensional de 16 posições, correspondendo exatamente a 1 rótulo para cada uma das 16 imagens.
* `device=device`: Aloca os rótulos na mesma memória dos dados de entrada.

---

## Etapa 3: Quem vai transformar a imagem em predição? (A Rede Sequencial)

### 1. O que esta etapa faz?

Constrói um modelo matemático que recebe os tensores da Etapa 2 e cospe notas de pontuação (**logits**) para cada uma das 3 classes.

### 2. Por que ela é necessária?

Uma imagem crua é apenas uma grade de números. Precisamos de camadas que extraiam padrões visuais (linhas, texturas, formas) e de uma camada final que decida a qual classe a foto pertence.
O container `nn.Sequential` encadeia essas camadas como uma esteira rolante direta: a saída da camada anterior entra na próxima.

### 3. Código e Explicação Detalhada:

```python
import torch.nn as nn

modelo = nn.Sequential(
    # Camada 1: Convolução 2D
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
  
    # Camada 2: Ativação Não-Linear
    nn.ReLU(inplace=True),
  
    # Camada 3: Subamostragem / Pooling
    nn.MaxPool2d(kernel_size=2, stride=2),
  
    # Camada 4: Pooling Global Adaptativo
    nn.AdaptiveAvgPool2d((1, 1)),
  
    # Camada 5: Achatamento Dimensional
    nn.Flatten(),
  
    # Camada 6: Decisão Linear Totalmente Conectada
    nn.Linear(in_features=16, out_features=3)
).to(device)

# Testando a passagem direta (Forward Pass)
predicoes_iniciais = modelo(x)
print(f"Formato da saída: {predicoes_iniciais.shape}") # torch.Size([16, 3])
```

#### O que cada método e seus parâmetros fazem:

* **`nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)`**:

  * *O que faz:* Desliza pequenas matrizes de pesos (filtros) sobre a imagem para detectar padrões visuais locais.
  * `in_channels=3`: Quantos canais de entrada a camada receberá. Como a imagem é RGB, são 3.
  * `out_channels=16`: Quantos filtros convolucionais diferentes a camada aprenderá. Cada filtro gera uma nova matriz de características, transformando os 3 canais de entrada em 16 canais na saída.
  * `kernel_size=3`: A dimensão espacial do filtro. Passar o número `3` significa um filtro quadrado de $3 \times 3$ pixels.
  * `padding=1`: Adiciona uma moldura de 1 pixel de zeros nas bordas externas da imagem. Isso permite que o filtro de tamanho 3 passe pelas bordas sem encolher o tamanho da imagem ($224 \times 224$ permanece $224 \times 224$).
* **`nn.ReLU(inplace=True)`**:

  * *O que faz:* A função de ativação linear retificada ($f(z) = \max(0, z)$). Zera qualquer valor negativo e mantém os positivos intactos. Isso introduz **não-linearidade** para que a rede aprenda padrões complexos e não seja apenas uma equação linear gigante.
  * `inplace=True`: Faz a operação matemática direto no mesmo espaço de memória do tensor recebido, sem alocar uma cópia extra. Economiza memória VRAM.
* **`nn.MaxPool2d(kernel_size=2, stride=2)`**:

  * *O que faz:* Divide a imagem em janelas $2 \times 2$ e seleciona apenas o maior valor de cada janela. Isso reduz a resolução espacial, descartando detalhes redundantes.
  * `kernel_size=2`: A janela de busca tem tamanho $2 \times 2$ pixels.
  * `stride=2`: O salto que a janela dá a cada passo. Andar de 2 em 2 pixels corta a altura e a largura exatamente pela metade ($224 \times 224 \to 112 \times 112$).
* **`nn.AdaptiveAvgPool2d((1, 1))`**:

  * *O que faz:* Calcula a média aritmética de todos os pixels restantes de cada canal, colapsando qualquer resolução espacial restante em uma dimensão de $1 \times 1$.
  * `(1, 1)`: O tamanho espacial desejado na saída $(H, W)$. Essa função é crucial porque transforma o formato $(16, 16, 112, 112)$ em exatamente $(16, 16, 1, 1)$, garantindo que o tamanho seja fixo mesmo se a resolução da foto de entrada mudar.
* **`nn.Flatten()`**:

  * *O que faz:* Converte tensores multidimensionais em um vetor plano 2D, mantendo o lote intacto. Transforma a matriz 4D $(16, 16, 1, 1)$ em uma matriz 2D $(16, 16)$, permitindo que os números entrem na camada linear seguinte.
* **`nn.Linear(in_features=16, out_features=3)`**:

  * *O que faz:* A camada clássica totalmente conectada ($y = xW^T + b$). Realiza uma combinação linear ponderada dos atributos extraídos.
  * `in_features=16`: Quantidade de entradas que cada imagem entrega (vem dos 16 canais da camada anterior).
  * `out_features=3`: Quantidade de saídas por amostra. Como temos 3 classes de classificação, precisamos de exatamente 3 pontuações (**logits**) por imagem.

> **Por que não colocar `Softmax` no final?**
> A função de custo que usaremos a seguir (`nn.CrossEntropyLoss`) aplica internamente a função Softmax combinada com o logaritmo por questões de estabilidade numérica. Inserir um Softmax manual no modelo faria o cálculo ser duplicado e quebraria a convergência. A rede deve sempre cuspir valores numéricos brutos (**logits**).

---

## Etapa 4: Como medir o erro e quem ajusta os pesos? (Loss & Optimizer)

### 1. O que esta etapa faz?

Instancia os dois elementos que conduzem o aprendizado: a régua que mede o erro e o algoritmo que altera os pesos do modelo.

### 2. Por que ela é necessária?

A rede da Etapa 3 foi inicializada com pesos aleatórios. Sem um "juiz" para dizer quão ruim foram os palpites e um "mecânico" para ajustar os pesos na direção certa, a rede nunca aprenderá.

### 3. Código e Explicação Detalhada:

```python
# 1. O CRITÉRIO DE ERRO (O JUIZ)
criterion = nn.CrossEntropyLoss()

# 2. O OTIMIZADOR (O MECÂNICO)
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)
```

#### O que `nn.CrossEntropyLoss()` faz?

Mede a distância estatística entre a probabilidade prevista pela rede e a classe verdadeira. Se a foto for da classe 0 e a rede deu a maior pontuação para a classe 2, a penalidade matemática (*Loss*) será alta.

#### O que o método `torch.optim.Adam(...)` faz?

Implementa o algoritmo de otimização Adam (*Adaptive Moment Estimation*). Ele calcula derivadas parciais e ajusta individualmente a taxa de atualização de cada peso da rede com base no histórico recente de gradientes.

#### O que os parâmetros do `Adam(...)` determinam?

* `modelo.parameters()`: Um gerador do PyTorch contendo a lista de todos os tensores de pesos ($W$) e vieses ($b$) declarados no modelo. Sem esse parâmetro, o otimizador não saberia quais variáveis ele deve modificar.
* `lr=0.001`: A **Taxa de Aprendizado** (*Learning Rate*). Controla o tamanho do passo que o otimizador dá a cada iteração. Passos grandes demais pulam o ponto de erro mínimo; passos pequenos demais demoram dias para convergir. O valor `0.001` ($10^{-3}$) é o ponto de partida padrão para o Adam.

---

## Etapa 5: A Engrenagem Girando (O Ciclo de Treino em 5 Passos)

### 1. O que esta etapa faz?

Executa repetidamente a sequência de treino que faz a máquina aprender.

### 2. Por que ela é necessária?

A inteligência artificial aprende por repetição e correção. Cada repetição sobre o lote é chamada de **época** (*epoch*). Em cada época, a rede repete estritamente a mesma sequência de **5 passos**.

### 3. Código e Explicação Detalhada:

```python
# Informa à rede que estamos em fase de treinamento
modelo.train()

print("Iniciando o treinamento:")
for epoca in range(10):
    # PASSO 1: Zerar os gradientes acumulados
    optimizer.zero_grad()
  
    # PASSO 2: Passar os dados pela rede (Forward Pass)
    predicoes = modelo(x)
  
    # PASSO 3: Calcular o erro cometido em relação aos rótulos reais
    loss = criterion(predicoes, y)
  
    # PASSO 4: Calcular derivadas parciais via Autograd (Backward Pass)
    loss.backward()
  
    # PASSO 5: Atualizar os pesos da rede na direção da melhora
    optimizer.step()
  
    # Exibe o valor do erro na época atual
    print(f"Época {epoca+1:02d} | Perda (Loss): {loss.item():.4f}")
```

#### O que cada um dos 5 passos faz:

1. **`optimizer.zero_grad()`**:

   * *O que faz:* Zera o atributo `.grad` de todos os parâmetros.
   * *Por que é obrigatório:* No PyTorch, os gradientes são acumulativos por padrão (`w.grad += novo_grad`). Isso é útil quando acumulamos gradientes de lotes pequenos, mas se você esquecer essa linha no treino comum, os gradientes do lote anterior se somarão aos novos indefinidamente, explodindo os pesos da rede até virarem `NaN` (não-número).
2. **`predicoes = modelo(x)`**:

   * *O que faz:* Alimenta as imagens no modelo e executa todas as camadas. Em segundo plano, o PyTorch grava em um grafo dinâmico cada operação aritmética realizada.
3. **`loss = criterion(predicoes, y)`**:

   * *O que faz:* Compara as predições com as respostas reais `y` e devolve um tensor contendo um único número: o erro médio daquele lote.
4. **`loss.backward()`**:

   * *O que faz:* O motor de diferenciação automática (**Autograd**) percorre o grafo computacional de trás para frente pela Regra da Cadeia. Para cada parâmetro $w$ da rede, ele calcula a derivada parcial $\frac{\partial \text{Loss}}{\partial w}$ e guarda esse resultado dentro do campo `w.grad`.
5. **`optimizer.step()`**:

   * *O que faz:* Lê os valores armazenados em `w.grad` e aplica a fórmula do otimizador para corrigir o valor de cada peso:

   $$
   w_{\text{novo}} = w_{\text{antigo}} - \text{lr} \times \text{gradiente}
   $$
6. **`loss.item()`**:

   * *O que faz:* O objeto `loss` é um tensor do PyTorch amarrado ao grafo de derivadas. O método `.item()` extrai o valor numérico primitivo puro do Python (`float`), desconectando-o do grafo para imprimir no terminal sem consumir memória VRAM desnecessariamente.

---

### O Script Plano Consolidado

Agora que você conhece o propósito e os parâmetros de cada linha, veja o script plano completo rodando de ponta a ponta:

```python
import torch
import torch.nn as nn

# 1. HARDWARE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. DADOS SIMULADOS (16 fotos 224x224 RGB, 3 classes)
x = torch.randn(16, 3, 224, 224, device=device)
y = torch.randint(low=0, high=3, size=(16,), device=device)

# 3. REDE SEQUENCIAL
modelo = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(in_features=16, out_features=3)
).to(device)

# 4. CRITÉRIO E OTIMIZADOR
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)

# 5. O CICLO DE TREINO
modelo.train()
for epoca in range(10):
    optimizer.zero_grad()
    predicoes = modelo(x)
    loss = criterion(predicoes, y)
    loss.backward()
    optimizer.step()
  
    print(f"Época {epoca+1:02d} | Perda (Loss): {loss.item():.4f}")
```

---

# PARTE 2: A Ponte (Por que o Script Plano não é suficiente no mundo real?)

O script plano demonstra o ciclo completo de forma limpa. No entanto, ao lidar com um problema real de visão computacional, ele apresenta **três limitações críticas**:

1. **Gargalo de Memória (Leitura em Disco):**Não podemos carregar 10.000 imagens reais de uma só vez na memória com um único tensor. O sistema ficaria sem RAM. Precisamos de uma estrutura que leia imagens do disco rígido sob demanda (**Lazy Loading**).
2. **Gargalo Arquitetural:**O `nn.Sequential` só aceita camadas enfileiradas em linha reta. Redes modernas utilizam caminhos paralelos ou conexões residuais (*Skip Connections*). Para isso, precisamos criar uma classe própria que herde de `nn.Module`.
3. **Engenharia de Software:**
   Em produção, a interface do usuário (ex: um app web), a lógica de treino e a definição da rede não podem residir no mesmo arquivo.

Por isso, na Parte 3, nós desacoplamos essas funções em uma arquitetura profissional.

---

# PARTE 3: A Arquitetura Modular Profissional

Organizamos o código em pastas com papéis isolados dentro do pacote `src/<pacote>/`:

```text
meu-projeto/
├── pyproject.toml              # Dependências e índice de hardware
├── .venv/                      # Ambiente virtual isolado
│
├── src/<pacote>/               # Código reutilizável da aplicação
│   ├── config.py               # 1. Centraliza caminhos, hiperparâmetros e hardware
│   │
│   ├── data/                   # 2. Pipeline de leitura e tratamento de imagens
│   │   ├── dataset.py          # Leitura sob demanda (torch.utils.data.Dataset)
│   │   └── transforms.py       # Redimensionamento e augmentations
│   │
│   ├── models/                 # 3. Arquitetura da rede neural pura
│   │   └── simple_cnn.py       # Modelo encapsulado em classe nn.Module
│   │
│   └── engine/                 # 4. Orquestração do ciclo de vida
│       ├── trainer.py          # Funções de treino e validação com acurácia
│       └── predictor.py        # Inferência de imagens avulsas para produção
│
└── src/main.py                 # Ponto de entrada fino que importa e executa
```

---

## 3.1 O Painel de Configuração

> 📁 **Arquivo:** `src/deep_learning/config.py`

Em vez de espalhar caminhos de pastas e números mágicos pelo código, centralizamos tudo:

```python
from pathlib import Path
import torch

# Path(__file__).resolve().parents[1]:
# .resolve() obtém o caminho absoluto deste arquivo sem ambiguidades
# .parents[1] sobe dois níveis de diretório para chegar confiavelmente na raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Diretórios físicos das partições do Dataset
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "Train"
VALIDATION_DATA_DIR = DATA_DIR / "Validation"
TEST_DATA_DIR = DATA_DIR / "Test"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Hardware detectado (GPU via ROCm/CUDA ou CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparâmetros de treinamento
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
```

---

## 3.2 Ingestão de Dados: `transforms.py`, `dataset.py` e `DataLoader`

### A) Os Pipelines de Transformação (PIL + NumPy Puro)

> 📁 **Arquivo:** `src/deep_learning/data/transforms.py`

As fotos no disco possuem resoluções variadas e pixels inteiros em $[0, 255]$. O PyTorch precisa de tensores uniformes $(3, 224, 224)$ com escala decimal $[0.0, 1.0]$. Fazemos isso diretamente com **PIL e NumPy**:

> [!IMPORTANT]
> **A Pegadinha dos Eixos:** O PIL e o NumPy organizam pixels no formato $(H, W, C)$ (Altura, Largura, Canais). No entanto, toda camada convolucional do PyTorch exige estritamente $(C, H, W)$ (Canais, Altura, Largura). A transposição `transpose(2, 0, 1)` é **obrigatória** antes de gerar o Tensor!

```python
import numpy as np
import torch
from PIL import Image, ImageOps
from deep_learning.config import IMAGE_SIZE


def train_transform(image: Image.Image) -> torch.Tensor:
    """Pipeline de Treino: Redimensionamento proporcional (Center Crop) + Tensor (C, H, W)."""
    # 1. PIL: Redimensiona mantendo a proporção original e corta o centro (sem achatar)
    image = ImageOps.fit(image, size=IMAGE_SIZE, method=Image.Resampling.BILINEAR)

    # 2. NumPy: Converte inteiros [0, 255] -> float32 [0.0, 1.0]
    # Formato original do NumPy: (H, W, C) = (224, 224, 3)
    arr = np.array(image, dtype=np.float32) / 255.0

    # 3. NumPy: Reordena eixos de (H, W, C) -> (C, H, W) = (3, 224, 224)
    arr = arr.transpose(2, 0, 1)

    # 4. PyTorch: Converte a matriz NumPy em Tensor sem cópia desnecessária de memória
    tensor = torch.from_numpy(arr)
    return tensor


def validation_transform(image: Image.Image) -> torch.Tensor:
    """Pipeline de Validação: Determinístico, idêntico em resolução e escala."""
    image = ImageOps.fit(image, size=IMAGE_SIZE, method=Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


def test_transform(image: Image.Image) -> torch.Tensor:
    """Pipeline de Teste: Idêntico à validação."""
    return validation_transform(image)
```

---

### B) Dataset com Leitura Preguiçosa

> 📁 **Arquivo:** `src/deep_learning/data/dataset.py`

Para evitar que a memória RAM do computador exploda carregando milhares de imagens de uma vez, criamos uma classe herdando de `torch.utils.data.Dataset`. Ela exige implementar dois métodos especiais do Python:

* **`__len__(self)`:** Devolve a quantidade total de fotos catalogadas no disco.
* **`__getitem__(self, idx)`:** Invocado na sintaxe `dataset[idx]`. Implementa o **Lazy Loading**: ele abre e processa **uma única imagem** no momento exato em que o lote precisa dela.

```python
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch

from deep_learning.config import (
    DATA_DIR,
    TRAIN_DATA_DIR,
    VALIDATION_DATA_DIR,
    TEST_DATA_DIR,
)
from deep_learning.data.transforms import (
    train_transform,
    validation_transform,
    test_transform,
)

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str | Path = DATA_DIR, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
  
        # Mapeia as subpastas como nomes de classes (ex: 'Healthy', 'Powdery', 'Rust')
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_index = {cls_name: i for i, cls_name in enumerate(self.classes)}
  
        # Guarda apenas o caminho do arquivo e o índice da classe (NUNCA pixels em memória!)
        self.samples = []
        for cls_name in self.classes:
            folder = self.root_dir / cls_name
            for img_path in folder.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, self.class_to_index[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
  
        # Leitura sob demanda: abre a foto e converte para RGB (descarta canal alfa de PNG)
        image = Image.open(img_path).convert("RGB")
  
        if self.transform is not None:
            image = self.transform(image)
  
        return image, label


# Instanciação das 3 partições passando o root_dir e o transform correspondente:
train_dataset = ImageFolderDataset(root_dir=TRAIN_DATA_DIR, transform=train_transform)
validation_dataset = ImageFolderDataset(root_dir=VALIDATION_DATA_DIR, transform=validation_transform)
test_dataset = ImageFolderDataset(root_dir=TEST_DATA_DIR, transform=test_transform)

print("Classes detectadas:", train_dataset.classes)
# Saída: ['Healthy', 'Powdery', 'Rust'] (Mapeamento: 0, 1, 2 idêntico nas 3 partições)
print(f"Amostras -> Treino: {len(train_dataset)} | Val: {len(validation_dataset)} | Teste: {len(test_dataset)}")
```

---

### C) O `DataLoader` (O Gerente de Lotes)

> 📁 **Arquivo:** `src/deep_learning/data/dataset.py` (ou `src/main.py`)

O `DataLoader` recebe o dataset individual e empacota as amostras em lotes `(Batch, C, H, W)` usando subprocessos paralelos da CPU:

```python
from torch.utils.data import DataLoader
import torch

# Loader de Treino: shuffle=True para embaralhar a cada época e evitar viés de ordem
loader_treino = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available() # Acelera cópia direta da RAM para a VRAM da GPU via DMA
)

# Loader de Validação: shuffle=False para avaliação justa e reprodutiva
loader_val = DataLoader(
    dataset=validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# Loader de Teste: shuffle=False para cálculo final das métricas
loader_teste = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)
```

---

## 3.3 A Rede Neural em Classe

> 📁 **Arquivo:** `src/deep_learning/models/simple_cnn.py`

Ao herdar de `nn.Module`, temos total liberdade arquitetural e separamos claramente o extrator de características da cabeça de classificação:

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        # super().__init__() é OBRIGATÓRIO: inicializa o rastreador de parâmetros do PyTorch
        super().__init__()
  
        # 1. EXTRATOR DE ATRIBUTOS CONVOLUCIONAIS
        self.features = nn.Sequential(
            # Bloco 1: (B, 3, 224, 224) -> (B, 32, 112, 112)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32), # Normaliza as médias dos canais, estabilizando o treino
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
  
            # Bloco 2: (B, 32, 112, 112) -> (B, 64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 3: (B, 64, 56, 56) -> (B, 128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
  
        # 2. POOLING GLOBAL ADAPTATIVO: reduz qualquer dimensão espacial restante para (1 x 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
  
        # 3. CABEÇA DE CLASSIFICAÇÃO LINEAR
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            # nn.Dropout: desliga aleatoriamente 40% das conexões durante o treino contra overfitting
            nn.Dropout(p=0.4),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define o caminho que os tensores percorrem dentro da rede."""
        x = self.features(x)
        x = self.gap(x)
        logits = self.classifier(x)
        return logits
```

---

## 3.4 O Motor de Treinamento e Avaliação

> 📁 **Arquivo:** `src/deep_learning/engine/trainer.py`

No código de produção, desacoplamos o treino da avaliação e adicionamos o cálculo da **acurácia**:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> tuple[float, float]:
    # model.train(): ativa o comportamento de aprendizado (Dropout e BatchNorm operantes)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        # O ciclo canônico
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        # Acumula a perda multiplicada pelo tamanho do lote para calcular a média exata da época
        running_loss += loss.item() * images.size(0)
  
        # torch.max(logits, dim=1): busca o índice da classe com a maior pontuação por linha
        # Descartamos as pontuações brutas com '_' e pegamos apenas os índices preditos (preds)
        _, preds = torch.max(logits, dim=1)
  
        # (preds == targets).sum().item(): conta quantas previsões foram exatamente corretas
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> tuple[float, float]:
    # model.eval(): congela BatchNorm e desativa o sorteio do Dropout para testes determinísticos
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # with torch.no_grad(): desliga a gravação do grafo de derivadas, economizando até 60% de VRAM
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total
    val_acc = (correct / total) * 100
    return val_loss, val_acc
```

---

## 3.5 Como Salvar e Restaurar Pesos (`state_dict`)

Nunca salve o objeto inteiro com `torch.save(model, "modelo.pth")`. Isso serializa via `pickle` o caminho do arquivo no seu computador e quebra se você mover ou renomear arquivos.

Em vez disso, salve apenas o **dicionário com os valores numéricos dos parâmetros (`state_dict`)**:

```python
# 1. SALVANDO APENAS OS PESOS
torch.save(modelo.state_dict(), "melhor_modelo.pth")

# 2. RESTAURANDO OS PESOS EM OUTRO SCRIPT
modelo = SimpleCNN(num_classes=3).to(device)

# map_location=device garante que um modelo treinado na GPU abra sem erros em uma máquina com CPU
modelo.load_state_dict(torch.load("melhor_modelo.pth", map_location=device))

# OBRIGATÓRIO: colocar imediatamente em modo de avaliação após carregar
modelo.eval()
```

---

## 3.6 Inferência em Produção

> 📁 **Arquivo:** `src/deep_learning/engine/predictor.py`

Para classificar uma imagem avulsa vinda de uma interface web ou API:

```python
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn

def predict_single_image(
    image_path: str | Path, 
    model: nn.Module, 
    transform, 
    class_names: list[str], 
    device: torch.device
) -> dict:
    model.eval()
  
    # 1. Abre a foto e aplica o mesmo pipeline de transformação do treino
    image = Image.open(image_path).convert("RGB")
    tensor_img = transform(image) # Retorna formato 3D: (3, 224, 224)
  
    # 2. unsqueeze(0): insere um eixo de lote unitário no índice 0 -> (1, 3, 224, 224)
    # Toda rede convolucional em PyTorch exige obrigatoriamente 4 dimensões
    tensor_batch = tensor_img.unsqueeze(0).to(device)
  
    with torch.no_grad():
        logits = model(tensor_batch)
        # torch.softmax: converte logits em probabilidades percentuais que somam 1.0 (dim=1)
        # squeeze(0): remove a dimensão de lote unitária de volta -> vetor 1D
        probs = torch.softmax(logits, dim=1).squeeze(0)
  
    top_prob, top_idx = torch.max(probs, dim=0)
  
    return {
        "classe": class_names[top_idx.item()],
        "confianca": top_prob.item() * 100,
        "todas_probabilidades": {
            cls: p.item() * 100 
            for cls, p in zip(class_names, probs)
        }
    }
```

---

## 3.7 O Maestro do Projeto: Orquestração e Avaliação Final de Teste

> 📁 **Arquivo:** `src/main.py`

### 1. O que este arquivo faz?

Ele é o ponto de entrada (*entry point*) da aplicação. Ele não define classes nem redes neurais novas; seu único papel é **importar e conectar** os módulos que construímos nas seções anteriores, orquestrando:

1. A inicialização do hardware, modelo, função de perda e otimizador.
2. O loop de épocas intercalando `train_one_epoch` e `evaluate` no conjunto de validação.
3. O salvamento automático do melhor modelo (*checkpointing*) com base na acurácia de validação.
4. A **avaliação cega final no conjunto de teste** (`test_data_loader`).

### 2. A Tríplice Divisão: Treino vs. Validação vs. Teste

Por que precisamos de três conjuntos separados e não apenas dois?

| Conjunto                           | Papel no Ciclo de Vida                                                                                             | Atualiza Pesos?                          | Frequência de Uso                                              |
| :--------------------------------- | :----------------------------------------------------------------------------------------------------------------- | :--------------------------------------- | :-------------------------------------------------------------- |
| **Treino (Train)**           | Aprender padrões visuais (convoluções e pesos lineares).                                                        | **Sim** (via `loss.backward()`). | Todo lote de cada época.                                       |
| **Validação (Validation)** | Avaliar generalização durante o treino e selecionar o melhor checkpoint (*early stopping* / *model saving*). | **Não** (apenas inferência).     | 1 vez ao final de cada época.                                  |
| **Teste (Test)**             | Fornecer a métrica final imparcial de desempenho do modelo em produção.                                         | **Não** (apenas inferência).     | **Apenas 1 vez**, após o término de todo o treinamento. |

> ⚠️ **Atenção Metodológica:**
> Se você usar o conjunto de validação para escolher o melhor modelo e também para reportar o resultado final no seu relatório/artigo, seus números estarão **otimistas demais (viciados)**. Você indiretamente "calibrou" o modelo para ter bom desempenho naquele subconjunto de 60 imagens. O conjunto de teste (150 imagens) nunca deve influenciar nenhuma decisão de treino.

---

### 3. Código Completo de `src/main.py`

```python
import torch
import torch.nn as nn

from deep_learning.config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MODEL_SAVE_PATH,
    CLASSES,
)
from deep_learning.data.dataset import (
    train_data_loader,
    validation_data_loader,
    test_data_loader,
)
from deep_learning.models.simple_cnn import SimpleCNN
from deep_learning.engine.trainer import train_one_epoch, evaluate
from deep_learning.engine.predictor import predict_single_image
from deep_learning.data.transforms import val_transform


def main():
    print(f"🚀 Iniciando pipeline no dispositivo: {DEVICE}")

    # 1. INSTANCIAÇÃO DOS COMPONENTES
    model = SimpleCNN(in_channels=3, num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    # 2. LOOP DE TREINAMENTO E VALIDAÇÃO
    print("\n--- INÍCIO DO TREINAMENTO ---")
    for epoch in range(NUM_EPOCHS):
        # Treino (com atualização de gradientes)
        train_loss, train_acc = train_one_epoch(
            model=model,
            data_loader=train_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        # Validação (sem gradientes, apenas para checagem)
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=validation_data_loader,
            criterion=criterion,
            device=DEVICE,
        )

        print(
            f"Época [{epoch + 1:02d}/{NUM_EPOCHS:02d}] | "
            f"Treino Loss: {train_loss:.4f} - Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2%}"
        )

        # 3. CHECKPOINT: Salva o modelo sempre que bater recorde de acurácia de validação
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ⭐ Novo recorde de validação! Pesos salvos em: {MODEL_SAVE_PATH}")

    print("\n--- FIM DO TREINAMENTO ---")

    # 4. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (TEST DATASET)
    print("\n🧪 Carregando o melhor checkpoint para avaliação final no CONJUNTO DE TESTE...")
    best_model = SimpleCNN(in_channels=3, num_classes=len(CLASSES)).to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    test_loss, test_acc = evaluate(
        model=best_model,
        data_loader=test_data_loader,
        criterion=criterion,
        device=DEVICE,
    )

    print(f"📊 Desempenho Real no Conjunto de Teste (150 imagens inéditas):")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2%}")

    # 5. TESTE DE INFERÊNCIA AVULSA (PRODUÇÃO)
    # Pega a primeira imagem de teste para validar o motor de predição
    sample_img_path = test_data_loader.dataset.samples[0][0]
    resultado = predict_single_image(
        image_path=sample_img_path,
        model=best_model,
        transform=val_transform,
        class_names=CLASSES,
        device=DEVICE,
    )

    print("\n🔎 Demonstração de Inferência com predictor.py:")
    print(f"   Arquivo: {sample_img_path}")
    print(f"   Diagnóstico: {resultado['classe']} ({resultado['confianca']:.2f}% de certeza)")


if __name__ == "__main__":
    main()
```

---

# GUIA DE BOLSO: Diagnóstico Rápido de Erros (Troubleshooting)

| Erro no Terminal                                | O que significa na prática                                                                        | O que fazer                                                                                                           |
| :---------------------------------------------- | :------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| `Expected 4D tensor, got 3D`                  | Você passou uma foto solta `(3, H, W)` para a rede sem a dimensão de lote.                     | Use `tensor.unsqueeze(0)` para transformar em `(1, 3, H, W)`.                                                     |
| `mat1 and mat2 shapes cannot be multiplied`   | A quantidade de canais que saem da convolução não é igual ao `in_features` da camada Linear. | Insira `nn.AdaptiveAvgPool2d((1, 1))` antes do `Flatten` para fixar a saída espacial em `1x1`.                 |
| `Tensor on cuda:0 not on cpu`                 | O modelo está na GPU, mas os dados estão na CPU (ou o contrário).                               | Certifique-se de que tanto o modelo quanto o lote tenham recebido `.to(device)`.                                    |
| `CUDA out of memory` / `HIP out of memory`  | O `batch_size` é grande demais para a memória da sua GPU.                                      | Reduza o `batch_size` (ex: de 32 para 16) e confira se usou `with torch.no_grad():` na validação.               |
| `Loss não diminui (estagnada)`               | Os gradientes estão acumulando loucamente ou os dados não foram normalizados.                    | Verifique se `optimizer.zero_grad()` é a primeira linha do loop e normalize os pixels para o intervalo $[0, 1]$. |
| `Resultados mudam a cada teste da mesma foto` | O `Dropout` continua sorteando e desligando conexões aleatórias.                               | Chame obrigatoriamente `model.eval()` antes de rodar qualquer teste ou predição.                                  |
