# AgroVision — Classificação de Patologias em Folhas com Deep Learning

Um projeto prático de Visão Computacional e Redes Neurais Convolucionais (CNNs) desenvolvido com PyTorch, estruturado desde os fundamentos matemáticos e manipulação de tensores até o ciclo completo de treinamento, avaliação e disponibilização em interface web interativa.

---

## Objetivo Educacional: O Caminho no Deep Learning

O objetivo central deste projeto foi consolidar dois meses de imersão intensiva em Deep Learning e Engenharia de Machine Learning. Em vez de utilizar soluções prontas de alto nível, cada etapa do pipeline foi construída entendendo a teoria e a mecânica por trás do código:

1. **Manipulação de Tensores e Imagens**:
   - Representação matricial de imagens coloridas (canais RGB).
   - Conversão de formatos entre bibliotecas: de `(Altura, Largura, Canais)` no NumPy/PIL para `(Canais, Altura, Largura)` exigido pelas camadas convolucionais do PyTorch.
   - Normalização dos valores dos pixels de inteiros `[0, 255]` para números de ponto flutuante `[0.0, 1.0]`.
   - Ajuste dimensional com corte proporcional (`ImageOps.fit`) para a resolução padrão de `(224, 224)` pixels.

2. **Ingestão de Dados Eficiente**:
   - Criação de `Dataset` customizado herdando de `torch.utils.data.Dataset` com *Lazy Loading* (leitura sob demanda), evitando sobrecarga de memória RAM.
   - Configuração de `DataLoader` com lotes (*batches* de 32 imagens), paralelismo com *workers* e travamento de páginas (`pin_memory=True`) para acelerar a transferência PCIe para a GPU.

3. **Aumento de Dados (Data Augmentation)**:
   - Aplicação de transformações estocásticas no conjunto de treino (rotação angular aleatória e espelhamento horizontal).
   - Expansão do espaço amostral para combater o *overfitting*, dado que filtros convolucionais não são naturalmente invariantes a rotações.

4. **Arquitetura de Rede Convolucional (`SimpleCNN`)**:
   - **Camadas Convolucionais (`Conv2d`)**: Extração de características espaciais locais com compartilhamento de pesos (kernel 3x3, stride 1, padding 1).
   - **Normalização em Lote (`BatchNorm2d`)**: Suavização do terreno da função de perda e estabilização de gradientes.
   - **Ativação Não-Linear (`ReLU`)**: Introdução de não-linearidade para aproximação de funções complexas, mitigando o problema do gradiente evanescente.
   - **Pooling (`MaxPool2d`)**: Redução da dimensionalidade espacial preservando as ativações mais fortes de cada filtro.
   - **Pooling Adaptativo Global (`AdaptiveAvgPool2d((1, 1))`)**: Colapso das dimensões espaciais antes do classificador linear, reduzindo a contagem de parâmetros e o risco de memorização.
   - **Regularização com `Dropout(0.4)`**: Desativação aleatória de neurônios no treino para evitar co-adaptação.

5. **Mecânica do Ciclo de Treino e Otimização**:
   - O fluxo completo de retropropagação (*backpropagation*):
     $$\text{Forward} \rightarrow \text{Cálculo da Perda (Loss)} \rightarrow \text{Zero Grad} \rightarrow \text{Backward} \rightarrow \text{Optimizer Step}$$
   - Uso de `CrossEntropyLoss` combinada diretamente com *logits* brutos (evitando instabilidades numéricas ao não aplicar *Softmax* previamente no modelo).
   - Diferenciação entre modos de execução: `model.train()` (com Dropout e BatchNorm ativos) versus `model.eval()` acompanhado de `torch.no_grad()` para inferência.

---

## Domínio do Problema: Doenças Foliares

O modelo classifica folhas em três categorias:

* **Saudável (Healthy)**: Tecido vegetal íntegro, sem patógenos aparentes.
* **Oídio (Powdery Mildew)**: Infecção fúngica superficial caracterizada por uma camada esbranquiçada/acinzentada pulverulenta.
* **Ferrugem (Rust)**: Pústulas salientes de tonalidade amarelo-alaranjada a castanho-ferrugem causadas por fungos biotróficos.

### Distribuição do Conjunto de Dados

A base de dados é balanceada entre as classes e dividida em três partições:

| Partição | Saudável | Oídio | Ferrugem | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Treino (Train)** | 458 | 430 | 434 | **1.322** |
| **Validação (Validation)** | 20 | 20 | 20 | **60** |
| **Teste (Test)** | 50 | 50 | 50 | **150** |
| **Total Geral** | **528** | **500** | **504** | **1.532** |

---

## Estrutura do Projeto

```text
.
├── data/                       # Conjuntos de dados (Train, Validation, Test)
├── guide/
│   └── questionario_50_questoes.md  # Questionário com 50 perguntas de fixação técnica
├── models/
│   └── model.pth               # Pesos treinados do modelo PyTorch
├── notebooks/                  # Notebooks experimentais (NumPy, PIL, PyTorch)
├── src/
│   └── deep_learning/
│       ├── config.py           # Configurações globais, diretórios, hiperparâmetros
│       ├── data/
│       │   ├── dataset.py      # Implementação do ImageFolderDataset
│       │   └── transforms.py   # Pipelines de pré-processamento e augmentation
│       ├── engine/
│       │   ├── predictor.py    # Pipeline de inferência individual
│       │   └── trainer.py      # Loops de treinamento e avaliação por época
│       ├── models/
│       │   ├── resnet_custom.py# Arquitetura residual experimental
│       │   └── simple_cnn.py   # Arquitetura principal SimpleCNN
│       └── ui/
│           ├── app.py          # Ponto de entrada da aplicação Streamlit (navegação)
│           └── pages/
│               ├── home.py     # Visão geral do projeto e das patologias
│               └── predict_single.py # Diagnóstico em tempo real e testes
├── pyproject.toml              # Gerenciador de dependências e metadados do projeto
└── README.md
```

---

## Como Executar o Projeto

### 1. Instalação de Dependências

Recomenda-se utilizar o [uv](https://github.com/astral-sh/uv) como gerenciador de ambientes Python:

```bash
uv sync
```

### 2. Iniciar a Interface Web (Streamlit)

Execute a aplicação Streamlit:

```bash
uv run streamlit run src/deep_learning/ui/app.py
```

Acesse no navegador:
**`http://localhost:8501`**

---

## Funcionalidades da Interface

* **Visão Geral**: Apresentação técnica do projeto, detalhes estruturados das três classes e panorama do conjunto de dados.
* **Diagnóstico Individual**:
  * **Upload de Imagem**: Suporte a fotos em formato JPG, JPEG ou PNG com limite de tamanho máximo configurado em 10 MB.
  * **Galeria de Amostras de Teste**: Seleção de amostras do conjunto de teste para validação direta.
  * **Inferência Otimizada**: Carregamento em cache (`@st.cache_resource`) mantendo o modelo em memória.
  * **Resultados**: Identificação da classe, percentual de confiança, distribuição de probabilidades e recomendações técnicas de manejo.

---

## Material de Apoio e Fixação

Na pasta [`guide/`](file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/guide), encontra-se o [`questionario_50_questoes.md`](file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/guide/questionario_50_questoes.md), cobrindo 50 afirmativas de Verdadeiro ou Falso divididas em 10 blocos temáticos:

1. Imagens, Cores e NumPy
2. Ingestão de Dados e Hardware
3. Aumento de Dados / Data Augmentation
4. Convolução e Geometria
5. Funções de Ativação
6. Pooling e Redução Dimensional
7. Normalização em Lote - BatchNorm
8. Regularização e Camadas Finais
9. Saídas, Erro e Função de Perda
10. Otimização e Ciclo de Treino
