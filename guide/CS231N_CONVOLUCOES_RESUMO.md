# 📑 Síntese CS231n (Stanford): Redes Convolucionais & Transfer Learning
> **Fonte:** [Stanford CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/) e [Transfer Learning](https://cs231n.github.io/transfer-learning/)  
> **Autores originais:** Andrej Karpathy, Justin Johnson e Fei-Fei Li.  
> **Objetivo:** Os trechos e conceitos mais vitais de Stanford traduzidos e comentados para o projeto de pesquisa em Visão Computacional.

---

# PARTE 1: Redes Convolucionais (CNNs)

## 1. Por que Redes Neurais Comuns (FC) Falham com Imagens?
> *"In CIFAR-10, images are only 32x32x3 [...], so a single neuron in the first layer would have 32x32x3 = 3,072 weights. [...] But clearly this does not scale to larger images. For example, an image of respectable size, e.g. 200x200x3, would lead to neurons that have 200x200x3 = 120,000 weights. Moreover, we would want several such neurons. Clearly, this full connectivity is wasteful and leads to severe overfitting."*

* **Explicação Detalhada da Aritmética:**
  * **Entradas (Pixels):** Uma imagem $200 \times 200 \times 3$ fornece **$120.000$ valores de entrada**.
  * **Pesos por Neurônio:** Em uma camada totalmente conectada (`Linear`), cada neurônio individual precisa de $120.000$ pesos ($+ 1$ viés).
  * **Multiplicação pelo Número de Neurônios (Hiperparâmetro):** Se você criar uma primeira camada simples com apenas $100$ neurônios, serão necessários:
    $$(120.000 \times 100) + 100 = \mathbf{12.000.100} \text{ parâmetros treináveis!}$$
  * Mais de 12 milhões de parâmetros apenas na primeira camada levam a **overfitting brutal** (a rede decora os pixels específicos do treino) e inviabilizam o treino de imagens maiores.

> [!NOTE]
> **O Contraste com Convoluções:**
> Para essa mesma imagem de 3 canais, uma camada convolucional `nn.Conv2d(3, 32, kernel_size=3)` precisa de apenas:
> $$32 \times (3 \times 3 \times 3 + 1) = \mathbf{896} \text{ parâmetros!}$$
> 896 contra 12 milhões, com a vantagem de manter a integridade bidimensional e a invariância por translação.

---

## 2. Volumes 3D e Conectividade Local (*Receptive Field*)
> *"Unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. [...] Neurons in a layer will only be connected to a small region of the layer before it. The spatial extent of this connectivity is a hyperparameter called the receptive field of the neuron (filter size)."*

* **O Pulo do Gato:** Em vez de olhar a imagem inteira de uma vez, cada neurônio olha apenas para uma janelinha local (ex: $3 \times 3$ pixels).
* **Regra de Profundidade:** A conexão é pequena no espaço 2D ($3 \times 3$), mas atravessa **todos os canais de cor** (se for RGB, o filtro tem profundidade 3: $3 \times 3 \times 3$).

---

## 3. Compartilhamento de Pesos (*Parameter Sharing*)
> *"If one feature is useful to compute at some spatial position (x, y), then it should also be useful to compute at a different position (x2, y2). [...] In other words, we constrain the neurons in each depth slice to use the same weights and bias."*

* **Por que é genial:** Se o filtro aprendeu a detectar a textura de um fungo no topo da folha, ele não precisa reaprender essa textura no meio ou na base da folha. O mesmo filtro desliza pela imagem inteira.
* **Impacto Numérico (AlexNet):** Reduziu os parâmetros da primeira camada de **105 milhões para apenas 35 mil**.

---

## 4. A Fórmula Matemática de Ouro do Tamanho Espacial
$$\text{Tamanho de Saída} = \frac{W - F + 2P}{S} + 1$$
* $W$ = Dimensão de entrada (ex: $224$)
* $F$ = Tamanho do Filtro (ex: $3$)
* $P$ = *Zero-Padding* (zeros adicionados nas bordas)
* $S$ = *Stride* (passo do filtro)

> **Regra Prática de Stanford:** Para manter a resolução espacial intacta ($224 \to 224$), use sempre:  
> $$P = \frac{F - 1}{2} \quad \text{quando } S = 1$$  
> Para $F = 3$, use $P = 1$ e $S = 1$. A imagem nunca é "comida" nas bordas!

---

## 5. O Segredo de Design: 3 camadas $3 \times 3$ vs. 1 camada $7 \times 7$
> *"Suppose that you stack three 3x3 CONV layers on top of each other. Each neuron on the 3rd layer has a 7x7 view of the input.  
> Suppose instead we used a single 7x7 CONV layer. These have the same spatial extent, but with disadvantages:  
> 1. The three 3x3 layers contain three non-linearities (ReLUs) instead of one, making features more expressive.  
> 2. The 7x7 layer has $49 C^2$ parameters. The three 3x3 layers have only $27 C^2$ parameters."*

* Três convoluções pequenas têm o mesmo alcance de uma grande, mas gastam **quase metade dos parâmetros** e inserem **três ReLUs**, aprendendo padrões muito mais ricos. É por isso que ResNet e VGG usam filtros $3 \times 3$ do início ao fim.

---

## 6. Max Pooling: Redução de Resolução
> *"Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting.  
> The most common form is a pooling layer of size 2x2 with a stride of 2, discarding 75% of the activations."*

* Reduz a largura e altura pela metade ($112 \to 56$), mantendo apenas o sinal mais intenso de cada bloco $2 \times 2$. A profundidade de canais não é alterada.

---

# PARTE 2: Transfer Learning (A Lição Definitiva de Stanford)

Na página dedicada a **Transfer Learning**, Stanford apresenta o que se tornou a bíblia prática da área:

> *"In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest."*

---

## 1. Os Dois Níveis de Transfer Learning

### Nível 1: A Rede como Extrator de Características Fixo (*Fixed Feature Extractor*)
* Você pega uma rede treinada no ImageNet (ex: **ResNet18**).
* **Congela todos os pesos convolucionais** (`requires_grad = False`).
* Arranca fora a última camada linear de 1000 classes e coloca uma nova com as classes do seu projeto (ex: 3 doenças de folha).
* Treina **apenas** essa última camada linear.
* *Vantagem:* Treina em poucos minutos e é impossível estragar os pesos pré-treinados.

### Nível 2: Ajuste Fino (*Fine-Tuning*)
* Você não substitui apenas a última camada; você também **descongela algumas das últimas camadas convolucionais** para continuarem aprendendo.
* **O Pulo do Gato de Stanford:**
  > *"It is common to use a smaller learning rate (e.g. 10x smaller than the new linear layer) for the ConvNet layers that are being fine-tuned, in order to preserve the good features that were learned on ImageNet while gently adapting them to the new data."*
  * Se você usar uma taxa de aprendizado normal na base convolucional, você destrói (*"catastrophic forgetting"*) o conhecimento que a rede levou semanas para aprender no ImageNet!

---

## 2. A Lendária Matriz dos 4 Quadrantes de Stanford

Como saber exatamente qual estratégia usar? Stanford criou a **Matriz de Decisão dos 4 Cenários**, baseada em dois fatores:
1. **O Tamanho do seu Dataset:** Pequeno (*Small*) vs. Grande (*Large*).
2. **A Similaridade dos seus Dados com o ImageNet:** Parecido (*Similar*) vs. Muito Diferente (*Different*).

```
                             SIMILAR AO IMAGENET
                                      │
                   QUADRANTE 1        │        QUADRANTE 2
                (Dataset Pequeno)     │     (Dataset Grande)
                                      │
               Treine apenas uma      │   Faça Fine-Tuning na
               camada Linear final.   │   rede inteira com taxa
               NÃO desfaça o          │   baixa. Muito seguro
               congelamento das       │   contra overfitting!
               convoluções!           │
  DATASET ────────────────────────────┼──────────────────────────── DATASET
  PEQUENO                             │                             GRANDE
                   QUADRANTE 3        │        QUADRANTE 4
                (Dataset Pequeno)     │     (Dataset Grande)
                                      │
               Extraia features de    │   Faça Fine-Tuning total
               camadas intermediárias │   ou treine do zero.
               (iniciais) da rede,    │   O dataset grande
               pois o topo é inútil.  │   evita overfitting.
                                      │
                                      │
                           MUITO DIFERENTE DO IMAGENET
```

---

## 3. Onde o nosso projeto AgroVision se encaixa?

Em projetos de patologia foliar e imagens agrícolas:
* **Tamanho do Dataset:** Geralmente **Pequeno a Moderado** (algumas centenas ou poucos milhares de fotos).
* **Similaridade:** **Moderadamente Parecido** (são fotos ópticas de plantas com texturas naturais, folhas e cores reais, bem parecidas com as plantas que existem no ImageNet).

### O Diagnóstico de Stanford para o Projeto:
> **Estamos no Quadrante 1 caminhando para o Quadrante 2:**  
> 1. Primeiro congelamos a base e treinamos apenas o classificador final (`model.fc`).
> 2. Quando a acurácia estabilizar, descongelamos **apenas o último bloco residual (`layer4` da ResNet)** com um Learning Rate minúsculo (`1e-5`) para fazer o ajuste fino das texturas de fungos.

---

## 📌 Resumo para Apresentação Técnica:

Em discussões técnicas ou bancas acadêmicas sobre a metodologia do modelo:

> *"Seguindo as diretrizes de Transfer Learning do CS231n de Stanford, evitamos treinar uma rede do zero devido à limitação de amostras no campo. Inicialmente utilizamos a ResNet pré-treinada no ImageNet como extrator de características fixo (congelando as convoluções). Em seguida, aplicamos Fine-Tuning progressivo com taxa de aprendizado diferencial ($10\times$ menor na base convolucional), garantindo que os extratores de bordas universais fossem preservados enquanto a rede se especializava nas patologias foliares."*
