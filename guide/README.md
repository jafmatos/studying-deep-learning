# 🌿 AgroVision: Trilha de Visão Computacional e Deep Learning

Este repositório reúne uma trilha completa de fundamentos teóricos e implementação prática em Visão Computacional aplicada ao diagnóstico foliar e patologias agrícolas.

O material aplica o **Princípio de Pareto (80/20)** — cobrindo os conceitos e ferramentas essenciais que resolvem 80% dos problemas reais de pesquisa e desenvolvimento sem sobrecarga desnecessária.

---

## 📚 Trilha Completa de Estudos (Passo a Passo)

```
[ GUIA 00: Teoria Essencial ] ──> [ GUIA 01: NumPy ] ──> [ GUIA 02: PyTorch ] ──> [ GUIA 03: TorchVision ]
(Álgebra, Cálculo, Lógica)        (Matrizes & Pixels)    (Redes & Treinamento)    (Transfer Learning & Dados)
```

1. **[GUIA 00: Fundamentos Teóricos Essenciais](GUIA_00_FUNDAMENTOS_TEORICOS.md)** *(Comece aqui!)*
   * Álgebra linear das imagens: Vetores, Matrizes, Tensores e Produto Escalar.
   * Cálculo intuitivo: Derivadas como sensibilidade, Gradiente Descendente e Regra da Cadeia (Backprop).
   * Lógica do neurônio, hiperparâmetros vs parâmetros aprendidos, ReLU e Entropia Cruzada.
   * Roteiro de vídeos (3Blue1Brown) e leituras chave de Stanford (CS231n).

2. **[GUIA 01: PIL (Pillow) e NumPy para Visão Computacional](GUIA_01_NUMPY_CV.md)**
   * **PIL (Pillow):** Leitura preguiçosa (*lazy*), metadados, `.show()`, modos (`RGB` vs `RGBA`), redimensionamento e recortes espaciais.
   * **A grande pegadinha:** `(Largura, Altura)` no PIL vs `(Altura, Largura)` no NumPy.
   * Representação de imagens como matrizes e a inversão `(H, W, C)` vs `(C, H, W)`.
   * Fatiamento (*Slicing*), máscaras de segmentação e ponte zero-copy com PyTorch.

3. **[GUIA 02: PyTorch Prático para Visão Computacional](GUIA_02_PYTORCH_CV.md)**
   * **Engenharia de Software:** Estrutura modular de pastas (`config.py`, `models/`, `data/`, `engine/`) e fluxo de dados.
   * **Tensores e Hardware:** Detecção de dispositivo (`cuda`, `mps`, `cpu`) e gestão de VRAM com `torch.no_grad()`.
   * **Construção de CNNs:** Camadas essenciais do `torch.nn` e o truque do `AdaptiveAvgPool2d((1, 1))`.
   * **Pipeline de Dados:** Criação de `Dataset` customizado (lazy loading com PIL) e `DataLoader` otimizado.
   * **Treinamento e Inferência:** Os 5 passos canônicos do loop, salvamento com `state_dict`, função de predição e troubleshooting 80/20.

4. **[GUIA 03: TorchVision: Modelos, Dados e Augmentation](GUIA_03_TORCHVISION_CV.md)**
   * Leitura automática de diretórios com `datasets.ImageFolder`.
   * Pipeline de *Data Augmentation* realista para dados de campo com `transforms`.
   * **Transfer Learning** progressivo com **ResNet18** e **MobileNetV3**.
   * Interpretabilidade com Grad-CAM e a transição para Detecção de Objetos com **YOLO**.

---

## 💻 Códigos Práticos do Repositório

| Arquivo | Descrição | Como Rodar |
| :--- | :--- | :--- |
| `guide/01_visao_basica.py` | Demonstra tensores e convolução manual de bordas | `python guide/01_visao_basica.py` |
| `guide/02_treino_classificador.py` | Script monolítico de demonstração de treino de CNN | `python guide/02_treino_classificador.py` |
| `guide/app.py` | Interface interativa em Streamlit com predição e mapas de ativação convolucional | `streamlit run guide/app.py` |
