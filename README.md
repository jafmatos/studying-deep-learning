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
   * Lógica do neurônio, por que usamos ReLU (não-linearidade) e Entropia Cruzada (*Cross-Entropy*).
   * Roteiro de vídeos (3Blue1Brown) e leituras chave de Stanford (CS231n).

2. **[GUIA 01: PIL (Pillow) e NumPy para Visão Computacional](GUIA_01_NUMPY_CV.md)**
   * **PIL (Pillow):** Leitura preguiçosa (*lazy*), metadados, `.show()`, modos (`RGB` vs `RGBA`), redimensionamento e recortes espaciais.
   * **A grande pegadinha:** `(Largura, Altura)` no PIL vs `(Altura, Largura)` no NumPy.
   * Representação de imagens como matrizes e a inversão `(H, W, C)` vs `(C, H, W)`.
   * Fatiamento (*Slicing*), máscaras de segmentação e ponte zero-copy com PyTorch.

3. **[GUIA 02: PyTorch para Deep Learning](GUIA_02_PYTORCH_CV.md)**
   * **Engenharia de Software:** Arquitetura modular de diretórios (`src/models`, `src/data`, `src/engine`, `config.py`) e separação de responsabilidades.
   * O motor matemático: Tensores, autograd dinâmico e aceleração por hardware (CPU/GPU).
   * Aritmética de Convoluções: Kernel, Stride, Padding e Pooling adaptativo.
   * Pipeline de dados com `Dataset` e `DataLoader` otimizados.
   * Loop científico de treino/validação com diagnóstico de *Overfitting* e *Early Stopping*.

4. **[GUIA 03: TorchVision: Modelos, Dados e Augmentation](GUIA_03_TORCHVISION_CV.md)**
   * Leitura automática de diretórios com `datasets.ImageFolder`.
   * Pipeline de *Data Augmentation* realista para dados de campo com `transforms`.
   * **Transfer Learning** progressivo com **ResNet18** e **MobileNetV3**.
   * Interpretabilidade com Grad-CAM e a transição para Detecção de Objetos com **YOLO**.

---

## 💻 Códigos Práticos do Repositório

| Arquivo | Descrição | Como Rodar |
| :--- | :--- | :--- |
| `AIGeneratedDemo/01_visao_basica.py` | Demonstra tensores e convolução manual de bordas | `python AIGeneratedDemo/01_visao_basica.py` |
| `AIGeneratedDemo/02_treino_classificador.py` | Script monolítico de demonstração de treino de CNN | `python AIGeneratedDemo/02_treino_classificador.py` |
| `app.py` | Interface interativa em Streamlit com predição e mapas de ativação convolucional | `streamlit run app.py` |
