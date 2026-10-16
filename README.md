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

2. **[GUIA 01: NumPy para Visão Computacional](GUIA_01_NUMPY_CV.md)**
   * Representação de imagens `(H, W, C)` vs `(C, H, W)`.
   * Fatiamento (*Slicing*) e recorte de lesões/ROIs em folhas.
   * Normalização, padronização e ponte de memória zero-copy com Tensores PyTorch.

3. **[GUIA 02: PyTorch para Deep Learning](GUIA_02_PYTORCH_CV.md)**
   * A estrutura canônica de modelos herdando de `nn.Module`.
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
| `01_visao_basica.py` | Mostra como tensores funcionam e aplica convolução manual de bordas | `python 01_visao_basica.py` |
| `02_treino_classificador.py` | Treina uma CNN real com PyTorch em 20 segundos e salva os pesos | `python 02_treino_classificador.py` |
| `app.py` | Interface interativa em Streamlit com predição e mapas de ativação convolucional | `streamlit run app.py` |
