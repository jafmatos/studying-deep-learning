# 📘 Guia Técnico: PIL (Pillow) e NumPy para Visão Computacional (Os 20% Fundamentais)

> **Projeto Âncora:** Diagnóstico de Patologias em Folhas e Culturas Agrícolas (AgroVision)
> **Objetivo:** Dominar a base computacional de manipulação de imagens digitais (PIL) e matrizes numéricas (NumPy) que você realmente usará no laboratório de pesquisa em Deep Learning.

---

## 1. PIL (Pillow): A Porta de Entrada da Visão Computacional

Antes de uma imagem se tornar uma matriz numérica no NumPy ou um tensor no PyTorch, ela existe como um arquivo no disco (`.png`, `.jpg`, `.webp`). A biblioteca padrão absoluta do ecossistema Python para abrir, inspecionar, converter e salvar imagens é o **Pillow** (`PIL` - *Python Imaging Library*).

O próprio **TorchVision** usa o PIL internamente por padrão para carregar imagens em datasets como `datasets.ImageFolder`.

### A) Leitura Preguiçosa (*Lazy Loading*) e Metadados

Ao executar `Image.open()`, o Pillow **não carrega todos os pixels na memória RAM imediatamente**. Ele lê apenas o cabeçalho do arquivo para extrair os metadados fundamentais, economizando recursos em pipelines que lidam com milhares de fotos:

```python
from PIL import Image

img = Image.open("data/divina-misericordia.png")

print("Formato do arquivo:", img.format) # Ex: 'PNG', 'JPEG'
print("Modo de cores:", img.mode)       # Ex: 'RGB', 'RGBA', 'L'
print("Dimensões (W, H):", img.size)     # (Largura, Altura) -> Ex: (726, 908)
```

### B) A Pegadinha Nº 1: `(Largura, Altura)` vs `(Altura, Largura)`

Essa é uma das maiores fontes de bugs para quem começa em Visão Computacional:

* No **PIL**: `img.size` devolve `(Largura, Altura)` — o eixo X vem primeiro!
* No **NumPy**: `arr.shape` devolve `(Altura, Largura, Canais)` — as linhas (eixo Y) vêm primeiro!
* No **PyTorch**: `tensor.shape` exige `(Canais, Altura, Largura)` ou `(Batch, Canais, Altura, Largura)`.

> [!WARNING]
> Confundir Largura com Altura ao redimensionar ou fatiar imagens causa distorções silenciosas de proporção (*aspect ratio*) ou quebras de dimensão na entrada da rede.

### C) Os Modos de Cores (`.mode`) e a Conversão Obrigatória `.convert('RGB')`

* `'RGB'`: 3 canais de 8 bits (Vermelho, Verde, Azul). É o formato que as redes neurais convolucionais esperam.
* `'RGBA'`: 4 canais (RGB + **A**lpha / canal de transparência). Muito comum em arquivos `.png`.
* `'L'`: 1 canal de 8 bits (Escala de cinza / *Luminance*, 0 a 255).

**Por que o canal Alpha quebra redes neurais?**
Se você carregar um PNG com transparência (`RGBA`), a matriz resultante terá **4 canais**. Se a sua CNN foi projetada para receber 3 canais (`in_channels=3`), o PyTorch lançará uma exceção imediata:
`RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input to have 3 channels, but got 4 channels instead!`

**A Regra de Ouro em Pipelines Profissionais:**

```python
# SEMPRE converta explicitamente para 'RGB':
img = Image.open("caminho.png").convert("RGB")
```

### D) Exibição e Depuração Rápida

* `img.show()`: Cria um arquivo temporário e abre a imagem no visualizador padrão do seu sistema operacional. É a forma mais rápida de inspecionar imagens via terminal/desktop.
* Em interfaces (como Streamlit), você passa a imagem diretamente: `st.image(img)`.

### E) As 4 Operações de Geometria Essenciais no PIL

```python
# 1. Redimensionar forçando dimensões exatas (Largura, Altura)
img_224 = img.resize((224, 224), Image.Resampling.LANCZOS)

# 2. Thumbnail: Reduz mantendo o aspecto original sem distorcer (modifica in-place)
img_thumb = img.copy()
img_thumb.thumbnail((300, 300))

# 3. Recorte Espacial (Crop): Requer caixa delimitadora (left, top, right, bottom)
# Coordenadas: (x1, y1, x2, y2)
largura, altura = img.size
caixa = (largura // 4, altura // 4, 3 * largura // 4, 3 * altura // 4)
crop_centro = img.crop(caixa)

# 4. Rotação e Espelhamento
img_girada = img.rotate(45, expand=True) # expand=True evita cortar os cantos da imagem
img_espelhada = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

# 5. Salvamento no disco
img_224.save("saida.jpg", quality=95)
```

### F) A Ponte Bidirecional: PIL $\leftrightarrow$ NumPy

```python
import numpy as np

# PIL -> NumPy (uint8: 0 a 255, formato HWC)
arr = np.array(img)

# NumPy -> PIL (utePermite voltar da matriz numérica para salvar ou exibir)
img_reconstruida = Image.fromarray(arr)
```

---

## 2. A Anatomia de uma Imagem como Tensor 3D (NumPy)

Para a máquina, uma imagem digital colorida não tem forma, cor ou contexto: é uma grade discreta de números organizados em 3 eixos: **Altura ($H$)**, **Largura ($W$)** e **Canais de Cor ($C$)**.

```python
import numpy as np
from PIL import Image

# 1. Carregando imagem e convertendo para matriz
img = Image.open("data/divina-misericordia.png").convert("RGB")
arr = np.array(img)

print("Tipo primitivo:", arr.dtype)       # uint8 (inteiros sem sinal: 0 a 255)
print("Formato da matriz (Shape):", arr.shape) # (H, W, C)
print("Número de dimensões:", arr.ndim)   # 3
print("Total de pixels/valores:", arr.size)
```

### O Perigo do Tipo `uint8` em Cálculos Matemáticos

O tipo padrão de imagens no NumPy é `uint8` (1 byte por valor, de $0$ a $255$).

* Se você somar $250 + 10$ no formato `uint8`, o valor dá a volta (*overflow*) e vira **$4$**!
* Por isso, **antes de qualquer operação matemática, normalização ou cálculo de gradientes**, a imagem deve ser convertida para ponto flutuante (`float32`):

```python
# FORMA CORRETA: Cast para float32 antes de normalizar
arr_float = arr.astype(np.float32) / 255.0
print("Valores agora entre:", arr_float.min(), "e", arr_float.max()) # 0.0 e 1.0
```

---

## 3. Cores: RGB vs. BGR (A Clássica Armadilha do OpenCV)

No ecossistema Python de visão computacional, existem dois padrões concorrentes:

* **PIL, Matplotlib, PyTorch, Streamlit:** Usam a ordem **RGB** (Canal 0 = Red, Canal 1 = Green, Canal 2 = Blue).
* **OpenCV (`cv2`):** Carrega por padrão histórico como **BGR** (Canal 0 = Blue, Canal 1 = Green, Canal 2 = Red).

Se você abrir a foto de uma folha com OpenCV e tentar exibi-la no Matplotlib ou passá-la para o PyTorch sem converter, uma folha verde saudável ficará azulada ou arroxeada, arruinando a acurácia da rede.

```python
# Inversão manual de canais com NumPy slicing:
# Pega todas as linhas (:), todas as colunas (:), e inverte os canais (::-1)
bgr_para_rgb = arr[:, :, ::-1]
```

---

## 4. Fatiamento Espacial e Máscaras de Segmentação (ROIs)

Em patologia vegetal e agricultura de precisão, o agrônomo frequentemente fotografa a planta com terra, gravetos, mãos ou vasos ao fundo. O NumPy permite recortar a região de interesse (*Region of Interest* - ROI) ou criar máscaras de segmentação.

### A) Recorte Espacial (Crop de Lesão)

```python
# Sintaxe: matriz[y_inicio:y_fim, x_inicio:x_fim, canais]
# Exemplo: Isolando uma mancha de fungo no terço superior direito da folha
lesao_fungo = arr[50:180, 320:450, :]

# ATENÇÃO À MEMÓRIA: Slicing no NumPy cria uma visualização (view) compartilhada.
# Se alterar lesao_fungo, arr original será alterado!
# Para criar uma cópia isolada na memória RAM:
lesao_isolada = lesao_fungo.copy()
```

### B) Máscara Booleana de Cor (Segmentação por Predominância de Verde)

Tecnicamente, essa máscara **não sabe o que é uma folha** — ela apenas localiza **onde existe predominância da cor verde** na foto (filtrando pixels onde o canal Verde $G$ é superior ao Vermelho $R$ e ao Azul $B$).

Em um cenário controlado (ex: uma folha sobre uma bancada branca ou fundo neutro), essa regra permite isolar a área verde da folha. Porém, se houver grama, mato ou outro objeto verde na cena, a regra selecionará tudo indistintamente, pois filtros de cor são "cegos" para forma e contexto espacial:

```python
# Separando os canais
R = arr_float[:, :, 0]
G = arr_float[:, :, 1]
B = arr_float[:, :, 2]

# Condição: Verde maior que Vermelho E maior que Azul (com intensidade mínima)
mascara_verde = (G > R) & (G > B) & (G > 0.2) # Matriz booleana de True/False

# Contando a porcentagem de pixels verdes na imagem:
percentual_verde = (np.count_nonzero(mascara_verde) / (mascara_verde.shape[0] * mascara_verde.shape[1])) * 100
print(f"Presença de pixels verdes na imagem: {percentual_verde:.2f}%")

# Substituindo tudo o que não for verde por preto:
imagem_segmentada = arr_float.copy()
imagem_segmentada[~mascara_verde] = 0.0 # O til (~) inverte a máscara booleana
```

---

## 5. O Abismo de Memória: NumPy `(H, W, C)` vs. PyTorch `(C, H, W)`

Por que o PyTorch exige os canais na frente?
As placas de vídeo (GPUs) realizam convoluções lendo os pixels em blocos contíguos de memória. Ter todos os valores do canal Vermelho juntos, seguidos pelos do Verde e do Azul, acelera brutalmente o produto escalar dos filtros convolucionais.

### A Diferença Fatal entre `transpose()` e `reshape()`

* **`np.transpose()` / `arr.transpose(2, 0, 1)`:** Altera a ordem dos eixos geométricos mantendo a integridade espacial dos dados.
* **`arr.reshape(3, H, W)`:** **NUNCA USE PARA ISSO!** O `reshape` lê a memória linearmente e a reorganiza. O canal 0 receberá uma mistura caótica de linhas de cores diferentes, destruindo a imagem em ruído irreconhecível.

```python
# 1. Forma correta: Transpor os eixos (2 -> 0, 0 -> 1, 1 -> 2)
arr_pytorch = np.transpose(arr_float, (2, 0, 1))
print("Formato correto para PyTorch:", arr_pytorch.shape) # (3, 400, 600)

# 2. Reverter de volta para Matplotlib/Streamlit:
arr_visualizacao = np.transpose(arr_pytorch, (1, 2, 0))
print("Formato para exibição:", arr_visualizacao.shape)   # (400, 600, 3)
```

### Contiguidade de Memória (`ascontiguousarray`)

Após um `transpose`, os dados na memória não estão mais dispostos sequencialmente em blocos contíguos. O PyTorch lançará um erro se você tentar criar um tensor diretamente. A solução padrão:

```python
arr_contiguo = np.ascontiguousarray(arr_pytorch)
```

---

## 6. Dimensão de Lote (Batch Dimension)

Modelos de Deep Learning nunca aceitam uma imagem solta de 3 dimensões $(C, H, W)$. Eles operam sobre tensores de 4 dimensões:

$$
\text{Lote} = (B, C, H, W)
$$

Onde $B$ é o tamanho do lote (*Batch Size*). Para inferir uma única imagem de teste:

```python
# Adicionar dimensão de batch (B = 1):
lote_1 = np.expand_dims(arr_contiguo, axis=0) # Shape: (1, 3, 400, 600)
# Ou de forma mais pythônica:
lote_2 = arr_contiguo[np.newaxis, ...]        # Shape: (1, 3, 400, 600)

# Remover a dimensão após predição:
imagem_unica = np.squeeze(lote_1, axis=0)     # Retorna para (3, 400, 600)
```

---

## 7. A Ponte Zero-Copy: NumPy $\leftrightarrow$ PyTorch

Para evitar gargalos de processamento, a transferência entre NumPy e PyTorch pode ser feita sem copiar bytes na memória RAM:

```python
import torch

# 1. NumPy -> PyTorch (Zero-Copy: apontam para o mesmo endereço de memória!)
tensor_folha = torch.from_numpy(arr_contiguo)

# CUIDADO: Modificar arr_contiguo agora alterará tensor_folha automaticamente!

# 2. PyTorch -> NumPy (Após inferência com a rede neural)
# É obrigatório chamar:
# .detach() -> Desconecta do grafo de derivadas automáticas
# .cpu()    -> Traz da GPU/VRAM para a memória RAM do sistema
# .numpy()  -> Converte para array NumPy
predicoes_numpy = tensor_folha.detach().cpu().numpy()
```

---

## 📌 Checklist do Pesquisador em Visão Computacional (PIL & NumPy 80/20)

| Problema                                         | Solução                                 | Por que é vital?                                                 |
| :----------------------------------------------- | :---------------------------------------- | :---------------------------------------------------------------- |
| **PNG com 4 canais (Alpha) quebrando CNN** | `Image.open(...).convert('RGB')`        | Descarta transparência e garante 3 canais exatos                 |
| **Confundir Largura e Altura**             | PIL: `(W, H)` vs NumPy: `(H, W)`      | Evita distorções no resize e recortes desalinhados              |
| **Overflow de pixels**                     | `.astype(np.float32) / 255.0`           | Evita que valores $> 255$ reiniciem para zero em `uint8`      |
| **Padrão de canais errado**               | `np.transpose(arr, (2, 0, 1))`          | PyTorch só aceita $(C, H, W)$ para convoluções               |
| **Matriz não contígua**                  | `np.ascontiguousarray(...)`             | Impede erros de ponteiro na GPU                                   |
| **Imagem individual rejeitada**            | `arr[np.newaxis, ...]`                  | Transforma imagem 3D no lote 4D $(1, C, H, W)$                  |
| **Fundo neutro com solo/bancada**          | Máscaras booleanas `(G > R) & (G > B)` | Filtra pixels verdes em cenários controlados (sem mato no fundo) |
