# 📘 Guia Técnico: NumPy para Visão Computacional (Os 20% Fundamentais)
> **Projeto Âncora:** Diagnóstico de Patologias em Folhas e Culturas Agrícolas (AgroVision)  
> **Objetivo:** Dominar a base matemática e computacional de manipulação de matrizes de imagem que você realmente usará no laboratório de pesquisa em Deep Learning.

---

## 1. A Anatomia de uma Imagem como Tensor 3D

Para a máquina, uma imagem digital colorida não tem forma, cor ou contexto: é uma grade discreta de números organizados em 3 eixos: **Altura ($H$)**, **Largura ($W$)** e **Canais de Cor ($C$)**.

```python
import numpy as np
from PIL import Image

# 1. Carregando imagem
img = Image.open("folha_cafe.jpg") # Ex: imagem de 400x600 pixels
arr = np.array(img)

print("Tipo primitivo:", arr.dtype)       # uint8 (inteiros sem sinal: 0 a 255)
print("Formato da matriz (Shape):", arr.shape) # (400, 600, 3) -> (H, W, C)
print("Número de dimensões:", arr.ndim)   # 3
print("Total de pixels/valores:", arr.size) # 400 * 600 * 3 = 720.000 valores
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

## 2. Cores: RGB vs. BGR (A Clássica Armadilha do OpenCV)

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

## 3. Fatiamento Espacial e Máscaras de Segmentação (ROIs)

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

### B) Máscara Booleana para Isolar Clorofila (Folha Verde vs Fundo)
Podemos usar álgebra booleana de arrays para descobrir onde a folha está, filtrando pixels onde o canal Verde (G) predomina sobre o Vermelho (R) e o Azul (B):

```python
# Separando os canais
R = arr_float[:, :, 0]
G = arr_float[:, :, 1]
B = arr_float[:, :, 2]

# Condição para folha verde: Verde maior que Vermelho E maior que Azul
mascara_folha = (G > R) & (G > B) & (G > 0.2) # Matriz booleana de True/False

# Contando a porcentagem de área foliar saudável na imagem:
percentual_verde = (np.count_nonzero(mascara_folha) / (mascara_folha.shape[0] * mascara_folha.shape[1])) * 100
print(f"Área verde identificada: {percentual_verde:.2f}%")

# Substituindo o fundo (tudo que não for folha) por preto:
folha_segmentada = arr_float.copy()
folha_segmentada[~mascara_folha] = 0.0 # O til (~) inverte a máscara booleana
```

---

## 4. O Abismo de Memória: NumPy `(H, W, C)` vs. PyTorch `(C, H, W)`

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

## 5. Dimensão de Lote (Batch Dimension)

Modelos de Deep Learning nunca aceitam uma imagem solta de 3 dimensões $(C, H, W)$. Eles operam sobre tensores de 4 dimensões:
$$\text{Lote} = (B, C, H, W)$$
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

## 6. Calculando a Média e Desvio Padrão Reais do seu Dataset

Em projetos de pesquisa agrícola, muitos desenvolvedores usam cegamente a média do ImageNet (`mean=[0.485, 0.456, 0.406]`). Mas o ImageNet é composto por carros, cachorros e prédios. Se você estiver trabalhando com café ou soja, as imagens são predominantemente verdes e marrons!

Como calcular a média e desvio padrão exatos do seu dataset com NumPy:

```python
import glob

caminhos = glob.glob("dataset_folhas/**/*.jpg", recursive=True)

medias = []
desvios = []

for caminho in caminhos[:100]: # Amostra de 100 imagens
    img_np = np.array(Image.open(caminho).convert('RGB')).astype(np.float32) / 255.0
    # Média e desvio por canal (eixos 0 e 1 = Altura e Largura)
    medias.append(img_np.mean(axis=(0, 1)))
    desvios.append(img_np.std(axis=(0, 1)))

media_dataset = np.mean(medias, axis=0)
desvio_dataset = np.mean(desvios, axis=0)

print("Média exata dos canais [R, G, B]:", media_dataset)
print("Desvio padrão exato [R, G, B]:", desvio_dataset)
# Use esses valores no torchvision.transforms.Normalize()!
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

## 📌 Checklist do Pesquisador em Visão Computacional (NumPy 80/20)

| Problema | Solução NumPy | Por que é vital? |
| :--- | :--- | :--- |
| **Overflow de pixels** | `.astype(np.float32) / 255.0` | Evita que valores $> 255$ reiniciem para zero |
| **Padrão de canais errado** | `np.transpose(arr, (2, 0, 1))` | PyTorch só aceita $(C, H, W)$ para convoluções |
| **Matriz não contígua** | `np.ascontiguousarray(...)` | Impede erros de ponteiro na GPU |
| **Imagem individual rejeitada** | `arr[np.newaxis, ...]` | Transforma imagem 3D no lote 4D $(1, C, H, W)$ |
| **Fundo ruidoso atrapalhando** | Máscaras booleanas `(G > R) & (G > B)` | Segmenta o tecido foliar e descarta solo/vaso |
