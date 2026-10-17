"""
01_visao_basica.py
Demonstração dos fundamentos de Visão Computacional com PyTorch:
1. Formato de imagem em PyTorch: (C, H, W) vs OpenCV/Numpy (H, W, C)
2. Como uma camada de convolução (nn.Conv2d) detecta padrões (bordas, linhas)
"""

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

def criar_imagem_exemplo():
    """Cria uma imagem sintética com formas geométricas para visualização."""
    img = Image.new('RGB', (128, 128), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Círculo
    draw.ellipse([20, 20, 60, 60], fill=(220, 50, 50), outline=(0, 0, 0), width=2)
    # Retângulo
    draw.rectangle([70, 40, 110, 100], fill=(50, 100, 220), outline=(0, 0, 0), width=2)
    # Linha diagonal
    draw.line([10, 110, 118, 110], fill=(30, 30, 30), width=4)
    return img

def main():
    print("=" * 60)
    print(" FUNDAMENTOS DE VISÃO COMPUTACIONAL COM PYTORCH")
    print("=" * 60)

    # 1. Carregar imagem
    pil_img = criar_imagem_exemplo()
    print("\n1. Imagem criada com PIL:")
    print(f"   Tamanho (Largura x Altura): {pil_img.size}")

    # 2. Converter para Tensor PyTorch
    # No PyTorch o formato padrão é sempre: (Batch, Canais, Altura, Largura) -> (B, C, H, W)
    np_img = np.array(pil_img).astype(np.float32) / 255.0  # Normalizado entre 0.0 e 1.0
    
    # Transpor de (H, W, C) para (C, H, W)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1)
    
    # Adicionar dimensão de Batch: (1, C, H, W)
    batch_img = tensor_img.unsqueeze(0)
    print(f"   Shape do Tensor PyTorch: {batch_img.shape} -> (Batch, Canais, Altura, Largura)")

    # 3. Criar uma convolução para detecção de bordas verticais e horizontais (Filtro Sobel)
    # Uma CNN aprende esses filtros automaticamente durante o treinamento!
    sobel_v = torch.tensor([
        [-1.,  0.,  1.],
        [-2.,  0.,  2.],
        [-1.,  0.,  1.]
    ]).view(1, 1, 3, 3)

    sobel_h = torch.tensor([
        [-1., -2., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  2.,  1.]
    ]).view(1, 1, 3, 3)

    # Converter imagem para escala de cinza para aplicar o filtro simples
    gray_img = batch_img.mean(dim=1, keepdim=True)  # (1, 1, 128, 128)

    # Aplicar convolução manual (usando F.conv2d)
    borda_v = torch.nn.functional.conv2d(gray_img, sobel_v, padding=1)
    borda_h = torch.nn.functional.conv2d(gray_img, sobel_h, padding=1)
    bordas = torch.sqrt(borda_v**2 + borda_h**2)

    # 4. Salvar gráfico explicativo
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(pil_img)
    axes[0].set_title("1. Imagem Original (RGB)")
    axes[0].axis('off')

    axes[1].imshow(gray_img.squeeze().numpy(), cmap='gray')
    axes[1].set_title("2. Escala de Cinza (Tensor 1x128x128)")
    axes[1].axis('off')

    axes[2].imshow(bordas.squeeze().detach().numpy(), cmap='inferno')
    axes[2].set_title("3. Mapa de Características (Bordas)\nFiltro Convolucional")
    axes[2].axis('off')

    plt.tight_layout()
    output_path = "exemplo_convolucao.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\n[OK] Gráfico explicativo salvo em: {output_path}")
    print("\nConclusão visual:")
    print("-> Em Machine Learning clássico, você precisaria calcular essas bordas manualmente.")
    print("-> Em Deep Learning, as camadas Conv2d aprendem os melhores filtros sozinhas a partir dos dados!")
    print("=" * 60)

if __name__ == '__main__':
    main()
