"""
app.py
Interface Streamlit para inferência e interpretabilidade de Visão Computacional.
Carrega a SimpleCNN treinada em Fashion-MNIST e visualiza o que a rede enxerga.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import pandas as pd

# 1. Configuração da Página
st.set_page_config(
    page_title="Deep Learning CV - AgroVision", page_icon="👁️", layout="wide"
)

CLASSES = [
    "Camiseta/Top",
    "Calça",
    "Suéter",
    "Vestido",
    "Casaco",
    "Sandália",
    "Camisa",
    "Tênis",
    "Bolsa",
    "Bota",
]


# 2. Definição da Arquitetura (Idêntica ao treino)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        out = self.classifier(f2)
        return out, f1


@st.cache_resource
def carregar_modelo():
    model = SimpleCNN()
    try:
        model.load_state_dict(
            torch.load("modelo_cnn_fashion.pth", map_location=torch.device("cpu"))
        )
        model.eval()
        return model, True
    except FileNotFoundError:
        return model, False


@st.cache_data
def carregar_amostras_teste():
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return test_ds


# 3. Cabeçalho
st.title("👁️ Visão Computacional com PyTorch & Streamlit")
st.caption(
    "Do Clássico ao Deep Learning: Demonstração Prática de Inferência e Mapas de Características"
)

model, model_loaded = carregar_modelo()

if not model_loaded:
    st.warning(
        "⚠️ O modelo ainda não foi treinado. Execute no terminal: `python 02_treino_classificador.py`"
    )

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("1. Selecione ou Envie uma Imagem")
    metodo = st.radio(
        "Origem da Imagem:",
        ["Amostra do Test Set", "Upload de Imagem"],
        horizontal=True,
    )

    imagem_tensor = None
    imagem_pil = None
    label_real = None

    if metodo == "Amostra do Test Set":
        test_data = carregar_amostras_teste()
        idx = st.slider("Índice da Amostra de Teste:", 0, 500, 15)
        tensor_bruto, label_idx = test_data[idx]
        label_real = CLASSES[label_idx]

        # Converter tensor para PIL para exibição
        img_np = (tensor_bruto.squeeze().numpy() * 255).astype(np.uint8)
        imagem_pil = Image.fromarray(img_np, mode="L")

        # Normalizar para inferência ([-1, 1])
        imagem_tensor = ((tensor_bruto - 0.5) / 0.5).unsqueeze(0)

        st.image(
            imagem_pil, caption=f"Imagem Real: {label_real} (28x28 pixels)", width=180
        )
    else:
        uploaded_file = st.file_uploader(
            "Envie uma imagem (PNG/JPG):", type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            raw_img = Image.open(uploaded_file).convert("L").resize((28, 28))
            imagem_pil = raw_img
            img_arr = np.array(raw_img).astype(np.float32) / 255.0
            imagem_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)
            imagem_tensor = (imagem_tensor - 0.5) / 0.5
            st.image(
                imagem_pil,
                caption="Imagem Carregada (Redimensionada para 28x28)",
                width=180,
            )

with col_right:
    st.subheader("2. Predição da Rede Neural")
    if imagem_tensor is not None and model_loaded:
        with torch.no_grad():
            logits, feature_maps = model(imagem_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        pred_idx = np.argmax(probs)
        pred_nome = CLASSES[pred_idx]
        confianca = probs[pred_idx] * 100

        # Alerta de resultado
        st.metric(
            label="Classe Predita",
            value=f"{pred_nome}",
            delta=f"{confianca:.1f}% Confiança",
        )
        if label_real:
            if pred_nome == label_real:
                st.success(f"Acerto! O rótulo real é de fato **{label_real}**.")
            else:
                st.error(f"Erro: O rótulo real era **{label_real}**.")

        # Gráfico de probabilidades
        df_probs = pd.DataFrame(
            {"Classe": CLASSES, "Probabilidade (%)": [round(p * 100, 2) for p in probs]}
        ).sort_values(by="Probabilidade (%)", ascending=True)

        st.bar_chart(df_probs.set_index("Classe"))

        # 3. Visão interna: O que a Convolução aprendeu?
        st.markdown("---")
        st.subheader("3. O que a CNN enxerga internamente? (Feature Maps da Conv1)")
        st.caption(
            "Estes são os 16 canais gerados pela primeira camada convolucional. Veja como cada filtro captura bordas e relevos diferentes!"
        )

        cols = st.columns(8)
        f_maps = feature_maps.squeeze().numpy()  # (16, 14, 14)
        for i in range(16):
            col_target = cols[i % 8]
            col_target.image(f_maps[i], caption=f"Filtro {i+1}", clamp=True, width=50)

st.markdown("---")
st.info(
    "💡 **Dica Técnica:** Este projeto demonstra o ciclo completo de Deep Learning: carregamento de tensores, arquitetura modular em PyTorch, métricas de confiança softmax e interpretabilidade através dos mapas de ativação."
)
