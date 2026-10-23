from pathlib import Path
import numpy
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch

from deep_learning.config import (
    DEVICE,
    IMAGE_SIZE,
    MODEL_PATH,
    ROOT_DIR,
    TEST_DATA_DIR,
)
from deep_learning.engine.predictor import predict_single_image, transform
from deep_learning.models.simple_cnn import SimpleCNN

# Configurações de validação
CLASSES = ["Healthy", "Powdery", "Rust"]
MAX_FILE_SIZE_MB = 10

# Recomendações técnicas por patologia
RECOMMENDATIONS = {
    "Healthy": {
        "title": "Folha Saudável",
        "description": "Nenhum sinal aparente de infecção fúngica ou estresse patológico detectado.",
        "actions": [
            "Manter o cronograma de irrigação e adubação equilibrada.",
            "Continuar o monitoramento periódico da lavoura.",
            "Preservar a aeração adequada entre os dosséis vegetativos.",
        ],
        "type": "success",
    },
    "Powdery": {
        "title": "Oídio Detectado (Powdery Mildew)",
        "description": "Presença característica de manchas pulverulentas brancas/acinzentadas na superfície da folha.",
        "actions": [
            "Evitar irrigação por aspersão nas horas mais frias do dia.",
            "Promover poda de aeração em áreas muito adensadas para reduzir umidade residual.",
            "Avaliar aplicação de fungicidas ou caldas recomendadas.",
            "Isolar plantas severamente atacadas para evitar a dispersão de esporos.",
        ],
        "type": "warning",
    },
    "Rust": {
        "title": "Ferrugem Detectada (Rust)",
        "description": "Presença de pústulas alaranjadas a castanho-avermelhadas típicas de fungos biotróficos.",
        "actions": [
            "Realizar inspeção detalhada e avaliar aplicação de fungicidas sistêmicos.",
            "Remover e descartar folhas basais com alto grau de infecção.",
            "Desinfetar ferramentas após o manejo das plantas afetadas.",
            "Monitorar a dispersão em áreas a favor do vento.",
        ],
        "type": "error",
    },
}


@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Carrega os pesos do modelo em cache."""
    model = SimpleCNN(in_channels=3, num_classes=len(CLASSES)).to(DEVICE)

    weights_path = None
    if MODEL_PATH.exists():
        weights_path = MODEL_PATH
    elif (ROOT_DIR / "melhor_modelo.pth").exists():
        weights_path = ROOT_DIR / "melhor_modelo.pth"

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    return None


st.title("Diagnóstico Foliar")
st.markdown(
    "Envie uma imagem ou selecione uma amostra de teste para análise automatizada."
)

# Carregamento do modelo
model = load_trained_model()

if model is None:
    st.error(
        "Não foi possível carregar o modelo de classificação. Verifique os arquivos de pesos."
    )
    st.stop()

# Layout em duas colunas principais
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Entrada da Imagem")

    tab_upload, tab_samples = st.tabs(["Upload de Arquivo", "Amostras de Teste"])

    selected_image = None
    image_origin = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Selecione uma imagem (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help=f"Tamanho máximo permitido: {MAX_FILE_SIZE_MB} MB.",
            max_upload_size=MAX_FILE_SIZE_MB,
        )
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(
                    f"O arquivo enviado tem {file_size_mb:.2f} MB e excede o limite máximo permitido de {MAX_FILE_SIZE_MB} MB."
                )
            else:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_origin = f"{uploaded_file.name} ({file_size_mb:.2f} MB)"

    with tab_samples:
        st.caption("Amostras do conjunto de teste:")

        sample_options = {}
        if TEST_DATA_DIR.exists():
            for cls_dir in sorted(TEST_DATA_DIR.iterdir()):
                if cls_dir.is_dir():
                    images_in_cls = list(cls_dir.glob("*.jpg")) + list(
                        cls_dir.glob("*.png")
                    )
                    for i, img_path in enumerate(images_in_cls[:3], 1):
                        sample_options[f"{cls_dir.name} - Amostra {i}"] = img_path

        if sample_options:
            selected_sample_label = st.selectbox(
                "Amostra:",
                options=list(sample_options.keys()),
                index=0,
            )
            if uploaded_file is None:
                sample_path = sample_options[selected_sample_label]
                selected_image = Image.open(sample_path).convert("RGB")
                image_origin = selected_sample_label
        else:
            st.info("Nenhuma amostra encontrada no diretório de testes.")

    # Pré-visualização da imagem
    if selected_image is not None:
        st.markdown(f"**Pré-visualização:** {image_origin}")
        st.image(selected_image, use_container_width=True)

with col_result:
    st.subheader("Diagnóstico")

    if selected_image is None:
        st.info("Selecione ou envie uma imagem à esquerda para iniciar o diagnóstico.")
    else:
        if st.button("Executar Diagnóstico", type="primary", use_container_width=True):
            with st.spinner("Processando diagnóstico..."):
                # Pré-processamento
                fitted_image = ImageOps.fit(
                    selected_image, IMAGE_SIZE, method=Image.Resampling.BILINEAR
                )
                image_numpy = numpy.array(fitted_image)

                # Inferência
                result = predict_single_image(
                    model=model,
                    classes=CLASSES,
                    device=DEVICE,
                    tensor=image_numpy,
                    transform=transform,
                )

                st.session_state["last_result"] = {
                    "predicted_class": result["predicted_class"],
                    "certainty": result["certainty"],
                    "probabilities": result["probabilities"],
                }

        # Exibição dos resultados
        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            pred_cls = res["predicted_class"]
            cert = res["certainty"]
            probs = res["probabilities"]

            rec = RECOMMENDATIONS.get(pred_cls, {})

            # Box de resultado por gravidade
            title_text = f"**{rec.get('title')}** — Confiança: {cert * 100:.1f}%"
            if rec.get("type") == "success":
                st.success(title_text)
            elif rec.get("type") == "warning":
                st.warning(title_text)
            else:
                st.error(title_text)

            st.caption(rec.get("description", ""))

            # Gráfico de probabilidades
            st.markdown("**Distribuição de Probabilidades:**")
            df_probs = pd.DataFrame(
                {
                    "Classe": list(probs.keys()),
                    "Probabilidade (%)": [round(p * 100, 2) for p in probs.values()],
                }
            ).set_index("Classe")

            st.bar_chart(df_probs, y="Probabilidade (%)", color="#2e7d32")

            # Orientações agronômicas
            with st.expander("Recomendações Técnicas", expanded=True):
                for action in rec.get("actions", []):
                    st.markdown(f"- {action}")
