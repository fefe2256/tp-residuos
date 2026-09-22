"""
Front en Streamlit: el usuario sube una foto o video con residuos, y la app
devuelve la imagen/video con cada objeto segmentado y coloreado según el
contenedor de reciclaje correspondiente.

Uso:
    streamlit run app/app.py
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent / "src"))
from inference import ResiduosSegmenter  # noqa: E402
from color_mapping import CONTAINERS  # noqa: E402

DEFAULT_WEIGHTS = str(
    Path(__file__).parent.parent / "runs" / "segment" / "residuos_yolov8seg_etapa2" / "weights" / "best.pt"
)

st.set_page_config(page_title="Clasificador de Residuos", page_icon="♻️", layout="centered")

st.title("♻️ Clasificador de Residuos por Visión por Computadora")
st.write(
    "Subí una foto o un video con residuos y el sistema va a detectar cada objeto, "
    "identificar su material y mostrarte a qué contenedor corresponde."
)

with st.expander("🎨 Convención de colores de contenedores"):
    for container in CONTAINERS.values():
        st.markdown(f"{container.emoji} **{container.label}**")


@st.cache_resource
def load_segmenter(weights_path: str, device: str | None):
    return ResiduosSegmenter(weights_path, device=device)


st.sidebar.header("Configuración")
weights_path = st.sidebar.text_input("Path del modelo entrenado (.pt)", value=DEFAULT_WEIGHTS)
device_choice = st.sidebar.selectbox("Dispositivo", options=["Auto", "cpu", "mps", "cuda"], index=0)
conf_threshold = st.sidebar.slider("Umbral de confianza", min_value=0.1, max_value=0.95, value=0.5, step=0.05)

device_arg = None if device_choice == "Auto" else device_choice

if not Path(weights_path).exists():
    st.warning(
        f"No se encontró el modelo en `{weights_path}`. "
        "Entrená primero con `python src/train.py` o ajustá el path en la barra lateral."
    )
    st.stop()

segmenter = load_segmenter(weights_path, device_arg)
segmenter.conf = conf_threshold

tab_imagen, tab_video = st.tabs(["📷 Imagen", "🎥 Video"])

with tab_imagen:
    uploaded_image = st.file_uploader("Subí una imagen", type=["jpg", "jpeg", "png"], key="img")

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Procesando imagen..."):
            overlay_rgb, detections = segmenter.predict_image(img_bgr)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with col2:
            st.image(overlay_rgb, caption="Segmentado", use_container_width=True)

        st.subheader("Detecciones")
        if detections:
            for d in detections:
                st.markdown(f"- {d['label']}")
        else:
            st.info("No se detectaron residuos con el umbral de confianza actual.")

with tab_video:
    uploaded_video = st.file_uploader("Subí un video", type=["mp4", "avi", "mov"], key="vid")

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            input_path = tmp_in.name

        output_path = str(Path(tempfile.gettempdir()) / "resultado_segmentado.mp4")

        with st.spinner("Procesando video (puede tardar unos minutos)..."):
            segmenter.predict_video(input_path, output_path, sample_every_n=2)

        st.video(output_path)

st.sidebar.markdown("---")
st.sidebar.caption("Trabajo Práctico Final — Visión por Computadora II (CEIA)")
