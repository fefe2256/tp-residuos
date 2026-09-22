"""
Inferencia sobre imagen o video: detecta y segmenta cada residuo, y superpone
la máscara coloreada según el contenedor de reciclaje correspondiente.

Compatible con Mac (MPS), CUDA y CPU.

Uso como script:
    python src/inference.py --source foto.jpg --weights runs/segment/residuos_yolov8seg_etapa2/weights/best.pt
    python src/inference.py --source video.mp4 --weights best.pt --output resultado.mp4

Uso como módulo (para el front de Streamlit):
    from inference import ResiduosSegmenter
    segmenter = ResiduosSegmenter("best.pt")
    annotated_img, detections = segmenter.predict_image(img_bgr)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from color_mapping import get_container_for_material, format_detection_label
from device_utils import get_device_str

CONF_THRESHOLD = 0.5


class ResiduosSegmenter:
    def __init__(self, weights_path: str, device: str | None = None, conf: float = CONF_THRESHOLD):
        self.device = get_device_str(force=device)
        self.model = YOLO(weights_path)
        self.conf = conf
        print(f"[ResiduosSegmenter] Modelo cargado en dispositivo: {self.device}")

    def predict_image(self, img_bgr: np.ndarray):
        """
        Corre inferencia sobre una imagen (array BGR de OpenCV) y devuelve:
        - imagen anotada con máscaras coloreadas según contenedor (RGB, lista para mostrar)
        - lista de detecciones: [{"material":..., "confianza":..., "contenedor":...}, ...]
        """
        results = self.model.predict(
            source=img_bgr, device=self.device, conf=self.conf, verbose=False
        )
        r = results[0]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        detections = []

        if r.masks is None:
            return img_rgb, detections

        h, w = img_rgb.shape[:2]

        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            material = self.model.names[cls_id]

            mask = r.masks.data[i].detach().cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            container = get_container_for_material(material)
            color = np.array(container.color_rgb, dtype=np.uint8)

            colored_mask = np.zeros_like(img_rgb)
            colored_mask[mask == 1] = color
            overlay = np.where(mask[..., None] == 1,
                                cv2.addWeighted(overlay, 0.5, colored_mask, 0.5, 0),
                                overlay)

            detections.append({
                "material": material,
                "confianza": conf,
                "contenedor": container.label,
                "color": container.color_name,
                "label": format_detection_label(material, conf),
            })

        return overlay, detections

    def predict_video(self, source_path: str, output_path: str, sample_every_n: int = 1):
        """
        Procesa un video frame por frame (o cada N frames) y guarda el resultado
        con las máscaras coloreadas superpuestas.
        """
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        last_overlay_bgr = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                overlay_rgb, _ = self.predict_image(frame)
                last_overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            writer.write(last_overlay_bgr if last_overlay_bgr is not None else frame)
            frame_idx += 1

        cap.release()
        writer.release()
        print(f"✅ Video procesado guardado en: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia de segmentación de residuos")
    parser.add_argument("--source", type=str, required=True, help="Imagen o video de entrada")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint del modelo entrenado")
    parser.add_argument("--output", type=str, default=None, help="Path de salida (solo para video)")
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda (default: auto)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Umbral de confianza")
    return parser.parse_args()


def main():
    args = parse_args()
    segmenter = ResiduosSegmenter(args.weights, device=args.device, conf=args.conf)

    source_path = Path(args.source)
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if source_path.suffix.lower() in video_exts:
        output = args.output or str(source_path.with_name(source_path.stem + "_resultado.mp4"))
        segmenter.predict_video(str(source_path), output)
    else:
        img_bgr = cv2.imread(str(source_path))
        overlay_rgb, detections = segmenter.predict_image(img_bgr)

        for d in detections:
            print(d["label"])

        output = args.output or str(source_path.with_name(source_path.stem + "_resultado.jpg"))
        cv2.imwrite(output, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        print(f"✅ Imagen procesada guardada en: {output}")


if __name__ == "__main__":
    main()
