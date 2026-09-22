"""
Entrenamiento (fine-tuning) de YOLOv8-seg sobre el dataset TACO preparado,
con seguimiento de experimentos en Weights & Biases.

Compatible con Mac (Apple Silicon -> MPS), GPU NVIDIA (CUDA) y CPU.

Uso:
    python src/train.py
    python src/train.py --epochs 100 --imgsz 640
    python src/train.py --device cpu      # forzar CPU
    python src/train.py --device mps      # forzar GPU de Mac
    python src/train.py --device cuda     # forzar GPU NVIDIA

Estrategia de fine-tuning en dos etapas (justificación en el paper):
    Etapa 1: backbone congelado, se entrenan solo las capas de detección/
             segmentación finales, con LR más alto -> adaptación rápida.
    Etapa 2: se descongela todo el modelo y se afina con LR bajo -> ajuste
             fino de las features al dominio específico de residuos.
"""

import argparse
from pathlib import Path

import wandb
from ultralytics import YOLO

from device_utils import get_device_str

DATA_YAML = Path(__file__).parent.parent / "data" / "taco_yolo" / "data.yaml"
WANDB_PROJECT = "residuos-segmentacion"


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv8-seg para clasificación de residuos")
    parser.add_argument("--model", type=str, default="yolov8s-seg.pt",
                         help="Checkpoint pre-entrenado base (ej: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt)")
    parser.add_argument("--epochs", type=int, default=60, help="Épocas totales (se reparten entre las 2 etapas)")
    parser.add_argument("--freeze-epochs", type=int, default=10,
                         help="Épocas de la Etapa 1 (backbone congelado)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen de entrada")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                         help="Forzar dispositivo: cpu | mps | cuda (default: auto-detección)")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Path al data.yaml")
    parser.add_argument("--project", type=str, default="runs/segment", help="Carpeta de salida de Ultralytics")
    parser.add_argument("--name", type=str, default="residuos_yolov8seg", help="Nombre del experimento")
    parser.add_argument("--no-wandb", action="store_true", help="Desactivar logging a Weights & Biases")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device_str(force=args.device)
    print(f"[train] Dispositivo a utilizar: {device}")

    if not Path(args.data).exists():
        print(f"No se encontró {args.data}. Corré antes: python data/prepare_dataset.py")
        return

    if not args.no_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            name=args.name,
            config={
                "model_base": args.model,
                "epochs": args.epochs,
                "freeze_epochs": args.freeze_epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": device,
            },
        )

    model = YOLO(args.model)

    # -----------------------------------------------------------------
    # Etapa 1: backbone congelado, LR más alto, adaptación rápida
    # -----------------------------------------------------------------
    print("\n=== Etapa 1: fine-tuning con backbone congelado ===")
    model.train(
        data=args.data,
        epochs=args.freeze_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=f"{args.name}_etapa1",
        freeze=10,          # congela las primeras 10 capas (backbone) de YOLOv8
        lr0=1e-3,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Etapa 2: modelo completo descongelado, LR bajo, ajuste fino
    # -----------------------------------------------------------------
    remaining_epochs = max(args.epochs - args.freeze_epochs, 1)
    print(f"\n=== Etapa 2: fine-tuning completo ({remaining_epochs} épocas) ===")

    # Se retoma desde el mejor checkpoint de la etapa 1
    best_ckpt_stage1 = Path(args.project) / f"{args.name}_etapa1" / "weights" / "best.pt"
    model = YOLO(str(best_ckpt_stage1)) if best_ckpt_stage1.exists() else model

    results = model.train(
        data=args.data,
        epochs=remaining_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=f"{args.name}_etapa2",
        freeze=None,
        lr0=1e-4,
        exist_ok=True,
    )

    print("\n✅ Entrenamiento finalizado.")
    print(f"   Mejor checkpoint: {Path(args.project) / f'{args.name}_etapa2' / 'weights' / 'best.pt'}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
