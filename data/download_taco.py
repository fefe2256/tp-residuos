"""
Descarga y organiza el dataset TACO (Trash Annotations in Context).

TACO no aloja las imágenes directamente en el repo (están en Flickr), por lo
que se usa el script oficial de descarga del propio repositorio.

Uso:
    python data/download_taco.py

Salida esperada:
    data/TACO/
    ├── data/                  # imágenes descargadas, organizadas por batch
    └── data/annotations.json  # anotaciones COCO originales (60 categorías)
"""

import subprocess
import sys
from pathlib import Path

TACO_REPO_URL = "https://github.com/pedropro/TACO.git"
TARGET_DIR = Path(__file__).parent / "TACO"


def run(cmd: list[str], cwd: Path | None = None):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Error ejecutando: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    if TARGET_DIR.exists():
        print(f"El directorio {TARGET_DIR} ya existe. Si querés re-descargar, "
              f"borralo primero.")
    else:
        run(["git", "clone", TACO_REPO_URL, str(TARGET_DIR)])

    requirements_file = TARGET_DIR / "requirements.txt"
    if requirements_file.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])

    download_script = TARGET_DIR / "download.py"
    if not download_script.exists():
        print(f"No se encontró {download_script}. Revisá la estructura del repo TACO.")
        sys.exit(1)

    print("\nDescargando imágenes desde Flickr (puede tardar varios minutos)...")
    run([sys.executable, str(download_script)], cwd=TARGET_DIR)

    print("\n✅ Descarga completa.")
    print(f"   Anotaciones COCO en: {TARGET_DIR / 'data' / 'annotations.json'}")
    print(f"   Imágenes en:         {TARGET_DIR / 'data'}")
    print("\nSiguiente paso: python data/prepare_dataset.py")


if __name__ == "__main__":
    main()
