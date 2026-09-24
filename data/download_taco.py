"""
Descarga y organiza el dataset TACO (Trash Annotations in Context).

Por defecto baja las 1500 imágenes del mirror oficial de Zenodo
(https://zenodo.org/records/3587843, un TACO.zip de ~2.7 GB). El script
oficial de TACO las baja de Flickr, pero por links caídos consigue menos del
10%; esa opción queda disponible con --flickr.

Las anotaciones (annotations.json) salen del repo de TACO, que se clona si no
existe.

Uso:
    python data/download_taco.py              # Zenodo (recomendado)
    python data/download_taco.py --keep-zip   # no borra TACO.zip al terminar
    python data/download_taco.py --flickr     # script oficial de Flickr

Salida esperada:
    data/TACO/
    ├── data/batch_*/          # imágenes, organizadas por batch
    └── data/annotations.json  # anotaciones COCO originales (60 categorías)
"""

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

TACO_REPO_URL = "https://github.com/pedropro/TACO.git"
TARGET_DIR = Path(__file__).parent / "TACO"
IMAGES_DIR = TARGET_DIR / "data"

ZENODO_URL = "https://zenodo.org/api/records/3587843/files/TACO.zip/content"
ZENODO_MD5 = "e9149407d883e21a8d224feef8210920"
ZIP_PATH = Path(__file__).parent / "TACO_zenodo.zip"


def run(cmd: list[str], cwd: Path | None = None):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Error ejecutando: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def descargar_zip():
    """Baja TACO.zip de Zenodo (si no está ya bajado y completo) y valida el MD5."""
    if ZIP_PATH.exists():
        print(f"Verificando {ZIP_PATH.name} ya existente...")
        if md5_of(ZIP_PATH) == ZENODO_MD5:
            print("✅ Zip completo, no hace falta volver a bajarlo.")
            return
        print("⚠️  El zip existente está incompleto o corrupto; se vuelve a bajar.")

    print(f"\nDescargando TACO.zip de Zenodo (~2.7 GB, puede tardar bastante)...")
    tmp_path = ZIP_PATH.with_suffix(".zip.part")
    with requests.get(ZENODO_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print("Verificando MD5...")
    if md5_of(tmp_path) != ZENODO_MD5:
        print("⚠️  El MD5 no coincide: la descarga quedó corrupta. Volvé a correr el script.",
              file=sys.stderr)
        sys.exit(1)
    tmp_path.rename(ZIP_PATH)


def extraer_imagenes() -> int:
    """Extrae solo TACO/data/batch_*/<imagen> a data/TACO/data/, salteando las existentes."""
    extraidas = 0
    with zipfile.ZipFile(ZIP_PATH) as z:
        miembros = [
            m for m in z.infolist()
            if m.filename.startswith("TACO/data/batch_")
            and not m.is_dir()
            and not Path(m.filename).name.startswith(".")  # .DS_Store y similares
        ]
        for m in tqdm(miembros, desc="Extrayendo imágenes"):
            destino = IMAGES_DIR / Path(m.filename).relative_to("TACO/data")
            if destino.exists() and destino.stat().st_size == m.file_size:
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(destino, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            extraidas += 1
    return extraidas


def descargar_flickr():
    """Script oficial de TACO. Solo necesita Pillow y requests, que ya están en requirements.txt."""
    download_script = TARGET_DIR / "download.py"
    if not download_script.exists():
        print(f"No se encontró {download_script}. Revisá la estructura del repo TACO.")
        sys.exit(1)
    print("\nDescargando imágenes desde Flickr (puede tardar varios minutos)...")
    run([sys.executable, str(download_script)], cwd=TARGET_DIR)


def main():
    parser = argparse.ArgumentParser(description="Descarga del dataset TACO")
    parser.add_argument("--flickr", action="store_true",
                        help="Usar el script oficial de Flickr (baja <10%% de las imágenes)")
    parser.add_argument("--keep-zip", action="store_true",
                        help="No borrar TACO_zenodo.zip después de extraerlo")
    args = parser.parse_args()

    if TARGET_DIR.exists():
        print(f"El directorio {TARGET_DIR} ya existe; se usa el clon actual.")
    else:
        run(["git", "clone", TACO_REPO_URL, str(TARGET_DIR)])

    if args.flickr:
        descargar_flickr()
    else:
        descargar_zip()
        extraidas = extraer_imagenes()
        print(f"✅ {extraidas} imágenes nuevas extraídas.")
        if not args.keep_zip:
            ZIP_PATH.unlink()
            print(f"   Se borró {ZIP_PATH.name} (usá --keep-zip para conservarlo).")

    n_imagenes = sum(1 for p in IMAGES_DIR.glob("batch_*/*") if p.is_file())
    print("\n✅ Descarga completa.")
    print(f"   Anotaciones COCO en: {IMAGES_DIR / 'annotations.json'}")
    print(f"   Imágenes en:         {IMAGES_DIR} ({n_imagenes} archivos)")
    print("\nSiguiente paso: python data/prepare_dataset.py")


if __name__ == "__main__":
    main()
