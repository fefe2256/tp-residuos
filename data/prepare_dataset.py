"""
Prepara el dataset TACO para entrenar con YOLOv8-seg:

1. Carga las anotaciones COCO originales (60 categorías finas de TACO).
2. Agrupa esas categorías en las clases macro del proyecto:
   plastico, papel_carton, vidrio, metal, organico, otros.
3. Convierte las anotaciones (polígonos de segmentación) al formato YOLO-seg
   (un .txt por imagen, con líneas: class_id x1 y1 x2 y2 ... xn yn normalizado).
4. Genera el data.yaml necesario para entrenar con Ultralytics.

Uso:
    python data/prepare_dataset.py

Salida:
    data/taco_yolo/
    ├── images/{train,val}/...
    ├── labels/{train,val}/...
    └── data.yaml
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------

TACO_DIR = Path(__file__).parent / "TACO" / "data"
ANNOTATIONS_PATH = TACO_DIR / "annotations.json"
OUTPUT_DIR = Path(__file__).parent / "taco_yolo"
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Clases macro finales del proyecto (orden = class_id en YOLO)
MACRO_CLASSES = ["plastico", "papel_carton", "vidrio", "metal", "organico", "otros"]

# Mapeo de categorías originales de TACO (super_category en TACO) -> clase macro.
# TACO ya trae una jerarquía de "supercategory" que simplifica bastante esto;
# se ajusta manualmente lo que no calce perfecto.
SUPERCATEGORY_TO_MACRO = {
    "Plastic bag & wrapper": "plastico",
    "Plastic container": "plastico",
    "Other plastic": "plastico",
    "Straw": "plastico",
    "Styrofoam piece": "plastico",
    "Lid": "plastico",
    "Bottle": "plastico",       # se corrige a "vidrio" si el material es vidrio (ver nota abajo)
    "Bottle cap": "plastico",
    "Cup": "plastico",
    "Paper": "papel_carton",
    "Paper bag": "papel_carton",
    "Carton": "papel_carton",
    "Glass jar": "vidrio",
    "Broken glass": "vidrio",
    "Can": "metal",
    "Aluminium foil": "metal",
    "Metal bottle cap": "metal",
    "Pop tab": "metal",
    "Scrap metal": "metal",
    "Food waste": "organico",
    "Cigarette": "otros",
    "Unlabeled litter": "otros",
    "Blister pack": "otros",
    "Squeezable tube": "otros",
    "Battery": "otros",
    "Shoe": "otros",
}


def load_coco_annotations(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_category_mapping(coco: dict) -> dict[int, str]:
    """
    Devuelve {category_id_original: clase_macro} usando supercategory como
    puente. Cualquier categoría no mapeada cae en "otros".
    """
    cat_id_to_macro = {}
    unmapped = []
    for cat in coco["categories"]:
        supercategory = cat.get("supercategory", "")
        macro = SUPERCATEGORY_TO_MACRO.get(supercategory)
        if macro is None:
            unmapped.append((cat["id"], cat["name"], supercategory))
            macro = "otros"
        cat_id_to_macro[cat["id"]] = macro

    if unmapped:
        print(f"[prepare_dataset] {len(unmapped)} categorías sin mapeo explícito, "
              f"asignadas a 'otros'. Revisar si conviene remapear alguna manualmente:")
        for cid, name, super_ in unmapped[:15]:
            print(f"   - id={cid} name='{name}' supercategory='{super_}'")
        if len(unmapped) > 15:
            print(f"   ... y {len(unmapped) - 15} más.")

    return cat_id_to_macro


def polygon_to_yolo_seg_line(class_id: int, segmentation: list[float],
                              img_w: int, img_h: int) -> str | None:
    """
    Convierte un polígono COCO (lista plana [x1,y1,x2,y2,...]) a una línea
    de formato YOLO-seg: "class_id x1 y1 x2 y2 ... xn yn" con coordenadas
    normalizadas [0,1].
    """
    if len(segmentation) < 6:  # menos de 3 puntos, no es un polígono válido
        return None

    coords = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] / img_w
        y = segmentation[i + 1] / img_h
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")

    return f"{class_id} " + " ".join(coords)


def main():
    if not ANNOTATIONS_PATH.exists():
        print(f"No se encontró {ANNOTATIONS_PATH}.")
        print("Corré primero: python data/download_taco.py")
        return

    random.seed(RANDOM_SEED)

    print("Cargando anotaciones COCO...")
    coco = load_coco_annotations(ANNOTATIONS_PATH)

    cat_id_to_macro = build_category_mapping(coco)
    macro_to_class_id = {name: i for i, name in enumerate(MACRO_CLASSES)}

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # --- Split ESTRATIFICADO por clase minoritaria presente en cada imagen ---
    # Un split aleatorio por imagen puede desbalancear clases raras (ej. vidrio)
    # entre train y val por pura casualidad. En su lugar, agrupamos las imágenes
    # según la clase macro MÁS ESCASA que contienen, y hacemos el split 80/20
    # DENTRO de cada grupo, para que cada clase quede proporcionalmente
    # repartida en ambos conjuntos.

    # 1) Solo consideramos imágenes que realmente se descargaron
    valid_image_ids = [
        image_id for image_id in anns_by_image.keys()
        if (TACO_DIR / images_by_id[image_id]["file_name"]).exists()
    ]

    # 2) Contamos instancias totales por clase macro (para saber cuál es "rara")
    total_instances_per_macro = defaultdict(int)
    image_macro_classes = {}
    for image_id in valid_image_ids:
        macros_en_imagen = set()
        for ann in anns_by_image[image_id]:
            macro = cat_id_to_macro.get(ann["category_id"], "otros")
            macros_en_imagen.add(macro)
        image_macro_classes[image_id] = macros_en_imagen
        for macro in macros_en_imagen:
            total_instances_per_macro[macro] += 1  # conteo por imagen, no por instancia

    # 3) Para cada imagen, la "clase de estratificación" es la más rara que contiene
    def clase_mas_rara(image_id):
        macros = image_macro_classes[image_id]
        if not macros:
            return "otros"
        return min(macros, key=lambda m: total_instances_per_macro[m])

    grupos = defaultdict(list)
    for image_id in valid_image_ids:
        grupos[clase_mas_rara(image_id)].append(image_id)

    # 4) Split 80/20 dentro de cada grupo por separado
    val_ids = set()
    for macro, ids_grupo in grupos.items():
        ids_grupo = ids_grupo.copy()
        random.shuffle(ids_grupo)
        n_val_grupo = max(1, int(len(ids_grupo) * VAL_SPLIT)) if len(ids_grupo) > 1 else 0
        val_ids.update(ids_grupo[:n_val_grupo])

    # Crear estructura de carpetas
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_counts = defaultdict(int)
    class_counts_by_split = {"train": defaultdict(int), "val": defaultdict(int)}
    skipped_no_valid_polygon = 0

    for image_id, anns in anns_by_image.items():
        img_info = images_by_id[image_id]
        split = "val" if image_id in val_ids else "train"

        src_img_path = TACO_DIR / img_info["file_name"]
        if not src_img_path.exists():
            continue  # imagen no descargada (falló en Flickr, por ejemplo)

        dst_img_name = f"{image_id}_{src_img_path.name}"
        dst_img_path = OUTPUT_DIR / "images" / split / dst_img_name
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)

        label_lines = []
        for ann in anns:
            macro_class = cat_id_to_macro.get(ann["category_id"], "otros")
            class_id = macro_to_class_id[macro_class]

            segmentation = ann.get("segmentation")
            if not segmentation or not isinstance(segmentation, list):
                continue
            # segmentation puede tener múltiples polígonos por objeto (huecos, etc.)
            for poly in segmentation:
                line = polygon_to_yolo_seg_line(
                    class_id, poly, img_info["width"], img_info["height"]
                )
                if line:
                    label_lines.append(line)
                    class_counts[macro_class] += 1
                    class_counts_by_split[split][macro_class] += 1
                else:
                    skipped_no_valid_polygon += 1

        label_path = OUTPUT_DIR / "labels" / split / f"{image_id}_{src_img_path.stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    # data.yaml para Ultralytics
    data_yaml = OUTPUT_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(MACRO_CLASSES)}\n")
        f.write(f"names: {MACRO_CLASSES}\n")

    print("\n✅ Dataset preparado en formato YOLO-seg.")
    print(f"   Salida: {OUTPUT_DIR}")
    print(f"   data.yaml: {data_yaml}")
    print(f"\nImágenes: train={len(valid_image_ids) - len(val_ids & set(valid_image_ids))}, "
          f"val={len(val_ids & set(valid_image_ids))}")
    print("\nDistribución de instancias por clase macro (train / val):")
    for cls in MACRO_CLASSES:
        n_train = class_counts_by_split["train"].get(cls, 0)
        n_val = class_counts_by_split["val"].get(cls, 0)
        pct_val = (n_val / (n_train + n_val) * 100) if (n_train + n_val) > 0 else 0
        print(f"   {cls:15s}: train={n_train:5d}  val={n_val:5d}  ({pct_val:.0f}% en val)")
    if skipped_no_valid_polygon:
        print(f"\n({skipped_no_valid_polygon} polígonos inválidos/pequeños fueron omitidos)")

    print("\nSiguiente paso: python src/train.py")


if __name__ == "__main__":
    main()
