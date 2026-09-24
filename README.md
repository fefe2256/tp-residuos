# ♻️ Clasificación y Segmentación de Residuos Reciclables

**Trabajo Práctico Final — Visión por Computadora II (CEIA)**

Sistema que, a partir de una foto o video, detecta y segmenta cada residuo
presente en la escena, identifica su material (plástico, papel/cartón,
vidrio, metal, orgánico) y lo señala con el color del contenedor de
reciclaje correspondiente — funcionando incluso con múltiples residuos
simultáneos en una misma imagen.

---

## Tabla de contenidos

- [Objetivo y motivación](#objetivo-y-motivación)
- [Arquitectura de la solución](#arquitectura-de-la-solución)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Compatibilidad de hardware](#compatibilidad-de-hardware-mac--linux--windows)
- [Dataset](#dataset)
- [Uso](#uso)
- [Tracking de experimentos con W&B](#tracking-de-experimentos-con-weights--biases)
- [Evaluación](#evaluación)
- [Equipo](#equipo)
- [Roadmap / trabajo futuro](#roadmap--trabajo-futuro)
- [Referencias](#referencias)

---

## Objetivo y motivación

Las plantas de reciclaje y los puntos verdes municipales reciben residuos
mezclados que hoy se separan mayormente a mano o con sistemas mecánicos poco
precisos. Este proyecto propone automatizar esa clasificación mediante un
modelo de segmentación de instancias que identifica **qué es** cada residuo
y **dónde está exactamente** dentro de la imagen (no solo si existe, como
haría un clasificador simple), permitiendo en el futuro integrarse a un
brazo robótico o a una señal de control sobre una cinta transportadora.

## Arquitectura de la solución

| Componente | Elección | Justificación breve |
|---|---|---|
| **Dataset** | [TACO](https://github.com/pedropro/TACO) (Trash Annotations in Context) | Fotos reales de basura, ya anotadas con máscaras de segmentación en formato COCO |
| **Tarea** | Segmentación de instancias | Necesitamos distinguir objetos individuales superpuestos, no solo regiones de material |
| **Modelo** | YOLOv8-seg, vía transfer learning | Dataset pequeño (~1500 img) → inviable entrenar desde cero; se parte de pesos pre-entrenados en COCO |
| **Estrategia de entrenamiento** | Fine-tuning en 2 etapas (backbone congelado → modelo completo) | Evita que los gradientes de las capas nuevas distorsionen las features ya aprendidas antes de estabilizar el modelo |
| **Tracking de experimentos** | Weights & Biases | Permite comparar corridas entre los integrantes del equipo desde un dashboard centralizado |
| **Post-procesamiento** | Mapeo material → color de contenedor | Traduce la predicción técnica a una acción clara ("a qué tacho va") |
| **Front** | Streamlit | Carga de imagen/video y visualización de resultados sin necesidad de un front a medida |

## Estructura del repositorio

```
proyecto-residuos/
├── README.md
├── requirements.txt
├── pyproject.toml                  # dependencias para uv (mismas que requirements.txt + grupo dev)
├── uv.lock                         # versiones exactas resueltas por uv
├── .python-version                 # Python 3.13 (el que usó uv)
├── .gitignore
├── .env.example                    # plantilla de variables de entorno (el .env real no se versiona)
├── TP_Final_VxC2_PaperVF.docx      # informe/paper final del TP
│
├── data/
│   ├── download_taco.py            # clona TACO y baja las imágenes (Zenodo; --flickr opcional)
│   ├── prepare_dataset.py          # agrupa 60 categorías -> 6 clases macro; COCO -> YOLO-seg
│   ├── TACO/                       # (generado) dataset original, no versionado
│   └── taco_yolo/                  # (generado) dataset ya preparado para entrenar
│
├── notebooks/
│   ├── 00_TP_Informe_Principal.ipynb   # notebook autocontenido: instala deps, descarga TACO, entrena, evalúa, prueba imágenes nuevas
│   ├── 01_EDA_TACO.ipynb               # EDA detallado y aislado del dataset
│   └── 02_Comparacion_MaskRCNN.ipynb   # comparación de YOLOv8-seg vs. Mask R-CNN sobre el mismo dataset
│
├── src/
│   ├── device_utils.py             # auto-detección de dispositivo: CUDA / MPS (Mac) / CPU
│   ├── train.py                    # entrenamiento (fine-tuning en 2 etapas) con logging a W&B
│   ├── inference.py                # inferencia sobre imagen o video + post-procesamiento de color
│   └── color_mapping.py            # lógica material -> color de contenedor de reciclaje
│
├── app/
│   └── app.py                      # front en Streamlit
│
├── imagenes_prueba/                # imágenes de ejemplo para probar la inferencia y el front
│
└── runs/                           # (generado) checkpoints, curvas y métricas de cada entrenamiento
```

## Instalación

### Opción A: con [uv](https://docs.astral.sh/uv/) (recomendada)

```bash
git clone <url-del-repo>
cd proyecto-residuos
uv sync                       # crea .venv con Python 3.13 e instala todo desde uv.lock
cp .env.example .env          # luego editar .env y completar WANDB_API_KEY
```

Después anteponé `uv run` a cualquier comando, sin activar nada:

```bash
uv run python data/download_taco.py
uv run python data/prepare_dataset.py
uv run python src/train.py --epochs 60 --freeze-epochs 10
uv run streamlit run app/app.py
uv run jupyter lab            # jupyter, ipykernel y torchmetrics vienen en el grupo dev
```

> Con uv **no hace falta** correr la celda `!pip install -r ../requirements.txt`
> del notebook principal (ni el `!pip install torchmetrics` del notebook 02):
> todo ya queda instalado con `uv sync`. Elegí el kernel de `.venv` en Jupyter/VS Code.
> Si agregás una dependencia, usá `uv add <paquete>` y sumala también a
> `requirements.txt`.

### Opción B: con pip

```bash
git clone <url-del-repo>
cd proyecto-residuos
python -m venv venv
source venv/bin/activate      # en Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # luego editar .env y completar WANDB_API_KEY
```

> El archivo `.env` **no se versiona** (está en `.gitignore`); se usa para
> guardar la API key de Weights & Biases y evitar tener que correr
> `wandb login` manualmente en cada máquina. Si preferís no usar W&B,
> podés saltearte este paso y correr el entrenamiento con `--no-wandb`
> (ver [Tracking de experimentos](#tracking-de-experimentos-con-weights--biases)).

> Si vas a usar el **notebook principal** (`00_TP_Informe_Principal.ipynb`,
> ver [Uso](#uso)), alcanza con crear y activar el entorno virtual — el
> `pip install -r requirements.txt` también lo corre la primera celda del
> notebook. Correrlo acá igual no molesta (`pip` detecta lo ya instalado),
> pero no es obligatorio si vas a instalar todo desde el notebook.

## Compatibilidad de hardware (Mac / Linux / Windows)

Todo el código (`src/device_utils.py`) detecta automáticamente el mejor
dispositivo disponible, en este orden de prioridad:

1. **CUDA** — GPU NVIDIA (Linux/Windows con GPU dedicada).
2. **MPS** — GPU integrada de Mac con Apple Silicon (M1/M2/M3/M4).
3. **CPU** — fallback universal, funciona en cualquier máquina (más lento).

No hace falta instalar nada distinto según el sistema operativo: `torch`
incluye soporte MPS de fábrica desde la versión 2.0. Todos los scripts
(`train.py`, `inference.py`) y el notebook principal aceptan forzar un
dispositivo puntual si hace falta:

```bash
python src/train.py --device cpu     # CPU (debug, cualquier máquina)
python src/train.py --device mps     # GPU de Mac (Apple Silicon)
python src/train.py --device cuda    # GPU NVIDIA
```

> **Nota sobre entrenar en Mac:** MPS funciona para inferencia sin problemas,
> pero entrenar YOLOv8-seg en MPS puede ser considerablemente más lento que
> en CUDA, y Ultralytics tiene bugs puntuales conocidos con MPS en
> operaciones de segmentación. Si el entrenamiento falla o es muy lento en
> Mac, la alternativa recomendada es entrenar en Google Colab (GPU T4
> gratuita) y bajar los pesos (`best.pt`) para correr inferencia y el front
> localmente.

## Dataset

**TACO (Trash Annotations in Context):** https://github.com/pedropro/TACO

TACO no aloja las imágenes en el repo. El script oficial las baja de Flickr,
pero por links caídos consigue menos del 10%, así que `download_taco.py`
usa por defecto el **mirror de Zenodo** (https://zenodo.org/records/3587843):
baja `TACO.zip` (~2.7 GB, se valida el MD5), extrae las 1500 imágenes y
borra el zip. Hacen falta ~5.5 GB libres durante la descarga.

```bash
python data/download_taco.py            # Zenodo (--keep-zip para conservar el zip, --flickr para el script oficial)
python data/prepare_dataset.py
```

El primer script clona el repo de TACO (de ahí sale
`data/TACO/data/annotations.json`, anotaciones COCO, 60 categorías finas)
y deja las imágenes en `data/TACO/data/batch_*/`. El segundo
agrupa las 60 categorías finas en las 6 clases macro y convierte el
formato COCO al formato YOLO-seg que espera Ultralytics, dejando el
dataset listo en `data/taco_yolo/`.

Las 60 categorías originales se agrupan en **6 clases macro** para tener
suficientes ejemplos por clase durante el entrenamiento:

| Clase macro | Contenedor asignado |
|---|---|
| `plastico` | 🟡 Amarillo |
| `papel_carton` | 🔵 Azul |
| `vidrio` | 🟢 Verde |
| `metal` | 🟡 Amarillo |
| `organico` | ⚫ Gris / marrón |
| `otros` | ⚫ Gris / marrón |

El mapeo completo de categoría fina → clase macro está en
`data/prepare_dataset.py` (diccionario `SUPERCATEGORY_TO_MACRO`), y el
mapeo de clase macro → color de contenedor en `src/color_mapping.py`.

> **Nota sobre el código de colores en Argentina.** La Ley 25.916 (GIRSU)
> fija a nivel nacional solo la separación mínima entre residuos secos
> reciclables y húmedos/orgánicos, **sin establecer colores específicos** —
> cada municipio define los suyos. La tabla de arriba corresponde al
> esquema de clasificación fina por material más extendido en islas
> ecológicas de empresas y municipios (azul=papel/cartón,
> amarillo=plástico/metal, verde=vidrio, gris/marrón=orgánico), que es el
> que necesitamos porque nuestro modelo predice material específico, no
> solo "seco vs. húmedo". Este esquema **no es universal**: por ejemplo,
> en CABA el sistema es más simple (un único contenedor verde para todo lo
> reciclable seco, y negro para húmedos, sin separar por material). Si el
> proyecto se adapta a una localidad o planta con otro código de colores,
> alcanza con editar el diccionario `MATERIAL_TO_CONTAINER` en
> `src/color_mapping.py`.

## Uso

Hay dos formas de correr el proyecto: **todo desde el notebook principal**
(recomendado si es la primera vez, o si preferís ver cada paso ejecutado
con su output antes de seguir), o **por scripts sueltos desde terminal**
(más práctico para re-entrenar muchas veces con distintos hiperparámetros,
o para dejar el entrenamiento corriendo en background).

### Opción A: todo desde el notebook (recomendado para arrancar)

```
notebooks/00_TP_Informe_Principal.ipynb
```

Este notebook es autocontenido: instala las dependencias, descarga TACO,
prepara el dataset, entrena, evalúa y prueba con imágenes nuevas, todo
corriendo celda por celda sin salir a la terminal. Solo hace falta, antes
de abrirlo, tener creado y activado el entorno virtual (ver
[Instalación](#instalación)) y seleccionarlo como kernel del notebook.

También incluye la sección de EDA y las decisiones de preprocesamiento
documentadas en markdown junto al código que las ejecuta — es el mismo
notebook que sirve de base para redactar el informe/paper.

### Opción B: por scripts sueltos desde terminal

Útil para volver a entrenar con otra configuración sin re-ejecutar todo el
notebook, o para correr inferencia/la app en el día a día una vez que ya
se entrenó el modelo.

**1. Descargar y preparar el dataset**

```bash
python data/download_taco.py
python data/prepare_dataset.py
```

El segundo script imprime la distribución final de instancias por clase
macro — revisar que ninguna quede demasiado desbalanceada respecto al resto.

**2. Entrenar el modelo**

```bash
python src/train.py --epochs 60 --freeze-epochs 10
```

Parámetros principales (ver `python src/train.py --help` para el resto):

| Parámetro | Default | Descripción |
|---|---|---|
| `--model` | `yolov8s-seg.pt` | Checkpoint base (`n`/`s`/`m`/`l`/`x`, de más rápido a más preciso) |
| `--epochs` | `60` | Épocas totales (se reparten entre las 2 etapas de fine-tuning) |
| `--freeze-epochs` | `10` | Épocas de la Etapa 1 (backbone congelado) |
| `--imgsz` | `640` | Tamaño de imagen de entrada |
| `--batch` | `16` | Batch size |
| `--no-wandb` | — | Desactiva el logging a W&B |

**3. Correr inferencia sobre una imagen o video**

```bash
python src/inference.py --source foto.jpg --weights runs/segment/residuos_yolov8seg_etapa2/weights/best.pt
python src/inference.py --source video.mp4 --weights best.pt --output resultado.mp4
```

**4. Correr la app (Streamlit)**

```bash
streamlit run app/app.py
```

Permite subir una foto o video, ajustar el umbral de confianza y ver el
resultado segmentado con los colores de contenedor superpuestos.

## Tracking de experimentos con Weights & Biases

El entrenamiento loguea automáticamente a W&B (sin código adicional más
allá de tener la sesión iniciada), aprovechando la integración nativa de
Ultralytics:

```bash
# Una sola vez por máquina (pide un API key gratuito de https://wandb.ai/authorize)
wandb login
```

Esto permite comparar entre los integrantes del equipo distintas
configuraciones (modelo base, cantidad de épocas de congelamiento,
learning rate) desde un dashboard centralizado, en vez de gráficos sueltos
en cada computadora. Para desactivarlo puntualmente: `--no-wandb`.

## Evaluación

Métricas reportadas (generadas automáticamente por Ultralytics al validar):

- **mAP@50 y mAP@50-95**, tanto para las cajas (`box`) como para las
  máscaras (`mask`).
- **Matriz de confusión** por clase macro, para identificar qué materiales
  se confunden entre sí (ej. plástico vs. metal por brillo similar).
- **Comparación Etapa 1 vs. Etapa 2**, para justificar la estrategia de
  fine-tuning en dos pasos frente a un fine-tuning directo.

Todo esto se genera automáticamente en `runs/segment/<nombre_corrida>/` y
se visualiza inline en `notebooks/00_TP_Informe_Principal.ipynb`.

## Equipo

- Integrante 1
- Integrante 2
- Integrante 3
- Integrante 4

## Roadmap / trabajo futuro

Capacidades opcionales que suman valor por sobre el eje obligatorio del TP:

- [ ] Sub-clasificación de vidrio por color físico (verde / ámbar /
      transparente), relevante para plantas reales que separan vidrio por color.
- [ ] Depth estimation para estimar el volumen aproximado de cada residuo.
- [ ] Object tracking sobre video simulando una cinta transportadora, para
      evitar contar el mismo objeto dos veces en frames consecutivos.
- [ ] Comparación contra un baseline de segmentación clásica (Watershed /
      K-Means) para cuantificar la mejora que aporta el enfoque de deep learning.

## Referencias

- Pedro F. Proença, Pedro Simões. *TACO: Trash Annotations in Context for
  Litter Detection*. https://github.com/pedropro/TACO
- Documentación de Ultralytics YOLOv8: https://docs.ultralytics.com
- Documentación de Weights & Biases: https://docs.wandb.ai
- Ley Nº 25.916 de Gestión Integral de Residuos Domiciliarios (GIRSU),
  Argentina — separación mínima seco/húmedo a nivel nacional.
- Plásticos Roca. *Código de Colores de Reciclaje en Argentina: Guía Ley
  25.916 y CABA*. https://www.plasticosroca.com/blog/codigo-colores-reciclaje-argentina-ley-25916
