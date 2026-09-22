"""
Mapeo de material detectado -> color de contenedor de reciclaje.

Convención de colores utilizada:

    Verde   -> Vidrio
    Azul    -> Papel / Cartón
    Amarillo-> Plástico / Metal / Tetrabrik
    Gris    -> No reciclable / Orgánico

IMPORTANTE: en Argentina no existe un código de colores único a nivel
nacional. La Ley 25.916 (GIRSU) solo exige separar secos reciclables de
húmedos/orgánicos, sin fijar colores. La convención de arriba corresponde
al esquema de clasificación FINA por material, habitual en islas
ecológicas de empresas y municipios -- pero, por ejemplo, en CABA el
sistema real es más simple (un solo contenedor verde para todo lo seco
reciclable, y negro para húmedos). Ajustar MATERIAL_TO_CONTAINER más abajo
si el municipio/planta de destino usa otro código de colores.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContainerInfo:
    color_name: str
    color_rgb: tuple[int, int, int]   # para dibujar máscaras (RGB)
    color_bgr: tuple[int, int, int]   # para dibujar con OpenCV (BGR)
    emoji: str
    label: str


# Definición de contenedores por color
CONTAINERS = {
    "verde": ContainerInfo("verde", (46, 204, 113), (113, 204, 46), "🟢", "Tacho Verde"),
    "azul": ContainerInfo("azul", (52, 152, 219), (219, 152, 52), "🔵", "Tacho Azul"),
    "amarillo": ContainerInfo("amarillo", (241, 196, 15), (15, 196, 241), "🟡", "Tacho Amarillo"),
    "gris": ContainerInfo("gris", (149, 165, 166), (166, 165, 149), "⚫", "Tacho Gris / No reciclable"),
}

# Mapeo de clase de material (las 5-6 clases macro del modelo) -> contenedor
MATERIAL_TO_CONTAINER = {
    "vidrio": "verde",
    "papel": "azul",
    "carton": "azul",
    "papel_carton": "azul",
    "plastico": "amarillo",
    "metal": "amarillo",
    "organico": "gris",
    "otros": "gris",
}


def get_container_for_material(material: str) -> ContainerInfo:
    """
    Dada una clase de material predicha por el modelo, devuelve la
    información del contenedor correspondiente (color, emoji, etc).

    Si el material no está mapeado, devuelve el contenedor "gris" por defecto
    (no reciclable / a revisar manualmente).
    """
    key = material.strip().lower().replace(" ", "_")
    container_key = MATERIAL_TO_CONTAINER.get(key, "gris")
    return CONTAINERS[container_key]


def format_detection_label(material: str, confidence: float) -> str:
    """
    Arma el string de etiqueta para mostrar en el front, ej:
    "🟡 Plástico → Tacho Amarillo (95%)"
    """
    container = get_container_for_material(material)
    return f"{container.emoji} {material.capitalize()} → {container.label} ({confidence * 100:.0f}%)"


if __name__ == "__main__":
    # Pequeño self-test
    ejemplos = ["plastico", "vidrio", "papel_carton", "metal", "organico", "desconocido"]
    for m in ejemplos:
        print(format_detection_label(m, 0.9))
