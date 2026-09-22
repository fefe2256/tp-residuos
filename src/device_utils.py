"""
Utilidad para seleccionar el dispositivo de cómputo disponible.

Soporta:
- CUDA   -> GPU NVIDIA (Linux/Windows, o Mac con eGPU raro)
- MPS    -> GPU integrada de Mac con Apple Silicon (M1/M2/M3/M4)
- CPU    -> fallback universal

Uso:
    from device_utils import get_device
    device = get_device()               # auto-detección
    device = get_device(force="cpu")    # forzar CPU (útil para debug)
    device = get_device(force="mps")    # forzar MPS
    device = get_device(force="cuda")   # forzar CUDA
"""

import torch


def get_device(force: str | None = None, verbose: bool = True) -> torch.device:
    """
    Devuelve el mejor dispositivo disponible, o el forzado si se especifica.

    Parameters
    ----------
    force : str | None
        "cuda", "mps" o "cpu" para forzar un dispositivo puntual.
        None (default) para auto-detectar en orden: cuda -> mps -> cpu.
    verbose : bool
        Si True, imprime qué dispositivo se eligió y por qué.

    Returns
    -------
    torch.device
    """
    if force is not None:
        force = force.lower()
        if force == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se forzó CUDA pero no hay GPU NVIDIA disponible.")
        if force == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Se forzó MPS pero no está disponible (¿estás en Mac con Apple Silicon?).")
        device = torch.device(force)
        if verbose:
            print(f"[device_utils] Dispositivo forzado: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        if verbose:
            print(f"[device_utils] Usando CUDA -> {name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[device_utils] Usando MPS (GPU de Apple Silicon)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[device_utils] No se detectó GPU. Usando CPU (será más lento).")

    return device


def get_device_str(force: str | None = None) -> str:
    """
    Igual que get_device pero devuelve un string, útil para pasarle
    el argumento `device` directamente a Ultralytics (YOLO), que espera
    "cpu", "mps" o un índice de GPU como 0.
    """
    device = get_device(force=force, verbose=False)
    if device.type == "cuda":
        return "0"  # Ultralytics espera el índice de GPU, no "cuda"
    return device.type  # "mps" o "cpu"


if __name__ == "__main__":
    d = get_device()
    print(f"torch.device resultante: {d}")
    print(f"String para Ultralytics: {get_device_str()}")
