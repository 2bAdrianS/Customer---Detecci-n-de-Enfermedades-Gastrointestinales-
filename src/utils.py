from pathlib import Path

# Clases usadas del dataset Kvasir
CLASES_KVASIR = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "polyps",
    "ulcerative-colitis",
]

def asegurar_dir(p) -> Path:
    """
    Crea el directorio si no existe y retorna Path(p).
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
