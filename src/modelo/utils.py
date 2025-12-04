from pathlib import Path
CLASES_KVASIR = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "polyps",
    "ulcerative-colitis",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
]

def asegurar_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)
