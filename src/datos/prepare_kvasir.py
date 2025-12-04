# preparación dataset - import path robusto
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse, zipfile, shutil, random
from pathlib import Path
import cv2
from src.utils import CLASES_KVASIR, asegurar_dir

def extraer_si_hace_falta(origen_dir: Path, data_dir: Path):
    # detecta zip o carpetas ambos para prevenir errores humanos
    raw_dir = data_dir / "raw"
    if raw_dir.exists() and any(raw_dir.iterdir()):
        return raw_dir
    asegurar_dir(raw_dir)
    zip_path = origen_dir / "kvasir-dataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
    else:
        # copiar directorios ya extraídos, deja que el programa se mueva solo
        for c in CLASES_KVASIR:
            src = origen_dir / c
            if src.exists():
                dst = raw_dir / c
                if dst.exists(): shutil.rmtree(dst)
                shutil.copytree(src, dst)
    return raw_dir

def hacer_split(raw_dir: Path, out_dir: Path, val=0.15, test=0.15):
    split_dir = out_dir / "split"
    if split_dir.exists() and any(split_dir.iterdir()): return split_dir
    for parte in ["train","val","test"]:
        for c in CLASES_KVASIR:
            asegurar_dir(split_dir/parte/c)

    for c in CLASES_KVASIR:
        clase_dir = raw_dir / c
        if not clase_dir.exists(): 
            # clases faltantes se ignoran para evitar errores humanos
            continue
        imgs = [p for p in clase_dir.iterdir() if p.suffix.lower() in [".jpg",".png",".jpeg"]]
        random.shuffle(imgs)
        n = len(imgs); n_val = int(n*val); n_test = int(n*test)
        cortes = {"train": imgs[n_val+n_test:], "val": imgs[:n_val], "test": imgs[n_val:n_val+n_test]}
        for parte, lista in cortes.items():
            for src in lista:
                img = cv2.imread(str(src))
                if img is None: continue
                img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(split_dir/parte/c/src.name), img)
    return split_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="carpeta Dataset")
    ap.add_argument("--salida", required=True, help="carpeta data")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()
    out = Path(args.salida); asegurar_dir(out)
    raw = extraer_si_hace_falta(Path(args.origen), out)
    hacer_split(raw, out, args.val, args.test)
    print("OK: datos listos")