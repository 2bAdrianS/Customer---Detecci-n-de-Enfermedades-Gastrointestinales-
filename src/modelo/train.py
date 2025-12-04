# entrenamiento import path robusto
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

from src.modelo.lightning_model import KvasirLit
from src.utils import asegurar_dir

def get_tfms():
    # aumentos simples
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

import torch

def main(args):
    split = Path("data/split")
    assert split.exists(), "Falta split, ejecuta prepare_kvasir.py"
    tfm = get_tfms()
    train_ds = datasets.ImageFolder(split/"train", transform=tfm)
    val_ds   = datasets.ImageFolder(split/"val",   transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = KvasirLit(lr=args.lr)
    maxep = args.epochs
    ckpt_dir = Path("models"); asegurar_dir(ckpt_dir)

    trainer = pl.Trainer(
        max_epochs=maxep,
        accelerator="auto",
        devices=1,
        log_every_n_steps=5,
        enable_checkpointing=False
    )
    trainer.fit(model, train_dl, val_dl)

    # guardar peso sencillo por si las dudas
    torch.save(model.state_dict(), args.salida)
    print(f"OK: modelo guardado en {args.salida}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--salida", type=str, default="models/kvasir_resnet18.pt")
    ap.add_argument("--tensorboard", type=int, default=0)
    args = ap.parse_args()
    main(args)
