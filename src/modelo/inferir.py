# inferencia import path robusto
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torchvision import transforms
from PIL import Image
from src.modelo.lightning_model import KvasirLit
from src.utils import CLASES_KVASIR
import numpy as np


_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def cargar_modelo(peso="models/kvasir_resnet18.pt"):
    m = KvasirLit(lr=1e-3)
    m.load_state_dict(torch.load(peso, map_location="cpu"))
    m.eval()
    return m

def predecir(m, ruta_img: str):
    im = Image.open(ruta_img).convert("RGB")
    x = _tf(im).unsqueeze(0)
    with torch.no_grad():
        y = m(x)
        p = torch.softmax(y, dim=1).numpy()[0]
    idx = int(np.argmax(p))
    return CLASES_KVASIR[idx], float(p[idx])
