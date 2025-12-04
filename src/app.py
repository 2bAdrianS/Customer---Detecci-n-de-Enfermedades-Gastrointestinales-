# V5 - gráficas guardables

import os, sys, io, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

import streamlit as st

from src.modelo.inferir import cargar_modelo, predecir
from src.utils import CLASES_KVASIR, asegurar_dir


st.set_page_config(
    page_title="Detección de enfermedades gastrointestinales",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

THEME = {
    "bg": "#0e0f13",
    "surface": "#171a21",
    "accent": "#4e41a1",
    "text": "#e5e7eb",
    "muted": "#9ca3af",
}

st.markdown(f"""
<style>
:root {{
  --bg: {THEME['bg']};
  --surface: {THEME['surface']};
  --accent: {THEME['accent']};
  --text: {THEME['text']};
  --muted: {THEME['muted']};
}}
.stApp {{
  background: var(--bg) !important;
  color: var(--text);
}}
.block-container {{ max-width: 1100px; padding-top: 2rem; }}

h1, h2, h3 {{ color: var(--text); }}
p, span, label {{ color: var(--muted); }}

/* Tarjeta del uploader */
[data-testid="stFileUploader"] > section {{
  background: var(--surface);
  border: 1px solid #2a2f3a;
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 4px 24px rgba(0,0,0,.35);
}}

/* Zona de arrastre (traducción y estilo) */
div[data-testid="stFileUploadDropzone"] > div {{ position: relative; }}
div[data-testid="stFileUploadDropzone"] > div * {{ font-size: 0 !important; }}
div[data-testid="stFileUploadDropzone"] > div::after {{
  content: "Arrastra y suelta la imagen aquí";
  position: absolute; inset: 0;
  display: grid; place-items: center;
  font-size: 0.95rem !important; font-weight: 600;
  color: rgba(250,250,250,.92);
}}
div[data-testid="stFileUploader"] small {{ font-size: 0 !important; }}
div[data-testid="stFileUploader"] small::after {{
  content: "Límite 200MB por archivo · JPG, PNG, JPEG";
  font-size: 0.75rem !important; opacity: .75;
}}
div[data-testid="stFileUploader"] button span {{ font-size: 0 !important; }}
div[data-testid="stFileUploader"] button::after {{
  content: "Examinar archivos";
  font-size: 0.85rem !important;
}}

button[kind="primary"] {{
  background: var(--accent) !important;
  color: #fff !important;
  border-radius: 12px !important;
  border: none !important;
}}
button[kind="secondary"] {{ border-radius: 12px !important; }}

[data-testid="stImage"] img {{
  border-radius: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}}

#MainMenu, footer {{ display: none; }}
header [data-testid="stHeaderActionElements"] {{ display:none; }}
</style>
""", unsafe_allow_html=True)
# fin UI


# Imagen Cargadad
st.title("Detección de enfermedades gastrointestinales")
#st.caption("ResNet18 + Transfer Learning (Kvasir)")

peso = Path("models/kvasir_resnet18.pt")
if not peso.exists():
    st.error("No existe el modelo. Ejecuta `run_app.bat` para auto-entrenar.")
    st.stop()

modelo = cargar_modelo(peso)

sub = st.file_uploader(
    "Sube tu imagen endoscópica",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

if sub is not None and peso.exists():
    tmp = Path("data/tmp.jpg")
    tmp.parent.mkdir(exist_ok=True, parents=True)
    with open(tmp, "wb") as f:
        f.write(sub.getbuffer())

    etiqueta, prob = predecir(modelo, str(tmp))
    st.image(str(tmp), caption=f"Predicción: {etiqueta} — confianza {prob:.2f}")
    st.success(f"Clase detectada: {etiqueta}")
    st.progress(min(1.0, prob))
    st.write("Clases:", CLASES_KVASIR)


st.markdown("---")
st.header("📊 Evaluación del modelo")

# métricas/gráficas

# mismas transformaciones que el software usa en inferencia
TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_split(split: str):
    base = Path("data/split") / split
    if not base.exists():
        st.error(f"No encuentro el split `{split}` en {base}. "
                 "Ejecuta el script de preparación/entrenamiento primero.")
        st.stop()
    ds = datasets.ImageFolder(str(base), transform=TFM)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    return ds, dl

@st.cache_data(show_spinner=False)
def infer_dataset(split: str, pesos_mtime: float):
    """Devuelve y_true (N,), y_pred (N,), y_proba (N,C) para el split dado.
       Cachea por (split, fecha_mod_pesos) para no recalcular si no cambia el modelo.
    """
    ds, dl = load_split(split)
    modelo.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = modelo(xb)  # el LightningModule devuelve logits
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            y_true.append(yb.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            y_proba.append(prob.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_proba = np.concatenate(y_proba)
    return y_true, y_pred, y_proba, ds.classes  # classes por si cambia el orden


def _save_fig(fig, base_name: str, ext: str = "png") -> Path:
    out = asegurar_dir("reports")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(out) / f"{stamp}_{base_name}.{ext.lower()}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(y_true, y_pred, labels, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i,j]:.0f}" if not normalize else f"{cm[i,j]*100:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", color="black")

    title = "Matriz de confusión (normalizada)" if normalize else "Matriz de confusión"
    ax.set_title(title)
    return fig


def plot_f1_bars(y_true, y_pred, labels):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1 = [rep.get(str(i), {}).get("f1-score", 0.0) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, f1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("F1-score")
    ax.set_xlim(0, 1)
    ax.set_title("F1 por clase")
    for i, v in enumerate(f1):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center")
    return fig


def plot_roc_ovr(y_true, y_proba, labels):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for i, name in enumerate(labels):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curvas ROC (one-vs-rest)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    return fig


def plot_pr_curves(y_true, y_proba, labels):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for i, name in enumerate(labels):
        y_bin = (y_true == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        ap = average_precision_score(y_bin, y_proba[:, i])
        ax.plot(rec, prec, label=f"{name} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title("Curvas Precisión–Recall")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    return fig


def plot_training_curves():
    logs = Path("lightning_logs")
    if not logs.exists():
        return None
    versions = sorted(logs.glob("version_*"), key=lambda p: p.stat().st_mtime)
    if not versions:
        return None
    metrics = versions[-1] / "metrics.csv"
    if not metrics.exists():
        return None
    df = pd.read_csv(metrics)

    # última medición por época de cada métrica
    def last_by_epoch(col):
        x = df[["epoch", col]].dropna()
        return x.groupby("epoch")[col].last()

    # columnas posibles - flexible a nombres
    acc_cols  = [c for c in df.columns if c.endswith("_acc")]
    loss_cols = [c for c in df.columns if c.endswith("_loss")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes

    for c in sorted(acc_cols):
        s = last_by_epoch(c)
        ax1.plot(s.index, s.values, marker="o", label=c.replace("_", " "))
    ax1.set_title("Accuracy por época")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Accuracy"); ax1.grid(alpha=.25); ax1.legend(fontsize=8)

    for c in sorted(loss_cols):
        s = last_by_epoch(c)
        ax2.plot(s.index, s.values, marker="o", label=c.replace("_", " "))
    ax2.set_title("Loss por época")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Loss"); ax2.grid(alpha=.25); ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ejecución panel
col1, col2 = st.columns([1,1])
with col1:
    split = st.radio("Conjunto a evaluar", options=["val", "test"], index=0, horizontal=True)
with col2:
    fmt = st.selectbox("Formato para exportar", options=["PNG", "JPG"], index=0)

if st.button("Calcular y generar gráficas", type="primary"):
    with st.spinner("Calculando métricas…"):
        y_true, y_pred, y_proba, labels = infer_dataset(split, peso.stat().st_mtime)

    # Matriz de confusión - conteos
    fig1 = plot_confusion(y_true, y_pred, labels, normalize=False)
    path1 = _save_fig(fig1, f"confusion_{split}", ext=fmt)
    st.pyplot(fig1)
    st.download_button("Descargar matriz (conteos)", data=open(path1, "rb").read(),
                       file_name=path1.name, mime=f"image/{fmt.lower()}")

    # Matriz de confusióbn normalizada
    fig2 = plot_confusion(y_true, y_pred, labels, normalize=True)
    path2 = _save_fig(fig2, f"confusion_norm_{split}", ext=fmt)
    st.pyplot(fig2)
    st.download_button("Descargar matriz normalizada", data=open(path2, "rb").read(),
                       file_name=path2.name, mime=f"image/{fmt.lower()}")

    # F1 por clase
    fig3 = plot_f1_bars(y_true, y_pred, labels)
    path3 = _save_fig(fig3, f"f1_por_clase_{split}", ext=fmt)
    st.pyplot(fig3)
    st.download_button("Descargar F1 por clase", data=open(path3, "rb").read(),
                       file_name=path3.name, mime=f"image/{fmt.lower()}")

    # Curvas ROC
    fig4 = plot_roc_ovr(y_true, y_proba, labels)
    path4 = _save_fig(fig4, f"roc_ovr_{split}", ext=fmt)
    st.pyplot(fig4)
    st.download_button("Descargar ROC one-vs-rest", data=open(path4, "rb").read(),
                       file_name=path4.name, mime=f"image/{fmt.lower()}")

    # Curvas Precisión – Recal
    fig5 = plot_pr_curves(y_true, y_proba, labels)
    path5 = _save_fig(fig5, f"pr_curves_{split}", ext=fmt)
    st.pyplot(fig5)
    st.download_button("Descargar curvas P–R", data=open(path5, "rb").read(),
                       file_name=path5.name, mime=f"image/{fmt.lower()}")

    # Curvas de entrenamiento - deben tener logs
    fig6 = plot_training_curves()
    if fig6 is not None:
        path6 = _save_fig(fig6, f"training_curves", ext=fmt)
        st.pyplot(fig6)
        st.download_button("Descargar curvas de entrenamiento", data=open(path6, "rb").read(),
                           file_name=path6.name, mime=f"image/{fmt.lower()}")
    else:
        st.info("No encontré `lightning_logs/**/metrics.csv` para dibujar las curvas de entrenamiento.")

    st.success("¡Listo! Las gráficas también quedaron guardadas en la carpeta `reports/`.")