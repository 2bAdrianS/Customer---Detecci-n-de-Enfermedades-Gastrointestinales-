# modelo lightning import path robusto
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch, torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.classification import MulticlassAccuracy
from src.utils import CLASES_KVASIR

class KvasirLit(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feats = base.fc.in_features
        base.fc = nn.Linear(in_feats, len(CLASES_KVASIR))
        self.modelo = base
        self.crit = nn.CrossEntropyLoss()
        self.acc = MulticlassAccuracy(num_classes=len(CLASES_KVASIR), average='macro')

    def forward(self, x):
        return self.modelo(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def paso(self, batch, etapa):
        x,y = batch
        yhat = self(x)
        loss = self.crit(yhat,y)
        acc = self.acc(torch.softmax(yhat,dim=1), y)
        self.log(f"{etapa}_loss", loss, prog_bar=True)
        self.log(f"{etapa}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.paso(batch,"train")

    def validation_step(self, batch, batch_idx):
        self.paso(batch,"val")
