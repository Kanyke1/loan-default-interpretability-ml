import argparse, json, math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import pandas as pd
from pathlib import Path
from ..data.loaders import load_dataset, stratified_split
from sklearn.preprocessing import StandardScaler
from ..utils.io import ensure_dir

class MLP(pl.LightningModule):
    def __init__(self, in_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.bauroc = BinaryAUROC()
        self.baprc = BinaryAveragePrecision()
        self.lr = lr

    def forward(self, x): return self.net(x)

    def step(self, batch, stage):
        x, y = batch
        p = self(x).squeeze(1)
        loss = nn.BCELoss()(p, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.bauroc.update(p, y.int())
        self.baprc.update(p, y.int())
        return loss

    def training_step(self, batch, _): return self.step(batch, "train")
    def validation_step(self, batch, _): return self.step(batch, "val")
    def test_step(self, batch, _): return self.step(batch, "test")

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.bauroc.compute(), prog_bar=True)
        self.log("val_auprc", self.baprc.compute(), prog_bar=True)
        self.bauroc.reset(); self.baprc.reset()

    def on_test_epoch_end(self):
        self.log("test_auroc", self.bauroc.compute(), prog_bar=True)
        self.log("test_auprc", self.baprc.compute(), prog_bar=True)
        self.bauroc.reset(); self.baprc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def to_tensor_dataset(X, y):
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    return TensorDataset(X, y)

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    art = Path(cfg["artifacts_dir"]); ensure_dir(art.as_posix())

    X, y = load_dataset(cfg["dataset"]["path"], cfg["target"], cfg["dataset"]["id_column"])
    Xtr, ytr, Xva, yva, Xte, yte = stratified_split(X, y, cfg["test_size"], cfg["val_size"], cfg["random_seed"])

    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns)
    Xva_s = pd.DataFrame(scaler.transform(Xva), columns=Xva.columns)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns)

    train_ds = to_tensor_dataset(Xtr_s, ytr)
    val_ds   = to_tensor_dataset(Xva_s, yva)
    test_ds  = to_tensor_dataset(Xte_s, yte)

    model = MLP(in_dim=Xtr_s.shape[1], lr=cfg["training"]["lr"])
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        enable_checkpointing=False,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    trainer.fit(model, DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True),
                      DataLoader(val_ds, batch_size=cfg["training"]["batch_size"]))
    trainer.test(model, DataLoader(test_ds, batch_size=cfg["training"]["batch_size"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
