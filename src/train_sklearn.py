import argparse, json
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from .data.loaders import load_dataset, stratified_split
from .features.transform import build_preprocessor
from .models.sklearn_models import ModelZoo
from .eval.metrics import all_metrics
from .utils.io import ensure_dir, save_json

def train_and_eval(X, y, cfg, name, model):
    pre = build_preprocessor(X, scale_numeric=cfg["features"].get("scale_numeric", True))
    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X, y)
    return pipe

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    art_dir = Path(cfg["artifacts_dir"]); ensure_dir(art_dir.as_posix())

    X, y = load_dataset(cfg["dataset"]["path"], cfg["target"], cfg["dataset"]["id_column"])
    Xtr, ytr, Xva, yva, Xte, yte = stratified_split(X, y, cfg["test_size"], cfg["val_size"], cfg["random_seed"])

    zoo = ModelZoo(cfg["models"].get("logreg", {}))
    models = {
        "logreg": zoo.logreg(),
        "tree": ModelZoo(cfg["models"].get("tree", {})).tree(),
        "xgb": ModelZoo(cfg["models"].get("xgb", {})).xgb()
    }

    metrics = {}
    preds = pd.DataFrame()
    preds["y_true"] = yte.reset_index(drop=True)

    for name, m in models.items():
        pipe = train_and_eval(Xtr.append(Xva), ytr.append(yva), cfg, name, m)
        proba = pipe.predict_proba(Xte)[:,1]
        cur = all_metrics(yte.values, proba)
        metrics[name] = cur
        preds[f"prob_{name}"] = proba

    # default: pick best by AUROC
    best_name = max(metrics, key=lambda k: metrics[k]["auroc"])
    preds["y_prob"] = preds[f"prob_{best_name}"]
    # Attach features used for optional fairness grouping if basic columns exist
    if "age" in Xte.columns:
        preds["age"] = Xte["age"].reset_index(drop=True)

    save_json(metrics, art_dir / "metrics_test.json")
    preds.to_csv(art_dir / "predictions_test.csv", index=False)
    print("Saved:", art_dir / "metrics_test.json", "and predictions_test.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
