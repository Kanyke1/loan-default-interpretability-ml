import argparse, json, pandas as pd
from pathlib import Path
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import roc_auc_score, f1_score
from ..utils.io import save_json

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    art = Path(cfg["artifacts_dir"])
    # Placeholder: read predictions with a 'group' column if available
    preds_path = art / "predictions_test.csv"
    if not preds_path.exists():
        print("No predictions found. Run training first.")
        return
    df = pd.read_csv(preds_path)
    if "group" not in df.columns:
        df["group"] = (df["age"] < df["age"].median()).astype(int) if "age" in df.columns else 0
    mf = MetricFrame(
        metrics={"auroc": roc_auc_score, "f1": f1_score, "selection_rate": selection_rate},
        y_true=df["y_true"],
        y_pred=(df["y_prob"] >= 0.5).astype(int),
        sensitive_features=df["group"]
    )
    out = {"by_group": mf.by_group.to_dict(), "overall": mf.overall.to_dict()}
    save_json(out, art / "fairness.json")
    print("Saved fairness report ->", art / "fairness.json")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
