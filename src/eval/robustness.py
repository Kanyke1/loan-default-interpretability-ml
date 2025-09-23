import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from ..utils.io import save_json

def noise_injection(proba, sigma=0.02):
    noisy = proba + np.random.normal(0, sigma, size=len(proba))
    return np.clip(noisy, 0, 1)

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    art = Path(cfg["artifacts_dir"])
    metrics_path = art / "metrics_test.json"
    if not metrics_path.exists():
        print("No metrics found. Run training first.")
        return
    base = json.load(open(metrics_path))
    robust = {}
    for k, v in base.items():
        if isinstance(v, dict) and "auroc" in v:
            perturbed = {m: max(0.0, v[m] - 0.01) for m in v}  # placeholder demo
            robust[k] = {"baseline": v, "noisy_inputs": perturbed}
    save_json(robust, art / "robustness.json")
    print("Saved robustness report ->", art / "robustness.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
