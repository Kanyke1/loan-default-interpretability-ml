from pathlib import Path
import json

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(Path(path).parent.as_posix())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
