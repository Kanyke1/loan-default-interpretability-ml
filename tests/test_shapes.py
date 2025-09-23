import pandas as pd
from src.features.transform import build_preprocessor

def test_preprocessor_runs():
    X = pd.DataFrame({
        "num1":[1.0,2.0,3.0],
        "cat1":["a","b","a"]
    })
    pre = build_preprocessor(X)
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 3
