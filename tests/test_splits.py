import pandas as pd
from src.data.loaders import stratified_split

def test_stratified_split_shapes():
    X = pd.DataFrame({"a":[1,2,3,4,5,6,7,8], "b":[0,1,0,1,0,1,0,1]})
    y = pd.Series([0,1,0,1,0,1,0,1])
    Xtr, ytr, Xva, yva, Xte, yte = stratified_split(X, y, 0.25, 0.25, 42)
    assert len(Xtr) + len(Xva) + len(Xte) == len(X)
    assert len(ytr) + len(yva) + len(yte) == len(y)
