import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path: str, target: str, id_column: str | None = None):
    df = pd.read_csv(path)
    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def stratified_split(X, y, test_size=0.2, val_size=0.2, random_state=1337):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), stratify=y_temp, random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
