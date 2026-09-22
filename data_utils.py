from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "HR", "SysABP", "DiasABP", "MeanABP",
    "RespRate", "Temp", "SpO2", "Glucose"
]

BASE_DIR = Path(__file__).resolve().parent.parent

def parse_hour(t):
    h, m = t.split(":")
    return min(int(h), 47)

def load_outcomes(path):
    df = pd.read_csv(path)
    return dict(zip(df["RecordID"], df["In-hospital_death"]))

def load_patient(file_path):
    df = pd.read_csv(file_path)

    df = df[df["Parameter"].isin(FEATURES)].copy()

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df.loc[df["Value"] < 0, "Value"] = np.nan

    df["Hour"] = df["Time"].apply(parse_hour)

    x = np.full((48, len(FEATURES)), np.nan)

    feat_idx = {f: i for i, f in enumerate(FEATURES)}

    for _, row in df.iterrows():
        h = int(row["Hour"])
        j = feat_idx[row["Parameter"]]
        x[h, j] = row["Value"]

    return x

def build_dataset(data_dir, outcome_file):
    outcomes = load_outcomes(outcome_file)

    X, y = [], []

    for fp in Path(data_dir).glob("*.txt"):
        rid = int(fp.stem)

        if rid not in outcomes:
            continue

        x = load_patient(fp)
        X.append(x)
        y.append(outcomes[rid])

    X = np.array(X)
    y = np.array(y)

    return X, y

def impute(X):
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i])

        df = df.ffill()
        df = df.fillna(df.mean())
        df = df.fillna(0)

        X[i] = df.values

    return X

def normalize(X):
    mean = X.mean(axis=(0,1))
    std = X.std(axis=(0,1)) + 1e-6
    return (X - mean) / std