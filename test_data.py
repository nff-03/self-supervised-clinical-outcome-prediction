import numpy as np

from data_utils import build_dataset, impute, normalize

X, y = build_dataset("../data/set-a", "../data/Outcomes-a.txt")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Example y:", y[:10])

X = impute(X)
X = normalize(X)

mean = X.mean(axis=(0,1))
std = X.std(axis=(0,1))

print("Feature means:", mean)
print("Feature stds:", std)

print("Min:", X.min())
print("Max:", X.max())
print(X[0])

print("Any NaNs left?", np.isnan(X).any())