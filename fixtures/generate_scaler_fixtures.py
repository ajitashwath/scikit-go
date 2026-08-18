import json
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "standard_scaler_fixtures.json")

np.random.seed(42)

fixtures = {}

# Synthetic multi-feature dataset
n_samples, n_features = 100, 4
X_synth = np.random.randn(n_samples, n_features) * 2.0 + 3.0
X_synth[:, 2] = X_synth[:, 2].astype(int).astype(float)  # discrete-ish column

scaler_synth = StandardScaler()
X_synth_scaled = scaler_synth.fit_transform(X_synth)
X_synth_inverse = scaler_synth.inverse_transform(X_synth_scaled)

fixtures["synthetic"] = {
    "description": "100x4 random data with non-unit scale/offset",
    "X": X_synth.tolist(),
    "mean": scaler_synth.mean_.tolist(),
    "scale": scaler_synth.scale_.tolist(),
    "var": scaler_synth.var_.tolist(),
    "X_scaled": X_synth_scaled.tolist(),
    "X_inverse": X_synth_inverse.tolist(),
}

# Dataset with a zero-variance (constant) column, the sklearn scale_=1.0 edge case
X_const = np.random.randn(50, 3)
X_const[:, 1] = 7.0  # constant feature

scaler_const = StandardScaler()
X_const_scaled = scaler_const.fit_transform(X_const)
X_const_inverse = scaler_const.inverse_transform(X_const_scaled)

fixtures["constant_column"] = {
    "description": "50x3 data where feature 1 is constant (scale_ stays 1.0)",
    "X": X_const.tolist(),
    "mean": scaler_const.mean_.tolist(),
    "scale": scaler_const.scale_.tolist(),
    "var": scaler_const.var_.tolist(),
    "X_scaled": X_const_scaled.tolist(),
    "X_inverse": X_const_inverse.tolist(),
}

# Single feature
X_single = np.arange(1.0, 11.0).reshape(-1, 1)

scaler_single = StandardScaler()
X_single_scaled = scaler_single.fit_transform(X_single)
X_single_inverse = scaler_single.inverse_transform(X_single_scaled)

fixtures["single_feature"] = {
    "description": "10x1 data, single feature 1..10",
    "X": X_single.tolist(),
    "mean": scaler_single.mean_.tolist(),
    "scale": scaler_single.scale_.tolist(),
    "var": scaler_single.var_.tolist(),
    "X_scaled": X_single_scaled.tolist(),
    "X_inverse": X_single_inverse.tolist(),
}

with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

for name, fx in fixtures.items():
    print(f"{name}: {fx['description']}")
    print(f"Mean = {fx['mean']}")
    print(f"Scale = {fx['scale']}")
    print(f"Var = {fx['var']}")
