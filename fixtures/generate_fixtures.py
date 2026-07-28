import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

np.random.seed(42)

fixtures = {}

n_samples, n_features = 100, 3
X_synth = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, -2.0, 0.5])
true_intercept = 4.0
noise = np.random.randn(n_samples) * 0.01
y_synth = X_synth @ true_coef + true_intercept + noise

model_synth = LinearRegression()
model_synth.fit(X_synth, y_synth)
pred_synth = model_synth.predict(X_synth)

fixtures["synthetic"] = {
    "description": "100x3 synthetic data, known coefficients, tiny noise",
    "X": X_synth.tolist(),
    "y": y_synth.tolist(),
    "coef": model_synth.coef_.tolist(),
    "intercept": float(model_synth.intercept_),
    "predictions": pred_synth.tolist(),
    "r2_score": float(model_synth.score(X_synth, y_synth)),
}

diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

model_diabetes = LinearRegression()
model_diabetes.fit(X_diabetes, y_diabetes)
pred_diabetes = model_diabetes.predict(X_diabetes)

fixtures["diabetes"] = {
    "description": "sklearn diabetes dataset (442x10), real-world regression benchmark",
    "X": X_diabetes.tolist(),
    "y": y_diabetes.tolist(),
    "coef": model_diabetes.coef_.tolist(),
    "intercept": float(model_diabetes.intercept_),
    "predictions": pred_diabetes.tolist(),
    "r2_score": float(model_diabetes.score(X_diabetes, y_diabetes)),
}

X_simple = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_simple = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # y = 2x + 1 exactly

model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)
pred_simple = model_simple.predict(X_simple)

fixtures["simple_exact"] = {
    "description": "5x1 data with exact linear relationship y=2x+1, no noise",
    "X": X_simple.tolist(),
    "y": y_simple.tolist(),
    "coef": model_simple.coef_.tolist(),
    "intercept": float(model_simple.intercept_),
    "predictions": pred_simple.tolist(),
    "r2_score": float(model_simple.score(X_simple, y_simple)),
}

with open("linear_regression_fixtures.json", "w") as f:
    json.dump(fixtures, f, indent=2)

for name, fx in fixtures.items():
    print(f"{name}: {fx['description']}")
    print(f"Coef = {fx['coef']}, Intercept = {fx['intercept']:.6f}, R2 = {fx['r2_score']:.6f}")