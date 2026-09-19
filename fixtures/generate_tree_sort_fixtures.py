import json
import os

import numpy as np
from sklearn.utils._sorting import _py_simultaneous_sort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tree", "testdata", "sort_fixtures.json")

rng = np.random.default_rng(99)
cases = []


def add(name, values):
    values = np.asarray(values, dtype=np.float32)
    v = values.copy()
    idx = np.arange(len(v), dtype=np.intp)
    if len(v):
        _py_simultaneous_sort(v, idx, len(v), use_three_way_partition=True)
    cases.append(
        {
            "name": name,
            "values": values.tolist(),
            "sorted_values": v.tolist(),
            "sorted_indices": idx.tolist(),
        }
    )


add("empty", [])
add("single", [3.5])
for n in (2, 5, 15, 16, 17, 31, 64, 100, 257, 1000):
    add(f"random_{n}", rng.normal(size=n))
    add(f"ties_{n}", rng.integers(0, 4, size=n))
    add(f"few_distinct_{n}", rng.integers(0, 2, size=n))
    add(f"sorted_{n}", np.arange(n))
    add(f"reversed_{n}", np.arange(n)[::-1])
    add(f"constant_{n}", np.full(n, 2.0))
    half = np.arange(n // 2)
    add(f"organ_pipe_{n}", np.concatenate([half, half[::-1]]))
    add(f"sawtooth_{n}", np.arange(n) % 7)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(cases, f)

print("sort fixtures written to", OUT, "(%d cases)" % len(cases))
