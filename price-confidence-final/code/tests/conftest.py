"""Shared fixtures. A small synthetic dataset is generated once per test session (seconds, not minutes)."""
import json
import os
import sys

import pytest

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

import generate_dataset  # noqa: E402
import train_models  # noqa: E402

SMALL_N = 40
SMALL_SEED = 123


@pytest.fixture(scope="session")
def small_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("small")
    generate_dataset.run(SMALL_SEED, "default", SMALL_N, str(d))
    return str(d)


@pytest.fixture(scope="session")
def small_tables(small_dir):
    return train_models.load(small_dir)


@pytest.fixture(scope="session")
def small_model(small_tables, tmp_path_factory):
    P, W, Mk, R, man = small_tables
    m = train_models.fit(P, W, Mk, R, man)
    path = tmp_path_factory.mktemp("model") / "model.json"
    path.write_text(json.dumps(m))
    return m, str(path)


@pytest.fixture(scope="session")
def shipped_model():
    with open(os.path.join(CODE, "model.json")) as f:
        return json.load(f)
