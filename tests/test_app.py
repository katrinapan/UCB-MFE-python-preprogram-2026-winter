"""Check the prediction contract, failures, and in-process latency."""

import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data():
    return json.loads((ROOT / "example_request.json").read_text(encoding="utf-8"))


@pytest.fixture
def hw4_reference():
    return json.loads(
        Path(__file__).with_name("hw4_reference.json").read_text(encoding="utf-8")
    )


def test_predict_valid_input(client, valid_input_data, hw4_reference):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float)
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == hw4_reference["model_version"]

    trained_columns = ["r21", "r63", "r126"]
    frame = pd.DataFrame([valid_input_data], columns=trained_columns, dtype="float64")
    artifact = joblib.load(ROOT / "model.joblib")
    direct = artifact["pipeline"].predict(frame).item()
    assert body["model_version"] == artifact["model_version"]
    assert body["prediction"] == pytest.approx(direct, rel=1e-10, abs=1e-12)
    assert body["prediction"] == pytest.approx(
        hw4_reference["prediction"], rel=1e-10, abs=1e-12
    )


def test_predict_uses_named_features_and_training_clip(client, valid_input_data):
    payload = {"r126": 0, "r63": valid_input_data["r63"], "r21": 100.0}
    frame = pd.DataFrame(
        [[payload["r21"], payload["r63"], payload["r126"]]], columns=["r21", "r63", "r126"]
    )
    pipeline = app.state.pipeline
    limits = pipeline.named_steps["clip"].kw_args
    assert payload["r21"] > limits["upper"]["r21"]
    clipped = frame.clip(limits["lower"], limits["upper"], axis=1)
    expected = pipeline[1:].predict(clipped).item()

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(expected, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize(
    "raw_input",
    [
        "{}",
        '{"r21":0,"r63":0}',
        '{"r21":0,"r63":0,"r126":0,"extra":0}',
        "[0,0,0]",
        "null",
        (ROOT / "example_invalid_request.json").read_text(encoding="utf-8"),
        '{"r21":true,"r63":0,"r126":0}',
        '{"r21":null,"r63":0,"r126":0}',
        '{"r21":NaN,"r63":0,"r126":0}',
        '{"r21":Infinity,"r63":0,"r126":0}',
        '{"r21":-Infinity,"r63":0,"r126":0}',
        '{"r21":1e400,"r63":0,"r126":0}',
        '{"r21":-1.01,"r63":0,"r126":0}',
        '{"r21":0,',
    ],
    ids=[
        "empty",
        "missing-feature",
        "extra-feature",
        "array",
        "null-body",
        "numeric-string",
        "boolean",
        "null-feature",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "overflow",
        "below-minimum",
        "malformed-json",
    ],
)
def test_predict_invalid_input(client, raw_input, monkeypatch):
    calls = []

    def unexpected_predict(frame):
        calls.append(frame)
        raise AssertionError("Invalid request reached the model")

    monkeypatch.setattr(app.state.pipeline, "predict", unexpected_predict)
    response = client.post(
        "/predict", content=raw_input, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    assert calls == []


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def broken_predict(_frame):
        raise RuntimeError("simulated estimator failure")

    monkeypatch.setattr(app.state.pipeline, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Inference failed"}
    assert "simulated estimator failure" in caplog.text


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    elapsed_ms = []
    for _ in range(30):
        start = perf_counter()
        response = client.post("/predict", json=valid_input_data)
        elapsed_ms.append((perf_counter() - start) * 1_000)
        assert response.status_code == 200

    p95 = sorted(elapsed_ms)[math.ceil(0.95 * len(elapsed_ms)) - 1]
    print(
        f"TestClient, n={len(elapsed_ms)}: median={median(elapsed_ms):.3f} ms, p95={p95:.3f} ms"
    )
