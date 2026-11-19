# HW5 · Serve and containerize the HW4 return model

This service runs version `hw4-ridge-final-2016-04-01-v1` of the fixed model used for the HW4 final-period experiment. It predicts the gross adjusted-price return from the next session's close to the first trading close on or after 90 calendar days from entry. The forecast is a decimal, with `0.03` representing a 3% return.

## API contract

Send a JSON object to `POST /predict`. All three keys are required. The request accepts keys in any order and supplies a float64 pandas DataFrame to the model in the training order below.

| Training order | Field | Type and range | Definition |
| --- | --- | --- | --- |
| 1 | `r21` | Finite JSON number, ≥ −1 | Adjusted close at signal / adjusted close 21 trading sessions earlier − 1 |
| 2 | `r63` | Finite JSON number, ≥ −1 | Same return over 63 trading sessions |
| 3 | `r126` | Finite JSON number, ≥ −1 | Same return over 126 trading sessions |

Returns use decimal units. Positive adjusted prices imply a return above −1 in exact arithmetic. The inclusive bound at −1 allows the floating-point limit. Large positive returns may occur and are handled by the original training-derived clip, so the API imposes no upper cap. JSON integers and floating-point numbers are accepted. Extra keys, missing keys, arrays, numeric strings, booleans, nulls, non-finite values and values below −1 receive HTTP 422.

A successful response returns HTTP 200 with a finite numeric `prediction` and a string `model_version`. An unexpected inference error returns HTTP 500 with `{"detail":"Inference failed"}`. The exception is recorded in server logs. Only the fitted pipeline runs during inference. Clients are responsible for computing features from the same adjusted-price convention and point-in-time market calendar as HW4.

## Model and reference

The original HW4 run fitted a ridge regression with `alpha=1000` after clipping each feature to its training 1st and 99th percentiles and standardizing it with `StandardScaler`. Training targets were clipped to their own training 1st and 99th percentiles. The exported model uses all 6,812 eligible entries from 2010–2015 with known targets and exits before April 1, 2016.

HW4 did not save its fitted estimator to disk. The export reproduces the original HW4 data selection, feature calculations, and ridge fit on the full course equities database. It was run in the original Python 3.9.6 training environment. The clipping bounds, scaler and ridge coefficients, and fixed reference were checked against an export from the original HW4 code. The export attached the learned clipping limits to the fitted scaler and Ridge as one joblib pipeline. `requirements.txt` pins the original NumPy, pandas, scikit-learn, SciPy, joblib and threadpoolctl versions for Python 3.12 serving.

`example_request.json` contains the raw features for ticker AAM at its March 31, 2016 signal close, with entry April 1 and exit June 30. `tests/hw4_reference.json` records the score `0.0307423007789469` (about 3.07423%) from the original HW4 scoring path before the serving pipeline was assembled. The test compares that fixed score, a direct call to the saved pipeline and the API response with relative tolerance `1e-10` and absolute tolerance `1e-12` to allow for floating-point differences across platforms.

The original HW4 backtest stopped at an unresolved 2024 quote gap. That occurred after model fitting and scoring. The exported model represents the same fit used in that experiment.

The fitted binary `model.joblib` ships with this submission. The service resolves it relative to `app.py` and loads it once at startup per worker process. The serving environment requires neither DuckDB nor the course database. Use only a trusted copy of the joblib artifact.

`course_data.py` recreates the original HW4 feature and target rows from `equities_top2000.duckdb`, available in bCourses. To regenerate the artifact and reference in an activated Python 3.12 environment, install the pinned dependencies and the training-only DuckDB package.

```sh
python -m pip install -r requirements.txt "duckdb==1.4.5"
python export_hw4.py /path/to/equities_top2000.duckdb
```

To regenerate just the example and reference from the existing fitted model, run the following.

```sh
python make_reference.py /path/to/equities_top2000.duckdb
```

## Run locally

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make lint test
python -m pytest -q -s tests/test_app.py::test_predict_performance
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

View OpenAPI docs at `http://127.0.0.1:8000/docs`. Stop local Uvicorn before running the container on the same port.

## Local verification

All 18 tests passed, and Ruff passed its lint and format checks. On an Apple M5 Pro with 24 GiB RAM, macOS 27.0.1 and Python 3.12.16, 30 sequential in-process TestClient requests after five warm-ups took a median of **1.395 ms** and a nearest-rank p95 of **1.730 ms**. Each request was timed with `time.perf_counter`, excluding startup and real network I/O. See [the verification transcript](verification.txt).

## Build and check the container

```sh
make build
make run
```

From another shell in this folder, run these requests. The expected statuses are 200 and 422, respectively.

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @example_request.json
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @example_invalid_request.json
```

Run `docker stop mfe-hw5` to stop the first container. Run `make run` again with the same local image, then repeat the valid request. The build context allows only the Dockerfile, dependency list, API module and fitted artifact.

The image built with `make build` on Linux arm64. A real localhost request returned HTTP 200 with the recorded HW4 prediction, an invalid request returned HTTP 422, and a new container from the same image returned the identical prediction and model version. See [the Docker transcript](container_evidence.txt).

## Before deployment

Verify market data lineage, corporate-action adjustments and point-in-time feature availability before using forecasts. Log model versions and monitor latency, errors, feature drift and realized accuracy after the forecast horizon. Sign and verify release artifacts, build from a pinned base-image digest, and add authentication, TLS, rate limits and resource limits at the deployment boundary. Set a latency target after measuring concurrent traffic and load-test it on the target deployment hardware.

## Attribution

The FastAPI lifespan skeleton, pytest fixture structure, Dockerfile base and Makefile were adapted from the [course HW5 template](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/tree/main/Homeworks/HW5/template). The original HW4 experiment supplies the feature definitions, and its data loader and fitting logic were adapted into `course_data.py` and `export_hw4.py`. AI assisted with the model export, API, validation tests, container and documentation.
