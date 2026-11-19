"""Regenerate the example request and the original HW4 scoring reference."""

import argparse
import json
from pathlib import Path

import joblib

from course_data import FEATURES, TEST_END, TEST_START, load_samples

ROOT = Path(__file__).resolve().parent


def write_reference(samples, ridge, bounds, *, training_rows, model_version):
    evaluation = samples.loc[samples["entry"].between(TEST_START, TEST_END)].copy()
    features = list(FEATURES)
    clipped = evaluation[features].clip(bounds["lower"], bounds["upper"], axis=1)
    evaluation["score"] = ridge.predict(clipped)
    row = evaluation.dropna(subset=["target"]).sort_values(["entry", "symbol"]).iloc[0]
    request = {name: float(row[name]) for name in features}
    reference = {
        "symbol": str(row["symbol"]),
        "signal_date": str(row["signal"].date()),
        "entry_date": str(row["entry"].date()),
        "exit_date": str(row["exit"].date()),
        "training_rows": training_rows,
        "prediction": float(row["score"]),
        "model_version": model_version,
    }
    for path, value in (
        (ROOT / "example_request.json", request),
        (ROOT / "tests" / "hw4_reference.json", reference),
    ):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"input": request, **reference}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="Path to the full course DuckDB file")
    args = parser.parse_args()
    artifact = joblib.load(ROOT / "model.joblib")
    pipeline = artifact["pipeline"]
    write_reference(
        load_samples(args.database),
        pipeline[1:],
        pipeline.named_steps["clip"].kw_args,
        training_rows=artifact["training_rows"],
        model_version=artifact["model_version"],
    )


if __name__ == "__main__":
    main()
