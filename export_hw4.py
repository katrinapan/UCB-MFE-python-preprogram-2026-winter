"""Reproduce and export the original HW4 experiment's fitted model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from course_data import FEATURES, TEST_START, load_samples
from make_reference import write_reference

ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "hw4-ridge-final-2016-04-01-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="Path to the full course DuckDB file")
    args = parser.parse_args()

    samples = load_samples(args.database)
    train = samples.loc[
        samples["entry"].between("2010-01-01", "2015-12-31")
        & samples["exit"].lt(pd.Timestamp(TEST_START))
        & samples["target"].notna()
    ]
    features = train[list(FEATURES)]
    target = train["target"]
    limits = features.quantile([0.01, 0.99])
    lower, upper = limits.loc[0.01], limits.loc[0.99]
    target_lower, target_upper = target.quantile([0.01, 0.99])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1000))
    ridge.fit(features.clip(lower, upper, axis=1), target.clip(target_lower, target_upper))

    # Capture the HW4 score before attaching clipping to the serving pipeline.
    bounds = {"lower": lower, "upper": upper}
    write_reference(
        samples, ridge, bounds, training_rows=len(train), model_version=MODEL_VERSION
    )
    clip = FunctionTransformer(
        pd.DataFrame.clip,
        kw_args={**bounds, "axis": "columns"},
        feature_names_out="one-to-one",
    ).fit(features.iloc[:1])
    pipeline = Pipeline([("clip", clip), *ridge.steps])
    joblib.dump(
        {"pipeline": pipeline, "model_version": MODEL_VERSION, "training_rows": len(train)},
        ROOT / "model.joblib",
        compress=3,
    )


if __name__ == "__main__":
    main()
