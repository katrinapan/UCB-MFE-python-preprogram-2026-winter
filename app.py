"""Serve the fitted HW4 90-calendar-day return model."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

ARTIFACT = Path(__file__).resolve().with_name("model.joblib")
FEATURES = ("r21", "r63", "r126")
logger = logging.getLogger(__name__)

TrailingReturn = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1.0)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r21: TrailingReturn = Field(description="21-session adjusted-price return, as a decimal")
    r63: TrailingReturn = Field(description="63-session adjusted-price return, as a decimal")
    r126: TrailingReturn = Field(description="126-session adjusted-price return, as a decimal")


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load(ARTIFACT)
    app.state.pipeline = artifact["pipeline"]
    app.state.model_version = artifact["model_version"]
    yield


app = FastAPI(title="HW4 stock return model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # Echoing a rejected NaN or infinity would make the error itself invalid JSON.
    details = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    features = pd.DataFrame([data.model_dump()], columns=FEATURES, dtype="float64")
    try:
        prediction = request.app.state.pipeline.predict(features)
        return OutputData(
            prediction=float(prediction.item()),
            model_version=request.app.state.model_version,
        )
    except Exception as exc:
        logger.exception("HW4 inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from exc
