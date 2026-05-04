from __future__ import annotations

import io
import json
import os
from typing import Any

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load the fitted preprocessor and classifier from the model directory.

    Args:
        model_dir: SageMaker model directory containing serialized artifacts.

    Returns:
        Dictionary with loaded ``preprocessor`` and ``model`` objects.

    Raises:
        FileNotFoundError: If either model artifact is missing.
    """
    artifact_paths = {
        "model": os.path.join(model_dir, "model.joblib"),
        "preprocessor": os.path.join(model_dir, "preprocessor.joblib"),
        "label_mapping": os.path.join(model_dir, "label_mapping.json"),
    }
    missing_paths = [path for path in artifact_paths.values() if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(f"Missing model artifacts: {missing_paths}")

    with open(artifact_paths["label_mapping"], encoding="utf-8") as file:
        label_mapping = json.load(file)

    return {
        "model": joblib.load(artifact_paths["model"]),
        "preprocessor": joblib.load(artifact_paths["preprocessor"]),
        "id_to_class": label_mapping["id_to_class"],
    }


def _records_to_frame(records: Any) -> pd.DataFrame:
    """Convert JSON records to a feature DataFrame.

    Args:
        records: JSON-decoded object containing one record or a list of records.

    Returns:
        DataFrame with raw feature columns in training order.

    Raises:
        ValueError: If required feature columns are missing.
    """
    frame = pd.DataFrame([records] if isinstance(records, dict) else records)
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")
    return frame[FEATURE_COLUMNS]


def input_fn(request_body: str | bytes, content_type: str) -> pd.DataFrame:
    """Parse an inference request into raw model features.

    Args:
        request_body: Request payload from the SageMaker endpoint.
        content_type: MIME type of the request payload.

    Returns:
        DataFrame containing raw, untransformed features.

    Raises:
        ValueError: If the content type or payload schema is unsupported.
    """
    body = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    normalized_content_type = content_type.split(";")[0].strip().lower()

    if normalized_content_type == "application/json":
        return _records_to_frame(json.loads(body))
    if normalized_content_type == "text/csv":
        frame = pd.read_csv(io.StringIO(body))
        missing_columns = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"CSV payload is missing columns: {missing_columns}")
        return frame[FEATURE_COLUMNS]

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: pd.DataFrame, model: dict[str, Any]) -> list[str]:
    """Run preprocessing and prediction.

    Args:
        input_data: Raw feature DataFrame.
        model: Dictionary returned by ``model_fn``.

    Returns:
        Predicted class names.
    """
    transformed_features = model["preprocessor"].transform(input_data)
    predictions = model["model"].predict(transformed_features)
    return [model["id_to_class"][str(int(prediction))] for prediction in predictions]


def output_fn(prediction: list[str], accept: str) -> tuple[str, str]:
    """Serialize predictions for the endpoint response.

    Args:
        prediction: Predicted class names.
        accept: Requested response MIME type.

    Returns:
        Response body and MIME type.

    Raises:
        ValueError: If the requested response MIME type is unsupported.
    """
    normalized_accept = accept.split(";")[0].strip().lower()
    if normalized_accept in {"application/json", "*/*"}:
        return json.dumps({"predictions": prediction}), "application/json"
    if normalized_accept == "text/csv":
        return "\n".join(map(str, prediction)), "text/csv"

    raise ValueError(f"Unsupported accept type: {accept}")
