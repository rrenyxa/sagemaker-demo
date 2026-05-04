import json
import os
import pathlib
import tarfile
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def _safe_extract_archive(archive: tarfile.TarFile, destination: str) -> None:
    """Extract a tar archive while blocking path traversal members.

    Parameters:
        archive: Open tar archive to extract.
        destination: Directory where archive members should be extracted.

    Returns:
        None.

    Raises:
        ValueError: If an archive member resolves outside ``destination``.
    """
    destination_path = os.path.abspath(destination)
    for member in archive.getmembers():
        member_path = os.path.abspath(os.path.join(destination_path, member.name))
        if os.path.commonpath([destination_path, member_path]) != destination_path:
            raise ValueError(f"Unsafe archive member path: {member.name}")
    archive.extractall(destination_path)


def _resolve_model_path(model_dir: str) -> str:
    """Return the local path to the trained ``model.joblib`` artifact.

    Parameters:
        model_dir: Directory mounted by the SageMaker Processing job.

    Returns:
        Path to the extracted or directly available ``model.joblib`` file.

    Raises:
        FileNotFoundError: If neither ``model.joblib`` nor ``model.tar.gz``
            exists in the provided directory.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    if os.path.exists(model_path):
        return model_path

    archive_path = os.path.join(model_dir, "model.tar.gz")
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Model artifact not found in {model_dir}")

    extract_dir = os.path.join(model_dir, "extracted")
    pathlib.Path(extract_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract_archive(archive, extract_dir)

    extracted_model_path = os.path.join(extract_dir, "model.joblib")
    if not os.path.exists(extracted_model_path):
        raise FileNotFoundError(f"model.joblib not found inside {archive_path}")
    return extracted_model_path


def evaluate() -> None:
    """Evaluate the trained classifier and write SageMaker metrics JSON.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the model artifact or test dataset is missing.
    """
    if os.path.exists("/opt/ml/processing/model"):
        model_path = "/opt/ml/processing/model/model.joblib"
        test_path = "/opt/ml/processing/test/test.csv"
        output_dir = "/opt/ml/processing/evaluation"
    else:
        model_path = "data/model/model.joblib"
        test_path = "data/test/test.csv"
        output_dir = "data/evaluation"

    if not os.path.exists(model_path):
        model_path = _resolve_model_path(os.path.dirname(model_path))
    model = joblib.load(model_path)

    df = pd.read_csv(test_path, header=None)
    y_test = df.iloc[:, 0]
    X_test = df.iloc[:, 1:]

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"Final Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    report_dict: dict[str, Any] = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": float(acc),
                "standard_deviation": "NaN",
            },
        },
    }

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation.json")

    with open(evaluation_path, "w", encoding="utf-8") as file:
        json.dump(report_dict, file)

    print(f"Evaluation report saved to {evaluation_path}")


if __name__ == "__main__":
    evaluate()
