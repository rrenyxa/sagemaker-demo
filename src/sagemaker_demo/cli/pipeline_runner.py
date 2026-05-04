from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from sagemaker_demo.pipeline import get_pipeline


def _get_required_env(name: str) -> str:
    """Return a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Environment variable value.

    Raises:
        RuntimeError: If the variable is missing or empty.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_optional_float_env(name: str) -> Optional[float]:
    """Parse an optional float environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Parsed float value, or ``None`` when the variable is not configured.

    Raises:
        ValueError: If the value cannot be parsed as a float.
    """
    value = os.getenv(name)
    return None if not value else float(value)


def _get_optional_int_env(name: str) -> Optional[int]:
    """Parse an optional integer environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Parsed integer value, or ``None`` when the variable is not configured.

    Raises:
        ValueError: If the value cannot be parsed as an integer.
    """
    value = os.getenv(name)
    return None if not value else int(value)


def main() -> None:
    """Create or update the SageMaker Pipeline and start one execution.

    Returns:
        None.
    """
    load_dotenv()

    role_arn = _get_required_env("SAGEMAKER_ROLE_ARN")
    region = _get_required_env("AWS_DEFAULT_REGION")
    bucket = _get_required_env("SAGEMAKER_BUCKET")

    pipeline = get_pipeline(
        role=role_arn,
        region=region,
        default_bucket=bucket,
        pipeline_name=os.getenv("SAGEMAKER_PIPELINE_NAME", "PenguinTrainingPipeline"),
        model_package_group_name=os.getenv(
            "SAGEMAKER_MODEL_PACKAGE_GROUP",
            "PenguinClassifierPackageGroup",
        ),
        pipeline_artifacts_prefix=os.getenv(
            "SAGEMAKER_ARTIFACTS_PREFIX",
            "sagemaker-demo/pipelines",
        ),
    )

    parameters: dict[str, str | int | float] = {}
    if input_data := os.getenv("SAGEMAKER_INPUT_DATA"):
        parameters["InputData"] = input_data
    if accuracy_threshold := _get_optional_float_env("SAGEMAKER_ACCURACY_THRESHOLD"):
        parameters["AccuracyThreshold"] = accuracy_threshold
    if max_depth := _get_optional_int_env("SAGEMAKER_MAX_DEPTH"):
        parameters["MaxDepth"] = max_depth
    if n_estimators := _get_optional_int_env("SAGEMAKER_N_ESTIMATORS"):
        parameters["NEstimators"] = n_estimators
    if processing_instance_type := os.getenv("SAGEMAKER_PROCESSING_INSTANCE_TYPE"):
        parameters["ProcessingInstanceType"] = processing_instance_type
    if training_instance_type := os.getenv("SAGEMAKER_TRAINING_INSTANCE_TYPE"):
        parameters["TrainingInstanceType"] = training_instance_type
    if model_approval_status := os.getenv("SAGEMAKER_MODEL_APPROVAL_STATUS"):
        parameters["ModelApprovalStatus"] = model_approval_status


    print("Upserting pipeline to SageMaker...")
    pipeline.upsert(role_arn=role_arn)

    print("Starting pipeline execution...")
    execution = pipeline.start(parameters=parameters)
    print(f"Execution started. ARN: {execution.arn}")


if __name__ == "__main__":
    main()
