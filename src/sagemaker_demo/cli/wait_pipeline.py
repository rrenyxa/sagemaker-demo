from __future__ import annotations

import os
import time
from typing import Any, Final

import boto3
from dotenv import load_dotenv

SUCCEEDED_STATUS: Final[str] = "Succeeded"
FAILED_STATUSES: Final[set[str]] = {"Failed", "Stopped"}
DEFAULT_POLL_SECONDS: Final[int] = 60
DEFAULT_TIMEOUT_SECONDS: Final[int] = 7_200


def get_required_env(name: str) -> str:
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


def get_int_env(name: str, default: int) -> int:
    """Return an integer environment variable with a default.

    Args:
        name: Environment variable name.
        default: Value used when the variable is missing.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the value cannot be parsed as an integer.
    """
    value = os.getenv(name)
    return default if not value else int(value)


def get_latest_pipeline_execution_arn(client: Any, pipeline_name: str) -> str:
    """Return the latest SageMaker Pipeline execution ARN.

    Args:
        client: SageMaker boto3 client.
        pipeline_name: SageMaker Pipeline name.

    Returns:
        Latest pipeline execution ARN.

    Raises:
        RuntimeError: If the pipeline has no executions.
    """
    response = client.list_pipeline_executions(
        PipelineName=pipeline_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    executions = response.get("PipelineExecutionSummaries", [])
    if not executions:
        raise RuntimeError(f"No executions found for pipeline: {pipeline_name}")

    return executions[0]["PipelineExecutionArn"]


def wait_for_pipeline_execution(
    client: Any,
    execution_arn: str,
    *,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Wait until a SageMaker Pipeline execution reaches a terminal state.

    Args:
        client: SageMaker boto3 client.
        execution_arn: SageMaker Pipeline execution ARN.
        poll_seconds: Seconds between status checks.
        timeout_seconds: Maximum wait time in seconds.

    Returns:
        None.

    Raises:
        TimeoutError: If the execution does not finish before timeout.
        RuntimeError: If the execution finishes with a failed status.
    """
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        response = client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn,
        )
        status = response["PipelineExecutionStatus"]
        print(f"Pipeline execution status: {status}")

        if status == SUCCEEDED_STATUS:
            return
        if status in FAILED_STATUSES:
            failure_reason = response.get("FailureReason", "No failure reason returned.")
            raise RuntimeError(
                f"Pipeline execution failed with status {status}: {failure_reason}"
            )

        time.sleep(poll_seconds)

    raise TimeoutError(
        f"Timed out after {timeout_seconds} seconds waiting for {execution_arn}"
    )


def main() -> None:
    """Wait for the current or latest SageMaker Pipeline execution.

    Returns:
        None.
    """
    load_dotenv()

    region = get_required_env("AWS_DEFAULT_REGION")
    pipeline_name = get_required_env("SAGEMAKER_PIPELINE_NAME")
    poll_seconds = get_int_env("SAGEMAKER_PIPELINE_POLL_SECONDS", DEFAULT_POLL_SECONDS)
    timeout_seconds = get_int_env(
        "SAGEMAKER_PIPELINE_TIMEOUT_SECONDS",
        DEFAULT_TIMEOUT_SECONDS,
    )

    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE") or None,
        region_name=region,
    )
    client = session.client("sagemaker")

    execution_arn = os.getenv("SAGEMAKER_PIPELINE_EXECUTION_ARN") or (
        get_latest_pipeline_execution_arn(client, pipeline_name)
    )

    print(f"Waiting for pipeline execution: {execution_arn}")
    wait_for_pipeline_execution(
        client,
        execution_arn,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    print("Pipeline execution succeeded.")


if __name__ == "__main__":
    main()
