from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

DEFAULT_SMOKE_TEST_PAYLOAD = [
    {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750,
        "sex": "male",
    }
]


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
    """
    value = os.getenv(name)
    return default if not value else int(value)


def make_unique_name(prefix: str) -> str:
    """Build a SageMaker-safe resource name with a UTC timestamp suffix.

    Args:
        prefix: Human-readable resource name prefix.

    Returns:
        Unique resource name.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_prefix = "".join(char if char.isalnum() or char == "-" else "-" for char in prefix)
    return f"{safe_prefix}-{timestamp}"[:63].strip("-")


def get_latest_approved_model_package_arn(
    client: Any,
    model_package_group_name: str,
) -> str:
    """Return the latest approved model package ARN from Model Registry.

    Args:
        client: SageMaker boto3 client.
        model_package_group_name: SageMaker Model Package Group name.

    Returns:
        Latest approved Model Package ARN.

    Raises:
        RuntimeError: If the package group has no approved versions.
    """
    paginator = client.get_paginator("list_model_packages")
    for page in paginator.paginate(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    ):
        packages = page.get("ModelPackageSummaryList", [])
        if packages:
            return packages[0]["ModelPackageArn"]

    raise RuntimeError(
        f"No approved model packages found in group: {model_package_group_name}"
    )


def create_model_from_package(
    client: Any,
    *,
    model_name: str,
    model_package_arn: str,
    role_arn: str,
) -> None:
    """Create a SageMaker Model resource from a Model Registry package.

    Args:
        client: SageMaker boto3 client.
        model_name: Name for the new SageMaker Model resource.
        model_package_arn: Approved Model Package ARN.
        role_arn: SageMaker execution role ARN.

    Returns:
        None.
    """
    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        Containers=[{"ModelPackageName": model_package_arn}],
    )


def create_endpoint_config(
    client: Any,
    *,
    endpoint_config_name: str,
    model_name: str,
    instance_type: str,
    initial_instance_count: int,
    variant_name: str,
) -> None:
    """Create a SageMaker EndpointConfig for one production variant.

    Args:
        client: SageMaker boto3 client.
        endpoint_config_name: EndpointConfig resource name.
        model_name: SageMaker Model resource name.
        instance_type: Endpoint instance type.
        initial_instance_count: Number of endpoint instances.
        variant_name: Production variant name.

    Returns:
        None.
    """
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": variant_name,
                "ModelName": model_name,
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )


def endpoint_exists(client: Any, endpoint_name: str) -> bool:
    """Return whether a SageMaker endpoint exists.

    Args:
        client: SageMaker boto3 client.
        endpoint_name: Endpoint name.

    Returns:
        ``True`` when the endpoint exists, otherwise ``False``.

    Raises:
        ClientError: If AWS returns an unexpected error.
    """
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def create_or_update_endpoint(
    client: Any,
    *,
    endpoint_name: str,
    endpoint_config_name: str,
) -> str:
    """Create a new endpoint or update an existing endpoint.

    Args:
        client: SageMaker boto3 client.
        endpoint_name: Endpoint name.
        endpoint_config_name: EndpointConfig name to deploy.

    Returns:
        Deployment action: ``created`` or ``updated``.
    """
    if endpoint_exists(client, endpoint_name):
        client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        return "updated"

    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    return "created"


def wait_for_endpoint(
    client: Any,
    endpoint_name: str,
    *,
    poll_seconds: int = 30,
) -> None:
    """Wait until an endpoint reaches ``InService``.

    Args:
        client: SageMaker boto3 client.
        endpoint_name: Endpoint name.
        poll_seconds: Seconds between status checks.

    Returns:
        None.

    Raises:
        RuntimeError: If endpoint creation or update fails.
    """
    while True:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(f"Endpoint status: {status}")

        if status == "InService":
            return
        if status in {"Failed", "OutOfService"}:
            reason = response.get("FailureReason", "No failure reason returned.")
            raise RuntimeError(f"Endpoint failed with status {status}: {reason}")

        time.sleep(poll_seconds)


def run_smoke_test(runtime_client: Any, endpoint_name: str) -> dict[str, Any]:
    """Invoke the endpoint with one representative request.

    Args:
        runtime_client: SageMaker Runtime boto3 client.
        endpoint_name: Endpoint name.

    Returns:
        Parsed JSON response from the endpoint.
    """
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(DEFAULT_SMOKE_TEST_PAYLOAD),
    )
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def main() -> None:
    """Deploy the latest approved model package to a SageMaker endpoint.

    Returns:
        None.
    """
    load_dotenv()

    region = get_required_env("AWS_DEFAULT_REGION")
    role_arn = get_required_env("SAGEMAKER_ROLE_ARN")
    model_package_group_name = get_required_env("SAGEMAKER_MODEL_PACKAGE_GROUP")
    endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "penguin-classifier")
    instance_type = os.getenv("SAGEMAKER_ENDPOINT_INSTANCE_TYPE", "ml.m5.large")
    initial_instance_count = get_int_env("SAGEMAKER_ENDPOINT_INITIAL_INSTANCE_COUNT", 1)
    variant_name = os.getenv("SAGEMAKER_ENDPOINT_VARIANT_NAME", "variant-1")

    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE") or None,
        region_name=region,
    )
    client = session.client("sagemaker")
    runtime_client = session.client("sagemaker-runtime")

    model_package_arn = get_latest_approved_model_package_arn(
        client,
        model_package_group_name,
    )
    model_name = make_unique_name(f"{endpoint_name}-model")
    endpoint_config_name = make_unique_name(f"{endpoint_name}-config")

    print(f"Using approved model package: {model_package_arn}")
    print(f"Creating model: {model_name}")
    create_model_from_package(
        client,
        model_name=model_name,
        model_package_arn=model_package_arn,
        role_arn=role_arn,
    )

    print(f"Creating endpoint config: {endpoint_config_name}")
    create_endpoint_config(
        client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        variant_name=variant_name,
    )

    action = create_or_update_endpoint(
        client,
        endpoint_name=endpoint_name,
        endpoint_config_name=endpoint_config_name,
    )
    print(f"Endpoint {action}: {endpoint_name}")

    wait_for_endpoint(client, endpoint_name)
    prediction = run_smoke_test(runtime_client, endpoint_name)
    print(f"Smoke test response: {json.dumps(prediction, indent=2)}")


if __name__ == "__main__":
    main()
