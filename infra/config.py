from __future__ import annotations

from dataclasses import dataclass

import pulumi


@dataclass(frozen=True, slots=True)
class InfrastructureConfig:
    """Pulumi configuration for the SageMaker demo.

    Args:
        project_name: Short project identifier used in resource names and tags.
        environment: Pulumi stack name, for example ``dev``.
        artifacts_prefix: S3 prefix for project artifacts.
        raw_data_key: S3 key for the raw dataset.
        force_destroy_bucket: Whether Pulumi may delete a non-empty bucket.
        tags: Common AWS resource tags.
    """

    project_name: str
    pipeline_name: str
    environment: str
    artifacts_prefix: str
    raw_data_key: str
    force_destroy_bucket: bool
    model_package_group_name: str
    tags: dict[str, str]


def load_config() -> InfrastructureConfig:
    """Load typed project configuration from Pulumi config.

    Returns:
        Parsed infrastructure configuration.
    """
    config = pulumi.Config()
    stack = pulumi.get_stack()
    project_name = config.get("projectName") or "sagemaker-demo"

    return InfrastructureConfig(
        project_name=project_name,
        environment=stack,
        artifacts_prefix=config.get("artifactsPrefix") or "sagemaker-demo",
        raw_data_key=config.get("rawDataKey") or "sagemaker-demo/raw/penguins.csv",
        pipeline_name=config.get("pipelineName") or "PenguinTrainingPipeline",
        force_destroy_bucket=config.get_bool("forceDestroyBucket") or False,
        model_package_group_name=(
            config.get("modelPackageGroupName") or "PenguinClassifierPackageGroup"
        ),
        tags={
            "Project": project_name,
            "Environment": stack,
            "ManagedBy": "Pulumi",
        },
    )
