from __future__ import annotations

from dataclasses import dataclass

import pulumi
import pulumi_aws as aws

from infra.config import InfrastructureConfig

@dataclass(frozen=True, slots=True)
class SageMakerResources:
    """SageMaker registry resources.

    Args:
        model_package_group: SageMaker Model Package Group.
        model_package_group_name: Resolved Model Package Group name.
    """

    model_package_group: aws.sagemaker.ModelPackageGroup
    model_package_group_name: pulumi.Output[str]


def create_sagemaker_resources(config:InfrastructureConfig):
    """Create SageMaker resources managed as infrastructure.

    Args:
        config: Typed infrastructure configuration.

    Returns:
        Created SageMaker registry resources.
    """

    model_package_group = aws.sagemaker.ModelPackageGroup(
        resource_name="model-package-group",
        model_package_group_name=config.model_package_group_name,
        model_package_group_description="Model versions for the penguin classifier SageMaker pipeline.",
        tags=config.tags
    )

    return SageMakerResources(
        model_package_group=model_package_group,
        model_package_group_name=model_package_group.model_package_group_name
        )