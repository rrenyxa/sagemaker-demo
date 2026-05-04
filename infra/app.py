from __future__ import annotations

import pulumi

from infra.config import load_config
from infra.storage import create_storage
from infra.iam import create_sagemaker_role
from infra.sagemaker import create_sagemaker_resources


def deploy() -> None:
    """Provision and export infrastructure resources.

    Returns:
        None.
    """
    config = load_config()
    storage = create_storage(config)
    sagemaker_role = create_sagemaker_role(bucket=storage.bucket,
                                           tags=config.tags)
    sagemaker_resources = create_sagemaker_resources(config=config)

    pulumi.export("sagemaker_bucket", storage.bucket_name)
    pulumi.export("sagemaker_input_data", storage.raw_data_uri)
    pulumi.export("sagemaker_role_arn", sagemaker_role.role_arn)
    pulumi.export("sagemaker_model_package_group", 
                  sagemaker_resources.model_package_group_name)
    pulumi.export("sagemaker_pipeline_name", config.pipeline_name)
    pulumi.export("sagemaker_artifacts_prefix", config.artifacts_prefix)
