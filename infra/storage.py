from __future__ import annotations

from dataclasses import dataclass

import pulumi
import pulumi_aws as aws
from infra.config import InfrastructureConfig

@dataclass(frozen=True, slots=True)
class StorageResources:
    """S3 resources for SageMaker data and artifacts.

    Args:
        bucket: S3 bucket resource.
        bucket_name: Resolved S3 bucket name.
        raw_data_uri: S3 URI consumed by the SageMaker Pipeline.
    """

    bucket: aws.s3.Bucket
    bucket_name: pulumi.Output[str]
    raw_data_uri: pulumi.Output[str]

def create_storage(config:InfrastructureConfig) ->StorageResources:
    """Create an encrypted private S3 bucket for SageMaker.

    Args:
        config: Typed infrastructure configuration.

    Returns:
        Created storage resources.
    """

    bucket = aws.s3.Bucket(
        resource_name=f"{config.project_name}-artifacts",
        force_destroy=config.force_destroy_bucket,
        tags=config.tags
    )

    aws.s3.BucketPublicAccessBlock(
        "artifacts-bucket-public-access-block",
        bucket=bucket.id,
        block_public_acls=True,
        block_public_policy=True,
        ignore_public_acls=True,
        restrict_public_buckets=True,
    )

    aws.s3.BucketServerSideEncryptionConfiguration(
        "artifacts-bucket-encryption",
        bucket=bucket.id,
        rules=[
            aws.s3.BucketServerSideEncryptionConfigurationRuleArgs(
                apply_server_side_encryption_by_default=(
                    aws.s3.BucketServerSideEncryptionConfigurationRuleApplyServerSideEncryptionByDefaultArgs(
                        sse_algorithm="AES256",
                    )
                ),
            )
        ],
    )

    raw_data_uri = pulumi.Output.concat("s3://", bucket.bucket, "/", config.raw_data_key)

    return StorageResources(
        bucket=bucket,
        bucket_name=bucket.bucket,
        raw_data_uri=raw_data_uri,
    )