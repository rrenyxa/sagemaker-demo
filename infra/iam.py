from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pulumi
import pulumi_aws as aws


@dataclass(frozen=True, slots=True)
class SageMakerRoleResources:
    """IAM resources required by SageMaker jobs and endpoints.

    Args:
        role: IAM role assumed by SageMaker.
        role_arn: Resolved IAM role ARN.
    """

    role: aws.iam.Role
    role_arn: pulumi.Output[str]


def to_json(policy: dict[str, Any]) -> str:
    """Serialize an IAM policy document to compact JSON.

    Args:
        policy: IAM policy document.

    Returns:
        Compact JSON string.
    """
    return json.dumps(policy, separators=(",", ":"), sort_keys=True)


def create_sagemaker_role(bucket: aws.s3.Bucket, tags: dict[str, str]) -> SageMakerRoleResources:
    """Create a SageMaker execution role with S3, logs, ECR, and SageMaker access.

    Args:
        bucket: Project artifacts bucket.
        tags: Common AWS tags.

    Returns:
        Created SageMaker IAM role resources.
    """
    role = aws.iam.Role(
        "sagemaker-execution-role",
        assume_role_policy=to_json(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        ),
        tags=tags,
    )

    aws.iam.RolePolicy(
        "sagemaker-execution-policy",
        role=role.id,
        policy=pulumi.Output.all(bucket.arn, role.arn).apply(
            lambda args: to_json(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "ProjectBucketReadWrite",
                            "Effect": "Allow",
                            "Action": [
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:DeleteObject",
                                "s3:AbortMultipartUpload",
                                "s3:ListMultipartUploadParts",
                            ],
                            "Resource": f"{args[0]}/*",
                        },
                        {
                            "Sid": "ProjectBucketList",
                            "Effect": "Allow",
                            "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                            "Resource": args[0],
                        },
                        {
                            "Sid": "SageMakerAccess",
                            "Effect": "Allow",
                            "Action": "sagemaker:*",
                            "Resource": "*",
                        },
                        {
                            "Sid": "EcrPull",
                            "Effect": "Allow",
                            "Action": [
                                "ecr:GetAuthorizationToken",
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:BatchGetImage",
                            ],
                            "Resource": "*",
                        },
                        {
                            "Sid": "CloudWatchLogs",
                            "Effect": "Allow",
                            "Action": [
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:DescribeLogStreams",
                                "logs:PutLogEvents",
                                "cloudwatch:PutMetricData",
                            ],
                            "Resource": "*",
                        },
                        {
                            "Sid": "PassSelfToSageMaker",
                            "Effect": "Allow",
                            "Action": "iam:PassRole",
                            "Resource": args[1],
                            "Condition": {
                                "StringEquals": {
                                    "iam:PassedToService": "sagemaker.amazonaws.com"
                                }
                            },
                        },
                    ],
                }
            )
        ),
    )

    return SageMakerRoleResources(role=role, role_arn=role.arn)
