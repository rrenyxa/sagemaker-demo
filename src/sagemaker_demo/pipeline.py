from __future__ import annotations

from typing import Optional, Sequence

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

FRAMEWORK_VERSION = "1.2-1"
DEFAULT_INFERENCE_INSTANCES = ("ml.m5.xlarge",)
DEFAULT_TRANSFORM_INSTANCES = ("ml.m5.xlarge",)
PROCESSING_INPUT_DIR = "/opt/ml/processing/input"
PROCESSING_TRAIN_DIR = "/opt/ml/processing/train"
PROCESSING_VALIDATION_DIR = "/opt/ml/processing/validation"
PROCESSING_TEST_DIR = "/opt/ml/processing/test"
PROCESSING_EVALUATION_DIR = "/opt/ml/processing/evaluation"
PROCESSING_MODEL_DIR = "/opt/ml/processing/model"
PROCESSING_PREPROCESSOR_DIR = "/opt/ml/processing/preprocessor"
PROCESSING_METADATA_DIR = "/opt/ml/processing/metadata"
PROCESSING_DEPLOYABLE_MODEL_DIR = "/opt/ml/processing/deployable_model"
PROCESSING_SOURCE_DIR = "/opt/ml/processing/source"


def get_pipeline(
    *,
    region: Optional[str] = None,
    role: Optional[str] = None,
    default_bucket: Optional[str] = None,
    pipeline_name: str = "PenguinTrainingPipeline",
    model_package_group_name: str = "PenguinClassifierPackageGroup",
    pipeline_artifacts_prefix: str = "sagemaker-demo/pipelines",
    supported_inference_instances: Sequence[str] = DEFAULT_INFERENCE_INSTANCES,
    supported_transform_instances: Sequence[str] = DEFAULT_TRANSFORM_INSTANCES,
) -> Pipeline:
    """Create the SageMaker Pipeline definition for the penguins classifier.

    Parameters:
        region: AWS region for the SageMaker session. Uses the active boto3
            configuration when omitted.
        role: IAM execution role ARN used by SageMaker jobs. When omitted,
            the SDK tries to infer it from the current SageMaker environment.
        default_bucket: S3 bucket for pipeline artifacts. SageMaker creates or
            resolves the default bucket when omitted.
        pipeline_name: Name of the SageMaker Pipeline resource.
        model_package_group_name: Model Registry package group for approved
            model versions.
        pipeline_artifacts_prefix: S3 prefix for SDK uploads and pipeline job
            outputs inside ``default_bucket``.
        supported_inference_instances: Instance types declared as supported for
            future real-time endpoint deployments from the Model Registry.
        supported_transform_instances: Instance types declared as supported for
            future batch transform jobs from the Model Registry.

    Returns:
        A SageMaker ``Pipeline`` object ready for ``upsert`` and ``start``.

    Examples:
        >>> pipeline = get_pipeline(role="arn:aws:iam::123456789012:role/SageMakerRole")
        >>> pipeline.upsert(role_arn="arn:aws:iam::123456789012:role/SageMakerRole")
        >>> execution = pipeline.start()
    """
    boto_session = boto3.Session(region_name=region)
    pipeline_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=default_bucket,
        default_bucket_prefix=f"{pipeline_artifacts_prefix}/{pipeline_name}/sdk",
    )
    bucket = default_bucket or pipeline_session.default_bucket()
    role_arn = role or sagemaker.get_execution_role(sagemaker_session=pipeline_session)
    code_location = f"s3://{bucket}/{pipeline_artifacts_prefix}/{pipeline_name}/code/train"
    inference_image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=boto_session.region_name,
        version=FRAMEWORK_VERSION,
        py_version="py3",
        instance_type="ml.m5.large",
    )
    execution_artifacts_prefix = Join(
        on="/",
        values=[
            "s3:/",
            bucket,
            pipeline_artifacts_prefix,
            pipeline_name,
            "executions",
            ExecutionVariables.PIPELINE_EXECUTION_ID,
        ],
    )

    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{bucket}/sagemaker-demo/raw/penguins.csv",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.75)
    test_split_ratio = ParameterFloat(name="TestSplitRatio", default_value=0.2)
    val_split_ratio = ParameterFloat(name="ValidationSplitRatio", default_value=0.2)
    n_estimators = ParameterInteger(name="NEstimators", default_value=100)
    max_depth = ParameterInteger(name="MaxDepth", default_value=5)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.t3.medium",
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.t3.medium",
    )

    preprocess_processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        instance_type=processing_instance_type,
        instance_count=1,
        role=role_arn,
        base_job_name="penguin-preprocess",
        sagemaker_session=pipeline_session,
    )
    step_process = ProcessingStep(
        name="PreprocessPenguinData",
        processor=preprocess_processor,
        inputs=[
            ProcessingInput(
                input_name="raw_data",
                source=input_data,
                destination=PROCESSING_INPUT_DIR,
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source=PROCESSING_TRAIN_DIR,
                destination=Join(
                    on="/",
                    values=[execution_artifacts_prefix, "preprocess", "train"],
                ),
            ),
            ProcessingOutput(
                output_name="validation",
                source=PROCESSING_VALIDATION_DIR,
                destination=Join(
                    on="/",
                    values=[execution_artifacts_prefix, "preprocess", "validation"],
                ),
            ),
            ProcessingOutput(
                output_name="test",
                source=PROCESSING_TEST_DIR,
                destination=Join(
                    on="/",
                    values=[execution_artifacts_prefix, "preprocess", "test"],
                ),
            ),
            ProcessingOutput(
                output_name="preprocessor",
                source=PROCESSING_PREPROCESSOR_DIR,
                destination=Join(
                    on="/",
                    values=[execution_artifacts_prefix, "preprocess", "preprocessor"],
                ),
            ),
            ProcessingOutput(
                output_name="metadata",
                source=PROCESSING_METADATA_DIR,
                destination=Join(
                    on="/",
                    values=[execution_artifacts_prefix, "preprocess", "metadata"],
                ),
            ),
        ],
        code="src/sagemaker_demo/preprocess.py",
        job_arguments=[
            "--test-split-ratio",
            Join(on="", values=[test_split_ratio]),
            "--val-split-ratio",
            Join(on="", values=[val_split_ratio]),
        ],
    )

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src/sagemaker_demo",
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        instance_type=training_instance_type,
        instance_count=1,
        role=role_arn,
        base_job_name="penguin-train",
        hyperparameters={
            "n-estimators": n_estimators,
            "max-depth": max_depth,
        },
        output_path=Join(on="/", values=[execution_artifacts_prefix, "train"]),
        code_location=code_location,
        sagemaker_session=pipeline_session,
        use_spot_instances=True,
        max_run=600,
        max_wait=1200,
    )
    step_train = TrainingStep(
        name="TrainPenguinModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),  
        },
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluate_processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        instance_type=processing_instance_type,
        instance_count=1,
        role=role_arn,
        base_job_name="penguin-evaluate",
        sagemaker_session=pipeline_session,
    )
    step_evaluate = ProcessingStep(
        name="EvaluatePenguinModel",
        processor=evaluate_processor,
        inputs=[
            ProcessingInput(
                input_name="model",
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination=PROCESSING_MODEL_DIR,
            ),
            ProcessingInput(
                input_name="test",
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination=PROCESSING_TEST_DIR,
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source=PROCESSING_EVALUATION_DIR,
                destination=Join(
                    on="/",
                    values=[execution_artifacts_prefix, "evaluate", "evaluation"],
                ),
            )
        ],
        code="src/sagemaker_demo/evaluate.py",
        property_files=[evaluation_report],
    )

    step_prepare_model = ProcessingStep(
        name="PrepareDeployableModel",
        processor=evaluate_processor,
        inputs=[
            ProcessingInput(
                input_name="model",
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination=PROCESSING_MODEL_DIR,
            ),
            ProcessingInput(
                input_name="preprocessor",
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "preprocessor"
                ].S3Output.S3Uri,
                destination=PROCESSING_PREPROCESSOR_DIR,
            ),
            ProcessingInput(
                input_name="metadata",
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "metadata"
                ].S3Output.S3Uri,
                destination=PROCESSING_METADATA_DIR,
            ),
            ProcessingInput(
                input_name="source",
                source="src/sagemaker_demo/inference.py",
                destination=PROCESSING_SOURCE_DIR,
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="deployable_model",
                source=PROCESSING_DEPLOYABLE_MODEL_DIR,
                destination=Join(
                    on="/",
                    values=[execution_artifacts_prefix, "prepare_model"],
                ),
            )
        ],
        code="src/sagemaker_demo/prepare_model.py",
    )
    deployable_model_data = Join(
        on="/",
        values=[
            step_prepare_model.properties.ProcessingOutputConfig.Outputs[
                "deployable_model"
            ].S3Output.S3Uri,
            "model.tar.gz",
        ],
    )
    deployable_model = Model(
        model_data=deployable_model_data,
        image_uri=inference_image_uri,
        role=role_arn,
        env={
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
        },
        sagemaker_session=pipeline_session,
    )

    evaluation_s3_uri = Join(
        on="/",
        values=[
            step_evaluate.properties.ProcessingOutputConfig.Outputs[
                "evaluation"
            ].S3Output.S3Uri,
            "evaluation.json",
        ],
    )
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_s3_uri,
            content_type="application/json",
        )
    )
    step_register = RegisterModel(
        name="RegisterPenguinModel",
        model=deployable_model,
        content_types=["application/json", "text/csv"],
        response_types=["application/json", "text/csv"],
        inference_instances=list(supported_inference_instances),
        transform_instances=list(supported_transform_instances),
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_fail = FailStep(
        name="FailOnLowAccuracy",
        error_message="Model accuracy is below the configured threshold.",
    )
    step_condition = ConditionStep(
        name="CheckAccuracyThreshold",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="multiclass_classification_metrics.accuracy.value",
                ),
                right=accuracy_threshold,
            )
        ],
        if_steps=[step_prepare_model, step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            model_approval_status,
            accuracy_threshold,
            test_split_ratio,
            val_split_ratio,
            n_estimators,
            max_depth,
            processing_instance_type,
            training_instance_type,
        ],
        steps=[
            step_process,
            step_train,
            step_evaluate,
            step_condition,
        ],
        sagemaker_session=pipeline_session,
    )
