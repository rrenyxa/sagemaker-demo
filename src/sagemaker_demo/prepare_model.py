from __future__ import annotations

import os
import pathlib
import shutil
import tarfile


MODEL_INPUT_DIR = "/opt/ml/processing/model"
PREPROCESSOR_INPUT_DIR = "/opt/ml/processing/preprocessor"
METADATA_INPUT_DIR = "/opt/ml/processing/metadata"
SOURCE_INPUT_DIR = "/opt/ml/processing/source"
OUTPUT_DIR = "/opt/ml/processing/deployable_model"


def _safe_extract_archive(archive: tarfile.TarFile, destination: str) -> None:
    """Extract a tar archive while blocking path traversal members.

    Args:
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


def _copy_required_file(source_path: str, destination_path: str) -> None:
    """Copy a required file and fail with a clear error if it is missing.

    Args:
        source_path: Existing source file path.
        destination_path: Target file path.

    Returns:
        None.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Required artifact not found: {source_path}")
    pathlib.Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def prepare_model() -> None:
    """Build a deployable SageMaker model artifact.

    The output archive contains the trained classifier, fitted preprocessor, and
    optional inference entrypoint when ``inference.py`` is supplied as an input.

    Returns:
        None.
    """
    staging_dir = os.path.join(OUTPUT_DIR, "staging")
    pathlib.Path(staging_dir).mkdir(parents=True, exist_ok=True)

    model_archive_path = os.path.join(MODEL_INPUT_DIR, "model.tar.gz")
    with tarfile.open(model_archive_path, "r:gz") as archive:
        _safe_extract_archive(archive, staging_dir)

    _copy_required_file(
        source_path=os.path.join(PREPROCESSOR_INPUT_DIR, "preprocessor.joblib"),
        destination_path=os.path.join(staging_dir, "preprocessor.joblib"),
    )
    _copy_required_file(
        source_path=os.path.join(METADATA_INPUT_DIR, "label_mapping.json"),
        destination_path=os.path.join(staging_dir, "label_mapping.json"),
    )

    inference_path = os.path.join(SOURCE_INPUT_DIR, "inference.py")
    if os.path.exists(inference_path):
        inference_destination = os.path.join(staging_dir, "code", "inference.py")
        pathlib.Path(inference_destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inference_path, inference_destination)

    deployable_archive_path = os.path.join(OUTPUT_DIR, "model.tar.gz")
    with tarfile.open(deployable_archive_path, "w:gz") as archive:
        for file_path in pathlib.Path(staging_dir).iterdir():
            archive.add(file_path, arcname=file_path.name)

    print(f"Deployable model artifact saved to {deployable_archive_path}")


if __name__ == "__main__":
    prepare_model()
