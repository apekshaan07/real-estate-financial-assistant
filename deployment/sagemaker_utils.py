"""Shared helpers for SageMaker deployments."""

from __future__ import annotations

import os
from pathlib import Path

import sagemaker


def upload_artifact_to_s3(artifact_path: Path, key_prefix: str, session: sagemaker.Session) -> str:
    """Upload a local model artifact to S3 and return the s3:// URI."""
    bucket = os.getenv("SAGEMAKER_BUCKET") or session.default_bucket()
    print(f"Uploading {artifact_path.name} to s3://{bucket}/{key_prefix}/")
    s3_uri = session.upload_data(
        path=str(artifact_path),
        bucket=bucket,
        key_prefix=key_prefix,
    )
    print(f"Model artifact uploaded to: {s3_uri}")
    return s3_uri
