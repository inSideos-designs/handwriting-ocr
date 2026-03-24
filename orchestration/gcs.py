import os
import tempfile
from google.cloud import storage


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def parse_gcs_path(path: str) -> tuple[str, str]:
    path = path.removeprefix("gs://")
    parts = path.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    blob = blob.rstrip("/")
    return bucket, blob


def get_client() -> storage.Client:
    return storage.Client()


def download_file(gcs_path: str, local_path: str) -> str:
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def upload_file(local_path: str, gcs_path: str) -> str:
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return gcs_path


def download_directory(gcs_prefix: str, local_dir: str) -> str:
    bucket_name, prefix = parse_gcs_path(gcs_prefix)
    if not prefix.endswith("/"):
        prefix += "/"
    client = get_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        relative = blob.name[len(prefix):]
        if not relative:
            continue
        local_path = os.path.join(local_dir, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
    return local_dir


def list_blobs(gcs_prefix: str) -> list[str]:
    bucket_name, prefix = parse_gcs_path(gcs_prefix)
    client = get_client()
    bucket = client.bucket(bucket_name)
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
